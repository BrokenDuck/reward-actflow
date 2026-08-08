"""ActFlow-R Algorithm 2: arithmetic-mixture replay.

    theta_{t+1} <- UpdateFlow(theta_t, D+, P, w, lambda)

Each minibatch composes two sources instead of reweighting one pool:

  - U_a: lambda*n points drawn UNIFORMLY from the frozen anchor pool P
  - U_d: (1-lambda)*n points drawn from the verified-valid buffer D+ with
    probability PROPORTIONAL TO the (clipped) weight w = exp(lcb^r / eta)

and takes an UNWEIGHTED flow-matching step over U_a union U_d.

The spec's pseudocode also weights the *loss* by clip(w) on top of drawing
U_d proportional to w — doing both induces w^2 on D+, not the stated target
q_t propto w * D+. This implementation applies the tilt once, in the
proposal, and drops the loss weighting; the clip ratio G still does work,
bounding how much a single point can dominate U_d's draws.

Because U_a is drawn from a pool that never changes, this targets the
arithmetic mixture p_{theta_{t+1}} ~= lambda*p_P + (1-lambda)*q_t, in contrast
to the geometric tilt p_theta * e^{r/beta} the other three trainers target,
which cannot place mass outside supp(p_theta). lambda=0 with w==1 recovers a
plain (unweighted, D+-only) flow-matching refit.

Driven entirely by `ActFlowRLoop`, which owns P and w (Algorithm 1 lines
1-7); this trainer only implements line 8. `collect()` returns a
`MixtureExperience` with `replay=None`; `ActFlowRLoop.prepare_experience()`
fills it in before `update()` runs — a wrapped-experience dataclass, not a
buffer held on the trainer, because nothing in `FineTuningAlgorithm` validates
experience type (`collect()`/`update()` are generic in `ExperienceT`) and
diffusiongym's algorithms are deliberately built against accumulating mutable
state on the trainer itself (`core/environment.py`'s `FlowEnvironment` "owns
no policies" for the same reason).
"""

import copy
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from diffusiongym.core import FlowDynamics, RolloutRequest, RolloutStorage
from diffusiongym.trainers.base import (
    FineTuningAlgorithm,
    FineTuningContext,
    FineTuningRequirements,
)
from diffusiongym.types import DDBatch
from torch import Generator, Tensor

type Conditioning = Mapping[str, Any]


def _index_conditioning(conditioning: Conditioning, idx: Tensor) -> dict:
    """Sub-select a conditioning dict by an index tensor (repeats allowed).

    Mirrors `orw_cfm._index_conditioning` — kept local rather than imported
    since it is not part of that trainer's public surface.
    """
    result: dict[str, Any] = {}
    for k, v in conditioning.items():
        if isinstance(v, torch.Tensor):
            result[k] = v[idx]
        elif isinstance(v, list):
            result[k] = [v[i] for i in idx.tolist()]
        else:
            result[k] = v
    return result


def _concat_conditioning(parts: list[Conditioning]) -> dict[str, Any]:
    """Concatenate same-keyed conditioning dicts, mirroring `Batch.concat`."""
    if len(parts) == 1:
        return dict(parts[0])
    merged: dict[str, Any] = {}
    for key in parts[0]:
        values = [p[key] for p in parts]
        if isinstance(values[0], torch.Tensor):
            merged[key] = torch.cat(values, dim=0)
        elif isinstance(values[0], list):
            merged[key] = [item for v in values for item in v]
        else:
            merged[key] = values[0]
    return merged


def _clip_and_normalize_weights(weights: Tensor, *, clip_ratio: float) -> Tensor:
    """Clip `w` to `[1/G, G]` (Algorithm 2's clip), then normalise to a
    proposal distribution over D+ — sum to 1, not mean 1. Mean-1 normalisation
    is for a *loss* weight; this tilt is applied once, in the proposal (see
    the module docstring), so what's needed here is a categorical
    distribution, which sums to 1.
    """
    if weights.numel() == 0:
        return weights
    clipped = weights.clamp(1.0 / clip_ratio, clip_ratio)
    return clipped / clipped.sum().clamp_min(1e-12)


@dataclass(frozen=True)
class ReplaySource[StateT]:
    """`P`, `D+` and `w`, as of the iteration `MixtureReplay.update()` runs.

    Owned and constructed by `ActFlowRLoop` — see the module docstring.
    `weights` is raw (unclipped, unnormalised) `w` on `data`, one per row.
    """

    anchors: StateT
    anchor_conditioning: Conditioning
    data: StateT
    data_conditioning: Conditioning
    weights: Tensor


@dataclass
class MixtureExperience[StateT]:
    """`EndpointExperience`-shaped experience, plus the replay source.

    `collect()` returns this with `replay=None`; `ActFlowRLoop` fills it in
    (via `dataclasses.replace`) before `update()` runs.
    """

    latent: StateT
    rewards: Tensor
    valid: Tensor | None
    conditioning: Conditioning
    replay: ReplaySource[StateT] | None = None


class MixtureReplay[StateT: DDBatch, RawT](
    FineTuningAlgorithm[StateT, RawT, MixtureExperience[StateT]]
):
    """ActFlow-R's `UpdateFlow`: arithmetic-mixture replay (Algorithm 2).

    Parameters
    ----------
    anchor_frac:
        lambda — fraction of each minibatch drawn uniformly from the frozen
        anchor pool P. 0 draws only from D+ (reduces, under w==1, to a plain
        flow-matching refit); 1 draws only from P (theta never moves toward
        D+ at all).
    weight_clip:
        G — w is clipped to [1/G, G] before being turned into a sampling
        distribution over D+. G=1 clips every weight to the same value, i.e.
        uniform sampling of D+ regardless of w.
    steps_per_update:
        Gradient steps per call to `update()` (Algorithm 2's N_step). The
        same clipped/normalised proposal is reused across all of them — `w`
        is an Algorithm-1-line-7 quantity, computed once per outer iteration,
        not refit mid-inner-loop.
    batch_size:
        Minibatch size n.
    """

    def __init__(
        self,
        *,
        anchor_frac: float = 0.25,
        weight_clip: float = 20.0,
        steps_per_update: int = 50,
        batch_size: int = 64,
    ) -> None:
        if not (0.0 <= anchor_frac <= 1.0):
            raise ValueError(f"anchor_frac must be in [0, 1], got {anchor_frac}.")
        if weight_clip < 1.0:
            raise ValueError(f"weight_clip must be >= 1, got {weight_clip}.")
        if steps_per_update < 1:
            raise ValueError(f"steps_per_update must be >= 1, got {steps_per_update}.")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}.")

        self.anchor_frac = anchor_frac
        self.weight_clip = weight_clip
        self.steps_per_update = steps_per_update
        self.batch_size = batch_size

    @property
    def requirements(self) -> FineTuningRequirements:
        return FineTuningRequirements(
            needs_reference_policy=False,
            # SMC guidance (reward_actflow.smc_guidance) needs transition
            # noise to decorrelate resampled particles — see core/smc.py's
            # module docstring — so this algorithm always trains under an SDE,
            # whether or not a given run actually turns guidance on.
            needs_stochastic_rollout=True,
            rollout_storage=RolloutStorage(),
        )

    def collect(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        dynamics: FlowDynamics[StateT],
        n: int,
        time_grid: Tensor,
        conditioning: Conditioning,
        generator: Generator | None = None,
    ) -> MixtureExperience[StateT]:
        """Plain SDE rollout of the current policy.

        `ActFlowRLoop` overrides collection with SMC guidance under
        `--guidance smc` (see `reward_actflow.smc_guidance`); this is what
        `--guidance none`, and any caller outside `reward_actflow`, gets.
        """
        request = RolloutRequest(
            time_grid=time_grid,
            storage=self.requirements.rollout_storage,
            evaluate_reward=True,
        )
        rollout = context.sde_sampler.rollout(
            environment=context.environment,
            model=context.policies.rollout,
            dynamics=dynamics,
            n=n,
            conditioning=conditioning,
            request=request,
            generator=generator,
        )
        assert rollout.reward is not None
        return MixtureExperience(
            latent=rollout.terminal_latent,
            rewards=rollout.reward.rewards,
            valid=rollout.reward.valid,
            conditioning=rollout.conditioning,
        )

    def update(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
        experience: MixtureExperience[StateT],
    ) -> Mapping[str, float]:
        replay = experience.replay
        if replay is None:
            raise ValueError(
                "MixtureReplay.update() needs a ReplaySource. This algorithm "
                "is driven by ActFlowRLoop, which owns P and w and attaches "
                "them via prepare_experience() before update() runs — a bare "
                "diffusiongym loop cannot supply them."
            )

        env = context.environment
        model = context.policies.train
        opt = context.optimizer
        device = model.device

        anchors = replay.anchors.to(device)
        data = replay.data.to(device)
        n_anchors = len(anchors)
        n_data = len(data)

        raw_weights = replay.weights.to(device)
        proposal = _clip_and_normalize_weights(raw_weights, clip_ratio=self.weight_clip)

        n_anchor_draw = round(self.anchor_frac * self.batch_size)
        n_data_draw = self.batch_size - n_anchor_draw

        total_loss = 0.0
        total_anchor_frac = 0.0
        steps_taken = 0
        steps_skipped = 0

        for _ in range(self.steps_per_update):
            parts: list[StateT] = []
            cond_parts: list[Conditioning] = []
            drawn_anchor = 0

            if n_anchor_draw > 0 and n_anchors > 0:
                idx_a = torch.randint(0, n_anchors, (n_anchor_draw,), device=device)
                parts.append(anchors.index_select(idx_a))
                cond_parts.append(
                    _index_conditioning(replay.anchor_conditioning, idx_a)
                )
                drawn_anchor = n_anchor_draw

            if n_data_draw > 0 and n_data > 0:
                idx_d = torch.multinomial(proposal, n_data_draw, replacement=True)
                parts.append(data.index_select(idx_d))
                cond_parts.append(_index_conditioning(replay.data_conditioning, idx_d))

            if not parts:
                # Both sources empty for this draw (e.g. D+ still empty at
                # t=0 with anchor_frac=0) — nothing to train on this step.
                steps_skipped += 1
                continue

            x_data_b = parts[0] if len(parts) == 1 else type(anchors).concat(parts)
            cond_b = _concat_conditioning(cond_parts)

            batch = env.make_forward_batch(x_data_b, conditioning=cond_b)
            pred_v = env.predict_velocity(
                model, x_t=batch.x_t, t=batch.t, conditioning=cond_b
            )
            per_sample_loss = env.velocity_error(pred_v, batch.target_velocity)

            loss = per_sample_loss.mean()  # unweighted — see module docstring

            opt.zero_grad()
            loss.backward()
            if hasattr(model, "parameters"):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # ty: ignore[call-non-callable]
            opt.step()

            total_loss += loss.item()
            total_anchor_frac += drawn_anchor / max(len(x_data_b), 1)
            steps_taken += 1

        denom = max(steps_taken, 1)
        w_clipped_raw = raw_weights.clamp(1.0 / self.weight_clip, self.weight_clip)
        clip_fraction = (
            float((w_clipped_raw != raw_weights).float().mean().item())
            if n_data > 0
            else 0.0
        )
        ess_postclip = (
            float((1.0 / (proposal**2).sum().clamp_min(1e-12)).item())
            if n_data > 0
            else 0.0
        )

        return {
            "loss": total_loss / denom,
            "anchor_frac_actual": total_anchor_frac / denom,
            "steps_skipped": float(steps_skipped),
            "w_clip_fraction": clip_fraction,
            "ess_postclip": ess_postclip,
            "ess_frac_postclip": ess_postclip / max(n_data, 1),
        }

    def synchronize_rollout_policy(
        self,
        *,
        context: FineTuningContext[StateT, RawT],
    ) -> None:
        """Hard-copy train -> rollout every call.

        `ActFlowRLoop` runs exactly one `collect()` + one `update()` per
        outer iteration, matching base ActFlow's own fused-loop assumption
        (`explore.py`'s module docstring). Without this, the next iteration's
        `collect()` would keep drawing from `theta_0` forever — only
        `policies.train` receives gradients.
        """
        train = context.policies.train
        rollout = context.policies.rollout
        if hasattr(train, "state_dict") and hasattr(rollout, "load_state_dict"):
            rollout.load_state_dict(copy.deepcopy(train.state_dict()))
