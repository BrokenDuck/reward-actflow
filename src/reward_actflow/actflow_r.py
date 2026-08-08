"""ActFlow-R: reward-directed active flow expansion.

ActFlow-R extends base ActFlow (`explore.py`, Algorithm 1) with a task
reward `r~` and a new `UpdateFlow`:

```
D_0 = {}, E_0 = {}
P <- p_data if available, else {x ~ p_1^{theta_0}}       (frozen anchor pool)
for t = 0 ... T-1:
  1: fit validity uncertainty sigma^v_t from D_t          (as in base ActFlow)
  2: fit reward surrogate (mu^r_t, sigma^r_t) from E_t    (valid points only)
  3: a_t(x) = zeta_t*sigma^v_t(x) + (1-zeta_t)*(mu^r_t + beta^r*sigma^r_t)(x)
  4: self-generate X_{t+1} ~ argmax_q E_q[a_t] - beta*KL(q || p^theta_t)
  5: y <- v(x), x in X_{t+1};           D_{t+1} = D_t + {(x,y)}
  6: u <- r~(x), x in X+_{t+1};         E_{t+1} = E_t + {(x,u)}
  7: w(x) <- exp(lcb^r_t(x) / eta), x in D+_{t+1}
  8: theta_{t+1} <- UpdateFlow(theta_t, D+_{t+1}, P, w, lambda)
```

Two departures from base ActFlow, and why each needs what it needs:

**Line 4 is solved by SMC, not in the weights.** Base ActFlow solves line 4
"in the weights": sampling is a plain rollout, and the KL-regularized tilt is
whatever the reward-maximizing fine-tuner in line 7/8 achieves (see
`explore.py`'s module docstring). `UpdateFlow` here is `MixtureReplay`
(`trainers/mixture_replay.py`), a weighted flow-matching *regression* onto
points already in `P` and `D+` — its gradient has no `a_t` term in it at all.
So the acquisition needs a different channel into theta, and line 4 gets one:
`reward_actflow.smc_guidance` twists the SDE rollout by `exp(a_t/beta)` via
resampling (`diffusiongym.core.smc`), so `a_t` determines *which* points get
verified, rewarded, and replayed. `--guidance none` disables this and
reproduces Algorithm 2 exactly as written, with `zeta`/`beta_r` computed and
logged but inert.

**`UpdateFlow` targets an arithmetic mixture, not a geometric tilt.** See
`trainers/mixture_replay.py`'s module docstring.

Appendix extensions (A)-(E) are not implemented — only the diagnostics
(`diagnostics.py`) that say whether one is warranted. Extension (F), a soft
validity gate on `a_t`'s reward term (`rewards/acquisition.py`), *is*
implemented, off by default (`--validity_gate`): a calibration run on the toy
showed `best_reward_far` pinning exactly at a region boundary for hundreds of
iterations while `valid_rate` degraded — `mu^r` extrapolating past the edge
with no notion of validity, sending SMC guidance after a dead zone instead of
across the (locally reward-flat) corridor to the next region. That is direct
evidence for (F), not a hypothetical one of the six.
"""

import argparse
import dataclasses
import logging
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import matplotlib
import numpy as np
import torch
import yaml
from diffusiongym import FineTuningSetup

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reward_actflow.trainers  # noqa: F401  (registers mixture_replay)
from reward_actflow.diagnostics import (
    best_reward_beyond,
    cluster_count,
    coordinate_bounds,
    effective_sample_size,
    heldout_r2,
    per_cell_density_ratio,
)
from reward_actflow.explore import (
    ALGORITHMS,
    ActFlowLoop,
    ExploreConfig,
    assemble_setup,
    endpoints_of,
)
from reward_actflow.rewards.acquisition import ActFlowRAcquisitionReward
from reward_actflow.sampling import sample_policy
from reward_actflow.setups import setups as problem_setups
from reward_actflow.setups.problem_setup import ProblemSetup
from reward_actflow.smc_guidance import (
    GUIDANCE_MODES,
    Guidance,
    ResampleMethod,
    collect_with_guidance,
)
from reward_actflow.trainers.mixture_replay import MixtureExperience, ReplaySource
from reward_actflow.uncertainty import UncertaintyEstimator, uncertainty_estimators
from reward_actflow.uncertainty.gp import BACKENDS as GP_BACKENDS
from reward_actflow.utils import (
    Batch,
    filter_out_invalids,
    serialize_args,
    setup_logger,
)

#: `--algorithm` choices for ActFlow-R: the base four (for a comparison run
#: pairing the two-surrogate acquisition with an RL-style trainer) plus
#: `mixture_replay`, which is the algorithm this module exists for.
ALGORITHMS_R = (*ALGORITHMS, "mixture_replay")

type ZetaSchedule = Literal["constant", "linear", "cosine", "exp"]
#: Written out rather than derived via `get_args(ZetaSchedule)`: PEP 695
#: `type` aliases resolve through `.__value__`, and `get_args` on the alias
#: itself silently returns `()`.
ZETA_SCHEDULES: tuple[ZetaSchedule, ...] = ("constant", "linear", "cosine", "exp")


def zeta_at(schedule: ZetaSchedule, start: float, end: float, progress: float) -> float:
    """`zeta_t` at `progress = t / max(num_iters - 1, 1)` in `[0, 1]`."""
    progress = min(max(progress, 0.0), 1.0)
    match schedule:
        case "constant":
            return start
        case "linear":
            return start + (end - start) * progress
        case "cosine":
            import math

            return end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
        case "exp":
            # Geometric interpolation; falls back to linear if either endpoint
            # is non-positive, where a geometric path isn't defined.
            if start <= 0 or end <= 0:
                return start + (end - start) * progress
            return start * (end / start) ** progress
    raise ValueError(
        f"Unknown zeta_schedule {schedule!r}. Expected one of {ZETA_SCHEDULES}."
    )


@dataclass(frozen=True)
class ActFlowRConfig(ExploreConfig):
    """`ExploreConfig` plus ActFlow-R's acquisition, `UpdateFlow`, and
    guidance knobs.

    `gate`/`invalid_floor` are inherited but unused — those are base
    ActFlow's validity-reward gates, unrelated to `validity_gate` below,
    which is `a_t`'s own extension-(F) gate and off by default. They stay at
    their `ExploreConfig` defaults and play no role here.
    """

    ALLOWED_ALGORITHMS: ClassVar[tuple[str, ...]] = ALGORITHMS_R

    algorithm: str = "mixture_replay"

    # Acquisition: a_t = zeta*sigma^v + (1-zeta)*(mu^r + beta_r*sigma^r)
    zeta_start: float = 1.0
    zeta_end: float = 0.0
    zeta_schedule: ZetaSchedule = "linear"
    beta_r: float = 1.0
    #: Extension (F): gate a_t's reward term by sigmoid(mu^v), off by
    #: default. See `rewards/acquisition.py`'s module docstring for why.
    validity_gate: bool = False

    # UpdateFlow (MixtureReplay, Algorithm 2). Dedicated fields rather than
    # --algorithm_kwargs: unlike the base four's tuning knobs, these are
    # ActFlow-R's own primary tunables, not incidental per-algorithm details.
    anchor_frac: float = 0.25
    weight_clip: float = 20.0
    mixture_steps: int = 50
    mixture_batch: int = 64
    anchor_pool_size: int = 2048

    #: Tilt temperature in w = exp(lcb^r / eta). Smaller -> a sharper tilt
    #: toward high-lcb points; large eta -> w -> 1 (reduces to a plain refit).
    eta: float = 1.0

    # Line 4: solved by SMC guidance, not in the weights — see the module
    # docstring for why a weighted flow-matching refit alone cannot do this.
    guidance: Guidance = "smc"
    #: Inverse temperature for SMC guidance: particles are resampled toward
    #: exp(a_t / acq_beta).
    acq_beta: float = 1.0
    smc_ess_threshold: float = 0.5
    smc_resample: ResampleMethod = "systematic"

    # Diagnostics (diagnostics.py) — logged regardless of whether any
    # appendix extension is ever turned on; see that module's docstring.
    diag_radius_train: float = 0.25
    diag_radius_eval: float = 0.5
    diag_far_distance: float = 1.0
    diag_r2_every: int = 10

    def __post_init__(self):
        super().__post_init__()

        if not (0.0 <= self.zeta_start <= 1.0):
            raise ValueError(f"zeta_start must be in [0, 1], got {self.zeta_start}.")
        if not (0.0 <= self.zeta_end <= 1.0):
            raise ValueError(f"zeta_end must be in [0, 1], got {self.zeta_end}.")
        if self.zeta_schedule not in ZETA_SCHEDULES:
            raise ValueError(
                f"zeta_schedule must be one of {ZETA_SCHEDULES}, got "
                f"{self.zeta_schedule!r}."
            )
        if self.beta_r < 0:
            raise ValueError(f"beta_r must be non-negative, got {self.beta_r}.")
        if not (0.0 <= self.anchor_frac <= 1.0):
            raise ValueError(f"anchor_frac must be in [0, 1], got {self.anchor_frac}.")
        if self.weight_clip < 1.0:
            raise ValueError(f"weight_clip must be >= 1, got {self.weight_clip}.")
        if self.mixture_steps < 1:
            raise ValueError(f"mixture_steps must be >= 1, got {self.mixture_steps}.")
        if self.mixture_batch < 1:
            raise ValueError(f"mixture_batch must be >= 1, got {self.mixture_batch}.")
        if self.anchor_pool_size < 1:
            raise ValueError(
                f"anchor_pool_size must be >= 1, got {self.anchor_pool_size}."
            )
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}.")
        if self.guidance not in GUIDANCE_MODES:
            raise ValueError(
                f"guidance must be one of {GUIDANCE_MODES}, got {self.guidance!r}."
            )
        if self.acq_beta <= 0:
            raise ValueError(f"acq_beta must be positive, got {self.acq_beta}.")

        # MixtureReplay's constructor kwargs are always derived from the
        # dedicated fields above, not from ALGORITHM_DEFAULTS/--algorithm_kwargs
        # — a user who wants a different anchor_frac passes --anchor_frac, not
        # --algorithm_kwargs '{"anchor_frac": ...}'. --algorithm_kwargs still
        # passes through anything else, for any future MixtureReplay param
        # that never gets a dedicated flag.
        if self.algorithm == "mixture_replay":
            derived = {
                "anchor_frac": self.anchor_frac,
                "weight_clip": self.weight_clip,
                "steps_per_update": self.mixture_steps,
                "batch_size": self.mixture_batch,
            }
            extra = {k: v for k, v in self.algorithm_kwargs.items() if k not in derived}
            object.__setattr__(self, "algorithm_kwargs", {**derived, **extra})


def _reward_surrogate_args(args: dict[str, Any]) -> dict[str, Any]:
    """Overlay `--reward_gp_*` onto the `gp_*` keys `GPUncertaintyEstimator`
    expects, so the reward surrogate can use different GP hyperparameters
    (e.g. lengthscale) than the validity surrogate without a second copy of
    every `--gp_*` flag. Unset (`None`) `--reward_gp_*` flags mean "inherit
    the validity surrogate's setting".
    """
    overlay = dict(args)
    for key, value in args.items():
        if key.startswith("reward_gp_") and value is not None:
            overlay[key.removeprefix("reward_")] = value
    return overlay


def build_actflow_r_setup(
    problem: ProblemSetup,
    config: ActFlowRConfig,
    args: dict[str, Any],
    device: torch.device,
) -> tuple[FineTuningSetup, UncertaintyEstimator, UncertaintyEstimator]:
    """Like `explore.build_setup`, but with two surrogates bound to
    `actflow/acquisition` instead of one bound to `actflow/uncertainty`.

    The `FlowFeatureExtractor` — `phi_s^t` — is shared between the two
    surrogates (both live on the same representation); their feature
    *caches* are not, since each `UncertaintyEstimator` owns its own.
    """
    setup, extractor, feat_dim = assemble_setup(
        problem,
        reward_id="actflow/acquisition",
        reward_kwargs={"beta_r": config.beta_r, "validity_gate": config.validity_gate},
        algorithm=config.algorithm,
        algorithm_kwargs=config.algorithm_kwargs,
        num_steps=config.num_steps,
        ft_lr=config.ft_lr,
        feat_timestep=config.feat_timestep,
        device=device,
        noise_scale=config.noise_scale,
    )

    estimator_cls = uncertainty_estimators[args["uncertainty_estimator"]]
    validity_uncertainty = estimator_cls(
        extractor, feat_dim=feat_dim, device=device, args=args
    )
    reward_uncertainty = estimator_cls(
        extractor, feat_dim=feat_dim, device=device, args=_reward_surrogate_args(args)
    )

    reward = setup.environment.reward
    if not isinstance(reward, ActFlowRAcquisitionReward):
        raise TypeError(
            f"Expected an ActFlowRAcquisitionReward, got {type(reward).__name__}."
        )
    reward.bind(estimator=validity_uncertainty, verifier=problem.validity)
    reward.bind_reward_surrogate(estimator=reward_uncertainty)
    return setup, validity_uncertainty, reward_uncertainty


class ActFlowRLoop(ActFlowLoop):
    """ActFlow-R's outer loop: two surrogates, SMC-guided self-generation,
    and arithmetic-mixture replay. See the module docstring for the full
    algorithm and for why line 4 needs SMC.
    """

    def __init__(
        self,
        problem: ProblemSetup,
        setup: FineTuningSetup,
        uncertainty: UncertaintyEstimator,
        reward_uncertainty: UncertaintyEstimator,
        config: ActFlowRConfig,
        logger: logging.Logger,
    ):
        super().__init__(
            problem=problem,
            setup=setup,
            uncertainty=uncertainty,
            config=config,
            logger=logger,
        )
        if not isinstance(self.reward, ActFlowRAcquisitionReward):
            raise TypeError(
                "ActFlowRLoop expects the environment's reward to be an "
                f"ActFlowRAcquisitionReward, got {type(self.reward).__name__}."
            )
        self.acquisition: ActFlowRAcquisitionReward = self.reward
        self.reward_uncertainty = reward_uncertainty
        self.config: ActFlowRConfig = config

        #: E_t: verified-valid batches queried for r~ so far. Parallel to
        #: self.observations (D_t) but shorter, and only ever grows with the
        #: valid subset of a batch, once r~ has actually been queried for it.
        self.reward_observations: list[Batch] = []

        #: w over D+_{t+1}, recomputed each iteration in post_collect() from
        #: the reward surrogate fit in pre_collect() — i.e. stale relative to
        #: this iteration's own r~ query, exactly as Algorithm 1 line 7 says.
        self._weights: torch.Tensor | None = None

        # P: the frozen anchor pool. Drawn here — after make() builds theta_0,
        # before any update_flow() call — so "never refreshed from theta_t" is
        # structural (__init__ runs once), not a promise the loop keeps by
        # discipline.
        device = setup.context.policies.train.device
        anchors = problem.anchor_latents(config.anchor_pool_size, device)
        anchor_source = "p_data"
        if anchors is None:
            anchors = sample_policy(
                setup.context,
                config.anchor_pool_size,
                dynamics=setup.dynamics,
                time_grid=setup.time_grid,
                policy="train",
            )
            anchor_source = "theta_0"
        self.anchors = anchors
        self.anchor_source = anchor_source
        self.logger.info(
            f"anchor pool: {len(self.anchors)} points from {anchor_source}"
        )
        torch.save(self.anchors.cpu(), config.folder / "anchors.pt")

    # ------------------------------------------------------------------
    # Hooks (Algorithm 1 lines 2-3, 6-8)
    # ------------------------------------------------------------------

    def pre_collect(self, iteration: int) -> dict[str, float]:
        """Lines 2-3: refit `(mu^r, sigma^r)` on `E_t` (valid only), set `zeta_t`."""
        if self.reward_observations:
            # task_rewards is `Tensor | None` on Batch in general, but every
            # entry here was set right before being appended in post_collect.
            self.reward_uncertainty.set_data(
                [b.latents for b in self.reward_observations],
                [b.task_rewards for b in self.reward_observations],  # ty: ignore[invalid-argument-type]
                [b.kwargs for b in self.reward_observations],
            )
        progress = iteration / max(self.config.num_iters - 1, 1)
        zeta = zeta_at(
            self.config.zeta_schedule,
            self.config.zeta_start,
            self.config.zeta_end,
            progress,
        )
        self.acquisition.set_zeta(zeta)
        return {"zeta": zeta}

    def collect(self) -> tuple[Batch, Any]:
        """Lines 4-6, via SMC guidance (line 4) in place of the inherited
        plain rollout — see the module docstring for why."""
        self.reward.clear_cache()
        experience = collect_with_guidance(
            guidance=self.config.guidance,
            context=self.setup.context,
            algorithm=self.setup.algorithm,
            dynamics=self.setup.dynamics,
            n=self.config.samples_per_iter,
            time_grid=self.setup.time_grid,
            conditioning={},
            acq_beta=self.config.acq_beta,
            ess_threshold=self.config.smc_ess_threshold,
            resample=self.config.smc_resample,
        )

        environment = self.setup.environment
        latents, rewards, conditioning = endpoints_of(experience)
        samples = environment.codec.decode(latents, conditioning=conditioning)
        if rewards is None:
            _, reward_batch = environment.evaluate_terminal(
                latents, conditioning=conditioning
            )
            rewards = reward_batch.rewards

        return self._record(latents, samples, rewards, conditioning), experience

    def post_collect(self, iteration: int, batch: Batch) -> dict[str, float]:
        """Lines 6-7: query `r~` on the valid subset, extend `E`, then compute
        `w` over `D+_{t+1}` from the surrogate fit in `pre_collect` — stale
        relative to this batch's own `r~`, exactly as the spec orders it.
        """
        valid_mask = batch.valids
        n_valid = int(valid_mask.sum().item())
        metrics: dict[str, float] = {"valid_count": float(n_valid)}

        if n_valid > 0:
            valid_batch = batch.select(valid_mask.nonzero(as_tuple=True)[0])
            valid_batch.task_rewards = self.problem.task_reward(
                valid_batch.samples, valid_batch.kwargs
            )
            self.reward_observations.append(valid_batch)
            metrics["task_reward_mean"] = valid_batch.task_rewards.mean().item()

        data = filter_out_invalids(self.observations)
        if len(data) > 0:
            mean_r, sigma_r = self.reward_uncertainty.mean_and_uncertainty(data.latents)
            lcb_r = mean_r - self.acquisition.beta_r * sigma_r
            self._weights = torch.exp(lcb_r / self.config.eta).detach()
        else:
            self._weights = torch.zeros(0)

        return metrics

    def prepare_experience(self, experience: Any) -> Any:
        """Line 8 prep: attach the replay source `(P, D+, w)` `MixtureReplay`
        needs — it is driven entirely by this loop, see
        `trainers/mixture_replay.py`'s module docstring."""
        data = filter_out_invalids(self.observations)
        weights = self._weights if self._weights is not None else torch.zeros(len(data))
        replay = ReplaySource(
            anchors=self.anchors,
            # Neither anchor_latents() nor sample_policy() (the two anchor
            # sources) return conditioning; {} is exact for the toy
            # (unconditioned) and a known simplification for a future
            # conditioned setup.
            anchor_conditioning={},
            data=data.latents,
            data_conditioning=data.kwargs,
            weights=weights,
        )
        if isinstance(experience, MixtureExperience):
            return dataclasses.replace(experience, replay=replay)
        return experience

    def update_flow(self, experience: Any) -> dict[str, float]:
        """Line 8, minus the reference re-anchor: `MixtureReplay` has no
        KL-to-current-iterate term (`requirements.needs_reference_policy` is
        `False`, so `make()` never builds one), so there is nothing to
        re-anchor. Kept as an explicit override rather than relying on
        `refresh_reference`'s no-op-without-a-reference fallback, so this
        stays correct if a future change adds one.
        """
        metrics = dict(
            self.setup.algorithm.update(
                context=self.setup.context, experience=experience
            )
        )
        self.setup.algorithm.synchronize_rollout_policy(context=self.setup.context)
        return metrics

    def visualize_iter(self, batch: Batch, iteration: int) -> dict[str, float]:
        """Prefers `problem.visualize_reward_sample` (reward field, anchor
        pool, this iteration's samples) over the inherited uncertainty/support
        figure; falls back to it for a setup that hasn't implemented the
        richer one.
        """
        result = self.problem.visualize_reward_sample(
            self.setup, self.uncertainty, self.reward_uncertainty, self.anchors, batch
        )
        if result is None:
            return super().visualize_iter(batch, iteration)

        fig, metrics = result
        directory = self.config.folder / "frames"
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{iteration:04d}.png", dpi=300)
        plt.close(fig)
        return metrics

    def extra_metrics(self, iteration: int, batch: Batch) -> dict[str, float]:
        metrics: dict[str, float] = {}
        if self.acquisition.last_reward_mean is not None:
            metrics["reward_mean_pred"] = (
                self.acquisition.last_reward_mean.mean().item()
            )
        if self.acquisition.last_reward_uncertainty is not None:
            metrics["reward_uncertainty"] = (
                self.acquisition.last_reward_uncertainty.mean().item()
            )
        metrics["reward_observations"] = float(
            sum(len(b) for b in self.reward_observations)
        )
        metrics["reward_label_std"] = self.reward_uncertainty.label_std
        metrics["reward_label_mean"] = self.reward_uncertainty.label_mean
        if self._weights is not None and len(self._weights) > 0:
            ess = effective_sample_size(self._weights)
            metrics["ess_preclip"] = ess
            metrics["ess_frac_preclip"] = ess / len(self._weights)

        metrics |= self._diagnostics(iteration, batch)
        return metrics

    def _diagnostics(self, iteration: int, batch: Batch) -> dict[str, float]:
        """The evidence `diagnostics.py` exists to provide — see its module
        docstring for which extension each diagnostic is the trigger for.

        Every key is set unconditionally (0 or nan when undefined), never
        omitted: `_flush_metrics` fixes the metrics.csv schema on the first
        write and silently drops any key absent then, so a diagnostic that
        only starts existing once D+/E_t are non-empty would otherwise
        vanish for the rest of the run if iteration 0 happened to have none.
        """
        metrics: dict[str, float] = {}

        data = filter_out_invalids(self.observations)
        if len(data) > 0:
            coords = self.problem.diagnostic_coordinates(data.latents)
            c_train = cluster_count(coords, self.config.diag_radius_train)
            c_eval = cluster_count(coords, self.config.diag_radius_eval)
        else:
            c_train = c_eval = 0
        metrics["clusters_r_train"] = float(c_train)
        metrics["clusters_r_eval"] = float(c_eval)
        metrics["cluster_ratio"] = c_eval / max(c_train, 1)

        if self.reward_observations:
            e_data = Batch.concat(self.reward_observations)
            anchor_coords = self.problem.diagnostic_coordinates(self.anchors)
            best, far_count = best_reward_beyond(
                e_data.task_rewards,  # ty: ignore[invalid-argument-type]
                self.problem.diagnostic_coordinates(e_data.latents),
                anchor_coords,
                self.config.diag_far_distance,
            )
        else:
            best, far_count = None, 0
        metrics["far_count"] = float(far_count)
        metrics["best_reward_far"] = best if best is not None else float("nan")

        r2 = None
        if (
            self.reward_observations
            and iteration % max(self.config.diag_r2_every, 1) == 0
        ):
            r2 = heldout_r2(
                self.reward_uncertainty,
                [b.latents for b in self.reward_observations],
                [b.task_rewards for b in self.reward_observations],  # ty: ignore[invalid-argument-type]
                [b.kwargs for b in self.reward_observations],
                seed=iteration,
            )
        metrics["reward_r2"] = r2 if r2 is not None else float("nan")

        if iteration % max(self.config.visualize_every, 1) == 0:
            # Reuses this iteration's own collected batch as the "current
            # policy" sample rather than paying for another draw — coarser
            # than a dedicated support sample, but free.
            current_coords = self.problem.diagnostic_coordinates(batch.latents)
            anchor_coords = self.problem.diagnostic_coordinates(self.anchors)
            limits = coordinate_bounds(current_coords, anchor_coords)
            density = per_cell_density_ratio(
                current_coords, anchor_coords, bins=50, limits=limits
            )
            metrics |= {f"density_{k}": v for k, v in density.items()}

        return metrics


def setup_and_run(args: argparse.Namespace):
    """Shared initialization and run logic, mirroring `explore.setup_and_run`."""
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ActFlowRConfig.construct_from_args(args)
    problem = problem_setups[args.problem_setup](vars(args), device=device)

    with open(config.folder / "args.yaml", "w") as f:
        yaml.safe_dump(serialize_args(args), f)

    logger = setup_logger(config.folder, args.verbose)
    logger.info("starting ActFlow-R...")

    setup, validity_uncertainty, reward_uncertainty = build_actflow_r_setup(
        problem, config, vars(args), device
    )

    loop = ActFlowRLoop(
        problem=problem,
        setup=setup,
        uncertainty=validity_uncertainty,
        reward_uncertainty=reward_uncertainty,
        config=config,
        logger=logger,
    )
    loop.run()


def main(args: argparse.Namespace):
    setup_and_run(args)


def add_actflow_r_args(parser: argparse.ArgumentParser):
    # Acquisition: a_t = zeta*sigma^v + (1-zeta)*(mu^r + beta_r*sigma^r)
    parser.add_argument("--zeta_start", type=float, default=1.0)
    parser.add_argument("--zeta_end", type=float, default=0.0)
    parser.add_argument(
        "--zeta_schedule", type=str, choices=ZETA_SCHEDULES, default="linear"
    )
    parser.add_argument(
        "--beta_r",
        type=float,
        default=1.0,
        help="Optimism coefficient on the reward surrogate's uncertainty.",
    )
    parser.add_argument(
        "--validity_gate",
        action="store_true",
        help=(
            "Extension (F): gate a_t's reward term by sigmoid(mu^v), the "
            "validity surrogate's own posterior mean. Off by default. Use "
            "when mu^r is extrapolating optimistically past a region "
            "boundary (best_reward_far pins at a fixed value for many "
            "iterations while valid_rate degrades)."
        ),
    )

    # UpdateFlow (MixtureReplay, Algorithm 2)
    parser.add_argument(
        "--anchor_frac",
        type=float,
        default=0.25,
        help="lambda: fraction of each minibatch drawn from the frozen anchor pool P.",
    )
    parser.add_argument(
        "--weight_clip",
        type=float,
        default=20.0,
        help="G: w is clipped to [1/G, G] before sampling D+.",
    )
    parser.add_argument(
        "--mixture_steps",
        type=int,
        default=50,
        help="N_step: gradient steps per outer iteration.",
    )
    parser.add_argument("--mixture_batch", type=int, default=64)
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="Tilt temperature in w = exp(lcb^r / eta).",
    )
    parser.add_argument("--anchor_pool_size", type=int, default=2048)

    # Line 4: SMC guidance
    parser.add_argument("--guidance", type=str, choices=GUIDANCE_MODES, default="smc")
    parser.add_argument(
        "--acq_beta",
        type=float,
        default=1.0,
        help="Inverse temperature for SMC guidance: particles resampled ~ exp(a_t/acq_beta).",
    )
    parser.add_argument("--smc_ess_threshold", type=float, default=0.5)
    parser.add_argument(
        "--smc_resample",
        type=str,
        choices=("systematic", "multinomial"),
        default="systematic",
    )

    # Reward surrogate GP overrides, namespaced: --gp_* still configures the
    # validity surrogate; unset (None) --reward_gp_* means "inherit that".
    parser.add_argument(
        "--reward_gp_kernel", type=str, choices=["linear", "rbf"], default=None
    )
    parser.add_argument("--reward_gp_lengthscale", type=float, default=None)
    parser.add_argument(
        "--reward_gp_backend", type=str, choices=GP_BACKENDS, default=None
    )
    parser.add_argument("--reward_gp_inducing", type=int, default=None)
    parser.add_argument("--reward_gp_grid_size", type=int, default=None)
    parser.add_argument("--reward_gp_grid_limit", type=float, default=None)

    # Diagnostics
    parser.add_argument("--diag_radius_train", type=float, default=0.25)
    parser.add_argument("--diag_radius_eval", type=float, default=0.5)
    parser.add_argument("--diag_far_distance", type=float, default=1.0)
    parser.add_argument("--diag_r2_every", type=int, default=10)


def build_parser() -> argparse.ArgumentParser:
    """Build the nested `<problem_setup> <uncertainty_estimator> [flags]` parser."""
    from reward_actflow.explore import add_global_args

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="problem_setup", required=True)

    for name, setup_cls in problem_setups.items():
        sub = subparsers.add_parser(name)
        uncertainty_subparsers = sub.add_subparsers(
            dest="uncertainty_estimator", required=True
        )

        for u_name, u_cls in uncertainty_estimators.items():
            u_sub = uncertainty_subparsers.add_parser(u_name)

            add_global_args(u_sub, algorithms=ALGORITHMS_R)
            add_actflow_r_args(u_sub)
            setup_cls.add_args(u_sub)
            u_cls.add_args(u_sub)

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
