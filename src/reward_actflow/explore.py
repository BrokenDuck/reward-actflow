"""Active Flow Expansion (ActFlow), Algorithm 1, on the diffusiongym `core/` stack.

```
D_0 = {}
for t = 0 ... T-1:
  3:  update surrogate uncertainty sigma_t from D_t
  4:  self-generate x_{t+1} ~ p_t in argmax_q E_{x~q}[sigma_t(phi_s^t(x))]
                                     - beta * KL(q || p^{theta_t})
  5:  y_{t+1} = v(x_{t+1})
  6:  D_{t+1} = D_t + {(x_{t+1}, y_{t+1})}
  7:  theta_{t+1} = UpdateFlow(theta_t, D_{t+1})
```

Line 4 is solved *in the weights*, not by inference-time guidance: sampling is a
plain rollout of `p^{theta_t}`, and the KL-regularized tilt is whatever the
fine-tuning algorithm in line 7 achieves. `UpdateFlow` is therefore a pluggable
slot — any of diffusiongym's four algorithms, chosen with `--algorithm`.

Each outer iteration is exactly one `collect()` + one `update()`, so the verifier
is queried `samples_per_iter` times per iteration and no more.

The KL anchor in line 4 is `p^{theta_t}`, the current iterate rather than the
frozen prior, so the reference policy is re-anchored to the train policy at the
end of every iteration (`refresh_reference`). Without that the algorithms that
carry a reference would be solving a differently-anchored problem than the one
written down, and pulling against the expansion they are meant to produce.
"""

import argparse
import csv
import json
import logging
import time
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Self

import diffusiongym
import matplotlib
import numpy as np
import torch
import yaml
from diffusiongym import FineTuningSetup
from diffusiongym.trainers.base import FineTuningContext
from diffusiongym.types import DDBatch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reward_actflow.rewards  # noqa: F401  (registers actflow/uncertainty)
from reward_actflow.rewards.base import SurrogateReward
from reward_actflow.rewards.uncertainty import (
    GATES,
    ActFlowUncertaintyReward,
    Gate,
)
from reward_actflow.sampling import sample_policy
from reward_actflow.setups import setups as problem_setups
from reward_actflow.setups.problem_setup import ProblemSetup
from reward_actflow.uncertainty import (
    FlowFeatureExtractor,
    UncertaintyEstimator,
    uncertainty_estimators,
)
from reward_actflow.utils import (
    Batch,
    serialize_args,
    setup_logger,
    write_video,
)

ALGORITHMS = ("diffusion_nft", "orw_cfm", "flow_grpo", "adjoint_matching")

#: Per-algorithm constructor overrides suited to the fused loop, where the
#: algorithm runs for `num_iters` iterations total rather than the several
#: hundred its own defaults assume. Two of these matter a lot:
#:
#: * DiffusionNFT's stock `ema_decay=0.995` lags the rollout policy by ~200
#:   iterations, so over a 100-iteration run it barely leaves the pretrained
#:   weights and the method looks inert.
#: * `inner_epochs=10` would give 1 000 gradient steps across the whole run,
#:   against the 50 000 the flow-matching refit it replaces used to take.
#:
#: ORW-CFM's `rollout_update_interval` is pinned to 1 (already its default)
#: because the fused loop calls `update()` once per outer iteration, so 1 is what
#: makes the rollout policy track the ActFlow iterate. Anything larger silently
#: makes the method non-online, which is the failure its own spec warns about.
ALGORITHM_DEFAULTS: dict[str, dict[str, Any]] = {
    "diffusion_nft": {
        "beta": 1.0,
        "ema_decay": 0.9,
        "inner_epochs": 50,
        "batch_size": 64,
    },
    "orw_cfm": {
        "temperature": 1.0,
        "alpha_w2": 0.5,
        "rollout_update_interval": 1,
        "steps_per_update": 50,
        "batch_size": 64,
    },
    "flow_grpo": {
        "group_size": 8,
        "ppo_epochs": 4,
        "ppo_batch_size": 64,
        "beta_kl": 1.0,
    },
    "adjoint_matching": {
        "lambda_reward": 1.0,
        "train_steps_per_iter": 50,
        "train_batch_size": 64,
    },
}


@dataclass(frozen=True)
class ExploreConfig:
    #: Algorithms `__post_init__` accepts. A `ClassVar`, not a field, so a
    #: subclass (`ActFlowRConfig`) can extend the set — e.g. with
    #: `"mixture_replay"` — without `ExploreConfig`'s own validation rejecting
    #: it. Left at the base four for `ExploreConfig` itself.
    ALLOWED_ALGORITHMS: ClassVar[tuple[str, ...]] = ALGORITHMS

    # Experiment directory
    folder: Path

    # Exploration sampling
    num_iters: int = 1000
    samples_per_iter: int = 64
    num_steps: int = 100

    # Surrogate
    feat_timestep: float = 0.9

    # Reward
    gate: Gate = "hard"
    invalid_floor: float = 1.0

    # Fine-tuning (UpdateFlow)
    algorithm: str = "diffusion_nft"
    algorithm_kwargs: dict[str, Any] = field(default_factory=dict)
    ft_lr: float = 1e-4

    #: SDE diffusion level for algorithms needing stochastic rollout (passed to
    #: `diffusiongym.make()`; ignored by algorithms using an ODE or memoryless
    #: dynamics). Governs how far a *single* rollout can wander from the
    #: deterministic trajectory before any reweighting happens — for
    #: `mixture_replay` under SMC guidance, this is the actual bottleneck on
    #: how far one iteration's exploration can reach, independent of ζ or the
    #: acquisition; raising it is the lever when `far_count` stays 0 for many
    #: iterations despite an aggressive acquisition.
    noise_scale: float = 0.75

    #: Leading iterations that run lines 3-6 but skip `UpdateFlow`, seeding `D`
    #: before the first policy update. With `D_0` empty the surrogate is a flat
    #: prior and line 4's objective has no maximizer, so an update at t=0 is at
    #: best a no-op. Adjoint Matching *requires* at least one: it differentiates
    #: the terminal cost, and a flat prior makes that gradient identically zero,
    #: which it rejects rather than silently training on nothing.
    warmup_iters: int = 0

    # Checkpointing
    ckpt_every: int = 100

    # Sampling and evaluation
    eval_samples: int = 0
    eval_every: int = 10
    video_fps: int = 4

    #: Iterations between figures. Each one draws 50k samples and evaluates the
    #: surrogate on a 10k-point grid, and that grid evaluation is the first
    #: posterior call after the refit, so it pays for rebuilding the GP's
    #: prediction cache against the whole replay buffer. Measured on a 400-step
    #: run, one iteration's figure cost 52.8s at t=199 against 0.13s for the
    #: policy update it is illustrating. Raise this for long runs.
    visualize_every: int = 1

    def __post_init__(self):
        self.folder.mkdir(parents=True, exist_ok=True)

        if not (0 <= self.feat_timestep <= 1):
            raise ValueError(
                f"feat_timestep must be in [0, 1], got {self.feat_timestep}"
            )
        if self.gate not in GATES:
            raise ValueError(f"gate must be one of {GATES}, got {self.gate!r}")
        if self.algorithm not in self.ALLOWED_ALGORITHMS:
            raise ValueError(
                f"algorithm must be one of {self.ALLOWED_ALGORITHMS}, got "
                f"{self.algorithm!r}"
            )
        if self.warmup_iters < 0:
            raise ValueError(
                f"warmup_iters cannot be negative, got {self.warmup_iters}"
            )
        if self.noise_scale <= 0:
            raise ValueError(f"noise_scale must be positive, got {self.noise_scale}")
        if self.algorithm == "adjoint_matching":
            if self.warmup_iters < 1:
                raise ValueError(
                    "adjoint_matching needs --warmup_iters >= 1. It validates "
                    "that the terminal cost has a non-zero gradient, and with "
                    "D_0 empty the surrogate is a flat prior whose gradient is "
                    "zero everywhere, so it would refuse the first iteration."
                )
            if self.gate == "soft":
                raise ValueError(
                    "adjoint_matching cannot use --gate soft: r = mu * sigma is "
                    "identically zero while the verifier keeps returning the "
                    "same answer, because z-scored constant labels give a flat "
                    "posterior mean, and a zero terminal-cost gradient is "
                    "rejected. Use --gate sigmoid, which degrades to sigma / 2 "
                    "instead of to zero."
                )

    @classmethod
    def _algorithm_defaults(cls, algorithm: str) -> dict[str, Any]:
        """Per-algorithm constructor overrides to merge under user kwargs.

        A classmethod hook, not a direct `ALGORITHM_DEFAULTS.get(...)` call, so
        a subclass can extend the lookup (e.g. `ActFlowRConfig` adding
        `MIXTURE_REPLAY_DEFAULTS`) without duplicating `construct_from_args`.
        """
        return ALGORITHM_DEFAULTS.get(algorithm, {})

    @classmethod
    def construct_from_args(cls, args: argparse.Namespace | dict) -> Self:
        args = vars(args) if isinstance(args, argparse.Namespace) else dict(args)

        name_mapping = {"dir": "folder"}
        config_fields = {f.name for f in cls.__dataclass_fields__.values()}

        clean_kwargs: dict[str, Any] = {}
        for key, value in args.items():
            target_key = name_mapping.get(key, key)
            if target_key in config_fields:
                clean_kwargs[target_key] = value

        # The ablation flags are just gate choices; resolving them here keeps the
        # reward with a single notion of what it is computing.
        if args.get("no_uncertainty"):
            clean_kwargs["gate"] = "validity"
        elif args.get("no_verifier"):
            clean_kwargs["gate"] = "raw"

        algorithm = clean_kwargs.get("algorithm", "diffusion_nft")
        clean_kwargs["algorithm_kwargs"] = {
            **cls._algorithm_defaults(algorithm),
            **(args.get("algorithm_kwargs") or {}),
        }

        return cls(**clean_kwargs)


def endpoints_of(experience: Any) -> tuple[Any, torch.Tensor | None, Mapping[str, Any]]:
    """Terminal latents, rewards and conditioning, whatever the experience type.

    The four algorithms return three different experience dataclasses:
    `EndpointExperience` carries the terminal latent directly, while
    `TrajectoryExperience` (Flow-GRPO) and `AdjointExperience` (Adjoint
    Matching) carry a whole `Rollout`. Only the endpoints are observations, so
    the loop is written against those and stays algorithm-agnostic.
    """
    if hasattr(experience, "latent"):
        return experience.latent, experience.rewards, experience.conditioning

    rollout = experience.rollout
    rewards = rollout.reward.rewards if rollout.reward is not None else None
    return rollout.terminal_latent, rewards, rollout.conditioning


def refresh_reference(context: FineTuningContext) -> bool:
    """Re-anchor the reference policy to the current train policy.

    This is the `p^{theta_t}` of Algorithm 1 line 4. `diffusiongym.make()` builds
    the reference once and never touches it again, which would anchor every
    iteration to `theta_0` instead.

    Returns whether an anchor was actually moved — algorithms with no reference
    policy (DiffusionNFT, and ORW-CFM without `alpha_w2`) make this a no-op.
    """
    reference: Any = context.policies.reference
    if reference is None:
        return False
    train: Any = context.policies.train
    reference.load_state_dict(train.state_dict())
    return True


class ActFlowLoop[D: DDBatch]:
    """The outer active-learning loop."""

    TIMING_HEADS = (
        "iteration",
        "eval",
        "surrogate",
        "sampling",
        "update",
        "visualize",
        "checkpoint",
    )

    def __init__(
        self,
        problem: ProblemSetup[D],
        setup: FineTuningSetup,
        uncertainty: UncertaintyEstimator[D],
        config: ExploreConfig,
        logger: logging.Logger,
    ):
        self.problem = problem
        self.setup = setup
        self.uncertainty = uncertainty
        self.config = config
        self.logger = logger

        self.observations: list[Batch[D]] = []
        self._timings: dict[str, float] = {}
        self._metrics_heads: list[str] | None = None

        reward = setup.environment.reward
        if not isinstance(reward, SurrogateReward):
            raise TypeError(
                "ActFlowLoop expects the environment's reward to be a "
                f"SurrogateReward (e.g. ActFlowUncertaintyReward), got "
                f"{type(reward).__name__}."
            )
        self.reward: SurrogateReward = reward

        train_policy: Any = setup.context.policies.train
        n_params = sum(p.numel() for p in train_policy.parameters() if p.requires_grad)
        self.logger.info(f"train policy parameters: {n_params:,}")
        self.logger.info(
            f"algorithm={config.algorithm} gate={config.gate} "
            f"kwargs={config.algorithm_kwargs}"
        )
        # Logged because choosing it wrong is quiet until it is catastrophic: an
        # exact GP is fine for a few hundred iterations and asks for ~16 GB at a
        # thousand, and nothing else in the output says which backend is live.
        self.logger.info(
            f"surrogate={type(uncertainty).__name__} "
            f"backend={uncertainty.args.get('gp_backend', 'n/a')} "
            f"lengthscale={uncertainty.args.get('gp_lengthscale', 'n/a')}"
        )

        self.timing_path = config.folder / "timings.csv"
        with open(self.timing_path, "w", newline="") as f:
            csv.writer(f).writerow(self.TIMING_HEADS)

        self.metrics_path = config.folder / "metrics.csv"

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    @contextmanager
    def _timer(self, name: str):
        t0 = time.perf_counter()
        yield
        self._timings[name] = time.perf_counter() - t0

    def _flush_timings(self, iteration: int):
        row = [iteration] + [
            self._timings.get(head, 0.0) for head in self.TIMING_HEADS[1:]
        ]
        with open(self.timing_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        self._timings = {}

    def _flush_metrics(self, iteration: int, metrics: dict[str, float]):
        """Append one row to `metrics.csv`, fixing the schema on the first write.

        Later keys that were not present on iteration 0 are dropped rather than
        silently shifting every column, which is what a plain `writerow` of
        `metrics.values()` would do once the algorithm's metric set changes.
        """
        if self._metrics_heads is None:
            self._metrics_heads = ["iteration"] + sorted(metrics)
            with open(self.metrics_path, "w", newline="") as f:
                csv.writer(f).writerow(self._metrics_heads)

        row = [iteration] + [
            metrics.get(head, float("nan")) for head in self._metrics_heads[1:]
        ]
        with open(self.metrics_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------

    def update_surrogate(self):
        """Line 3: refit sigma_t on everything observed so far."""
        if not self.observations:
            return
        self.uncertainty.set_data(
            [b.latents for b in self.observations],
            [b.valids.float() for b in self.observations],
            [b.kwargs for b in self.observations],
        )

    def warmup_collect(self) -> Batch[D]:
        """Lines 4-6 with the objective degenerate: draw from `p^{theta_t}`.

        With `D_t` still empty the surrogate is a flat prior, so line 4's argmax
        is just the current policy and there is nothing for `UpdateFlow` to do.
        Sampling directly rather than through `algorithm.collect()` also keeps
        Adjoint Matching out of an adjoint pass it would refuse (its
        zero-gradient check fires on exactly this flat prior).
        """
        self.reward.clear_cache()
        latents = sample_policy(
            self.setup.context,
            self.config.samples_per_iter,
            dynamics=self.setup.dynamics,
            time_grid=self.setup.time_grid,
        )
        samples, reward_batch = self.setup.environment.evaluate_terminal(
            latents, conditioning={}
        )
        return self._record(latents, samples, reward_batch.rewards, {})

    def _record(
        self,
        latents: D,
        samples: Any,
        rewards: torch.Tensor,
        conditioning: Mapping[str, Any],
    ) -> Batch[D]:
        """Line 5-6: attach verifier labels and fold the batch into `D`."""
        # If the reward already queried the verifier, reuse those labels rather
        # than paying for a second query — free here, but xTB later. Otherwise
        # ask now: the labels are what `D` is made of, whether or not the reward
        # happened to need them.
        labels = self.reward.last_verifier_labels
        if labels is None:
            labels = self.problem.validity(samples, dict(conditioning))

        batch = Batch.from_endpoints(
            latents=latents,
            samples=samples,
            rewards=rewards,
            valids=labels,
            conditioning=conditioning,
        )
        batch.latents = self.problem.postprocess_latents(batch)
        return batch

    def collect(self) -> tuple[Batch[D], Any]:
        """Lines 4-6: generate under `p^{theta_t}`, verify, and record.

        Returns the recorded batch and the raw experience, which `UpdateFlow`
        consumes directly — the algorithm's experience object carries more than
        `Batch` does (conditioning, per-step data for the trajectory methods).
        """
        self.reward.clear_cache()
        experience = self.setup.algorithm.collect(
            context=self.setup.context,
            dynamics=self.setup.dynamics,
            n=self.config.samples_per_iter,
            time_grid=self.setup.time_grid,
            conditioning={},
        )

        environment = self.setup.environment
        latents, rewards, conditioning = endpoints_of(experience)
        samples = environment.codec.decode(latents, conditioning=conditioning)

        if rewards is None:
            # No algorithm currently skips reward evaluation during collection,
            # but the experience contract does not promise it, and `D` should
            # carry a reward column regardless of how theta was updated.
            _, reward_batch = environment.evaluate_terminal(
                latents, conditioning=conditioning
            )
            rewards = reward_batch.rewards

        return self._record(latents, samples, rewards, conditioning), experience

    def update_flow(self, experience: Any) -> dict[str, float]:
        """Line 7: `UpdateFlow`, plus the two policy-anchor bookkeeping steps."""
        metrics = dict(
            self.setup.algorithm.update(
                context=self.setup.context, experience=experience
            )
        )
        self.setup.algorithm.synchronize_rollout_policy(context=self.setup.context)
        refresh_reference(self.setup.context)
        return metrics

    @torch.no_grad()
    def eval_model(self, iteration: int) -> dict[str, float]:
        n = self.config.eval_samples
        if n <= 0:
            return {}

        # The setup's own dynamics, not a forced ODE: a policy trained under an
        # SDE has to be evaluated under that SDE or the numbers describe a
        # distribution it was never optimized for.
        latents = sample_policy(
            self.setup.context,
            n,
            dynamics=self.setup.dynamics,
            time_grid=self.setup.time_grid,
        )
        samples = self.setup.environment.codec.decode(latents, conditioning={})
        valids = self.problem.validity(samples, {})

        directory = self.config.folder / "eval" / f"{iteration:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.problem.save_samples(samples, {}, directory)
        torch.save(valids.cpu(), directory / "valids.pt")

        metrics = self.problem.compute_metrics(samples, {})
        metrics["eval_valid"] = valids.float().mean().item()
        with open(directory / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

        return metrics

    def visualize_iter(self, batch: Batch[D], iteration: int) -> dict[str, float]:
        fig, metrics = self.problem.visualize_sample(
            self.setup, self.uncertainty, batch
        )

        directory = self.config.folder / "frames"
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{iteration:04d}.png", dpi=300)
        plt.close(fig)
        return metrics

    # ------------------------------------------------------------------
    # Extension hooks
    #
    # No-ops here, so base ActFlow is bit-identical whether or not a subclass
    # exists. `ActFlowRLoop` overrides these to fit a reward surrogate, query
    # the black-box task reward, and drive the arithmetic-mixture replay
    # weights — steps 2-3, 6-7 and 8 of ActFlow-R Algorithm 1 — without
    # duplicating `run()`'s timing/metrics/video machinery.
    # ------------------------------------------------------------------

    def pre_collect(self, iteration: int) -> dict[str, float]:
        """Called after `update_surrogate()`, before sampling."""
        return {}

    def post_collect(self, iteration: int, batch: Batch[D]) -> dict[str, float]:
        """Called after `batch` is appended to `self.observations`."""
        return {}

    def prepare_experience(self, experience: Any) -> Any:
        """Called on the collected experience just before `update_flow()`."""
        return experience

    def extra_metrics(self, iteration: int, batch: Batch[D]) -> dict[str, float]:
        """Called just before the iteration's metrics are flushed."""
        return {}

    def checkpoint(self, iteration: int):
        ckpt_dir = self.config.folder / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        train_policy: Any = self.setup.context.policies.train
        state_dict = train_policy.state_dict()
        torch.save(state_dict, ckpt_dir / "last.pt")
        if self.config.ckpt_every > 0 and iteration % self.config.ckpt_every == 0:
            torch.save(state_dict, ckpt_dir / f"ckpt_{iteration}.pt")

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def run(self):
        for i in range(self.config.num_iters):
            with self._timer("eval"):
                metrics: dict[str, float] = {}
                if self.config.eval_samples > 0 and i % self.config.eval_every == 0:
                    metrics |= self.eval_model(i)

            with self._timer("surrogate"):
                self.update_surrogate()
                metrics |= self.pre_collect(i)

            warming_up = i < self.config.warmup_iters

            with self._timer("sampling"):
                if warming_up:
                    batch, experience = self.warmup_collect(), None
                else:
                    batch, experience = self.collect()
                self.observations.append(batch)
                metrics |= self.post_collect(i, batch)

            with self._timer("update"):
                if experience is not None:
                    experience = self.prepare_experience(experience)
                    metrics |= self.update_flow(experience)

            with self._timer("visualize"):
                if i % max(self.config.visualize_every, 1) == 0:
                    metrics |= self.visualize_iter(batch, i)

            metrics["valid_rate"] = batch.valids.float().mean().item()
            metrics["reward_mean"] = batch.rewards.mean().item()
            if self.reward.last_uncertainty is not None:
                metrics["uncertainty"] = self.reward.last_uncertainty.mean().item()
            metrics["observations"] = float(sum(len(b) for b in self.observations))
            # Zero means every verifier answer so far has agreed, which makes
            # the surrogate's posterior mean flat and any mean-gated reward
            # uninformative. Cheap to log, and otherwise invisible.
            metrics["label_std"] = self.uncertainty.label_std
            if torch.cuda.is_available():
                # Per-iteration peak, not the run's high-water mark: the
                # cumulative counter can only ever rise, so it reads like a leak
                # even when usage is flat. `vram_held` is what actually persists
                # between iterations, and it grows as |D|^2 because the exact GP
                # caches a dense |D| x |D| train covariance.
                metrics["vram_peak"] = torch.cuda.max_memory_allocated() * 1e-9
                metrics["vram_held"] = torch.cuda.memory_allocated() * 1e-9
                torch.cuda.reset_peak_memory_stats()

            metrics |= self.extra_metrics(i, batch)

            self.logger.info(
                f"(iter={i:05d}) "
                + ", ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            )
            self._flush_metrics(i, metrics)

            with self._timer("checkpoint"):
                self.checkpoint(i)

            self._flush_timings(i)

        write_video(
            sorted(self.config.folder.glob("frames/*.png")),
            self.config.folder / "video.mp4",
            fps=self.config.video_fps,
        )


def assemble_setup(
    problem: ProblemSetup,
    *,
    reward_id: str,
    reward_kwargs: dict[str, Any],
    algorithm: str,
    algorithm_kwargs: dict[str, Any],
    num_steps: int,
    ft_lr: float,
    feat_timestep: float,
    device: torch.device,
    noise_scale: float = 0.75,
) -> tuple[FineTuningSetup, FlowFeatureExtractor, int]:
    """Build a `FineTuningSetup` and the `phi_s^t` feature extractor for it.

    Shared by `build_setup` (one surrogate, base ActFlow) and
    `actflow_r.build_actflow_r_setup` (two surrogates) — everything up to
    "what gets bound to the reward" is identical between them.

    `make()` rather than hand-wiring, because it derives the SDE/ODE profile, the
    interior time grid, the reference policy and the terminal cost from
    `algorithm.requirements`, and each of those four disagreeing with the
    algorithm is a silent failure rather than a crash.

    Returns the setup, the extractor, and its output feature dimensionality
    (probed once here, rather than by every caller).
    """
    setup = diffusiongym.make(
        modality=problem.modality_id,
        reward=reward_id,
        algorithm=algorithm,
        discretization_steps=num_steps,
        device=device,
        learning_rate=ft_lr,
        noise_scale=noise_scale,
        modality_kwargs=problem.modality_kwargs,
        reward_kwargs=reward_kwargs,
        algorithm_kwargs=algorithm_kwargs,
    )

    modality = diffusiongym.modality_registry.get(problem.modality_id)
    schedule = modality.instantiate(**problem.modality_kwargs).schedule()

    extractor = FlowFeatureExtractor(
        # phi_s^t is the representation under the *current* iterate.
        setup.context.policies.train,
        setup.environment.geometry,
        schedule,
        layer=problem.feature_layer,
        timestep=feat_timestep,
        postprocess=problem.postprocess_features,
    )

    # Probe the extractor to obtain the feature dimensionality.
    probe, _ = setup.environment.base_sampler.sample(1, conditioning={}, device=device)
    feat = extractor(probe)
    if not isinstance(feat, torch.Tensor):
        raise TypeError(
            f"Feature extractor output must be a torch.Tensor, got {type(feat)}"
        )
    if feat.ndim != 2:
        raise ValueError(
            f"Feature extractor output must be a 2D tensor, got {feat.ndim}D tensor"
        )

    return setup, extractor, feat.shape[1]


def build_setup(
    problem: ProblemSetup,
    config: ExploreConfig,
    args: dict[str, Any],
    device: torch.device,
) -> tuple[FineTuningSetup, UncertaintyEstimator]:
    """Assemble the environment, then bind the surrogate into its reward.

    The reward is constructed unbound and bound here: `make()` builds the flow
    model, the feature extractor needs that model, and the surrogate needs the
    extractor, so none of them can exist before this point.
    """
    setup, extractor, feat_dim = assemble_setup(
        problem,
        reward_id="actflow/uncertainty",
        reward_kwargs={"gate": config.gate, "invalid_floor": config.invalid_floor},
        algorithm=config.algorithm,
        algorithm_kwargs=config.algorithm_kwargs,
        num_steps=config.num_steps,
        ft_lr=config.ft_lr,
        feat_timestep=config.feat_timestep,
        device=device,
        noise_scale=config.noise_scale,
    )

    uncertainty = uncertainty_estimators[args["uncertainty_estimator"]](
        extractor,
        feat_dim=feat_dim,
        device=device,
        args=args,
    )

    reward = setup.environment.reward
    if not isinstance(reward, ActFlowUncertaintyReward):
        raise TypeError(
            f"Expected an ActFlowUncertaintyReward, got {type(reward).__name__}."
        )
    reward.bind(estimator=uncertainty, verifier=problem.validity)
    return setup, uncertainty


def setup_and_run(args: argparse.Namespace):
    """Shared initialization and run logic."""
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ExploreConfig.construct_from_args(args)
    problem = problem_setups[args.problem_setup](vars(args), device=device)

    with open(config.folder / "args.yaml", "w") as f:
        yaml.safe_dump(serialize_args(args), f)

    logger = setup_logger(config.folder, args.verbose)
    logger.info("starting...")

    setup, uncertainty = build_setup(problem, config, vars(args), device)

    loop = ActFlowLoop(
        problem=problem,
        setup=setup,
        uncertainty=uncertainty,
        config=config,
        logger=logger,
    )
    loop.run()


def build_parser(add_extra_args=None):
    """Build the nested `<problem_setup> <uncertainty_estimator> [flags]` parser."""
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="problem_setup", required=True)

    for name, setup_cls in problem_setups.items():
        sub = subparsers.add_parser(name)
        uncertainty_subparsers = sub.add_subparsers(
            dest="uncertainty_estimator", required=True
        )

        for u_name, u_cls in uncertainty_estimators.items():
            u_sub = uncertainty_subparsers.add_parser(u_name)

            add_global_args(u_sub)
            if add_extra_args is not None:
                add_extra_args(u_sub)
            setup_cls.add_args(u_sub)
            u_cls.add_args(u_sub)

    return parser


def add_global_args(parser, *, algorithms: tuple[str, ...] = ALGORITHMS):
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    # Exploration sampling
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--samples_per_iter", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=100)

    # Surrogate
    parser.add_argument("--feat_timestep", type=float, default=0.9)

    # Reward
    parser.add_argument("--gate", type=str, choices=GATES, default="hard")
    parser.add_argument("--invalid_floor", type=float, default=1.0)

    # UpdateFlow
    parser.add_argument(
        "--algorithm", type=str, choices=algorithms, default="diffusion_nft"
    )
    parser.add_argument(
        "--algorithm_kwargs",
        type=json.loads,
        default={},
        help="JSON overrides for the algorithm constructor, e.g. '{\"beta\": 0.5}'",
    )
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument(
        "--noise_scale",
        type=float,
        default=0.75,
        help=(
            "SDE diffusion level for stochastic-rollout algorithms (e.g. "
            "mixture_replay under --guidance smc). Bounds how far a single "
            "rollout can wander before any acquisition-based reweighting "
            "happens; raise it if exploration stalls (far_count stays 0) "
            "despite an aggressive acquisition/zeta schedule."
        ),
    )
    parser.add_argument(
        "--warmup_iters",
        type=int,
        default=0,
        help=(
            "Leading iterations that observe but do not update the policy, "
            "seeding D before the surrogate has anything to say. Required "
            "(>= 1) for adjoint_matching."
        ),
    )

    # Checkpointing
    parser.add_argument("--ckpt_every", type=int, default=100)

    # Evaluation
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument(
        "--visualize_every",
        type=int,
        default=1,
        help=(
            "Iterations between figures. The figure is the most expensive part "
            "of an iteration on a long run; raise this to keep one practical."
        ),
    )

    # Logging
    parser.add_argument("--video_fps", type=int, default=4)
