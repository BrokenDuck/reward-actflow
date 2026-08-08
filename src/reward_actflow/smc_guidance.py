"""Inference-time guidance for ActFlow-R's self-generation step (Algorithm 1
line 4), via SMC (`diffusiongym.core.smc`).

Solves "self-generate x ~ argmax_q E_q[a_t] - beta*KL(q || p_theta)" without
touching p_theta: particles are drawn from the model's own SDE and
periodically resampled in proportion to exp(a_t(x_hat1)/beta). This is what
gives ActFlow-R's acquisition (zeta, beta_r, sigma^v) a channel into theta at
all — a weighted flow-matching refit alone has none (see `actflow_r.py`'s
module docstring for the full argument).

`--guidance none` skips this file's SMC path entirely: `MixtureReplay.collect()`
(a plain SDE rollout) is used as-is, and a_t is computed and logged only —
Algorithm 2 exactly as written, with zeta/beta_r inert.

Kept out of `trainers/mixture_replay.py` so that trainer stays testable
without any surrogate at all (see its module docstring).
"""

from collections.abc import Mapping
from typing import Any, Literal

from diffusiongym.core import RolloutRequest
from diffusiongym.core.smc import SMCSampler
from diffusiongym.trainers.base import FineTuningContext
from torch import Generator, Tensor

from reward_actflow.rewards.acquisition import ActFlowRAcquisitionReward
from reward_actflow.trainers.mixture_replay import MixtureExperience

type Conditioning = Mapping[str, Any]
type Guidance = Literal["none", "smc"]
type ResampleMethod = Literal["systematic", "multinomial"]

GUIDANCE_MODES: tuple[Guidance, ...] = ("none", "smc")


def smc_collect(
    *,
    context: FineTuningContext,
    dynamics: Any,
    acq_beta: float,
    n: int,
    time_grid: Tensor,
    conditioning: Conditioning,
    ess_threshold: float = 0.5,
    resample: ResampleMethod = "systematic",
    generator: Generator | None = None,
) -> MixtureExperience:
    """Draw `n` particles from `p_theta` twisted by `a_t / acq_beta`.

    Reads the acquisition off `context.environment.reward` rather than taking
    it as a separate parameter — the two would otherwise have to be kept in
    sync by the caller, and only the environment's own reward instance is
    guaranteed bound to the surrogates `make()` wired in.

    Builds an `SMCSampler` from `context.sde_sampler`'s own geometry and
    kernel factory, so this needs no change to `FineTuningContext` or
    `make()`.
    """
    if acq_beta <= 0:
        raise ValueError(f"acq_beta must be positive, got {acq_beta}.")

    reward = context.environment.reward
    if not isinstance(reward, ActFlowRAcquisitionReward):
        raise TypeError(
            "smc_collect needs context.environment.reward to be an "
            f"ActFlowRAcquisitionReward, got {type(reward).__name__}."
        )

    sampler = SMCSampler(
        context.sde_sampler.geometry,
        context.sde_sampler.kernel_factory,
        ess_threshold=ess_threshold,
        resample=resample,
    )

    def log_potential(x1: Any, t: Tensor) -> Tensor:
        # Passing the latent as `sample` is exactly what `SoftGateTerminalCost`
        # already does (rewards/base.py) — safe here because a_t never
        # queries the verifier (ActFlowRAcquisitionReward's class docstring).
        return reward.score(sample=x1, latent=x1, conditioning=conditioning) / acq_beta

    rollout = sampler.rollout(
        environment=context.environment,
        model=context.policies.rollout,
        dynamics=dynamics,
        n=n,
        conditioning=conditioning,
        request=RolloutRequest(time_grid=time_grid, evaluate_reward=True),
        log_potential=log_potential,
        generator=generator,
    )
    assert rollout.reward is not None
    return MixtureExperience(
        latent=rollout.terminal_latent,
        rewards=rollout.reward.rewards,
        valid=rollout.reward.valid,
        conditioning=rollout.conditioning,
    )


def collect_with_guidance(
    *,
    guidance: Guidance,
    context: FineTuningContext,
    algorithm: Any,
    dynamics: Any,
    n: int,
    time_grid: Tensor,
    conditioning: Conditioning,
    acq_beta: float = 1.0,
    ess_threshold: float = 0.5,
    resample: ResampleMethod = "systematic",
    generator: Generator | None = None,
) -> MixtureExperience:
    """Collect one batch, guided by `a_t` (`guidance="smc"`) or not (`"none"`).

    `"none"` delegates to `algorithm.collect()` — whatever `MixtureReplay`
    would do on its own — so the default run reproduces Algorithm 2 exactly
    as written, with `zeta`/`beta_r` computed and logged but not steering
    anything (see the module docstring).
    """
    if guidance == "none":
        return algorithm.collect(
            context=context,
            dynamics=dynamics,
            n=n,
            time_grid=time_grid,
            conditioning=conditioning,
            generator=generator,
        )
    if guidance == "smc":
        return smc_collect(
            context=context,
            dynamics=dynamics,
            acq_beta=acq_beta,
            n=n,
            time_grid=time_grid,
            conditioning=conditioning,
            ess_threshold=ess_threshold,
            resample=resample,
            generator=generator,
        )
    raise ValueError(
        f"Unknown guidance {guidance!r}. Expected one of {GUIDANCE_MODES}."
    )
