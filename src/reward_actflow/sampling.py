"""Drawing samples from a policy, independent of how it is being trained."""

from typing import Any

import torch
from diffusiongym import make_time_grid
from diffusiongym.core.dynamics import FlowDynamics, ProbabilityFlowODE
from diffusiongym.core.rollout import RolloutRequest
from diffusiongym.trainers.base import FineTuningContext
from diffusiongym.types import DDBatch
from torch import Tensor


@torch.no_grad()
def sample_policy[D: DDBatch](
    context: FineTuningContext,
    n: int,
    *,
    dynamics: FlowDynamics | None = None,
    time_grid: Tensor | None = None,
    steps: int = 100,
    batch_size: int = 8192,
    policy: str = "train",
    conditioning: dict[str, Any] | None = None,
) -> D:
    """Draw `n` terminal latents from one of the context's policies.

    Pass the `dynamics` and `time_grid` the algorithm trains under. Defaulting to
    the probability-flow ODE regardless — which this used to do, on the reasoning
    that the SDE noise is for the *learner* and the ODE is what the model "really"
    represents — is wrong for a policy fine-tuned under an SDE. The two share
    marginals only while the velocity field is the true one for some path, and
    fine-tuning breaks that: the field is optimized against the SDE rollout only,
    so the ODE marginal is free to drift.

    It is not a subtle effect. On a Flow-GRPO policy from a 200-iteration run,
    the same weights gave a 0.017 valid rate and 0.048 coverage under the ODE
    against 0.900 and 0.260 under its own SDE — the difference between reading
    the run as a collapse and reading it as the best result so far.

    Reward evaluation is switched off: these draws are large (50k for a support
    histogram) and scoring them would mean that many surrogate posterior
    evaluations per frame, and would also clobber the reward's cached verifier
    labels for the iteration in flight.
    """
    model = getattr(context.policies, policy)
    device = model.device

    if dynamics is None:
        dynamics = ProbabilityFlowODE()
    if time_grid is None:
        time_grid = make_time_grid(steps, stochastic=dynamics.stochastic, device=device)

    sampler = context.sde_sampler if dynamics.stochastic else context.ode_sampler
    request = RolloutRequest(time_grid=time_grid, evaluate_reward=False)

    chunks: list[D] = []
    remaining = n
    while remaining > 0:
        size = min(batch_size, remaining)
        rollout = sampler.rollout(
            environment=context.environment,
            model=model,
            dynamics=dynamics,
            n=size,
            conditioning=dict(conditioning or {}),
            request=request,
        )
        chunks.append(rollout.terminal_latent.detach())
        remaining -= size

    return type(chunks[0]).concat(chunks)
