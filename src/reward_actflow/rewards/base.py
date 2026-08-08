"""Shared machinery for ActFlow acquisition rewards.

Both `ActFlowUncertaintyReward` (base ActFlow, Algorithm 1's validity-gated
uncertainty) and `ActFlowRAcquisitionReward` (ActFlow-R's optimism over two
surrogates) implement diffusiongym's `RewardEvaluator` protocol the same way:
score a batch, cache what the score depended on, and wrap it into a
`RewardBatch`. The cache exists because `collect()` evaluates the reward
internally and `EndpointExperience` does not carry `RewardBatch.metadata`
through — the loop reads `last_uncertainty` etc. off the reward object instead
of re-querying.

What differs between the two — what needs binding, and what "differentiable"
depends on — is specific to each reward and stays on the subclass rather than
being forced into a shared shape here.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any

from diffusiongym.core.reward import RewardBatch
from diffusiongym.types import DDBatch
from torch import Tensor

#: A black-box verifier `v(x)`, Algorithm 1 line 5. Shared here (rather than
#: defined once per reward module) because `ActFlowRAcquisitionReward.bind()`
#: accepts one for call-site uniformity with `ActFlowUncertaintyReward.bind()`
#: even though `a_t` never queries it.
type Verifier[D] = Callable[[D, dict[str, Any]], Tensor]


class SurrogateReward[D: DDBatch, RawT](ABC):
    """Common cache/dispatch for a reward built on one or more surrogates.

    Subclasses are constructed unbound: `make()` builds the flow model before
    the feature extractor (which needs the model) or a surrogate (which needs
    the extractor) can exist, so the caller wires dependencies up afterwards
    with whatever `bind*` method(s) the subclass exposes.
    """

    def __init__(self) -> None:
        # Populated by score(); see the module docstring for why this cache
        # exists. Only valid immediately after the call that produced it, so a
        # caller is expected to `clear_cache()` before whatever it wants to
        # read the results of — an algorithm that never evaluates the reward
        # (Adjoint Matching goes through the terminal cost instead) would
        # otherwise leave the previous iteration's values sitting here looking
        # current.
        self.last_verifier_labels: Tensor | None = None
        self.last_uncertainty: Tensor | None = None
        self.last_mean: Tensor | None = None

    def clear_cache(self) -> None:
        """Forget the last scored batch."""
        self.last_verifier_labels = None
        self.last_uncertainty = None
        self.last_mean = None

    @property
    @abstractmethod
    def is_bound(self) -> bool:
        """Whether every dependency `score()` needs has been attached."""

    @property
    @abstractmethod
    def is_differentiable(self) -> bool:
        """Whether `score()` is differentiable w.r.t. `latent`."""

    @abstractmethod
    def score(
        self,
        *,
        sample: RawT,
        latent: D,
        conditioning: Mapping[str, Any],
    ) -> Tensor:
        """Scalar reward per sample, shape `(n,)`."""

    def __call__(
        self,
        *,
        sample: RawT,
        latent: D,
        conditioning: Mapping[str, Any],
    ) -> RewardBatch:
        rewards = self.score(sample=sample, latent=latent, conditioning=conditioning)
        # `RewardBatch.valid` means "this reward is defined", not "the verifier
        # passed". Reporting verifier failures here would make DiffusionNFT drop
        # the invalid samples from the very reward statistics that are supposed
        # to separate them (`diffusion_nft.py:123-125`).
        return RewardBatch(
            rewards=rewards.detach(),
            valid=None,
            metadata={
                "verifier": self.last_verifier_labels,
                "uncertainty": self.last_uncertainty,
                "mean": self.last_mean,
            },
        )


class SoftGateTerminalCost[D: DDBatch]:
    """Differentiable terminal cost for Adjoint Matching.

    Adjoint Matching *minimizes* the terminal cost, so this returns `-r`. Getting
    that sign wrong trains the model to avoid uncertainty, silently. Works for
    any `SurrogateReward`, not just the `soft`/`sigmoid` gates the name
    originally referred to — kept as-is because it is imported by name from
    several call sites.
    """

    def __init__(self, reward: SurrogateReward[D, Any]) -> None:
        if not reward.is_differentiable:
            raise ValueError(
                f"{type(reward).__name__} has no gradient (it consults a "
                "black-box verifier or similar). Only a differentiable reward "
                "can supply a terminal cost."
            )
        self.reward = reward

    def __call__(
        self,
        terminal_latent: D,
        *,
        conditioning: Mapping[str, Any],
    ) -> Tensor:
        # Deliberately not under no_grad: the adjoint is d(cost)/d(x_K).
        return -self.reward.score(
            sample=terminal_latent,
            latent=terminal_latent,
            conditioning=conditioning,
        )
