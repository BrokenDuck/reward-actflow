"""The ActFlow acquisition reward: surrogate uncertainty, gated by validity.

This is the objective inside line 4 of Algorithm 1. Maximizing raw uncertainty
alone is not enough — uncertainty is just as high off the valid manifold as on
its frontier, so an unmodified `sigma` rewards a policy for leaving the region
entirely. Each gate below is a different way of saying "uncertain *and*
plausible".

Three gates, chosen by the `gate` argument:

  hard     r = sigma(x)               if v(x) else -invalid_floor
  mult     r = 1[v(x)] * sigma(x)
  soft     r = mu(x) * sigma(x)
  sigmoid  r = sigmoid(mu(x)) * sigma(x)

`hard` and `mult` consult the black-box verifier and are therefore not
differentiable, so Adjoint Matching cannot use them — `make()` refuses that
pairing with a clear message, which is the intended behaviour. `soft` and
`sigmoid` replace the verifier with the surrogate's own posterior mean over
validity labels, which is smooth and differentiable end to end.

`soft` and `sigmoid` differ only in how they behave before the verifier has
returned both answers. Labels are z-scored, so while every observation agrees
they all map to exactly 0, the posterior mean is flat, and `soft` collapses to
`r == 0` everywhere — a correct statement that constant labels carry no
information, but a dead objective, and one Adjoint Matching rejects outright
because the terminal-cost gradient is then zero. `sigmoid` degrades to
`sigma / 2` instead, i.e. falls back to pure uncertainty until validity becomes
informative. On a problem whose base model starts entirely inside the valid
region — the staircase toy is exactly that — `soft` cannot bootstrap at all and
`sigmoid` is the gate to use.

Two further gates exist only as ablations, one for each half of the product:

  raw        r = sigma(x)                             (--no_verifier)
  validity   r = sigma(x) if v(x) else -invalid_floor,
             with sigma pinned to 1                   (--no_uncertainty)
"""

from collections.abc import Mapping
from typing import Any, Literal

import torch
from diffusiongym.registry import reward_provider_registry
from diffusiongym.types import DDBatch
from torch import Tensor

from reward_actflow.rewards.base import SoftGateTerminalCost, SurrogateReward, Verifier
from reward_actflow.uncertainty import UncertaintyEstimator

type Gate = Literal["hard", "mult", "soft", "sigmoid", "raw", "validity"]

#: The gates that combine both signals. `raw` and `validity` are ablations.
GATES: tuple[Gate, ...] = ("hard", "mult", "soft", "sigmoid", "raw", "validity")

#: Gates whose reward is differentiable w.r.t. the terminal latent.
DIFFERENTIABLE_GATES: frozenset[str] = frozenset({"soft", "sigmoid", "raw"})

#: Gates that query the black-box verifier.
VERIFIER_GATES: frozenset[str] = frozenset({"hard", "mult", "validity"})


class ActFlowUncertaintyReward[D: DDBatch, RawT](SurrogateReward[D, RawT]):
    """`RewardEvaluator` scoring samples by validity-gated surrogate uncertainty.

    Constructed unbound, because of an ordering constraint: `make()` builds the
    flow model, the feature extractor needs that model, and the surrogate needs
    the extractor — none of which exist when the registry instantiates this. The
    caller wires it up afterwards with `bind()`.

    The surrogate is held by reference and refitted in place each iteration, so
    a frozen `FlowEnvironment` holding this object still sees a current
    `sigma_t`.
    """

    def __init__(self, *, gate: Gate = "hard", invalid_floor: float = 1.0) -> None:
        super().__init__()
        if gate not in GATES:
            raise ValueError(f"Unknown gate {gate!r}. Expected one of {GATES}.")
        if invalid_floor < 0:
            raise ValueError(
                f"invalid_floor must be non-negative, got {invalid_floor}."
            )

        self.gate: Gate = gate
        self.invalid_floor = invalid_floor

        self._estimator: UncertaintyEstimator[D] | None = None
        self._verifier: Verifier[RawT] | None = None

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def bind(
        self,
        *,
        estimator: UncertaintyEstimator[D],
        verifier: Verifier[RawT],
    ) -> None:
        """Attach the live surrogate and the black-box verifier."""
        self._estimator = estimator
        self._verifier = verifier

    @property
    def is_bound(self) -> bool:
        return self._estimator is not None and self._verifier is not None

    @property
    def uses_verifier(self) -> bool:
        """Whether scoring queries the black-box verifier."""
        return self.gate in VERIFIER_GATES

    @property
    def is_differentiable(self) -> bool:
        return self.gate in DIFFERENTIABLE_GATES

    def _require_binding(
        self,
    ) -> tuple[UncertaintyEstimator[D], Verifier[RawT]]:
        if self._estimator is None or self._verifier is None:
            raise RuntimeError(
                "ActFlowUncertaintyReward was used before bind(). Call "
                "reward.bind(estimator=..., verifier=...) after diffusiongym.make()."
            )
        return self._estimator, self._verifier

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(
        self,
        *,
        sample: RawT,
        latent: D,
        conditioning: Mapping[str, Any],
    ) -> Tensor:
        """Scalar reward per sample, shape `(n,)`.

        Differentiable w.r.t. `latent` when `gate == "soft"`.
        """
        estimator, verifier = self._require_binding()
        cond = dict(conditioning)

        mean, uncertainty = estimator.mean_and_uncertainty(latent, **cond)

        labels: Tensor | None = None
        if self.uses_verifier:
            labels = verifier(sample, cond).to(uncertainty.device)

        match self.gate:
            case "soft":
                rewards = mean * uncertainty
            case "sigmoid":
                rewards = torch.sigmoid(mean) * uncertainty
            case "raw":
                rewards = uncertainty
            case "mult":
                assert labels is not None
                rewards = labels.to(uncertainty.dtype) * uncertainty
            case "hard" | "validity":
                assert labels is not None
                # The `validity` ablation is the same shape with sigma pinned
                # to 1, so the reward carries validity and nothing else.
                signal = (
                    uncertainty if self.gate == "hard" else torch.ones_like(uncertainty)
                )
                floor = torch.full_like(uncertainty, -self.invalid_floor)
                rewards = torch.where(labels, signal, floor)

        self.last_verifier_labels = labels
        self.last_uncertainty = uncertainty.detach()
        self.last_mean = mean.detach()

        return rewards


@reward_provider_registry.register("actflow/uncertainty")
class UncertaintyRewardProvider:
    """Pairs the acquisition reward with its differentiable form, if it has one."""

    domain = "actflow"

    def __init__(self, *, gate: Gate = "hard", invalid_floor: float = 1.0) -> None:
        self._reward: ActFlowUncertaintyReward = ActFlowUncertaintyReward(
            gate=gate, invalid_floor=invalid_floor
        )

    def reward(self) -> ActFlowUncertaintyReward:
        # The same instance every time: the terminal cost wraps it, and the
        # caller binds the surrogate to whatever `make()` put in the environment.
        return self._reward

    def terminal_cost(self) -> SoftGateTerminalCost | None:
        if not self._reward.is_differentiable:
            return None
        return SoftGateTerminalCost(self._reward)
