"""ActFlow-R's acquisition reward: optimism over two surrogates.

The objective inside lines 3-4 of ActFlow-R's Algorithm 1:

    a_t(x) = zeta_t * sigma^v_t(phi_s^t(x))
             + (1 - zeta_t) * (mu^r_t + beta^r * sigma^r_t)(phi_s^t(x))

`sigma^v` is the validity surrogate's uncertainty (as in base ActFlow); `mu^r`,
`sigma^r` are the reward surrogate's posterior mean and uncertainty, fit on the
verified-valid subset only (Algorithm 1 line 2). Every `UncertaintyEstimator`
reports standard deviation (`uncertainty/gp.py`, `uncertainty/ensemble.py`), so
the two terms are on the same scale and `zeta` is a genuine interpolation
between them, not a sum of mismatched units.

By default `a_t` has no validity gate and never queries the black-box
verifier, so it is unconditionally differentiable w.r.t. `latent`. The
appendix's extension (F) — a *soft* validity gate on the reward term only —
is implemented as an opt-in (`validity_gate=True`), still off by default:

    a_t(x) = zeta*sigma^v(x) + (1-zeta) * g(x) * (mu^r(x) + beta_r*sigma^r(x))
    g(x)   = sigmoid(mu^v(x))   if validity_gate else 1

`mu^v` is the validity surrogate's own z-scored posterior mean, so `g` needs
no extra verifier query and stays differentiable. It exists because `mu^r` is
a smooth regressor with no notion of validity: near a region's boundary it
keeps extrapolating higher reward just past the edge, and unguarded, SMC
guidance chases that dead zone instead of paying for the (locally
reward-flat) detour needed to reach a disconnected higher-reward region — the
toy's own `top`-to-`bottom` corridor is exactly this shape. `g` suppresses
the reward term wherever the validity surrogate itself doubts the point is
reachable, without touching the `sigma^v` explore term (see
`sigmoid` in `rewards/uncertainty.py` for the same mu^v-as-soft-label idea
applied there).
"""

from collections.abc import Mapping
from typing import Any

import torch
from diffusiongym.registry import reward_provider_registry
from diffusiongym.types import DDBatch
from torch import Tensor

from reward_actflow.rewards.base import SoftGateTerminalCost, SurrogateReward, Verifier
from reward_actflow.uncertainty import UncertaintyEstimator


class ActFlowRAcquisitionReward[D: DDBatch, RawT](SurrogateReward[D, RawT]):
    """`RewardEvaluator` scoring `a_t = zeta*sigma^v + (1-zeta)*(mu^r + beta_r*sigma^r)`.

    Constructed unbound, the same ordering constraint as
    `ActFlowUncertaintyReward`: `make()` builds the flow model before the
    feature extractor (which needs the model) or a surrogate (which needs the
    extractor) can exist.

    `bind()` attaches the validity surrogate and mirrors
    `ActFlowUncertaintyReward.bind()`'s signature exactly, so assembly code
    that always calls `reward.bind(estimator=..., verifier=...)` works
    unchanged whichever reward it is wiring up — `verifier` is accepted only
    for that uniformity and is never used, since `a_t` has no validity gate.
    `bind_reward_surrogate()` is a separate call for the second estimator, so
    the two surrogates are wired (and can be re-fit) independently: the
    reward surrogate sees only the verified-valid subset and, by design, is
    one step stale relative to the validity surrogate (see `ActFlowRLoop`).
    """

    def __init__(self, *, beta_r: float = 1.0, validity_gate: bool = False) -> None:
        super().__init__()
        if beta_r < 0:
            raise ValueError(f"beta_r must be non-negative, got {beta_r}.")

        self.beta_r = beta_r
        self.validity_gate = validity_gate
        self._zeta = 1.0

        self._validity_estimator: UncertaintyEstimator[D] | None = None
        self._reward_estimator: UncertaintyEstimator[D] | None = None

        # Populated by score(), same "read instead of re-querying" reason as
        # the base class's cache — mirrors the *validity* surrogate's outputs
        # (last_uncertainty/last_mean), matching what ActFlowLoop.run()
        # already reads generically. The reward surrogate's own outputs get
        # their own fields so ActFlowRLoop's diagnostics can read both.
        self.last_reward_mean: Tensor | None = None
        self.last_reward_uncertainty: Tensor | None = None

    def clear_cache(self) -> None:
        super().clear_cache()
        self.last_reward_mean = None
        self.last_reward_uncertainty = None

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def bind(
        self,
        *,
        estimator: UncertaintyEstimator[D],
        verifier: Verifier[RawT] | None = None,
    ) -> None:
        """Attach the validity surrogate sigma^v. `verifier` is accepted and
        ignored — see the class docstring."""
        self._validity_estimator = estimator

    def bind_reward_surrogate(self, *, estimator: UncertaintyEstimator[D]) -> None:
        """Attach the reward surrogate `(mu^r, sigma^r)`, fit on `E_t`."""
        self._reward_estimator = estimator

    @property
    def is_bound(self) -> bool:
        return (
            self._validity_estimator is not None and self._reward_estimator is not None
        )

    @property
    def is_differentiable(self) -> bool:
        # Always differentiable w.r.t. latent — unlike ActFlowUncertaintyReward,
        # this never depends on a gate choice or the black-box verifier: even
        # the extension-(F) gate reads mu^v (a smooth posterior mean), not the
        # verifier itself.
        return True

    def set_zeta(self, zeta: float) -> None:
        """Set the explore/exploit weight for the next `score()` call.

        A mutable scalar read through the (frozen) `FlowEnvironment` this
        reward lives in — the same mechanism the surrogates already use to
        stay current without rebuilding the environment each iteration.
        """
        if not (0.0 <= zeta <= 1.0):
            raise ValueError(f"zeta must be in [0, 1], got {zeta}.")
        self._zeta = zeta

    @property
    def zeta(self) -> float:
        return self._zeta

    def _require_binding(
        self,
    ) -> tuple[UncertaintyEstimator[D], UncertaintyEstimator[D]]:
        if self._validity_estimator is None or self._reward_estimator is None:
            raise RuntimeError(
                "ActFlowRAcquisitionReward was used before both surrogates "
                "were bound. Call reward.bind(estimator=...) for the "
                "validity surrogate and reward.bind_reward_surrogate("
                "estimator=...) for the reward surrogate."
            )
        return self._validity_estimator, self._reward_estimator

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
        """`a_t(x)`, shape `(n,)`. Differentiable w.r.t. `latent`."""
        validity_estimator, reward_estimator = self._require_binding()
        cond = dict(conditioning)

        mean_v, sigma_v = validity_estimator.mean_and_uncertainty(latent, **cond)
        mean_r, sigma_r = reward_estimator.mean_and_uncertainty(latent, **cond)

        gate = torch.sigmoid(mean_v) if self.validity_gate else torch.ones_like(mean_v)
        acquisition = self._zeta * sigma_v + (1.0 - self._zeta) * gate * (
            mean_r + self.beta_r * sigma_r
        )

        self.last_verifier_labels = None
        self.last_uncertainty = sigma_v.detach()
        self.last_mean = mean_v.detach()
        self.last_reward_mean = mean_r.detach()
        self.last_reward_uncertainty = sigma_r.detach()

        return acquisition


@reward_provider_registry.register("actflow/acquisition")
class AcquisitionRewardProvider:
    """Pairs the ActFlow-R acquisition reward with its terminal cost.

    Unlike `UncertaintyRewardProvider`, `terminal_cost()` never returns
    `None`: `a_t` is unconditionally differentiable.
    """

    domain = "actflow"

    def __init__(self, *, beta_r: float = 1.0, validity_gate: bool = False) -> None:
        self._reward: ActFlowRAcquisitionReward = ActFlowRAcquisitionReward(
            beta_r=beta_r, validity_gate=validity_gate
        )

    def reward(self) -> ActFlowRAcquisitionReward:
        # The same instance every time: the terminal cost wraps it, and the
        # caller binds both surrogates to whatever build_actflow_r_setup put
        # in the environment.
        return self._reward

    def terminal_cost(self) -> SoftGateTerminalCost:
        return SoftGateTerminalCost(self._reward)
