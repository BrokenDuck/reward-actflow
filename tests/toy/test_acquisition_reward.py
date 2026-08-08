"""Tests for ActFlow-R's acquisition reward: optimism over two surrogates."""

import diffusiongym
import pytest
import torch

import reward_actflow.rewards  # noqa: F401  (registers actflow/acquisition)
from reward_actflow.rewards.acquisition import ActFlowRAcquisitionReward
from reward_actflow.rewards.base import SoftGateTerminalCost
from reward_actflow.uncertainty import UncertaintyEstimator


class _StubEstimator(UncertaintyEstimator):
    """Surrogate with hand-chosen outputs, so score() is checkable exactly."""

    def __init__(self, mean, uncertainty):
        self.mean = mean
        self.uncertainty = uncertainty
        self.num_observations = 1

    def _init_estimator(self): ...

    def _update_estimator(self, feats, labels): ...

    def _mean_and_uncertainty(self, feats):
        return self.mean, self.uncertainty

    def mean_and_uncertainty(self, latent, **conditioning):
        return self.mean, self.uncertainty


def _bound_reward(
    *, sigma_v, mu_r, sigma_r, beta_r=1.0, mu_v=None, validity_gate=False
):
    reward = ActFlowRAcquisitionReward(beta_r=beta_r, validity_gate=validity_gate)
    mu_v = torch.zeros_like(sigma_v) if mu_v is None else mu_v
    reward.bind(estimator=_StubEstimator(mu_v, sigma_v), verifier=None)
    reward.bind_reward_surrogate(estimator=_StubEstimator(mu_r, sigma_r))
    return reward


def test_acquisition_provider_is_registered():
    assert "actflow/acquisition" in diffusiongym.reward_provider_registry


def test_acquisition_reduces_to_pure_validity_uncertainty_at_zeta_one():
    sigma_v = torch.tensor([0.3, 1.7])
    reward = _bound_reward(
        sigma_v=sigma_v,
        mu_r=torch.tensor([5.0, -5.0]),
        sigma_r=torch.tensor([9.0, 9.0]),
        beta_r=2.0,
    )
    reward.set_zeta(1.0)
    a = reward.score(sample=None, latent=torch.zeros(2, 2), conditioning={})
    assert torch.allclose(a, sigma_v)


def test_acquisition_reduces_to_reward_ucb_at_zeta_zero():
    mu_r = torch.tensor([1.0, -2.0])
    sigma_r = torch.tensor([0.5, 4.0])
    beta_r = 1.5
    reward = _bound_reward(
        sigma_v=torch.tensor([0.3, 1.7]), mu_r=mu_r, sigma_r=sigma_r, beta_r=beta_r
    )
    reward.set_zeta(0.0)
    a = reward.score(sample=None, latent=torch.zeros(2, 2), conditioning={})
    assert torch.allclose(a, mu_r + beta_r * sigma_r)


def test_acquisition_interpolates_between_the_two_terms():
    reward = _bound_reward(
        sigma_v=torch.tensor([1.0]),
        mu_r=torch.tensor([0.0]),
        sigma_r=torch.tensor([0.0]),
    )
    reward.set_zeta(0.5)
    a = reward.score(sample=None, latent=torch.zeros(1, 2), conditioning={})
    assert torch.allclose(a, torch.tensor([0.5]))


def test_beta_r_raises_the_score_of_uncertain_reward_points():
    """Pessimism gets you `lcb`; the acquisition uses the opposite sign, so
    `beta_r` should raise, not lower, the score of an uncertain reward point —
    it is the optimism half of the algorithm."""
    low_beta = _bound_reward(
        sigma_v=torch.tensor([0.0]),
        mu_r=torch.tensor([0.0]),
        sigma_r=torch.tensor([2.0]),
        beta_r=0.1,
    )
    high_beta = _bound_reward(
        sigma_v=torch.tensor([0.0]),
        mu_r=torch.tensor([0.0]),
        sigma_r=torch.tensor([2.0]),
        beta_r=5.0,
    )
    low_beta.set_zeta(0.0)
    high_beta.set_zeta(0.0)
    a_low = low_beta.score(sample=None, latent=torch.zeros(1, 2), conditioning={})
    a_high = high_beta.score(sample=None, latent=torch.zeros(1, 2), conditioning={})
    assert a_high.item() > a_low.item()


def test_acquisition_never_queries_the_verifier():
    """No validity gate on a_t (extension F is off): score() must not need a
    verifier at all, unlike ActFlowUncertaintyReward's hard/mult/validity
    gates. Bound with verifier=None above; scoring must not raise."""
    reward = _bound_reward(
        sigma_v=torch.tensor([1.0]),
        mu_r=torch.tensor([0.0]),
        sigma_r=torch.tensor([0.0]),
    )
    a = reward.score(sample="unused", latent=torch.zeros(1, 2), conditioning={})
    assert torch.isfinite(a).all()


def test_acquisition_is_differentiable_regardless_of_zeta():
    reward = ActFlowRAcquisitionReward()
    latent = torch.tensor([[0.3, 0.4]], requires_grad=True)
    reward.bind(
        estimator=_StubEstimator(latent[:, 0], latent[:, 1].abs()), verifier=None
    )
    reward.bind_reward_surrogate(
        estimator=_StubEstimator(latent[:, 0], latent[:, 1].abs())
    )
    assert reward.is_differentiable

    cost = SoftGateTerminalCost(reward)(latent, conditioning={})
    grad = torch.autograd.grad(cost.sum(), latent)[0]
    assert torch.isfinite(grad).all()
    assert grad.abs().sum() > 0


def test_zeta_is_range_checked():
    reward = ActFlowRAcquisitionReward()
    with pytest.raises(ValueError, match="zeta"):
        reward.set_zeta(-0.1)
    with pytest.raises(ValueError, match="zeta"):
        reward.set_zeta(1.1)


def test_beta_r_is_range_checked():
    with pytest.raises(ValueError, match="beta_r"):
        ActFlowRAcquisitionReward(beta_r=-1.0)


def test_reward_raises_before_both_surrogates_are_bound():
    reward = ActFlowRAcquisitionReward()
    assert not reward.is_bound
    with pytest.raises(RuntimeError, match="before both surrogates"):
        reward.score(sample=None, latent=torch.zeros(2, 2), conditioning={})

    reward.bind(estimator=_StubEstimator(torch.zeros(2), torch.ones(2)), verifier=None)
    assert not reward.is_bound  # reward surrogate still missing
    with pytest.raises(RuntimeError, match="before both surrogates"):
        reward.score(sample=None, latent=torch.zeros(2, 2), conditioning={})

    reward.bind_reward_surrogate(
        estimator=_StubEstimator(torch.zeros(2), torch.ones(2))
    )
    assert reward.is_bound


def test_clear_cache_resets_reward_surrogate_fields_too():
    reward = _bound_reward(
        sigma_v=torch.tensor([1.0]),
        mu_r=torch.tensor([2.0]),
        sigma_r=torch.tensor([3.0]),
    )
    reward.score(sample=None, latent=torch.zeros(1, 2), conditioning={})
    assert reward.last_reward_mean is not None
    assert reward.last_reward_uncertainty is not None

    reward.clear_cache()
    assert reward.last_reward_mean is None
    assert reward.last_reward_uncertainty is None
    assert reward.last_uncertainty is None


# ---------------------------------------------------------------------------
# Extension (F): the soft validity gate on the reward term
# ---------------------------------------------------------------------------


def test_validity_gate_off_by_default_reduces_to_the_ungated_acquisition():
    mu_v = torch.tensor([-9.0, 9.0])  # would gate hard if it mattered
    gated_off = _bound_reward(
        sigma_v=torch.tensor([0.3, 0.3]),
        mu_r=torch.tensor([2.0, 2.0]),
        sigma_r=torch.tensor([1.0, 1.0]),
        mu_v=mu_v,
        validity_gate=False,
    )
    gated_off.set_zeta(0.0)
    a = gated_off.score(sample=None, latent=torch.zeros(2, 2), conditioning={})
    assert torch.allclose(a, torch.tensor([3.0, 3.0]))


def test_validity_gate_suppresses_the_reward_term_where_mu_v_is_low():
    """This is the mechanism extension (F) exists for: mu^r keeps
    extrapolating optimistically past a region's boundary, and unguarded that
    drags the acquisition toward a point the validity surrogate itself
    doubts is reachable. sigmoid(mu_v) should pull the score down toward the
    pure-explore (sigma_v) term as mu_v -> -inf, and leave it near the
    ungated value as mu_v -> +inf."""
    doubtful = _bound_reward(
        sigma_v=torch.tensor([0.3]),
        mu_r=torch.tensor([10.0]),
        sigma_r=torch.tensor([0.0]),
        mu_v=torch.tensor([-20.0]),
        validity_gate=True,
    )
    confident = _bound_reward(
        sigma_v=torch.tensor([0.3]),
        mu_r=torch.tensor([10.0]),
        sigma_r=torch.tensor([0.0]),
        mu_v=torch.tensor([20.0]),
        validity_gate=True,
    )
    doubtful.set_zeta(0.0)
    confident.set_zeta(0.0)
    a_doubtful = doubtful.score(sample=None, latent=torch.zeros(1, 2), conditioning={})
    a_confident = confident.score(
        sample=None, latent=torch.zeros(1, 2), conditioning={}
    )
    assert a_doubtful.item() == pytest.approx(0.0, abs=1e-6)
    assert a_confident.item() == pytest.approx(10.0, abs=1e-3)


def test_validity_gate_never_touches_the_explore_term():
    """The gate only scales (mu^r + beta_r*sigma^r); at zeta=1 the score is
    pure sigma^v regardless of the gate, or the paper's own zeta=1 reduction
    (base ActFlow) would silently change behaviour whenever (F) is on."""
    reward = _bound_reward(
        sigma_v=torch.tensor([0.7]),
        mu_r=torch.tensor([10.0]),
        sigma_r=torch.tensor([5.0]),
        mu_v=torch.tensor([-20.0]),
        validity_gate=True,
    )
    reward.set_zeta(1.0)
    a = reward.score(sample=None, latent=torch.zeros(1, 2), conditioning={})
    assert torch.allclose(a, torch.tensor([0.7]))


def test_validity_gate_is_differentiable():
    reward = ActFlowRAcquisitionReward(validity_gate=True)
    latent = torch.tensor([[0.3, 0.4]], requires_grad=True)
    reward.bind(
        estimator=_StubEstimator(latent[:, 0], latent[:, 1].abs()), verifier=None
    )
    reward.bind_reward_surrogate(
        estimator=_StubEstimator(latent[:, 0], latent[:, 1].abs())
    )
    assert reward.is_differentiable

    cost = SoftGateTerminalCost(reward)(latent, conditioning={})
    grad = torch.autograd.grad(cost.sum(), latent)[0]
    assert torch.isfinite(grad).all()
