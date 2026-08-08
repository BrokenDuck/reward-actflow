"""Tests for reward_actflow.diagnostics — the evidence for whether an
appendix extension is warranted, not the extensions themselves."""

import pytest
import torch

from reward_actflow.diagnostics import (
    best_reward_beyond,
    cluster_count,
    coordinate_bounds,
    effective_sample_size,
    heldout_r2,
    per_cell_density_ratio,
)
from reward_actflow.uncertainty.ensemble import EnsembleUncertaintyEstimator
from reward_actflow.uncertainty.gp import GPUncertaintyEstimator
from reward_actflow.uncertainty.uncertainty_estimator import FlowFeatureExtractor

DEVICE = torch.device("cpu")


def _gp_estimator(
    *, lengthscale: float = 0.3, feat_dim: int = 2
) -> GPUncertaintyEstimator:
    """A real GPUncertaintyEstimator on the toy's identity feature layer, so
    heldout_r2's `_extract_all`/`_update_estimator` calls exercise the real
    code path rather than a stub."""
    extractor = FlowFeatureExtractor(
        model=None,
        geometry=None,
        schedule=None,
        layer="input",
        timestep=0.9,
        postprocess=lambda x1, feats: feats.data,
    )
    return GPUncertaintyEstimator(
        extractor,
        feat_dim=feat_dim,
        device=DEVICE,
        args={"gp_kernel": "rbf", "gp_lengthscale": lengthscale, "gp_backend": "exact"},
    )


# ---------------------------------------------------------------------------
# effective_sample_size
# ---------------------------------------------------------------------------


def test_ess_of_uniform_weights_equals_n():
    w = torch.ones(10)
    assert effective_sample_size(w) == pytest.approx(10.0)


def test_ess_of_empty_is_zero():
    assert effective_sample_size(torch.zeros(0)) == 0.0


def test_ess_concentrated_weight_is_near_one():
    w = torch.zeros(10)
    w[0] = 1000.0
    w[1:] = 1e-6
    assert effective_sample_size(w) < 1.5


# ---------------------------------------------------------------------------
# cluster_count
# ---------------------------------------------------------------------------


def test_cluster_count_of_empty_is_zero():
    assert cluster_count(torch.zeros(0, 2), radius=0.1) == 0


def test_cluster_count_merges_nearby_points():
    coords = torch.tensor([[0.0, 0.0], [0.01, 0.0], [0.02, 0.0]])
    assert cluster_count(coords, radius=0.1) == 1


def test_cluster_count_separates_far_points():
    coords = torch.tensor([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    assert cluster_count(coords, radius=0.1) == 3


def test_cluster_count_is_radius_sensitive():
    """The whole point of reporting two radii: the same data can read as one
    cluster or several depending on the scale."""
    coords = torch.tensor([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert cluster_count(coords, radius=0.5) == 3
    assert cluster_count(coords, radius=1.5) == 1


# ---------------------------------------------------------------------------
# best_reward_beyond
# ---------------------------------------------------------------------------


def test_best_reward_beyond_returns_none_when_nothing_is_far():
    rewards = torch.tensor([1.0, 2.0])
    coords = torch.tensor([[0.0, 0.0], [0.1, 0.0]])
    anchors = torch.tensor([[0.0, 0.0]])
    best, count = best_reward_beyond(rewards, coords, anchors, distance=5.0)
    assert best is None
    assert count == 0


def test_best_reward_beyond_finds_the_best_far_point():
    rewards = torch.tensor([1.0, 5.0, 3.0])
    coords = torch.tensor([[0.0, 0.0], [10.0, 0.0], [10.0, 1.0]])
    anchors = torch.tensor([[0.0, 0.0]])
    best, count = best_reward_beyond(rewards, coords, anchors, distance=1.0)
    assert count == 2
    assert best == 5.0


def test_best_reward_beyond_empty_anchors_returns_none():
    rewards = torch.tensor([1.0])
    coords = torch.tensor([[0.0, 0.0]])
    anchors = torch.zeros(0, 2)
    best, count = best_reward_beyond(rewards, coords, anchors, distance=1.0)
    assert best is None
    assert count == 0


# ---------------------------------------------------------------------------
# coordinate_bounds
# ---------------------------------------------------------------------------


def test_coordinate_bounds_pads_the_range():
    a = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    lo, hi = coordinate_bounds(a, pad=0.5)
    assert lo == -1.5
    assert hi == 1.5


# ---------------------------------------------------------------------------
# per_cell_density_ratio
# ---------------------------------------------------------------------------


def test_density_ratio_disjoint_distributions_is_all_new_mass():
    anchor = torch.zeros(50, 2) - 5.0
    current = torch.zeros(50, 2) + 5.0
    result = per_cell_density_ratio(current, anchor, bins=20, limits=(-6.0, 6.0))
    assert result["new_mass_fraction"] == pytest.approx(1.0)
    assert result["anchor_mass_fraction"] == pytest.approx(0.0)


def test_density_ratio_identical_distributions_is_all_anchor_mass():
    torch.manual_seed(0)
    points = torch.randn(200, 2)
    result = per_cell_density_ratio(points, points, bins=10, limits=(-4.0, 4.0))
    assert result["anchor_mass_fraction"] == pytest.approx(1.0, abs=1e-6)
    assert result["new_mass_fraction"] == pytest.approx(0.0, abs=1e-6)


def test_density_ratio_rejects_high_dimensional_coordinates():
    a = torch.zeros(4, 3)
    try:
        per_cell_density_ratio(a, a, bins=10, limits=(-1.0, 1.0))
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "1-D or 2-D" in str(e)


# ---------------------------------------------------------------------------
# heldout_r2
# ---------------------------------------------------------------------------


def test_heldout_r2_returns_none_for_a_non_gp_estimator():
    extractor = FlowFeatureExtractor(
        model=None,
        geometry=None,
        schedule=None,
        layer="input",
        timestep=0.9,
        postprocess=lambda x1, feats: feats.data,
    )
    estimator = EnsembleUncertaintyEstimator(
        extractor, feat_dim=2, device=DEVICE, args={}
    )
    from diffusiongym.types import DDTensor

    latents = [DDTensor(torch.randn(10, 2))]
    rewards = [torch.randn(10)]
    result = heldout_r2(estimator, latents, rewards, [{}])
    assert result is None


def test_heldout_r2_returns_none_for_too_little_data():
    from diffusiongym.types import DDTensor

    estimator = _gp_estimator()
    latents = [DDTensor(torch.randn(3, 2))]
    rewards = [torch.randn(3)]
    assert heldout_r2(estimator, latents, rewards, [{}]) is None


def test_heldout_r2_is_high_for_a_smooth_learnable_signal():
    """A GP with a matching lengthscale should generalise well on a smooth
    reward that is roughly constant at that scale — the positive control for
    "the reward GP fits well", the complement of what triggers extension (C)."""
    from diffusiongym.types import DDTensor

    torch.manual_seed(0)
    x = torch.rand(60, 2) * 4 - 2  # spread over [-2, 2]^2
    reward = x[:, 0]  # smooth linear signal, easy for an RBF GP to fit

    estimator = _gp_estimator(lengthscale=1.0)
    r2 = heldout_r2(estimator, [DDTensor(x)], [reward], [{}], held_out_frac=0.3, seed=0)
    assert r2 is not None
    assert r2 > 0.5


def test_heldout_r2_does_not_mutate_the_live_estimator(tmp_path):
    """heldout_r2 fits a *throwaway* clone; the live estimator's own fit must
    be untouched by calling it."""
    from diffusiongym.types import DDTensor

    torch.manual_seed(0)
    x = torch.rand(20, 2)
    reward = x[:, 0]
    estimator = _gp_estimator()
    estimator.set_data([DDTensor(x)], [reward], [{}])
    observations_before = estimator.num_observations

    heldout_r2(estimator, [DDTensor(x)], [reward], [{}], seed=0)

    assert estimator.num_observations == observations_before
