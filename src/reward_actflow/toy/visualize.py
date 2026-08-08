"""Figures and coverage metrics for the 2-D staircase toy problem."""

from collections.abc import Callable
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusiongym import FineTuningSetup
from diffusiongym.types import DDTensor
from matplotlib.figure import Figure
from torch import Tensor

from reward_actflow.sampling import sample_policy
from reward_actflow.toy.validity import PLOT_LIMITS, staircase_validity

#: Density below which a grid cell counts as outside the model's support.
SUPPORT_THRESHOLD = 0.01

#: Grid resolution for the support histogram and the coverage metrics.
SUPPORT_BINS = 100


@torch.no_grad()
def sample_model(
    setup: FineTuningSetup,
    n: int,
    *,
    batch_size: int = 8192,
    policy: str = "train",
) -> torch.Tensor:
    """Draw `n` samples from a policy as a plain `(n, 2)` CPU tensor.

    Uses the setup's own dynamics and time grid, so a policy trained under an SDE
    is measured under that SDE. Forcing the ODE here reported a Flow-GRPO run at
    a 0.017 valid rate when its actual rate was 0.900.
    """
    latents: DDTensor = sample_policy(
        setup.context,
        n,
        dynamics=setup.dynamics,
        time_grid=setup.time_grid,
        batch_size=batch_size,
        policy=policy,
    )
    return latents.data.detach().cpu()


def _grid() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lo, hi = PLOT_LIMITS
    axis = torch.linspace(lo, hi, SUPPORT_BINS)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    return xx, yy, torch.stack([xx.flatten(), yy.flatten()], dim=-1)


def support_mask(samples: torch.Tensor) -> np.ndarray:
    """Boolean `(bins, bins)` mask of where the model puts appreciable mass."""
    lo, hi = PLOT_LIMITS
    hist, _, _ = np.histogram2d(
        samples[:, 0].numpy(),
        samples[:, 1].numpy(),
        bins=SUPPORT_BINS,
        density=True,
        range=[[lo, hi], [lo, hi]],
    )
    return hist >= SUPPORT_THRESHOLD


def coverage_metrics(samples: torch.Tensor) -> dict[str, float]:
    """How much of the valid set the model covers, and how cleanly.

    `support_coverage` alone is maximized by a model that spreads over the whole
    plane, and `support_precision` alone by one that never leaves its starting
    blob — they are only meaningful read together.
    """
    _, _, points = _grid()
    valid = staircase_validity(points).reshape(SUPPORT_BINS, SUPPORT_BINS).numpy()
    support = support_mask(samples)

    overlap = float((support & valid).sum())
    return {
        "support_coverage": overlap / max(float(valid.sum()), 1.0),
        "support_precision": overlap / max(float(support.sum()), 1.0),
        "support_area": float(support.sum()) / support.size,
    }


@torch.no_grad()
def plot_iteration(
    setup: FineTuningSetup,
    uncertainty: Any,
    batch_points: torch.Tensor,
    batch_valids: torch.Tensor,
    *,
    num_support_samples: int = 50_000,
) -> tuple[Figure, dict[str, float]]:
    """Two-panel snapshot: surrogate uncertainty, and where the model has mass.

    Returns the figure and the coverage metrics, so the caller does not have to
    draw the (expensive) support sample twice.
    """
    lo, hi = PLOT_LIMITS
    xx, yy, points = _grid()

    device = setup.context.policies.train.device
    _, sigma = uncertainty.mean_and_uncertainty(DDTensor(points).to(device))
    sigma = sigma.reshape(xx.shape).detach().cpu()

    fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)

    im = axes[0].imshow(
        sigma.T,
        extent=(lo, hi, lo, hi),
        origin="lower",
        cmap="YlGn",
        aspect="equal",
    )
    axes[0].set_title("Uncertainty")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    pts = batch_points.detach().cpu()
    valid = batch_valids.detach().cpu().bool()
    axes[0].scatter(pts[valid][:, 0], pts[valid][:, 1], s=2, alpha=1.0)
    axes[0].scatter(pts[~valid][:, 0], pts[~valid][:, 1], s=2, alpha=1.0)
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)

    samples = sample_model(setup, num_support_samples)
    support = support_mask(samples)
    axes[1].imshow(
        support.T,
        extent=(lo, hi, lo, hi),
        origin="lower",
        cmap="binary",
        aspect="equal",
    )
    axes[1].set_title(rf"Model support (p >= {SUPPORT_THRESHOLD})")

    validity_grid = staircase_validity(points).reshape(xx.shape)
    for ax in axes:
        ax.contour(xx, yy, validity_grid, levels=[0.5], colors="black", alpha=0.3)

    return fig, coverage_metrics(samples)


@torch.no_grad()
def plot_actflow_r_iteration(
    setup: FineTuningSetup,
    validity_uncertainty: Any,
    reward_fn: Callable[[Tensor], Tensor],
    anchors: Tensor,
    batch_points: Tensor,
    batch_valids: Tensor,
    *,
    num_support_samples: int = 50_000,
) -> tuple[Figure, dict[str, float]]:
    """Three-panel ActFlow-R snapshot.

    Left: the reward field, the frozen anchor pool `P`, and this iteration's
    samples — the moving parts of Algorithm 1 lines 4-7. Middle: the
    validity surrogate's uncertainty, as in `plot_iteration`. Right: the
    model's support (the "generable set") with the reward's contour lines
    overlaid, so whether support is drifting toward higher reward — not just
    growing — is visible frame to frame, and across a `video.mp4` made from
    them.
    """
    lo, hi = PLOT_LIMITS
    xx, yy, points = _grid()

    device = setup.context.policies.train.device
    _, sigma = validity_uncertainty.mean_and_uncertainty(DDTensor(points).to(device))
    sigma = sigma.reshape(xx.shape).detach().cpu()

    reward_grid = reward_fn(points).reshape(xx.shape).detach().cpu()

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), constrained_layout=True)

    im0 = axes[0].imshow(
        reward_grid.T,
        extent=(lo, hi, lo, hi),
        origin="lower",
        cmap="viridis",
        aspect="equal",
    )
    axes[0].set_title("Reward field")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    anchors_cpu = anchors.detach().cpu()
    axes[0].scatter(
        anchors_cpu[:, 0], anchors_cpu[:, 1], s=2, alpha=0.35, color="white", label="P"
    )
    pts = batch_points.detach().cpu()
    valid = batch_valids.detach().cpu().bool()
    axes[0].scatter(pts[valid][:, 0], pts[valid][:, 1], s=4, color="red", label="valid")
    axes[0].scatter(
        pts[~valid][:, 0],
        pts[~valid][:, 1],
        s=4,
        color="gray",
        alpha=0.6,
        label="invalid",
    )
    axes[0].set_xlim(lo, hi)
    axes[0].set_ylim(lo, hi)
    axes[0].legend(loc="upper right", fontsize=6, framealpha=0.5)

    im1 = axes[1].imshow(
        sigma.T,
        extent=(lo, hi, lo, hi),
        origin="lower",
        cmap="YlGn",
        aspect="equal",
    )
    axes[1].set_title("Validity uncertainty")
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    samples = sample_model(setup, num_support_samples)
    support = support_mask(samples)
    axes[2].imshow(
        support.T,
        extent=(lo, hi, lo, hi),
        origin="lower",
        cmap="binary",
        aspect="equal",
    )
    axes[2].contour(
        xx, yy, reward_grid, levels=6, cmap="viridis", alpha=0.6, linewidths=0.6
    )
    axes[2].set_title(rf"Model support (p >= {SUPPORT_THRESHOLD})")

    validity_grid = staircase_validity(points).reshape(xx.shape)
    for ax in axes:
        ax.contour(xx, yy, validity_grid, levels=[0.5], colors="black", alpha=0.3)

    return fig, coverage_metrics(samples)
