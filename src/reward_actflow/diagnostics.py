"""ActFlow-R diagnostics — logged regardless of whether any appendix
extension is ever turned on.

Extensions (A)-(E) are not implemented anywhere in this codebase; (F) (a soft
validity gate on `a_t`'s reward term, `rewards/acquisition.py`) is
implemented but stays default-off. Each is a documented response to a
specific symptom, and turning one on without evidence is exactly the "no
silent caps" mistake the appendix itself warns against — (F) was only turned
on after `best_reward_far` pinned at a region boundary for hundreds of
iterations. What is otherwise implemented is the evidence: five diagnostics,
each aimed at exactly one extension's stated trigger condition.

    diagnostic                          | motivates | function(s) here
    -------------------------------------|-----------|-----------------------
    valid-cluster count at two radii     | A, D      | cluster_count
    effective sample size of w           | tilt      | effective_sample_size
    best reward beyond d from p_data     | —         | best_reward_beyond
    per-cell density ratio vs P          | B         | per_cell_density_ratio
    reward-GP held-out R^2               | C         | heldout_r2

ESS of `w` is also computed inline by `MixtureReplay.update()` (post-clip)
and `ActFlowRLoop.extra_metrics()` (pre-clip); `effective_sample_size` here
is the single canonical formula both use.

Metric space is deliberately *not* `phi_s^t` (`ProblemSetup.diagnostic_coordinates`
instead) — the representation drifts with theta_t, so "the same cell" would
not mean the same thing between iterations.
"""

from typing import Any

import numpy as np
import torch
from torch import Tensor

from reward_actflow.uncertainty.gp import GPUncertaintyEstimator
from reward_actflow.uncertainty.uncertainty_estimator import UncertaintyEstimator


def effective_sample_size(weights: Tensor) -> float:
    """`(sum w)^2 / sum(w^2)` — 0 for an empty tensor, `n` for uniform `w`."""
    if weights.numel() == 0:
        return 0.0
    return float((weights.sum() ** 2 / (weights**2).sum().clamp_min(1e-12)).item())


def cluster_count(coords: Tensor, radius: float, *, subsample_cap: int = 4000) -> int:
    """Number of single-linkage connected components of `coords` at `radius`.

    A fixed radius makes the count comparable *across iterations*, but only
    at that one scale — the count alone can be an artefact of the radius
    chosen, which is exactly why this is always reported at two radii and
    read as a pair (`clusters_r_train`, `clusters_r_eval`), not as a single
    number. Subsampled above `subsample_cap`: this is O(n^2) in memory for
    the distance matrix and effectively O(n^2) in the union-find loop too, so
    it is a diagnostic for the toy's scale, not a scalable clustering routine.
    """
    n = coords.shape[0]
    if n == 0:
        return 0
    if n > subsample_cap:
        idx = torch.randperm(n)[:subsample_cap]
        coords = coords[idx]
        n = subsample_cap

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    chunk = 512
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        dist = torch.cdist(coords[start:end], coords)
        close = (dist < radius).nonzero(as_tuple=False)
        for offset, j in close.tolist():
            union(start + offset, j)

    return len({find(i) for i in range(n)})


def best_reward_beyond(
    rewards: Tensor,
    coords: Tensor,
    anchor_coords: Tensor,
    distance: float,
    *,
    chunk: int = 1024,
) -> tuple[float | None, int]:
    """Best task reward among points farther than `distance` from every
    anchor, and how many such points exist.

    Returns `(None, 0)` when there are none — `best_reward_beyond = nan`
    (nothing is out there) and `= 0.3` (something *bad* is out there) are
    opposite conclusions, so the caller must be able to tell them apart, and
    the count says which one this is.
    """
    n = coords.shape[0]
    if n == 0 or anchor_coords.shape[0] == 0:
        return None, 0

    min_dist = torch.full((n,), float("inf"), device=coords.device)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        min_dist[start:end] = (
            torch.cdist(coords[start:end], anchor_coords).min(dim=1).values
        )

    far_mask = min_dist > distance
    count = int(far_mask.sum().item())
    if count == 0:
        return None, 0
    return float(rewards[far_mask].max().item()), count


def coordinate_bounds(*tensors: Tensor, pad: float = 0.5) -> tuple[float, float]:
    """`(min, max)` over every value in `tensors`, padded — the fixed grid
    `per_cell_density_ratio` needs, derived from the data itself rather than
    a problem-specific plot range. Including the (frozen) anchor pool in the
    call keeps the range comparatively stable across iterations even as the
    range of `current` alone drifts.
    """
    values = torch.cat([t.flatten() for t in tensors])
    return float(values.min().item()) - pad, float(values.max().item()) + pad


def per_cell_density_ratio(
    current: Tensor,
    anchor: Tensor,
    *,
    bins: int,
    limits: tuple[float, float],
    min_count: int = 1,
) -> dict[str, float]:
    """Per-cell mass comparison of `current` (a fresh policy draw) against
    `anchor` (the frozen anchor pool `P`) over a fixed grid.

    Returns scalars, never the field:

    - `new_mass_fraction`: mass of `current` in cells `anchor` has none in —
      the direct measurement of mass appearing outside `supp(p^{theta_0})`,
      which a geometric tilt provably cannot produce.
    - `anchor_mass_fraction`: mass of `current` in cells `anchor` also
      covers — the empirical check of `p^{theta_{t+1}} >= lambda*p_P`.
    - `log_ratio_max` / `log_ratio_p95`: log10 density ratio, masked to
      cells both histograms have appreciable mass in — unmasked, this is the
      ratio of two numerical zeros almost everywhere, which reads as noise.

    `current`/`anchor` are `(n, k)`, `k` in `{1, 2}`.
    """
    k = current.shape[-1]
    lo, hi = limits
    current_np = current.detach().cpu().numpy()
    anchor_np = anchor.detach().cpu().numpy()

    if k == 1:
        cur_counts, _ = np.histogram(current_np[:, 0], bins=bins, range=(lo, hi))
        anc_counts, _ = np.histogram(anchor_np[:, 0], bins=bins, range=(lo, hi))
    elif k == 2:
        cur_counts, _, _ = np.histogram2d(
            current_np[:, 0], current_np[:, 1], bins=bins, range=[[lo, hi], [lo, hi]]
        )
        anc_counts, _, _ = np.histogram2d(
            anchor_np[:, 0], anchor_np[:, 1], bins=bins, range=[[lo, hi], [lo, hi]]
        )
    else:
        raise ValueError(
            f"per_cell_density_ratio only supports 1-D or 2-D coordinates, got k={k}."
        )

    n_current = max(int(cur_counts.sum()), 1)
    cur_mass = cur_counts / n_current
    anc_mass = anc_counts / max(int(anc_counts.sum()), 1)

    current_support = cur_counts >= min_count
    anchor_support = anc_counts >= min_count

    new_mass = float(cur_mass[current_support & ~anchor_support].sum())
    anchor_mass = float(cur_mass[current_support & anchor_support].sum())

    both = current_support & anchor_support
    if np.any(both):
        ratios = np.log10((cur_mass[both] + 1e-12) / (anc_mass[both] + 1e-12))
        log_ratio_max = float(np.max(ratios))
        log_ratio_p95 = float(np.percentile(ratios, 95))
    else:
        log_ratio_max = 0.0
        log_ratio_p95 = 0.0

    return {
        "new_mass_fraction": new_mass,
        "anchor_mass_fraction": anchor_mass,
        "log_ratio_max": log_ratio_max,
        "log_ratio_p95": log_ratio_p95,
    }


def heldout_r2(
    estimator: UncertaintyEstimator,
    latents: list[Any],
    task_rewards: list[Tensor],
    conditioning: list[dict[str, Any]],
    *,
    held_out_frac: float = 0.2,
    seed: int | None = None,
) -> float | None:
    """Held-out R^2 of a *throwaway* clone of `estimator`, fit on a random
    split and scored on the rest — the symptom check extension (C) is keyed
    on ("reward GP fits poorly at s_r").

    Returns `None` — not a crash, not a misleading `0.0` — when there is too
    little data for a meaningful split, when the held-out targets are
    constant (R^2 is undefined, not zero), or when `estimator` is not a
    `GPUncertaintyEstimator`: an ensemble's `_update_estimator` retrains 5
    MLPs for up to 1000 steps each, which is not a per-iteration cost.
    """
    if not isinstance(estimator, GPUncertaintyEstimator):
        return None

    all_rewards = torch.cat(task_rewards)
    n = all_rewards.shape[0]
    n_holdout = max(1, round(n * held_out_frac))
    if n - n_holdout < 2 or n_holdout < 1:
        return None

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    perm = torch.randperm(n, generator=generator)
    holdout_idx, train_idx = perm[:n_holdout], perm[n_holdout:]

    # Reuses estimator's own (cached, if static) features rather than a
    # second forward pass through the model.
    feats = estimator._extract_all(latents, conditioning)
    all_feats = torch.cat(feats, dim=0)

    throwaway = type(estimator)(
        estimator.feat_extractor,
        feat_dim=estimator.feat_dim,
        device=estimator.device,
        args=estimator.args,
    )
    train_labels = all_rewards[train_idx]
    label_mean = train_labels.mean()
    label_std = train_labels.std(correction=0).clamp_min(1e-8)
    throwaway._update_estimator(
        all_feats[train_idx], (train_labels - label_mean) / label_std
    )

    pred_mean_z, _ = throwaway._mean_and_uncertainty(all_feats[holdout_idx])
    pred_mean = pred_mean_z * label_std + label_mean

    truth = all_rewards[holdout_idx]
    ss_tot = float(((truth - truth.mean()) ** 2).sum().item())
    if ss_tot <= 1e-12:
        return None

    ss_res = float(((truth - pred_mean) ** 2).sum().item())
    return 1.0 - ss_res / ss_tot
