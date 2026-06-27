#!/usr/bin/env python
"""
Plot per-iteration coverage / diversity / validity / FID over the recursive
finetuning loop, aggregated across the 3 training seeds, in the exact visual
style of ``plotting_peptide.py`` (the molecule figures).

Each metric is saved as an independent PNG (matching colours, markers, figure
size via tueplots ``icml2024_half``, dashed silver y-grid, no legend):

  * Number of Clusters -- PCLM embedding-space sphere-exclusion clusters
                          (n_clusters_pclm_thr0.10), computed on the fly from
                          each variant's saved *_pclm_embeddings.npy.
  * Vendi              -- vendi_pclm_rbf_sigma0.5   (from the JSON)
  * Validity (%)       -- mean_batch_valid_rate * 100
  * FID                -- fid_morgan_vs_pretrained

Three series per panel + a pretrained reference line:

  * ActFlow  -- uncertainty_only_filtered       (dark violet, squares)
  * Rec-F    -- continued_pretraining_filtered   (dark teal, circles)
  * Rec-NF   -- synthetic: pretrained value at iter 0, then 0 (no filter ->
                invalid output); dashed orange.
  * Pre      -- pretrained reference (light-blue dashed axhline + CI band)

Curves show mean +/- 1 std (ddof=1) across the 3 seeds by default (set BAND_TYPE
to 'sem' or 'ci95' for standard error or an exact Student-t 95% CI instead),
with the band clipped at 0 (all metrics are non-negative).
Only iterations 0..12 are plotted; the x-axis is the iteration index * 100.

Usage:
  python scripts/plot_diversity_progression.py
"""

import argparse
import glob
import json
import os
import re

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tueplots import figsizes

# ---- paths ----
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EVAL_DIR = os.path.join(BASE_PATH, 'results', 'diversity_eval')

# ---- style (verbatim from plotting_peptide.py) ----
COLOR_PRE = (0.35, 0.60, 0.85)              # light blue
COLOR_OURS = (0.35, 0.00, 0.55)             # dark violet   (ActFlow)
COLOR_REC_NO_FILTER = (0.93, 0.55, 0.14)    # orange        (Rec-NF)
COLOR_REC_FILTER = (0.05, 0.42, 0.40)       # dark teal     (Rec-F)

STYLE = {
    'ActFlow':         dict(marker='s', markersize=4, linewidth=2.0, linestyle='-'),
    'Rec (filter)':    dict(marker='o', markersize=3, linewidth=1.2, linestyle='-'),
    'Rec (no filter)': dict(marker='D', markersize=3, linewidth=1.2, linestyle='--'),
    'Pre-trained':     dict(marker='*', markersize=6, linewidth=1.2, linestyle='--'),
}
CI_ALPHA = 0.2

# ---- coverage metric: PCLM embedding-space clusters (computed from embeddings,
# NOT the JSON's n_clusters_thr* which is Morgan/Tanimoto and saturates at 1000)
PCLM_THR = 0.10
MAX_ITER = 12
ITER_SCALE = 100  # x-axis = recursive iteration index * 100 (iter 1 -> 100, ...)

# Uncertainty band shown around the mean across seeds.
#   'std'  -> +/- 1 sample std (ddof=1)
#   'sem'  -> +/- 1 standard error (std / sqrt(n))
#   'ci95' -> exact Student-t 95% CI (t_{0.975,n-1} * std / sqrt(n))
BAND_TYPE = 'std'

# metric_key -> (json_key | None, ylabel, filename, scale)  (None => computed)
METRICS = [
    ('clusters', None,                       'Number of Clusters', 'peptide_clusters.png', 1.0),
    ('vendi',    'vendi_pclm_rbf_sigma0.5',  'Vendi',              'peptide_vendi.png',    1.0),
    ('validity', 'mean_batch_valid_rate',    'Validity (%)',       'peptide_validity.png', 100.0),
    ('fid',      'fid_morgan_vs_pretrained', 'FID',                'peptide_fid.png',      1.0),
]


# ---------------------------------------------------------------------------
# PCLM-space clustering
# ---------------------------------------------------------------------------

def _sphere_exclusion(emb, threshold):
    """Greedy sphere-exclusion (Leader algorithm) on L2-normalised cosine
    distance.  Returns the number of cluster centres."""
    n = emb.shape[0]
    if n == 0:
        return 0
    dists = 1.0 - (emb @ emb.T)
    available = np.ones(n, dtype=bool)
    n_centers = 0
    for i in range(n):
        if not available[i]:
            continue
        n_centers += 1
        available &= ~(dists[i] < threshold)
        available[i] = False
    return n_centers


def _compute_pclm_clusters(emb_path, thr):
    emb = np.load(emb_path)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
    return _sphere_exclusion(emb, thr)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_method_series(eval_dir, method_key, max_iter=MAX_ITER):
    """Load every seed directory matching *method_key* (latest per seed).

    Returns (by_seed, pretrained) where by_seed maps seed -> {'iters': arr,
    'clusters'/'vendi'/'validity'/'fid': arr} and pretrained maps the same
    metric keys -> list of per-seed pretrained values.
    """
    pattern = os.path.join(eval_dir, f'run_*_seed*_{method_key}')
    dirs = glob.glob(pattern)
    if not dirs:
        raise FileNotFoundError(f'No directories match {pattern}')

    by_seed_dir = {}
    for dpath in dirs:
        m = re.search(r'run_(\d+_\d+)_seed(\d+)_', os.path.basename(dpath))
        if not m:
            continue
        ts, seed = m.group(1), int(m.group(2))
        if seed not in by_seed_dir or ts > by_seed_dir[seed][0]:
            by_seed_dir[seed] = (ts, dpath)

    by_seed = {}
    pretrained = {mk: [] for mk, *_ in METRICS}
    for seed, (ts, dpath) in sorted(by_seed_dir.items()):
        with open(os.path.join(dpath, 'diversity_results_aggregated.json')) as f:
            d = json.load(f)
        emb_dir = os.path.join(dpath, 'seed_42')

        if 'pretrained' in d:
            for mk, jkey, *_ in METRICS:
                if jkey is None:
                    pre_npy = os.path.join(emb_dir, 'pretrained_pclm_embeddings.npy')
                    if os.path.exists(pre_npy):
                        pretrained[mk].append(_compute_pclm_clusters(pre_npy, PCLM_THR))
                else:
                    pretrained[mk].append(d['pretrained'][jkey]['mean'])

        rows = []
        for variant, metrics in d.items():
            mm = re.search(r'iter(\d+)$', variant)
            if not mm:
                continue
            it = int(mm.group(1))
            if it > max_iter:
                continue
            npy = os.path.join(emb_dir, f'{variant}_pclm_embeddings.npy')
            clusters = _compute_pclm_clusters(npy, PCLM_THR) if os.path.exists(npy) else np.nan
            rows.append((it, clusters,
                         metrics['vendi_pclm_rbf_sigma0.5']['mean'],
                         metrics['mean_batch_valid_rate']['mean'],
                         metrics['fid_morgan_vs_pretrained']['mean']))
        rows.sort()
        by_seed[seed] = {
            'iters':    np.array([r[0] for r in rows]),
            'clusters': np.array([r[1] for r in rows]),
            'vendi':    np.array([r[2] for r in rows]),
            'validity': np.array([r[3] for r in rows]),
            'fid':      np.array([r[4] for r in rows]),
        }
        print(f'  seed {seed}: {len(rows)} iters (<= {max_iter})  [{ts}]')

    return by_seed, pretrained


def _band(stds, n):
    """Half-width of the uncertainty band per BAND_TYPE (0 where n <= 1)."""
    stds = np.asarray(stds, dtype=float)
    n = np.asarray(n)
    if BAND_TYPE == 'std':
        half = stds
    elif BAND_TYPE == 'sem':
        half = stds / np.sqrt(n)
    elif BAND_TYPE == 'ci95':
        half = stats.t.ppf(0.975, np.maximum(n - 1, 1)) * stds / np.sqrt(n)
    else:
        raise ValueError(f'Unknown BAND_TYPE: {BAND_TYPE}')
    return np.where(n > 1, half, 0.0)


def aggregate_metric(by_seed, metric, max_iter=MAX_ITER):
    """Mean and uncertainty-band half-width per iteration across seeds
    (band defined by BAND_TYPE; n = #seeds with data at that iteration)."""
    iters = np.arange(max_iter + 1)
    matrix = np.full((len(by_seed), len(iters)), np.nan)
    for s, data in enumerate(by_seed.values()):
        lut = {it: i for i, it in enumerate(data['iters'])}
        for j, it in enumerate(iters):
            if it in lut:
                matrix[s, j] = data[metric][lut[it]]
    n = np.sum(~np.isnan(matrix), axis=0)
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0, ddof=1)
    return iters, means, _band(stds, n)


def _band_scalar(arr):
    arr = np.asarray(arr, dtype=float)
    n = len(arr)
    if n <= 1:
        return 0.0
    return float(_band(arr.std(ddof=1), n))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_metric(metric, ylabel, filename, scale, methods, recnf, pre_stats,
                out_dir, max_iter=MAX_ITER):
    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        # These metrics are all non-negative, so clip the symmetric CI band at 0.
        clip = lambda y: np.clip(y, 0.0, None)

        # Pretrained reference line + CI band.
        pre_mean, pre_ci = pre_stats[metric]
        s = STYLE['Pre-trained']
        ax.axhline(pre_mean * scale, color=COLOR_PRE, linestyle=s['linestyle'],
                   linewidth=s['linewidth'])
        if pre_ci > 0:
            ax.axhspan(clip(pre_mean - pre_ci) * scale, (pre_mean + pre_ci) * scale,
                       color=COLOR_PRE, alpha=CI_ALPHA)

        # Real methods: mean +/- band (BAND_TYPE; clipped at 0).
        for label, color, by_seed in methods:
            iters, means, ci = aggregate_metric(by_seed, metric, max_iter)
            x = iters * ITER_SCALE
            s = STYLE[label]
            ax.plot(x, means * scale, color=color, **s)
            ax.fill_between(x, clip(means - ci) * scale, (means + ci) * scale,
                            color=color, alpha=CI_ALPHA)

        # Synthetic Rec-NF: pretrained value at iter 0, then 0.
        s = STYLE['Rec (no filter)']
        recnf_y = np.where(recnf['iters'] == 0, pre_mean, 0.0)
        ax.plot(recnf['iters'] * ITER_SCALE, recnf_y * scale,
                color=COLOR_REC_NO_FILTER, **s)

        ax.set_xlabel('Iteration')
        ax.set_ylabel(ylabel)
        ax.grid(axis='y', linestyle='--', color='silver', linewidth=0.8)
        fig.savefig(os.path.join(out_dir, filename), dpi=150)
        plt.close(fig)
    print(f'Saved {os.path.join(out_dir, filename)}')


def main():
    parser = argparse.ArgumentParser(
        description='Per-iter coverage/diversity/validity/FID plots (molecule style)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--eval_dir', type=str, default=DEFAULT_EVAL_DIR)
    parser.add_argument('--actflow_key', type=str,
                        default='uncertainty_only_filtered_alpha0.005_gpfilt_all_iters')
    parser.add_argument('--recf_key', type=str,
                        default='continued_pretraining_filtered_alpha0.005_all_iters')
    parser.add_argument('--max_iter', type=int, default=MAX_ITER)
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(BASE_PATH, 'results', 'diversity_progression'))
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print('Loading ActFlow (uncertainty_only_filtered)...')
    actflow_seeds, actflow_pre = load_method_series(args.eval_dir, args.actflow_key, args.max_iter)
    print('Loading Rec-F (continued_pretraining_filtered)...')
    recf_seeds, recf_pre = load_method_series(args.eval_dir, args.recf_key, args.max_iter)

    methods = [
        ('ActFlow', COLOR_OURS, actflow_seeds),
        ('Rec (filter)', COLOR_REC_FILTER, recf_seeds),
    ]

    # Pretrained mean + band per metric (across every loaded seed dir).
    pre_stats = {}
    for mk, *_ in METRICS:
        vals = actflow_pre[mk] + recf_pre[mk]
        pre_stats[mk] = (float(np.mean(vals)) if vals else 0.0, _band_scalar(vals))

    recnf = {'iters': np.arange(args.max_iter + 1)}

    for mk, jkey, ylabel, filename, scale in METRICS:
        plot_metric(mk, ylabel, filename, scale, methods, recnf, pre_stats,
                    args.output_dir, args.max_iter)


if __name__ == '__main__':
    main()
