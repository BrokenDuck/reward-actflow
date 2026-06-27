"""Plot per-iter diversity-eval trajectories in the plotting.py style.

Reads the 6 (seed x mode) result dirs from a single alpha-sweep eval run
and produces individual per-metric figures (validity, PCLM coverage,
Vendi-PCLM diversity, FID-Morgan), aggregating across 3 training seeds
with 95% CIs (Student-t, df=2). PCLM cluster counts are computed on the
fly from each variant's saved *_pclm_embeddings.npy using sphere-exclusion
on cosine distance (thr=0.10).

Usage
-----
python plot_diversity_seeds.py \\
    --eval_dir results/diversity_eval \\
    --run_glob 'run_20260526_15045[56]_seed*_alpha0.1*_all_iters' \\
    --alpha 0.1 \\
    --out_dir results/diversity_eval/plots_alpha0.1
"""

import argparse
import glob
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from tueplots import figsizes


# ── Style matched to plotting.py ─────────────────────────────────────────────

COLOR_PRE = (0.35, 0.60, 0.85)              # light blue
COLOR_OURS = (0.35, 0.00, 0.55)             # dark violet -- UO_F (uncertainty-only)
COLOR_NO_UNC = (0.93, 0.55, 0.14)           # orange       -- CP_F (continued pretrain)

STYLE = {
    "UO_F (Ours)":   dict(marker="s", markersize=4, linewidth=2.0, linestyle="-"),
    "CP_F (Cont. Pretrain.)":  dict(marker="D", markersize=3, linewidth=1.2, linestyle="-"),
    "Pre-trained":   dict(marker="*", markersize=6, linewidth=1.2, linestyle="--"),
}
CI_ALPHA = 0.2
T_CRIT = float(stats.t.ppf(0.975, df=2))    # n=3 seeds


# ── Loading ──────────────────────────────────────────────────────────────────

VARIANT_RE = re.compile(
    r"^seed(\d+)_(uncertainty_only_filtered|continued_pretraining_filtered)"
    r"_alpha([\d.]+)_iter(\d+)$"
)


def _parse_variant(name: str):
    m = VARIANT_RE.match(name)
    if not m:
        return None
    return (int(m.group(1)), m.group(2), float(m.group(3)), int(m.group(4)))


def _sphere_exclusion(emb: np.ndarray, threshold: float) -> int:
    """Greedy sphere-exclusion (Leader algorithm) on L2-normalised cosine distance."""
    n = emb.shape[0]
    if n == 0:
        return 0
    sims = emb @ emb.T
    dists = 1.0 - sims
    available = np.ones(n, dtype=bool)
    n_centers = 0
    for i in range(n):
        if not available[i]:
            continue
        n_centers += 1
        available &= ~(dists[i] < threshold)
        available[i] = False
    return n_centers


def _compute_pclm_clusters(emb_path: str, thr: float) -> int:
    emb = np.load(emb_path)
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-9)
    return _sphere_exclusion(emb, thr)


def load_run_dirs(eval_dir: Path, run_glob: str, alpha: float, pclm_thr: float = 0.10):
    """Load metrics from all (seed, mode) result dirs matching run_glob.

    Returns a long-form DataFrame with columns:
      seed, mode, iter, n_valid, mean_batch_valid_rate,
      vendi_pclm_rbf_sigma0.5, fid_morgan_vs_pretrained,
      n_clusters_pclm_thr{pclm_thr}
    Plus a dict of pretrained reference values keyed by metric.
    """
    rows = []
    pretrained_ref = {}
    runs = sorted(eval_dir.glob(run_glob))
    if not runs:
        raise FileNotFoundError(f"No runs match {eval_dir/run_glob}")
    print(f"Found {len(runs)} result dirs at {eval_dir/run_glob}")
    for d in runs:
        seed_dir = d / "seed_42"
        csv = seed_dir / "diversity_summary.csv"
        if not csv.exists():
            print(f"  [skip] no diversity_summary.csv in {seed_dir}")
            continue
        df = pd.read_csv(csv)

        # Cache pretrained reference once (single row, identical across runs except
        # for floating-point noise — pick the first encountered).
        if not pretrained_ref:
            pre = df[df["name"] == "pretrained"]
            if len(pre):
                pretrained_ref["mean_batch_valid_rate"] = float(pre["mean_batch_valid_rate"].iloc[0])
                pretrained_ref["vendi_pclm_rbf_sigma0.5"] = float(pre["vendi_pclm_rbf_sigma0.5"].iloc[0])
                pretrained_ref["fid_morgan_vs_pretrained"] = 0.0
                # PCLM clusters @ thr — compute from pretrained_pclm_embeddings.npy
                pre_npy = seed_dir / "pretrained_pclm_embeddings.npy"
                if pre_npy.exists():
                    pretrained_ref[f"n_clusters_pclm_thr{pclm_thr}"] = _compute_pclm_clusters(
                        str(pre_npy), pclm_thr)

        for _, r in df.iterrows():
            parsed = _parse_variant(r["name"])
            if parsed is None:
                continue
            seed, mode, a, it = parsed
            if a != alpha:
                continue
            # Compute PCLM clusters from saved embeddings
            npy = seed_dir / f"{r['name']}_pclm_embeddings.npy"
            n_pclm_clust = _compute_pclm_clusters(str(npy), pclm_thr) if npy.exists() else np.nan
            rows.append({
                "seed": seed,
                "mode": mode,
                "iter": it,
                "mean_batch_valid_rate": float(r["mean_batch_valid_rate"]),
                "vendi_pclm_rbf_sigma0.5": float(r["vendi_pclm_rbf_sigma0.5"]),
                "fid_morgan_vs_pretrained": float(r["fid_morgan_vs_pretrained"]),
                f"n_clusters_pclm_thr{pclm_thr}": n_pclm_clust,
            })
    df_long = pd.DataFrame(rows).sort_values(["mode", "seed", "iter"]).reset_index(drop=True)
    print(f"Loaded {len(df_long)} variant rows; pretrained ref: {pretrained_ref}")
    return df_long, pretrained_ref


# ── Aggregation ──────────────────────────────────────────────────────────────

def aggregate(df: pd.DataFrame, mode: str, metric: str):
    """Return (iters, means, ci95_half) for one mode/metric, across seeds."""
    sub = df[df["mode"] == mode]
    g = sub.groupby("iter")[metric].agg(["mean", "std", "count"]).reset_index()
    iters = g["iter"].to_numpy()
    means = g["mean"].to_numpy()
    n = g["count"].to_numpy()
    std = g["std"].to_numpy()
    se = np.where(n > 1, std / np.sqrt(n), 0.0)
    half = np.where(n > 1, T_CRIT * se, 0.0)
    return iters, means, half


# ── Plotting ─────────────────────────────────────────────────────────────────

MODE_DISPLAY = {
    "uncertainty_only_filtered":     ("UO_F (Ours)",   COLOR_OURS),
    "continued_pretraining_filtered":("CP_F (Cont. Pretrain.)", COLOR_NO_UNC),
}


def _style(label: str) -> dict:
    return STYLE.get(label, dict(marker="o", markersize=3, linewidth=1.2, linestyle="-"))


def plot_metric(df: pd.DataFrame, pretrained_ref: dict, metric: str,
                ylabel: str, out_path: Path, scale: float = 1.0,
                ylim: tuple | None = None, pretrained_key: str | None = None,
                show_legend: bool = True):
    """Plot one metric vs iter for both modes + pretrained reference line."""
    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        # Pretrained reference (horizontal line, no CI band since n=1)
        pre_key = pretrained_key or metric
        if pre_key in pretrained_ref:
            s = _style("Pre-trained")
            ax.axhline(pretrained_ref[pre_key] * scale, color=COLOR_PRE,
                       linestyle=s["linestyle"], linewidth=s["linewidth"],
                       label="Pre-trained")

        for mode, (label, color) in MODE_DISPLAY.items():
            iters, means, half = aggregate(df, mode, metric)
            if len(iters) == 0 or np.all(np.isnan(means)):
                continue
            s = _style(label)
            ax.plot(iters, means * scale, color=color, label=label, **s)
            ax.fill_between(iters, (means - half) * scale, (means + half) * scale,
                            color=color, alpha=CI_ALPHA)

        ax.set_xlabel("AL Iteration")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", color="silver", linewidth=0.8)
        if show_legend:
            ax.legend(fontsize="small", framealpha=1.0, loc="best")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", type=Path, required=True,
                   help="Parent diversity_eval directory")
    p.add_argument("--run_glob", type=str, required=True,
                   help="Glob for run_* dirs (relative to --eval_dir)")
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument("--pclm_thr", type=float, default=0.10,
                   help="PCLM cosine-distance threshold for coverage (default 0.10)")
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--no_legend", action="store_true",
                   help="Suppress legend on every panel")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, pre = load_run_dirs(args.eval_dir, args.run_glob, args.alpha, args.pclm_thr)

    # Persist the long-form metrics for downstream reuse
    csv_out = args.out_dir / f"per_iter_metrics_alpha{args.alpha}.csv"
    df.to_csv(csv_out, index=False)
    print(f"Saved {csv_out}")

    pclm_col = f"n_clusters_pclm_thr{args.pclm_thr}"

    legend = not args.no_legend
    # Per-metric plots
    plot_metric(df, pre, "mean_batch_valid_rate",
                ylabel="Validity (%)",
                out_path=args.out_dir / "mol_validity.png",
                scale=100.0, ylim=(0, 105), show_legend=legend)
    plot_metric(df, pre, pclm_col,
                ylabel=f"# PCLM clusters (d={args.pclm_thr})",
                out_path=args.out_dir / "mol_coverage.png", show_legend=legend)
    plot_metric(df, pre, "vendi_pclm_rbf_sigma0.5",
                ylabel="Vendi-PCLM (σ=0.5)",
                out_path=args.out_dir / "mol_diversity.png", show_legend=legend)
    plot_metric(df, pre, "fid_morgan_vs_pretrained",
                ylabel="FID-Morgan vs pretrained ↓",
                out_path=args.out_dir / "mol_fid.png", show_legend=legend)

    print(f"\nAll plots saved to {args.out_dir}")


if __name__ == "__main__":
    main()
