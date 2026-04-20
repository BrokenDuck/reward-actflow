"""Local plotting script for molecule experiment results.

Subcommands
-----------
shared      Plot combined validity/novelty/diversity curves, bar charts, and Pareto plot
            (into data/shared_plots/).
            Each method flag points to a *parent* directory containing seed_* subdirectories,
            where each seed_* has eval/{iter:04d}/metrics.yaml files copied from the server.

File-system layout
------------------
data/
  act_flow/
    seed_42/
      eval/
        0000/metrics.yaml
        0050/metrics.yaml
        ...
    seed_43/
      eval/
        ...
  baseline_no_filter/
    seed_42/eval/...
  baseline_filter/
    seed_42/eval/...
  shared_plots/            ← output directory

Examples
--------
# Combined curves + bar charts + Pareto plot (aggregated over seeds)
python plotting.py shared \\
    --act_flow ./data/act_flow \\
    --baseline_no_filter ./data/baseline_no_filter \\
    --baseline_filter ./data/baseline_filter

# Only our method
python plotting.py shared --act_flow ./data/act_flow
"""

import argparse
import math
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt
from tueplots import figsizes

COLOR_PRE = (0.35, 0.60, 0.85)              # light blue
COLOR_OURS = (0.35, 0.00, 0.55)             # dark violet
COLOR_REC_NO_FILTER = (0.93, 0.55, 0.14)    # orange
COLOR_REC_FILTER = (0.05, 0.42, 0.40)       # dark teal

STYLE = {
    "ActFlow":          dict(marker="s", markersize=4, linewidth=2.0, linestyle="-"),
    "Rec (no filter)":  dict(marker="D", markersize=3, linewidth=1.2, linestyle="-"),
    "Rec (filter)":     dict(marker="o", markersize=3, linewidth=1.2, linestyle="-"),
    "Pre-trained":      dict(marker="*", markersize=6, linewidth=1.2, linestyle="--"),
}

SHORT_LEGEND = {
    "Pre-trained": "Pre", "ActFlow": "ActFlow",
    "Rec (no filter)": "Rec-NF", "Rec (filter)": "Rec-F",
}

CI_ALPHA = 0.2

def _style(label: str) -> dict:
    return STYLE.get(label, dict(marker="o", markersize=2, linewidth=1.0, linestyle="-"))

def _short(label: str) -> str:
    return SHORT_LEGEND.get(label, label)


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_single_eval_history(run_dir: Path) -> list[dict]:
    """Build eval history by reading all eval/{iter:04d}/metrics.yaml files."""
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        raise FileNotFoundError(f"No eval/ directory in {run_dir}")

    rows = []
    for iter_dir in sorted(eval_dir.iterdir()):
        if not iter_dir.is_dir():
            continue
        metrics_path = iter_dir / "metrics.yaml"
        if not metrics_path.exists():
            continue
        try:
            iteration = int(iter_dir.name)
        except ValueError:
            continue

        with open(metrics_path) as f:
            metrics = yaml.safe_load(f) or {}

        metrics["iteration"] = iteration
        rows.append(metrics)

    if not rows:
        raise FileNotFoundError(f"No eval/*/metrics.yaml files in {run_dir}")
    rows.sort(key=lambda r: r["iteration"])

    # The first eval is the pre-trained model (before any fine-tuning),
    # so remap it to iteration 0 regardless of its actual folder name.
    if rows:
        rows[0]["iteration"] = 0

    return rows


def _find_seed_dirs(method_dir: Path) -> list[Path]:
    """Return sorted list of seed_* directories that contain eval/ with metrics."""
    return sorted(
        [d for d in method_dir.iterdir()
         if d.is_dir() and d.name.startswith("seed_")
         and (d / "eval").exists()],
        key=lambda d: d.name,
    )


def load_multi_seed(method_dir: Path) -> list[list[dict]]:
    """Load eval histories from every seed_* subdirectory."""
    seeds = _find_seed_dirs(method_dir)
    if not seeds:
        raise FileNotFoundError(f"No seed_* dirs with eval/ in {method_dir}")
    histories = [_load_single_eval_history(s) for s in seeds]
    print(f"  Loaded {len(histories)} seeds from {method_dir}")
    return histories


def _resolve_key(row: dict, key: str) -> float | None:
    if key in row and row[key] is not None:
        return float(row[key])
    return None


def aggregate_metric(all_histories: list[list[dict]], key: str):
    """Aggregate a metric across seeds at each iteration.

    Returns (iters, means, ci95) arrays.
    """
    iters_set = sorted({int(row["iteration"]) for row in all_histories[0]})
    n_seeds = len(all_histories)
    n_iters = len(iters_set)

    matrix = np.full((n_seeds, n_iters), np.nan)
    for s, hist in enumerate(all_histories):
        iter_to_idx = {it: i for i, it in enumerate(iters_set)}
        for row in hist:
            it = int(row["iteration"])
            if it in iter_to_idx:
                val = _resolve_key(row, key)
                if val is not None:
                    matrix[s, iter_to_idx[it]] = val

    n_valid = np.sum(~np.isnan(matrix), axis=0)
    means = np.nanmean(matrix, axis=0)
    stds = np.nanstd(matrix, axis=0, ddof=1)
    ci95 = np.where(n_valid > 1, 1.96 * stds / np.sqrt(n_valid), 0.0)

    return np.array(iters_set), means, ci95


def _ci95_scalar(arr: np.ndarray) -> float:
    return 1.96 * arr.std(ddof=1) / math.sqrt(len(arr)) if len(arr) > 1 else 0.0


def _get_pre_stats(multi_runs):
    """Get mean and CI for pre-trained (iter 0) metrics across all seeds."""
    collected: dict[str, list[float]] = {}
    keys = {"model_valid": 100, "fid": 1, "n_clusters": 1, "vendi": 1}
    for all_histories, _, _ in multi_runs:
        for hist in all_histories:
            first = hist[0]
            if int(first["iteration"]) == 0:
                for key, scale in keys.items():
                    val = _resolve_key(first, key)
                    if val is not None:
                        collected.setdefault(key, []).append(val * scale)

    if "model_valid" not in collected:
        return None
    result = {}
    name_map = {"model_valid": "val", "fid": "fid", "n_clusters": "clust", "vendi": "vendi"}
    for key, prefix in name_map.items():
        if key in collected:
            arr = np.array(collected[key])
            result[f"{prefix}_mean"] = arr.mean()
            result[f"{prefix}_ci"] = _ci95_scalar(arr)
    return result


# ── Curve plots ──────────────────────────────────────────────────────────────

def plot_validity_curve(multi_runs, out_dir: Path):
    pre = _get_pre_stats(multi_runs)

    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        if pre is not None:
            s = _style("Pre-trained")
            ax.axhline(pre["val_mean"], color=COLOR_PRE, linestyle=s["linestyle"],
                       linewidth=s["linewidth"], label="Pre")
            if pre["val_ci"] > 0:
                ax.axhspan(pre["val_mean"] - pre["val_ci"],
                           pre["val_mean"] + pre["val_ci"],
                           color=COLOR_PRE, alpha=CI_ALPHA)

        for all_histories, label, color in multi_runs:
            iters, means, ci = aggregate_metric(all_histories, "model_valid")
            means_pct, ci_pct = means * 100, ci * 100
            s = _style(label)
            ax.plot(iters, means_pct, color=color, label=_short(label), **s)
            ax.fill_between(iters, means_pct - ci_pct, means_pct + ci_pct,
                            color=color, alpha=CI_ALPHA)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Validity (%)")
        ax.grid(axis="y", linestyle="--", color="silver", linewidth=0.8)
        ax.legend(fontsize="small", framealpha=1.0)
        fig.savefig(out_dir / "qm9_validity.png", dpi=150)
        plt.close(fig)
    print(f"Saved {out_dir / 'qm9_validity.png'}")


def plot_fid_curve(multi_runs, out_dir: Path):
    pre = _get_pre_stats(multi_runs)

    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        if pre is not None and "fid_mean" in pre:
            s = _style("Pre-trained")
            ax.axhline(pre["fid_mean"], color=COLOR_PRE, linestyle=s["linestyle"],
                       linewidth=s["linewidth"], label="Pre")
            if pre.get("fid_ci", 0) > 0:
                ax.axhspan(pre["fid_mean"] - pre["fid_ci"],
                           pre["fid_mean"] + pre["fid_ci"],
                           color=COLOR_PRE, alpha=CI_ALPHA)

        for all_histories, label, color in multi_runs:
            iters, means, ci = aggregate_metric(all_histories, "fid")
            s = _style(label)
            ax.plot(iters, means, color=color, label=_short(label), **s)
            ax.fill_between(iters, means - ci, means + ci,
                            color=color, alpha=CI_ALPHA)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("FID")
        ax.grid(axis="y", linestyle="--", color="silver", linewidth=0.8)
        fig.savefig(out_dir / "qm9_fid.png", dpi=150)
        plt.close(fig)
    print(f"Saved {out_dir / 'qm9_fid.png'}")


def plot_clusters_curve(multi_runs, out_dir: Path):
    pre = _get_pre_stats(multi_runs)

    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        if pre is not None and "clust_mean" in pre:
            s = _style("Pre-trained")
            ax.axhline(pre["clust_mean"], color=COLOR_PRE, linestyle=s["linestyle"],
                       linewidth=s["linewidth"], label="Pre")
            if pre.get("clust_ci", 0) > 0:
                ax.axhspan(pre["clust_mean"] - pre["clust_ci"],
                           pre["clust_mean"] + pre["clust_ci"],
                           color=COLOR_PRE, alpha=CI_ALPHA)

        for all_histories, label, color in multi_runs:
            iters, means, ci = aggregate_metric(all_histories, "n_clusters")
            s = _style(label)
            ax.plot(iters, means, color=color, label=_short(label), **s)
            ax.fill_between(iters, means - ci, means + ci,
                            color=color, alpha=CI_ALPHA)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Number of Clusters")
        ax.grid(axis="y", linestyle="--", color="silver", linewidth=0.8)
        fig.savefig(out_dir / "qm9_clusters.png", dpi=150)
        plt.close(fig)
    print(f"Saved {out_dir / 'qm9_clusters.png'}")


def plot_vendi_curve(multi_runs, out_dir: Path):
    pre = _get_pre_stats(multi_runs)

    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        if pre is not None and "vendi_mean" in pre:
            s = _style("Pre-trained")
            ax.axhline(pre["vendi_mean"], color=COLOR_PRE, linestyle=s["linestyle"],
                       linewidth=s["linewidth"], label="Pre")
            if pre.get("vendi_ci", 0) > 0:
                ax.axhspan(pre["vendi_mean"] - pre["vendi_ci"],
                           pre["vendi_mean"] + pre["vendi_ci"],
                           color=COLOR_PRE, alpha=CI_ALPHA)

        for all_histories, label, color in multi_runs:
            iters, means, ci = aggregate_metric(all_histories, "vendi")
            s = _style(label)
            ax.plot(iters, means, color=color, label=_short(label), **s)
            ax.fill_between(iters, means - ci, means + ci,
                            color=color, alpha=CI_ALPHA)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Vendi")
        ax.grid(axis="y", linestyle="--", color="silver", linewidth=0.8)
        fig.savefig(out_dir / "qm9_vendi.png", dpi=150)
        plt.close(fig)
    print(f"Saved {out_dir / 'qm9_vendi.png'}")


def _plot_generic_curve(multi_runs, metric: str, ylabel: str, filename: str,
                        out_dir: Path, scale: float = 1.0, ylim: tuple | None = None,
                        legend_kwargs: dict | None = None, show_legend: bool = True):
    """Generic metric-over-iterations curve plotter."""
    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        has_data = False
        for all_histories, label, color in multi_runs:
            iters, means, ci = aggregate_metric(all_histories, metric)
            if np.all(np.isnan(means)):
                continue
            has_data = True
            s = _style(label)
            ax.plot(iters, means * scale, color=color, label=_short(label), **s)
            ax.fill_between(iters, (means - ci) * scale, (means + ci) * scale,
                            color=color, alpha=CI_ALPHA)

        if not has_data:
            plt.close(fig)
            print(f"  No data for {metric}, skipping {filename}")
            return

        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis="y", linestyle="--", color="silver", linewidth=0.8)
        if show_legend:
            leg_kw = dict(fontsize="small", loc="best", framealpha=1.0)
            if legend_kwargs is not None:
                leg_kw.update(legend_kwargs)
            ax.legend(**leg_kw)
        fig.savefig(out_dir / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out_dir / filename}")


# ── Pareto plot (novelty vs validity) ────────────────────────────────────────

def plot_pareto(multi_runs, out_dir: Path):
    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        points = []

        pre = _get_pre_stats(multi_runs)
        if pre is not None and "clust_mean" in pre:
            s = _style("Pre-trained")
            points.append((pre["clust_mean"], pre["val_mean"],
                           pre.get("clust_ci", 0), pre["val_ci"],
                           "Pre", s["marker"], s["markersize"] ** 2 * 3, COLOR_PRE))

        for all_histories, label, color in multi_runs:
            final_vals, final_clusts = [], []
            for hist in all_histories:
                last = hist[-1]
                v = _resolve_key(last, "model_valid")
                c = _resolve_key(last, "n_clusters")
                if v is not None and c is not None:
                    final_vals.append(v * 100)
                    final_clusts.append(c)
            if not final_vals:
                continue
            arr_v, arr_c = np.array(final_vals), np.array(final_clusts)
            s = _style(label)
            points.append((arr_c.mean(), arr_v.mean(),
                           _ci95_scalar(arr_c), _ci95_scalar(arr_v),
                           _short(label), s["marker"], s["markersize"] ** 2 * 4, color))

        if not points:
            plt.close(fig)
            print("  No data for Pareto plot, skipping.")
            return

        LABEL_OFFSETS = {
            "Pre": (6, -10), "Rec-NF": (6, 6),
            "Rec-F": (6, -12), "ActFlow": (-4, -14),
        }
        LABEL_HA = {"Pre": "left", "Rec-NF": "left", "Rec-F": "left", "ActFlow": "center"}

        for xm, ym, xci, yci, name, mk, sz, col in points:
            ax.errorbar(xm, ym, xerr=xci, yerr=yci,
                        fmt="none", ecolor=col, capsize=3, capthick=1.2,
                        elinewidth=1.2, zorder=4)
            ax.scatter(xm, ym, marker=mk, s=sz, color=col, zorder=5)
            off = LABEL_OFFSETS.get(name, (8, 4))
            ha = LABEL_HA.get(name, "left")
            ax.annotate(name, (xm, ym), textcoords="offset points",
                        xytext=off, fontsize="small", color=col,
                        fontweight="bold", ha=ha)

        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        x_margin = max((max(xs) - min(xs)) * 0.25, 2)
        y_margin = max((max(ys) - min(ys)) * 0.25, 3)
        ax.set_xlim(min(xs) - x_margin, max(xs) + x_margin)
        ax.set_ylim(max(min(ys) - y_margin, 0), min(max(ys) + y_margin, 105))

        ax.set_xlabel("Number of Clusters")
        ax.set_ylabel("Validity (%)")
        ax.grid(axis="both", linestyle="--", color="silver", linewidth=0.8)
        fig.savefig(out_dir / "qm9_pareto.png", dpi=150)
        plt.close(fig)
    print(f"Saved {out_dir / 'qm9_pareto.png'}")


# ── Bar chart plots (final-iteration comparison) ─────────────────────────────

def _plot_bar_chart(bars: list[tuple[str, np.ndarray, tuple]],
                    ylabel: str, out_dir: Path, filename: str):
    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()
        labels = [_short(b[0]) for b in bars]
        means = [b[1].mean() for b in bars]
        n_seeds = [len(b[1]) for b in bars]
        ci95 = [1.96 * b[1].std(ddof=1) / math.sqrt(n) if n > 1 else 0.0
                for b, n in zip(bars, n_seeds)]
        colors = [b[2] for b in bars]

        x = np.arange(len(bars))
        ax.grid(axis="y", linestyle="--", color="silver", linewidth=0.8, zorder=0)
        ax.bar(x, means, yerr=ci95, capsize=4, color=colors, width=0.6, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize="small")
        ax.set_ylabel(ylabel)

        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
    print(f"Saved {out_dir / filename}")


def _build_final_bars(multi_runs, metric: str):
    """Build bars from the final iteration of each seed, including pre-trained (iter 0)."""
    pre_vals = []
    for all_histories, _, _ in multi_runs:
        for hist in all_histories:
            first = hist[0]
            if int(first["iteration"]) == 0:
                val = _resolve_key(first, metric)
                if val is not None:
                    pre_vals.append(val)

    bars = []
    if pre_vals:
        bars.append(("Pre-trained", np.array(pre_vals), COLOR_PRE))

    for all_histories, label, color in multi_runs:
        seed_vals = []
        for hist in all_histories:
            last = hist[-1]
            val = _resolve_key(last, metric)
            if val is not None:
                seed_vals.append(val)
        if seed_vals:
            bars.append((label, np.array(seed_vals), color))

    return bars if len(bars) > 1 else None


# ── CLI ──────────────────────────────────────────────────────────────────────

def _trim_histories(histories: list[list[dict]], max_iter: int) -> list[list[dict]]:
    """Keep only rows with iteration <= max_iter."""
    return [[row for row in hist if int(row["iteration"]) <= max_iter]
            for hist in histories]


def cmd_shared(args):
    multi_runs = []

    if args.act_flow:
        histories = load_multi_seed(args.act_flow)
        if args.max_iter is not None:
            histories = _trim_histories(histories, args.max_iter)
        multi_runs.append((histories, "ActFlow", COLOR_OURS))
    if args.baseline_no_filter:
        histories = load_multi_seed(args.baseline_no_filter)
        if args.max_iter is not None:
            histories = _trim_histories(histories, args.max_iter)
        multi_runs.append((histories, "Rec (filter)", COLOR_REC_FILTER))
    if args.baseline_filter:
        histories = load_multi_seed(args.baseline_filter)
        if args.max_iter is not None:
            histories = _trim_histories(histories, args.max_iter)
        multi_runs.append((histories, "Rec (no filter)", COLOR_REC_NO_FILTER))

    if not multi_runs:
        print("No runs provided.")
        return

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Iteration curves ──
    plot_validity_curve(multi_runs, out_dir)
    plot_fid_curve(multi_runs, out_dir)
    _plot_generic_curve(multi_runs, "sphere_exclusion_diversity",
                        "Sphere-Exclusion Diversity", "qm9_diversity.png", out_dir)
    plot_clusters_curve(multi_runs, out_dir)
    plot_vendi_curve(multi_runs, out_dir)

    # ── Pareto plot ──
    plot_pareto(multi_runs, out_dir)

    # ── Bar charts (final iteration) ──
    for metric, ylabel, filename in [
        ("sphere_exclusion_diversity", "Diversity", "qm9_diversity_bar.png"),
        ("n_clusters", "Clusters", "qm9_clusters_bar.png"),
        ("vendi", "Vendi", "qm9_vendi_bar.png"),
        ("fid", "FID", "qm9_fid_bar.png"),
    ]:
        bars = _build_final_bars(multi_runs, metric)
        if bars is not None:
            _plot_bar_chart(bars, ylabel, out_dir, filename)
        else:
            print(f"  No {metric} data found, skipping bar chart.")

    print(f"\nAll shared plots saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Molecule experiment plotting.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sh = sub.add_parser("shared", help="Plot combined curves + Pareto + bar charts")
    p_sh.add_argument("--act_flow", type=Path, default=None,
                      help="Parent dir with seed_* subdirs for ActFlow (ours)")
    p_sh.add_argument("--baseline_no_filter", type=Path, default=None,
                      help="Parent dir with seed_* subdirs for no_uncertainty (→ Rec-F)")
    p_sh.add_argument("--baseline_filter", type=Path, default=None,
                      help="Parent dir with seed_* subdirs for no_unc_no_ver (→ Rec-NF)")
    p_sh.add_argument("--max_iter", type=int, default=None,
                      help="Only plot up to this iteration (inclusive)")
    p_sh.add_argument("--out", type=Path, default=Path("data/shared_plots"))
    p_sh.set_defaults(func=cmd_shared)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
