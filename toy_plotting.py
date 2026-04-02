"""Local plotting script for toy experiment results.

Subcommands
-----------
generable   Plot generable-set images for a single run (into <run_dir>/local_plots/).
shared      Plot combined validity/coverage curves, BoN histograms, and Pareto plot (into data/shared_plots/).
            Each --method flag points to a *parent* directory containing seed_* subdirectories.

Examples
--------
# Generable set plots for a single seed
python toy_plotting.py generable ./data/act_flow/seed_42 --color ours

# Combined curves + BoN histograms + Pareto plot (aggregated over seeds)
python toy_plotting.py shared \
    --act_flow ./data/act_flow \
    --baseline_no_filter ./data/baseline_no_filter \
    --baseline_filter ./data/baseline_filter
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tueplots import figsizes

COLOR_PRE = (0.35, 0.60, 0.85)              # light blue
COLOR_OURS = (0.35, 0.00, 0.55)             # dark violet
COLOR_VALID_BG = (0.90, 0.99, 0.90)         # pastel green
COLOR_REC_NO_FILTER = (0.93, 0.55, 0.14)    # orange
COLOR_REC_FILTER = (0.05, 0.42, 0.40)       # dark teal

COLOR_MAP = {
    "ours": COLOR_OURS,
    "baseline_no_filter": COLOR_REC_NO_FILTER,
    "baseline_filter": COLOR_REC_FILTER,
}

STYLE = {
    "ActFlow":          dict(marker="s", markersize=4, linewidth=2.0, linestyle="-"),
    "Rec (no filter)":  dict(marker="D", markersize=3, linewidth=1.2, linestyle="-"),
    "Rec (filter)":     dict(marker="o", markersize=3, linewidth=1.2, linestyle="-"),
    "Pre-trained":      dict(marker="*", markersize=6, linewidth=1.2, linestyle="--"),
}

CI_ALPHA = 0.2

def _style(label: str) -> dict:
    return STYLE.get(label, dict(marker="o", markersize=2, linewidth=1.0, linestyle="-"))


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_single_eval_history(run_dir: Path) -> list[dict]:
    path = run_dir / "eval_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"No eval_history.csv in {run_dir}")
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            rows.append(parsed)
    return rows


def _find_seed_dirs(method_dir: Path) -> list[Path]:
    """Return sorted list of seed_* directories that contain eval_history.csv."""
    return sorted(
        [d for d in method_dir.iterdir()
         if d.is_dir() and d.name.startswith("seed_")
         and (d / "eval_history.csv").exists()],
        key=lambda d: d.name,
    )


def load_multi_seed(method_dir: Path) -> list[list[dict]]:
    """Load eval_history.csv from every seed_* subdirectory."""
    seeds = _find_seed_dirs(method_dir)
    if not seeds:
        raise FileNotFoundError(f"No seed_* dirs with eval_history.csv in {method_dir}")
    histories = [_load_single_eval_history(s) for s in seeds]
    print(f"  Loaded {len(histories)} seeds from {method_dir}")
    return histories


def _resolve_key(row: dict, key: str) -> float | None:
    """Look up a metric key. Returns None if absent."""
    if key in row:
        return float(row[key])
    return None


def aggregate_metric(all_histories: list[list[dict]], key: str):
    """Aggregate a metric across seeds at each iteration.

    Returns (iters, means, ci95) arrays, all in *original* scale (not x100).
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


def _get_pre_stats(multi_runs):
    """Get mean and CI for pre-trained (iter 0) validity and coverage across all seeds."""
    all_val, all_cov = [], []
    for all_histories, _, _ in multi_runs:
        for hist in all_histories:
            first = hist[0]
            if int(first["iteration"]) == 0:
                v = first.get("model_valid")
                c = _resolve_key(first, "generable_coverage")
                if v is not None:
                    all_val.append(float(v) * 100)
                if c is not None:
                    all_cov.append(float(c) * 100)

    if not all_val:
        return None
    arr_val, arr_cov = np.array(all_val), np.array(all_cov)
    ci_fn = lambda a: 1.96 * a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0
    return {
        "val_mean": arr_val.mean(), "val_ci": ci_fn(arr_val),
        "cov_mean": arr_cov.mean(), "cov_ci": ci_fn(arr_cov),
    }


def _load_bon(run_dir: Path, iteration: int) -> dict[str, np.ndarray] | None:
    path = run_dir / "eval" / f"{iteration:04d}" / "bon_results.npz"
    if not path.exists():
        return None
    return dict(np.load(path))


def _last_eval_iter(run_dir: Path) -> int | None:
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        return None
    iters = sorted(
        [int(d.name) for d in eval_dir.iterdir()
         if d.is_dir() and (d / "bon_results.npz").exists()],
    )
    return iters[-1] if iters else None


# ── Curve plots (combined across methods) ────────────────────────────────────

def plot_validity_curve(multi_runs, out_dir: Path):
    pre = _get_pre_stats(multi_runs)

    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        if pre is not None:
            s = _style("Pre-trained")
            ax.axhline(pre["val_mean"], color=COLOR_PRE, linestyle=s["linestyle"],
                        linewidth=s["linewidth"], label="Pre-trained")
            if pre["val_ci"] > 0:
                ax.axhspan(pre["val_mean"] - pre["val_ci"],
                           pre["val_mean"] + pre["val_ci"],
                           color=COLOR_PRE, alpha=CI_ALPHA)

        for all_histories, label, color in multi_runs:
            iters, means, ci = aggregate_metric(all_histories, "model_valid")
            means_pct, ci_pct = means * 100, ci * 100
            s = _style(label)
            ax.plot(iters, means_pct, color=color, label=label, **s)
            ax.fill_between(iters, means_pct - ci_pct, means_pct + ci_pct,
                            color=color, alpha=CI_ALPHA)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Validity (%)")
        ax.set_ylim(65, 105)
        ax.grid(axis="y", linestyle="--", color="silver", linewidth=0.8)
        fig.savefig(out_dir / "toy_validity.png", dpi=150)
        plt.close(fig)
    print(f"Saved {out_dir / 'toy_validity.png'}")


def plot_coverage_curve(multi_runs, out_dir: Path):
    pre = _get_pre_stats(multi_runs)

    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        SHORT_LEGEND = {
            "Pre-trained": "PRE", "ActFlow": "ActFlow",
            "Rec (no filter)": "Rec-NF", "Rec (filter)": "Rec-F",
        }

        if pre is not None:
            s = _style("Pre-trained")
            ax.axhline(pre["cov_mean"], color=COLOR_PRE, linestyle=s["linestyle"],
                        linewidth=s["linewidth"], label="PRE")
            if pre["cov_ci"] > 0:
                ax.axhspan(pre["cov_mean"] - pre["cov_ci"],
                           pre["cov_mean"] + pre["cov_ci"],
                           color=COLOR_PRE, alpha=CI_ALPHA)

        for all_histories, label, color in multi_runs:
            iters, means, ci = aggregate_metric(all_histories, "generable_coverage")
            means_pct, ci_pct = means * 100, ci * 100
            s = _style(label)
            ax.plot(iters, means_pct, color=color,
                    label=SHORT_LEGEND.get(label, label), **s)
            ax.fill_between(iters, means_pct - ci_pct, means_pct + ci_pct,
                            color=color, alpha=CI_ALPHA)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Coverage (%)")
        ax.set_ylim(-5, 105)
        ax.grid(axis="y", linestyle="--", color="silver", linewidth=0.8)
        ax.legend(fontsize="small", loc="lower right",
                  bbox_to_anchor=(1.0, 0.15), framealpha=1.0)
        fig.savefig(out_dir / "toy_coverage.png", dpi=150)
        plt.close(fig)
    print(f"Saved {out_dir / 'toy_coverage.png'}")


# ── Pareto plot (coverage vs validity) ───────────────────────────────────────

def plot_pareto(multi_runs, out_dir: Path):
    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        points = []  # (x_mean, y_mean, x_ci, y_ci, name, marker, size, color)

        pre = _get_pre_stats(multi_runs)
        if pre is not None:
            s = _style("Pre-trained")
            points.append((pre["cov_mean"], pre["val_mean"],
                           pre["cov_ci"], pre["val_ci"],
                           "PRE", s["marker"], s["markersize"] ** 2 * 3, COLOR_PRE))

        SHORT_NAME = {
            "ActFlow": "ActFlow", "Rec (no filter)": "Rec-NF",
            "Rec (filter)": "Rec-F",
        }
        for all_histories, label, color in multi_runs:
            final_vals, final_covs = [], []
            for hist in all_histories:
                last = hist[-1]
                v = last.get("model_valid")
                c = _resolve_key(last, "generable_coverage")
                if v is not None and c is not None:
                    final_vals.append(float(v) * 100)
                    final_covs.append(float(c) * 100)
            if not final_vals:
                continue
            arr_v, arr_c = np.array(final_vals), np.array(final_covs)
            n = len(arr_v)
            ci_fn = lambda a: 1.96 * a.std(ddof=1) / math.sqrt(len(a)) if len(a) > 1 else 0.0
            s = _style(label)
            points.append((arr_c.mean(), arr_v.mean(),
                           ci_fn(arr_c), ci_fn(arr_v),
                           SHORT_NAME.get(label, label),
                           s["marker"], s["markersize"] ** 2 * 4, color))

        LABEL_OFFSETS = {
            "PRE": (8, 6), "Rec-NF": (8, -10),
            "Rec-F": (8, -10), "ActFlow": (-8, -12),
        }
        LABEL_HA = {"PRE": "left", "ActFlow": "right"}

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
        x_margin = max((max(xs) - min(xs)) * 0.25, 5)
        y_margin = max((max(ys) - min(ys)) * 0.25, 3)
        ax.set_xlim(max(min(xs) - x_margin, -5), min(max(xs) + x_margin, 105))
        ax.set_ylim(max(min(ys) - y_margin, 0), min(max(ys) + y_margin, 105))

        ax.set_xlabel("Coverage (%)")
        ax.set_ylabel("Validity (%)")
        ax.grid(axis="both", linestyle="--", color="silver", linewidth=0.8)
        fig.savefig(out_dir / "toy_pareto.png", dpi=150)
        plt.close(fig)
    print(f"Saved {out_dir / 'toy_pareto.png'}")


# ── BoN histogram plots (aggregated across seeds) ───────────────────────────

def plot_bon_histogram(bars: list[tuple[str, np.ndarray, tuple]],
                       title: str, out_dir: Path, filename: str):
    """Bar chart comparing methods on a BoN metric.

    bars: list of (label, per_seed_means_array, color).
    """
    SHORT_BON_LABEL = {
        "Pre-trained": "PRE", "ActFlow": "ActFlow",
        "Rec (no filter)": "Rec-NF", "Rec (filter)": "Rec-F",
    }
    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()
        labels = [SHORT_BON_LABEL.get(b[0], b[0]) for b in bars]
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
        ax.set_ylabel("Reward")

        fig.savefig(out_dir / filename, dpi=150)
        plt.close(fig)
    print(f"Saved {out_dir / filename}")


def build_bon_bars(multi_runs):
    """Build BoN bars from CSV data, one value per seed.

    multi_runs: list of (all_histories, label, color).
    Returns (top1_bars, topk_bars) where each entry is (label, per_seed_array, color).
    """
    top1_bars, topk_bars = [], []

    pre_top1, pre_topk = [], []
    for all_histories, _, _ in multi_runs:
        for hist in all_histories:
            first = hist[0]
            if int(first["iteration"]) == 0:
                t1 = first.get("bon_avg_top1")
                tk = first.get("bon_avg_topk")
                if t1 is not None:
                    pre_top1.append(float(t1))
                if tk is not None:
                    pre_topk.append(float(tk))

    if not pre_top1:
        return None, None

    top1_bars.append(("Pre-trained", np.array(pre_top1), COLOR_PRE))
    topk_bars.append(("Pre-trained", np.array(pre_topk), COLOR_PRE))

    for all_histories, label, color in multi_runs:
        seed_top1, seed_topk = [], []
        for hist in all_histories:
            last = hist[-1]
            t1 = last.get("bon_avg_top1")
            tk = last.get("bon_avg_topk")
            if t1 is not None:
                seed_top1.append(float(t1))
            if tk is not None:
                seed_topk.append(float(tk))
        if seed_top1:
            top1_bars.append((label, np.array(seed_top1), color))
            topk_bars.append((label, np.array(seed_topk), color))

    if len(top1_bars) <= 1:
        return None, None

    return top1_bars, topk_bars


# ── Generable-set plots (per-method, into local_plots/) ─────────────────────

def plot_generable_set(run_dir: Path, color: tuple,
                       iteration: int | None = None):
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        print("No eval/ directory found, skipping generable set plot.")
        return

    out_dir = run_dir / "local_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_iters = sorted(
        [d for d in eval_dir.iterdir() if d.is_dir() and (d / "generable_set.npz").exists()],
        key=lambda d: int(d.name),
    )
    if not eval_iters:
        print("No generable_set.npz files found, skipping generable set plot.")
        return

    if iteration is not None:
        target = eval_dir / f"{iteration:04d}"
        if target not in eval_iters:
            print(f"Iteration {iteration} not found. Available: {[int(d.name) for d in eval_iters]}")
            return
        eval_iters = [target]

    for iter_dir in eval_iters:
        data = np.load(iter_dir / "generable_set.npz")
        support = data["support"]
        valid_mask = data["valid_mask"]

        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)

        n_blocks = 3
        plot_extent = (0, n_blocks, 0, n_blocks)

        valid_color = np.array([*COLOR_VALID_BG, 1.0])
        bg_color = np.array([1.0, 1.0, 1.0, 1.0])
        valid_rgba = np.where(valid_mask.T[..., None], valid_color, bg_color)
        ax.imshow(valid_rgba, extent=plot_extent, origin="lower", interpolation="nearest")

        iter_num = int(iter_dir.name)
        cell_color = [*COLOR_PRE, 1.0] if iter_num == 0 else [*color, 1.0]
        support_rgba = np.zeros((*support.T.shape, 4))
        support_rgba[support.T.astype(bool)] = cell_color
        ax.imshow(support_rgba, extent=plot_extent, origin="lower", interpolation="nearest")

        ticks = np.arange(0, n_blocks + 1, 1.0)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        out_path = out_dir / f"generable_set_{iter_num:04d}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def cmd_generable(args):
    color = COLOR_MAP[args.color]
    plot_generable_set(args.run_dir, color=color, iteration=args.generable_iter)


def cmd_shared(args):
    multi_runs = []

    if args.act_flow:
        histories = load_multi_seed(args.act_flow)
        multi_runs.append((histories, "ActFlow", COLOR_OURS))
    if args.baseline_no_filter:
        histories = load_multi_seed(args.baseline_no_filter)
        multi_runs.append((histories, "Rec (no filter)", COLOR_REC_NO_FILTER))
    if args.baseline_filter:
        histories = load_multi_seed(args.baseline_filter)
        multi_runs.append((histories, "Rec (filter)", COLOR_REC_FILTER))

    if not multi_runs:
        print("No runs provided.")
        return

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_validity_curve(multi_runs, out_dir)
    plot_coverage_curve(multi_runs, out_dir)
    plot_pareto(multi_runs, out_dir)

    top1_bars, topk_bars = build_bon_bars(multi_runs)
    if top1_bars is not None:
        plot_bon_histogram(top1_bars, "Best-of-N (Top-1)", out_dir, "toy_top1.png")
    else:
        print("No BoN data found, skipping BoN top-1 plot.")
    if topk_bars is not None:
        plot_bon_histogram(topk_bars, "Best-of-N (Top-K)", out_dir, "toy_topk.png")
    else:
        print("No BoN data found, skipping BoN top-K plot.")

    print(f"\nAll shared plots saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Toy experiment plotting.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_gen = sub.add_parser("generable", help="Plot generable-set images for a single run")
    p_gen.add_argument("run_dir", type=Path)
    p_gen.add_argument("--color", choices=COLOR_MAP.keys(), default="ours")
    p_gen.add_argument("--generable_iter", type=int, default=None)
    p_gen.set_defaults(func=cmd_generable)

    p_sh = sub.add_parser("shared", help="Plot combined curves + BoN + Pareto")
    p_sh.add_argument("--act_flow", type=Path, default=None,
                      help="Parent dir with seed_* subdirs for ActFlow")
    p_sh.add_argument("--baseline_no_filter", type=Path, default=None,
                      help="Parent dir with seed_* subdirs for baseline (no filter)")
    p_sh.add_argument("--baseline_filter", type=Path, default=None,
                      help="Parent dir with seed_* subdirs for baseline (filter)")
    p_sh.add_argument("--out", type=Path, default=Path("data/shared_plots"))
    p_sh.set_defaults(func=cmd_shared)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
