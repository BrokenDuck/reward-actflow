"""Reward Pareto-front plotting for GEOM molecule experiments.

Reads sample_metrics.csv files from geom_rewards/ directories and plots
HOMO vs LUMO Pareto frontiers for each method — matching the Pareto-front
style of adm.evaluation.pairwise_metric_scatter.

File-system layout
------------------
data/
  geom_rewards/
    pretrained/
      sample_metrics.csv
    ade_iter1000/
      sample_metrics.csv
    no_uncertainty_iter1000/
      sample_metrics.csv
    plots/            ← output directory

Examples
--------
python plotting_rewards.py \\
    --pretrained ./data/geom_rewards/pretrained \\
    --act_flow ./data/geom_rewards/ade_iter1000 \\
    --rec_f ./data/geom_rewards/no_uncertainty_iter1000 \\
    --out ./data/geom_rewards/plots
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tueplots import figsizes

N_REPEATS = 100
RNG_SEED = 42

COLOR_PRE = (0.35, 0.60, 0.85)
COLOR_OURS = (0.35, 0.00, 0.55)
COLOR_REC_NO_FILTER = (0.93, 0.55, 0.14)
COLOR_REC_FILTER = (0.05, 0.42, 0.40)

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

def _style(label: str) -> dict:
    return STYLE.get(label, dict(marker="o", markersize=2, linewidth=1.0, linestyle="-"))

def _short(label: str) -> str:
    return SHORT_LEGEND.get(label, label)


# ── Data loading ─────────────────────────────────────────────────────────────

def load_samples(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load valid samples with finite HOMO and LUMO from a sample_metrics.csv."""
    homo, lumo = [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["is_valid"] != "true":
                continue
            try:
                h, l = float(row["homo"]), float(row["lumo"])
            except (ValueError, TypeError, KeyError):
                continue
            if np.isfinite(h) and np.isfinite(l):
                homo.append(h)
                lumo.append(l)
    return np.array(homo), np.array(lumo)


# ── Pareto front (same logic as pairwise_metric_scatter.get_pareto_front) ────

def get_pareto_front(points: np.ndarray,
                     x_mode: str = "min",
                     y_mode: str = "min") -> np.ndarray:
    """Return Pareto-optimal points for the given optimisation directions."""
    processed = points.copy().astype(float)
    if x_mode == "max":
        processed[:, 0] *= -1
    if y_mode == "max":
        processed[:, 1] *= -1

    idx = np.lexsort((processed[:, 1], processed[:, 0]))
    sorted_pts = processed[idx]

    pareto_indices = []
    best_y = float("inf")
    for i, p in enumerate(sorted_pts):
        if p[1] < best_y:
            best_y = p[1]
            pareto_indices.append(idx[i])

    return points[pareto_indices]


# ── Fair subsampling ─────────────────────────────────────────────────────────

def _subsample(homo: np.ndarray, lumo: np.ndarray,
               n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Subsample without replacement to exactly *n* samples."""
    idx = rng.choice(len(homo), size=n, replace=False)
    return homo[idx], lumo[idx]


def _ci95(values: np.ndarray) -> float:
    n = len(values)
    return 1.96 * values.std(ddof=1) / math.sqrt(n) if n > 1 else 0.0


# ── Plot ─────────────────────────────────────────────────────────────────────

def plot_pareto_homo_lumo(methods: list[tuple], n_fair: int, out_dir: Path):
    """Single-panel Pareto front (maximize both HOMO and LUMO)."""
    rng = np.random.default_rng(RNG_SEED)

    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        for homo, lumo, label, color in methods:
            h, l = _subsample(homo, lumo, n_fair, rng)
            pts = np.column_stack([h, l])
            front = get_pareto_front(pts, x_mode="max", y_mode="max")
            s = _style(label)
            ax.plot(front[:, 0], front[:, 1], color=color,
                    linewidth=s["linewidth"], linestyle="-",
                    marker=s["marker"], markersize=s["markersize"],
                    label=_short(label), zorder=5)

        ax.set_xlabel("HOMO (eV)")
        ax.set_ylabel("LUMO (eV)")
        ax.grid(axis="both", linestyle="--", color="silver", linewidth=0.8)
        ax.legend(fontsize="small", framealpha=1.0)

        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "geom_pareto_homo_lumo.png", dpi=150,
                    bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out_dir / 'geom_pareto_homo_lumo.png'}")


def _pad_axes(ax, frac: float = 0.25):
    """Expand axis limits by *frac* of the data range on each side."""
    xl, xr = ax.get_xlim()
    yb, yt = ax.get_ylim()
    dx = (xr - xl) * frac
    dy = (yt - yb) * frac
    ax.set_xlim(xl - dx, xr + dx)
    ax.set_ylim(yb - dy, yt + dy)


def _plot_topk_independent(methods: list[tuple], n_fair: int, k: int, out_dir: Path):
    """Single marker per method: mean of top-k HOMO vs mean of top-k LUMO (independent)."""
    LABEL_OFFSETS = {
        "Pre": (6, -10), "ActFlow": (-6, -12),
        "Rec-F": (-6, 6), "Rec-NF": (6, 6),
    }
    LABEL_HA = {
        "Pre": "left", "ActFlow": "right",
        "Rec-F": "right", "Rec-NF": "left",
    }

    rng = np.random.default_rng(RNG_SEED)
    fname = f"geom_top{k}_homo_lumo.png"

    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        for homo, lumo, label, color in methods:
            xs, ys = [], []
            for _ in range(N_REPEATS):
                h, l = _subsample(homo, lumo, n_fair, rng)
                xs.append(np.sort(h)[-k:].mean())
                ys.append(np.sort(l)[-k:].mean())
            xs, ys = np.array(xs), np.array(ys)

            s = _style(label)
            name = _short(label)
            ax.errorbar(xs.mean(), ys.mean(),
                        xerr=_ci95(xs), yerr=_ci95(ys),
                        fmt="none", ecolor=color, capsize=3,
                        capthick=1.2, elinewidth=1.2, zorder=4)
            ax.scatter(xs.mean(), ys.mean(), marker=s["marker"],
                       s=s["markersize"] ** 2 * 4, color=color, zorder=5)
            off = LABEL_OFFSETS.get(name, (6, -10))
            ha = LABEL_HA.get(name, "left")
            ax.annotate(name, (xs.mean(), ys.mean()),
                        textcoords="offset points", xytext=off,
                        fontsize="small", color=color,
                        fontweight="bold", ha=ha)

        ax.set_xlabel("HOMO (eV)")
        ax.set_ylabel("LUMO (eV)")
        _pad_axes(ax)
        ax.grid(axis="both", linestyle="--", color="silver", linewidth=0.8)

        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out_dir / fname}")


def _plot_topk_joint(methods: list[tuple], n_fair: int, k: int, out_dir: Path):
    """Single marker per method: mean HOMO & LUMO of samples with top-k (HOMO+LUMO)."""
    LABEL_OFFSETS = {
        "Pre": (6, -10), "ActFlow": (-6, -12),
        "Rec-F": (-6, 6), "Rec-NF": (6, 6),
    }
    LABEL_HA = {
        "Pre": "left", "ActFlow": "right",
        "Rec-F": "right", "Rec-NF": "left",
    }

    rng = np.random.default_rng(RNG_SEED)
    fname = f"geom_top{k}_joint_homo_lumo.png"

    with plt.rc_context(figsizes.icml2024_half(nrows=1, ncols=1)):
        fig, ax = plt.subplots()

        for homo, lumo, label, color in methods:
            xs, ys = [], []
            for _ in range(N_REPEATS):
                h, l = _subsample(homo, lumo, n_fair, rng)
                score = h + l
                top_idx = np.argpartition(score, -k)[-k:]
                xs.append(h[top_idx].mean())
                ys.append(l[top_idx].mean())
            xs, ys = np.array(xs), np.array(ys)

            s = _style(label)
            name = _short(label)
            ax.errorbar(xs.mean(), ys.mean(),
                        xerr=_ci95(xs), yerr=_ci95(ys),
                        fmt="none", ecolor=color, capsize=3,
                        capthick=1.2, elinewidth=1.2, zorder=4)
            ax.scatter(xs.mean(), ys.mean(), marker=s["marker"],
                       s=s["markersize"] ** 2 * 4, color=color, zorder=5)
            off = LABEL_OFFSETS.get(name, (6, -10))
            ha = LABEL_HA.get(name, "left")
            ax.annotate(name, (xs.mean(), ys.mean()),
                        textcoords="offset points", xytext=off,
                        fontsize="small", color=color,
                        fontweight="bold", ha=ha)

        ax.set_xlabel("HOMO (eV)")
        ax.set_ylabel("LUMO (eV)")
        _pad_axes(ax)
        ax.grid(axis="both", linestyle="--", color="silver", linewidth=0.8)

        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved {out_dir / fname}")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="GEOM reward Pareto-front plotting (HOMO vs LUMO).")
    parser.add_argument("--pretrained", type=Path, default=None,
                        help="Dir with sample_metrics.csv for pre-trained model")
    parser.add_argument("--act_flow", type=Path, default=None,
                        help="Dir with sample_metrics.csv for ActFlow (ADE)")
    parser.add_argument("--rec_f", type=Path, default=None,
                        help="Dir with sample_metrics.csv for Rec (filter)")
    parser.add_argument("--rec_nf", type=Path, default=None,
                        help="Dir with sample_metrics.csv for Rec (no filter)")
    parser.add_argument("--out", type=Path, default=Path("data/geom_rewards/plots"))
    args = parser.parse_args()

    methods = []
    for arg, label, color in [
        (args.pretrained,  "Pre-trained",      COLOR_PRE),
        (args.act_flow,    "ActFlow",           COLOR_OURS),
        (args.rec_f,       "Rec (filter)",      COLOR_REC_FILTER),
        (args.rec_nf,      "Rec (no filter)",   COLOR_REC_NO_FILTER),
    ]:
        if arg is None:
            continue
        csv_path = arg / "sample_metrics.csv"
        homo, lumo = load_samples(csv_path)
        print(f"{_short(label)}: {len(homo)} valid samples from {csv_path}")
        methods.append((homo, lumo, label, color))

    if not methods:
        print("No methods provided.")
        return

    n_fair = min(len(h) for h, _, _, _ in methods)
    print(f"Fair subsampling: {n_fair} samples per method "
          f"({N_REPEATS} repeats, seed={RNG_SEED})")

    plot_pareto_homo_lumo(methods, n_fair, args.out)
    for k in [5, 10]:
        _plot_topk_independent(methods, n_fair, k, args.out)
        _plot_topk_joint(methods, n_fair, k, args.out)
    print(f"\nAll plots saved to {args.out}")


if __name__ == "__main__":
    main()
