"""Local plotting script for toy experiment results.

Usage:
    python toy_plotting.py <run_dir> [--out <output_dir>]

Example:
    python toy_plotting.py ./dps_13.0_ls_0.1_num_iters_500_ft_steps_150
    python toy_plotting.py ./dps_13.0_ls_0.1_num_iters_500_ft_steps_150 --out ./plots

Required files in <run_dir>:
    - eval_history.csv          (validity, coverage, etc. over iterations)
    - eval/<iter>/generable_set.npz  (support mask + valid mask per eval point)
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_eval_history(run_dir: Path) -> list[dict]:
    path = run_dir / "eval_history.csv"
    if not path.exists():
        raise FileNotFoundError(f"No eval_history.csv found in {run_dir}")
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


def plot_validity_curve(eval_history: list[dict], out_dir: Path):
    if "model_valid" not in eval_history[0]:
        return
    iters = [int(h["iteration"]) for h in eval_history]
    validity = [h["model_valid"] * 100 for h in eval_history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, validity, marker="o", color="steelblue")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validity (%)")
    ax.set_ylim(0, 105)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "validity_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'validity_curve.png'}")


def plot_coverage_curve(eval_history: list[dict], out_dir: Path):
    if "generable_coverage" not in eval_history[0]:
        return
    iters = [int(h["iteration"]) for h in eval_history]
    coverage = [h["generable_coverage"] * 100 for h in eval_history]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(iters, coverage, marker="o", color="seagreen")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Generable Coverage (%)")
    ax.set_ylim(0, 105)
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(out_dir / "coverage_curve.png", dpi=150)
    plt.close(fig)
    print(f"Saved {out_dir / 'coverage_curve.png'}")


def plot_generable_set(run_dir: Path, out_dir: Path, iteration: int | None = None):
    """Plot the generable set for a specific eval iteration (default: last available)."""
    eval_dir = run_dir / "eval"
    if not eval_dir.exists():
        print("No eval/ directory found, skipping generable set plot.")
        return

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
        xmin, xmax = float(data["xmin"]), float(data["xmax"])
        epsilon = float(data["epsilon"])

        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        ax.imshow(
            support.T,
            extent=(xmin, xmax, xmin, xmax),
            origin="lower",
            cmap="binary",
        )
        ax.set_title(f"Generable Set (p >= {epsilon})")

        bins = valid_mask.shape[0]
        x = np.linspace(xmin, xmax, bins)
        y = np.linspace(xmin, xmax, bins)
        X, Y = np.meshgrid(x, y, indexing="ij")
        ax.contour(X, Y, valid_mask.astype(float), levels=[0.5], colors="black", alpha=0.3)

        iter_num = int(iter_dir.name)
        out_path = out_dir / f"generable_set_{iter_num:04d}.png"
        fig.savefig(out_path, dpi=300)
        plt.close(fig)
        print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot toy experiment results locally.")
    parser.add_argument("run_dir", type=Path, help="Path to experiment run directory")
    parser.add_argument("--out", type=Path, default=None, help="Output directory for plots (default: <run_dir>/local_plots)")
    parser.add_argument("--generable_iter", type=int, default=None,
                        help="Specific iteration for generable set plot (default: all available)")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out or run_dir / "local_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_history = load_eval_history(run_dir)
    plot_validity_curve(eval_history, out_dir)
    plot_coverage_curve(eval_history, out_dir)
    plot_generable_set(run_dir, out_dir, iteration=args.generable_iter)

    print(f"\nAll plots saved to {out_dir}")


if __name__ == "__main__":
    main()
