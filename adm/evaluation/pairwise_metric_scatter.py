"""
Generate evaluation figures comparing multiple runs.

Produces three figure types:
  1. Property Combination Pareto fronts
  2. Mode retrieval (cumulative min/max)
  3. Property profiles (PCA density)

Usage:
    python -m adm.evaluation.pairwise_metric_scatter \
        --run /path/to/run1 eval_samples "Pre-Trained" \
        --run /path/to/run2 eval_samples "RS" \
        --run /path/to/run3 eval_samples "APT (Ours)" \
        --output figures/
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch
import yaml
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from adm.setups import setups as problem_setups


# Default color cycle for auto-assignment
DEFAULT_COLORS = [
    "black", "red", "green", "blue", "orange", "purple",
    "brown", "pink", "gray", "olive", "cyan",
]


class SampleCollection:
    def __init__(self, folder: Path, sample_folder: Path, label: str, color: str):
        self.folder = folder
        self.sample_folder = folder / sample_folder
        self.label = label
        self.color = color
        self.sample_metrics = pl.read_csv(self.sample_folder / "sample_metrics.csv")

        self.problem_setup = None
        self.samples = None
        self.kwargs = None

    def load_samples(self, device: torch.device):
        """Load the problem setup and actual sample objects from disk."""
        with open(self.folder / "args.yaml", "r") as f:
            exp_args = yaml.safe_load(f)

        self.problem_setup = problem_setups[exp_args["problem_setup"]](exp_args, device=device)
        self.samples, self.kwargs = self.problem_setup.load_samples(self.sample_folder)


def get_pareto_front(points: np.ndarray, x_mode: str = "min", y_mode: str = "min") -> np.ndarray:
    processed = points.copy().astype(float)
    if x_mode == "max":
        processed[:, 0] *= -1
    if y_mode == "max":
        processed[:, 1] *= -1

    idx = np.lexsort((processed[:, 1], processed[:, 0]))
    sorted_points = processed[idx]

    pareto_indices = []
    best_y = float("inf")
    for i, p in enumerate(sorted_points):
        if p[1] < best_y:
            best_y = p[1]
            pareto_indices.append(idx[i])

    return points[pareto_indices]


def to_float_numpy_with_nan(s: pl.Series) -> np.ndarray:
    return np.asarray(s.cast(pl.Float64).to_numpy(), dtype=float).reshape(-1)


def plot_pareto(collections: list[SampleCollection], metrics: list[str], output_dir: Path):
    """Property Combination Pareto fronts: for each x_metric, plot pareto fronts against all other metrics."""
    for x_metric in metrics:
        for x_mode in ["min", "max"]:
            fig, ax = plt.subplots(len(metrics), 2, figsize=(10, 5 * len(metrics)), sharey="row")

            for i, y_metric in enumerate(metrics):
                for j, y_mode in enumerate(["min", "max"]):
                    for coll in collections:
                        df = coll.sample_metrics
                        points = np.column_stack([df[x_metric].to_numpy(), df[y_metric].to_numpy()])
                        # Drop rows with NaN
                        mask = ~np.isnan(points).any(axis=1)
                        points = points[mask]
                        if len(points) == 0:
                            continue

                        pareto_front = get_pareto_front(points, x_mode=x_mode, y_mode=y_mode)
                        ax[i][j].plot(pareto_front[:, 0], pareto_front[:, 1],
                                      marker="o", color=coll.color, label=coll.label)
                        ax[i][j].set_xlabel(f"{x_metric} ({x_mode})")
                        ax[i][j].set_ylabel(f"{y_metric} ({y_mode})")

                        if i == 0 and j == 1:
                            ax[i][j].legend()

            fig.suptitle(f"Pareto fronts: x = {x_metric} ({x_mode})", fontsize=14)
            plt.tight_layout()
            fname = output_dir / f"pareto_{x_metric}_{x_mode}.png"
            fig.savefig(fname, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved {fname}")


def plot_mode_retrieval(collections: list[SampleCollection], metrics: list[str], output_dir: Path):
    """Mode retrieval: n vs cumulative min/max of each metric."""
    n_metrics = len(metrics)
    n_runs = 100
    z = 1.96

    fig, axes = plt.subplots(n_metrics, 2, figsize=(12, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = axes[np.newaxis, :]

    for row, metric in enumerate(metrics):
        for coll in collections:
            values = to_float_numpy_with_nan(coll.sample_metrics[metric])
            n = len(values)

            cum_max_runs = np.empty((n_runs, n), dtype=float)
            cum_min_runs = np.empty((n_runs, n), dtype=float)

            for r in range(n_runs):
                shuffled = np.random.permutation(values)

                sh_max = np.where(np.isnan(shuffled), -np.inf, shuffled)
                cmx = np.maximum.accumulate(sh_max)
                cmx = np.where(np.isneginf(cmx), np.nan, cmx)
                cum_max_runs[r] = cmx

                sh_min = np.where(np.isnan(shuffled), np.inf, shuffled)
                cmn = np.minimum.accumulate(sh_min)
                cmn = np.where(np.isposinf(cmn), np.nan, cmn)
                cum_min_runs[r] = cmn

            mean_max = np.nanmean(cum_max_runs, axis=0)
            se_max = np.nanstd(cum_max_runs, axis=0, ddof=1) / np.sqrt(n_runs)

            mean_min = np.nanmean(cum_min_runs, axis=0)
            se_min = np.nanstd(cum_min_runs, axis=0, ddof=1) / np.sqrt(n_runs)

            x = np.arange(1, n + 1)

            axes[row][0].plot(x, mean_min, label=coll.label, color=coll.color)
            axes[row][0].fill_between(x, mean_min - z * se_min, mean_min + z * se_min,
                                      color=coll.color, alpha=0.2)

            axes[row][1].plot(x, mean_max, label=coll.label, color=coll.color)
            axes[row][1].fill_between(x, mean_max - z * se_max, mean_max + z * se_max,
                                      color=coll.color, alpha=0.2)

        axes[row][0].set_title(f"{metric} (Expected Cumulative Min)")
        axes[row][0].set_ylabel(metric)
        axes[row][0].legend()
        axes[row][1].set_title(f"{metric} (Expected Cumulative Max)")

    plt.tight_layout()
    fname = output_dir / "mode_retrieval.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_property_profiles(collections: list[SampleCollection], metrics: list[str], output_dir: Path):
    """Property profiles: PCA density superlevel sets."""
    all_feats = []
    for coll in collections:
        df = coll.sample_metrics.filter(pl.col("is_valid")).select(pl.col(metrics))
        feats = df.to_numpy()
        feats = feats[~np.isnan(feats).any(axis=1), :]
        all_feats.append(feats)

    combined_feats = np.vstack(all_feats)
    reducer = make_pipeline(StandardScaler(), PCA(n_components=2)).fit(combined_feats)

    all_embeds = []
    for coll in collections:
        df = coll.sample_metrics.filter(pl.col("is_valid")).select(pl.col(metrics))
        feats = df.to_numpy()
        feats = feats[~np.isnan(feats).any(axis=1)]
        all_embeds.append(reducer.transform(feats))

    # Compute axis limits from all embeddings
    all_emb = np.vstack(all_embeds)
    x_margin = (all_emb[:, 0].max() - all_emb[:, 0].min()) * 0.2
    y_margin = (all_emb[:, 1].max() - all_emb[:, 1].min()) * 0.2
    xlim = (all_emb[:, 0].min() - x_margin, all_emb[:, 0].max() + x_margin)
    ylim = (all_emb[:, 1].min() - y_margin, all_emb[:, 1].max() + y_margin)

    fig, ax = plt.subplots(figsize=(10, 10))

    gridsize = 200
    alpha_density = 5e-4

    for coll, emb in zip(collections, all_embeds):
        kde = gaussian_kde(emb.T)
        xs = np.linspace(xlim[0], xlim[1], gridsize)
        ys = np.linspace(ylim[0], ylim[1], gridsize)
        xx, yy = np.meshgrid(xs, ys)
        zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

        ax.contourf(xx, yy, zz, levels=[alpha_density, zz.max()], alpha=0.3, colors=[coll.color])
        ax.plot([], [], color=coll.color, label=coll.label)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend()

    fname = output_dir / "property_profiles.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {fname}")


def plot_best_of_n_diversity(collections: list[SampleCollection], metrics: list[str], output_dir: Path):
    """Best-of-n diversity Pareto: E[f] vs diversity for varying n.

    For each sample metric and each n in ns:
      - Shuffle valid samples, take n*k, split into k groups of n
      - Pick the best sample per group (by max metric value) → k samples
      - Compute diversity metrics on those k samples via problem_setup.compute_metrics
      - Average over multiple random repeats
    Each n yields one point; connecting them traces a Pareto-like curve.
    """
    ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000]
    k = 100        # number of best-of-n samples (constant)
    repeats = 10   # random permutation repeats

    # Discover diversity metric names from the first collection
    first_coll = collections[0]
    probe_metrics = first_coll.problem_setup.compute_metrics(
        first_coll.samples[:k], first_coll.kwargs
    )
    diversity_metric_names = list(probe_metrics.keys())

    for metric in metrics:
        n_div = len(diversity_metric_names)
        fig, axes = plt.subplots(1, n_div, figsize=(6 * n_div, 5), squeeze=False)

        for coll in collections:
            # Get valid sample indices and their metric values
            df = coll.sample_metrics
            valid_df = df.filter(pl.col("is_valid") & pl.col(metric).is_not_null())
            valid_ids = valid_df["sample_id"].to_numpy()
            valid_values = valid_df[metric].to_numpy().astype(float)
            n_valid = len(valid_ids)

            # For each n, compute averaged E[metric] and diversity metrics
            mean_metric_by_n = []
            div_metrics_by_n = {name: [] for name in diversity_metric_names}
            ns_used = []

            for n in ns:
                if n * k > n_valid:
                    continue
                ns_used.append(n)

                rep_metric_vals = []
                rep_div_vals = {name: [] for name in diversity_metric_names}

                for _ in range(repeats):
                    perm = np.random.permutation(n_valid)[:n * k]
                    group_ids = valid_ids[perm]
                    group_vals = valid_values[perm]

                    # Split into k groups of n, pick argmax from each
                    group_ids = group_ids.reshape(k, n)
                    group_vals = group_vals.reshape(k, n)
                    best_in_group = group_vals.argmax(axis=1)
                    winning_ids = group_ids[np.arange(k), best_in_group]
                    winning_vals = group_vals[np.arange(k), best_in_group]

                    rep_metric_vals.append(winning_vals.mean())

                    # Extract actual samples at those indices and compute diversity
                    winning_samples = coll.samples[winning_ids.tolist()]
                    div = coll.problem_setup.compute_metrics(winning_samples, coll.kwargs)
                    for name in diversity_metric_names:
                        rep_div_vals[name].append(div[name])

                mean_metric_by_n.append(np.mean(rep_metric_vals))
                for name in diversity_metric_names:
                    div_metrics_by_n[name].append(np.mean(rep_div_vals[name]))

            # Plot one subplot per diversity metric
            for col_idx, div_name in enumerate(diversity_metric_names):
                ax = axes[0][col_idx]
                ax.plot(div_metrics_by_n[div_name], mean_metric_by_n,
                        marker="o", color=coll.color, label=coll.label)

                # Annotate n values
                for xi, yi, ni in zip(div_metrics_by_n[div_name], mean_metric_by_n, ns_used):
                    ax.annotate(str(ni), (xi, yi), textcoords="offset points",
                                xytext=(4, 4), fontsize=7, color=coll.color)

                ax.set_xlabel(div_name)
                ax.set_ylabel(f"E[{metric}] (best-of-n)")

        axes[0][0].legend()
        fig.suptitle(f"Best-of-n Diversity: {metric}", fontsize=14)
        plt.tight_layout()
        fname = output_dir / f"best_of_n_{metric}.png"
        fig.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


class RunAction(argparse.Action):
    """Parse --run FOLDER SAMPLE_FOLDER LABEL [COLOR] into a list of tuples."""
    def __call__(self, parser, namespace, values, option_string=None):
        runs = getattr(namespace, self.dest, None) or []
        if len(values) < 3:
            parser.error("--run requires at least 3 arguments: FOLDER SAMPLE_FOLDER LABEL [COLOR]")
        folder, sample_folder, label = values[0], values[1], values[2]
        color = values[3] if len(values) >= 4 else None
        runs.append((folder, sample_folder, label, color))
        setattr(namespace, self.dest, runs)


def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation comparison figures from multiple runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run", nargs="+", action=RunAction, dest="runs", metavar="ARG",
        help="FOLDER SAMPLE_FOLDER LABEL [COLOR]. Repeat for each run.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("figures"),
        help="Output directory for figures (default: figures/)",
    )
    parser.add_argument(
        "--x-metric", type=str, default=None,
        help="X metric for pareto plots. If not set, generates pareto plots for all metrics.",
    )
    parser.add_argument(
        "--skip-pareto", action="store_true",
        help="Skip pareto front plots (can be slow with many metrics).",
    )
    parser.add_argument(
        "--skip-mode-retrieval", action="store_true",
        help="Skip mode retrieval plots.",
    )
    parser.add_argument(
        "--skip-property-profiles", action="store_true",
        help="Skip property profile plots.",
    )
    parser.add_argument(
        "--skip-best-of-n", action="store_true",
        help="Skip best-of-n diversity plots.",
    )

    args = parser.parse_args()

    if not args.runs:
        parser.error("At least one --run is required.")

    # Build sample collections
    collections = []
    for i, (folder, sample_folder, label, color) in enumerate(args.runs):
        c = color or DEFAULT_COLORS[i % len(DEFAULT_COLORS)]
        collections.append(SampleCollection(Path(folder), Path(sample_folder), label, c))

    # Discover metrics from first collection (skip sample_id and is_valid)
    metrics = collections[0].sample_metrics.columns[2:]
    print(f"Metrics: {metrics}")

    args.output.mkdir(parents=True, exist_ok=True)

    if not args.skip_pareto:
        print("Generating pareto front plots...")
        if args.x_metric:
            # Only generate for the specified x_metric
            x_metrics_to_plot = [args.x_metric]
        else:
            x_metrics_to_plot = metrics

        for x_metric in x_metrics_to_plot:
            for x_mode in ["min", "max"]:
                fig, ax = plt.subplots(len(metrics), 2, figsize=(10, 5 * len(metrics)), sharey="row")
                if len(metrics) == 1:
                    ax = ax[np.newaxis, :]

                for i, y_metric in enumerate(metrics):
                    for j, y_mode in enumerate(["min", "max"]):
                        for coll in collections:
                            df = coll.sample_metrics
                            points = np.column_stack([df[x_metric].to_numpy(), df[y_metric].to_numpy()])
                            mask = ~np.isnan(points).any(axis=1)
                            points = points[mask]
                            if len(points) == 0:
                                continue

                            pareto_front = get_pareto_front(points, x_mode=x_mode, y_mode=y_mode)
                            ax[i][j].plot(pareto_front[:, 0], pareto_front[:, 1],
                                          marker="o", color=coll.color, label=coll.label)
                            ax[i][j].set_xlabel(f"{x_metric} ({x_mode})")
                            ax[i][j].set_ylabel(f"{y_metric} ({y_mode})")

                            if i == 0 and j == 1:
                                ax[i][j].legend()

                fig.suptitle(f"Pareto fronts: x = {x_metric} ({x_mode})", fontsize=14)
                plt.tight_layout()
                fname = args.output / f"pareto_{x_metric}_{x_mode}.png"
                fig.savefig(fname, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"  Saved {fname}")

    if not args.skip_mode_retrieval:
        print("Generating mode retrieval plots...")
        plot_mode_retrieval(collections, metrics, args.output)

    if not args.skip_property_profiles:
        print("Generating property profile plots...")
        plot_property_profiles(collections, metrics, args.output)

    if not args.skip_best_of_n:
        print("Loading samples for best-of-n diversity plots...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for coll in collections:
            coll.load_samples(device)
        print("Generating best-of-n diversity plots...")
        plot_best_of_n_diversity(collections, metrics, args.output)

    print("Done.")


if __name__ == "__main__":
    main()
