from argparse import ArgumentParser

import gpytorch
import torch
from diffusiongym.types import DDBatch

from reward_actflow.uncertainty.uncertainty_estimator import UncertaintyEstimator

#: Jitter added to the inducing-point Cholesky. The default (1e-6 for float32)
#: is not enough at small lengthscales, where the M x M inducing matrix is badly
#: conditioned, and gpytorch then warns once per posterior call.
_SPARSE_JITTER = 1e-4

#: `GridInterpolationKernel` is only sensible while the grid is affordable, which
#: is what caps its dimensionality: the grid has `grid_size ** feat_dim` points.
_MAX_GRID_DIMS = 4

BACKENDS = ("exact", "inducing", "grid")


class GPUncertaintyEstimator[D: DDBatch](UncertaintyEstimator[D]):
    """GP surrogate. Uncertainty is the posterior standard deviation.

    Three backends, chosen with `--gp_backend`, differing only in how they scale
    with the number of observations `n`:

    ``exact``
        Dense `n x n` covariance. Quadratic memory, so it stops being usable a
        few thousand observations in — at n=25,600 one posterior call over the
        plotting grid peaks at 4.0 GB, and 1000 ActFlow iterations at 64 samples
        would ask for roughly 16 GB of train covariance alone.
    ``inducing``
        `InducingPointKernel` on M points redrawn from the data each refit.
        O(M^2). Fastest of the three (0.03s to score a batch at n=64,000), and
        sound here because the inducing points come from the data, so away from
        them the posterior falls back to the prior rather than collapsing. It
        does flatten the explored region more than the exact GP does.
    ``grid``
        KISS-GP / `GridInterpolationKernel`. Near-exact in low dimensions and
        the most faithful of the three: measured at n=64,000 with grid_size=256,
        the posterior *variance* at an explored point is 0.910 against the
        exact GP's 0.954, where the inducing version reads 0.734 — figures
        measured pre-square-root, but the ranking (and hence the backend
        choice) is unaffected by the monotone `sqrt` this class now applies.
        Costs 630 MiB and 0.66s. Only valid for `feat_dim <= 4` — the grid
        holds `grid_size ** feat_dim` points — so it fits the 2-D toy and not
        learned molecular features.

    The grid is fixed by `--gp_grid_limit` rather than inferred from the data,
    because the data moves every iteration and a moving grid would silently
    change what sigma means. Features outside the box are clamped onto it: SKI
    raises on out-of-bounds input, and anything that far out is unexplored by
    definition, so the boundary's uncertainty — the prior — is the right answer
    for it anyway.
    """

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument(
            "--gp_kernel", type=str, choices=["linear", "rbf"], default="rbf"
        )
        parser.add_argument("--gp_lengthscale", type=float, default=0.1)
        parser.add_argument(
            "--gp_backend",
            type=str,
            choices=BACKENDS,
            default="exact",
            help=(
                "How the GP scales. 'exact' is only viable for a few thousand "
                "observations; use 'grid' for low-dimensional features and "
                "'inducing' otherwise."
            ),
        )
        parser.add_argument(
            "--gp_inducing",
            type=int,
            default=512,
            help="Inducing points, for --gp_backend inducing.",
        )
        parser.add_argument(
            "--gp_grid_size",
            type=int,
            default=256,
            help=(
                "Grid points per dimension, for --gp_backend grid. Spacing "
                "(2 * gp_grid_limit / gp_grid_size) should stay below "
                "--gp_lengthscale or the surrogate blurs the frontier."
            ),
        )
        parser.add_argument(
            "--gp_grid_limit",
            type=float,
            default=8.0,
            help=(
                "Half-width of the interpolation grid, for --gp_backend grid. "
                "Must comfortably contain everything the model generates."
            ),
        )

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        return str(self.args.get("gp_backend", "exact") or "exact")

    @property
    def num_inducing(self) -> int:
        return int(self.args.get("gp_inducing", 512) or 512)

    @property
    def grid_size(self) -> int:
        return int(self.args.get("gp_grid_size", 256) or 256)

    @property
    def grid_limit(self) -> float:
        return float(self.args.get("gp_grid_limit", 8.0) or 8.0)

    @property
    def is_sparse(self) -> bool:
        return self.backend == "inducing"

    @property
    def is_grid(self) -> bool:
        return self.backend == "grid"

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _init_estimator(self):
        if self.backend not in BACKENDS:
            raise ValueError(
                f"Unknown gp_backend {self.backend!r}. Expected one of {BACKENDS}."
            )
        if self.is_grid and self.feat_dim > _MAX_GRID_DIMS:
            raise ValueError(
                f"--gp_backend grid holds grid_size ** feat_dim points, which is "
                f"{self.grid_size}^{self.feat_dim} for these features. Use "
                f"--gp_backend inducing above {_MAX_GRID_DIMS} dimensions."
            )

        feats = torch.empty(0, self.feat_dim, device=self.device)
        labels = torch.empty(0, device=self.device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = self._build_model(feats, labels, inducing_points=None)

    def _build_model(
        self,
        feats: torch.Tensor,
        labels: torch.Tensor,
        inducing_points: torch.Tensor | None,
    ) -> "GPModel":
        model = GPModel(
            feats,
            labels,
            self.likelihood,
            kernel_type=self.args["gp_kernel"],
            lengthscale=self.args["gp_lengthscale"],
            inducing_points=inducing_points,
            grid_size=self.grid_size if self.is_grid else None,
            grid_limit=self.grid_limit,
            feat_dim=self.feat_dim,
        ).to(self.device)
        model.eval()
        self.likelihood.eval()
        return model

    def _confine(self, feats: torch.Tensor) -> torch.Tensor:
        """Clamp features into the interpolation grid.

        Only meaningful for the grid backend. `clamp` has zero gradient outside
        the box, which is the honest derivative: variance out there is the flat
        prior, so it really does not vary.
        """
        if not self.is_grid:
            return feats
        return feats.clamp(-self.grid_limit, self.grid_limit)

    def _select_inducing(self, feats: torch.Tensor) -> torch.Tensor:
        """A random subset of the observations, redrawn on every refit.

        Redrawing matters: the explored region grows over a run, and inducing
        points fixed at iteration 0 would summarize only where the model started.
        """
        n = feats.shape[0]
        if n <= self.num_inducing:
            return feats.clone()
        idx = torch.randperm(n, device=feats.device)[: self.num_inducing]
        return feats[idx].clone()

    def _update_estimator(self, feats: torch.Tensor, labels: torch.Tensor):
        feats = self._confine(feats)

        if not self.is_sparse:
            # The grid does not depend on the data, so like the exact GP this
            # only swaps the conditioning set.
            self.model.set_train_data(feats, labels, strict=False)
            return

        # Rebuilt rather than updated in place: the inducing points are baked
        # into the kernel, and they have to be redrawn as `D` grows.
        self.model = self._build_model(
            feats, labels, inducing_points=self._select_inducing(feats)
        )

    def _mean_and_uncertainty(
        self, feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with (
            gpytorch.settings.fast_pred_var(),
            gpytorch.settings.max_root_decomposition_size(500),
            gpytorch.settings.cholesky_jitter(_SPARSE_JITTER),
        ):
            posterior = self.likelihood(self.model(self._confine(feats)))

        # Standard deviation, not variance: every UncertaintyEstimator reports
        # uncertainty on the same scale so downstream UCB/LCB coefficients
        # (beta, eta, ...) are interpretable and transfer between backends —
        # see EnsembleUncertaintyEstimator, which has always reported std.
        # `clamp_min(0)` guards against a tiny negative variance from solver
        # roundoff, which `sqrt` would otherwise turn into a NaN.
        return posterior.mean, posterior.variance.clamp_min(0).sqrt()


class GPModel(gpytorch.models.ExactGP):
    """Zero-mean GP with a fixed-lengthscale kernel, optionally scaled up.

    Hyperparameters are set, never fitted — `set_train_data` only swaps the
    conditioning set — so the uncertainty is a pure "how close is this to
    something already observed" measure at the scale `lengthscale` sets.
    """

    def __init__(
        self,
        train_x,
        train_y,
        likelihood,
        kernel_type: str = "rbf",
        lengthscale: float = 0.1,
        inducing_points: torch.Tensor | None = None,
        grid_size: int | None = None,
        grid_limit: float = 8.0,
        feat_dim: int = 2,
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_type == "rbf":
            kernel = gpytorch.kernels.RBFKernel()
            kernel.lengthscale = torch.tensor(float(lengthscale))
        elif kernel_type == "linear":
            kernel = gpytorch.kernels.LinearKernel()
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        if inducing_points is not None and inducing_points.shape[0] > 0:
            kernel = gpytorch.kernels.InducingPointKernel(
                kernel, inducing_points=inducing_points, likelihood=likelihood
            )
        elif grid_size is not None:
            kernel = gpytorch.kernels.GridInterpolationKernel(
                kernel,
                grid_size=grid_size,
                num_dims=feat_dim,
                # One (low, high) pair per dimension. gpytorch annotates this as
                # a single pair but reads `len(grid_bounds)` against `num_dims`.
                grid_bounds=tuple(  # ty: ignore[invalid-argument-type]
                    (-grid_limit, grid_limit) for _ in range(feat_dim)
                ),
            )

        self.covar_module = kernel

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
