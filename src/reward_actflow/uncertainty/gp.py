import torch
import gpytorch
from diffusiongym import D
from argparse import ArgumentParser

from reward_actflow.uncertainty.uncertainty_estimator import UncertaintyEstimator


class GPUncertaintyEstimator(UncertaintyEstimator[D]):
    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument(
            "--gp_kernel", type=str, choices=["linear", "rbf"], default="rbf"
        )
        parser.add_argument("--gp_lengthscale", type=float, default=0.1)

    def _init_estimator(self):
        feats = torch.empty(0, self.feat_dim, device=self.device)
        labels = torch.empty(0, device=self.device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = GPModel(
            feats,
            labels,
            self.likelihood,
            kernel_type=self.args["gp_kernel"],
            lengthscale=self.args["gp_lengthscale"],
        ).to(self.device)

        self.model.eval()
        self.likelihood.eval()

    def _update_estimator(self, feats: torch.Tensor, labels: torch.Tensor):
        self.model.set_train_data(feats, labels, strict=False)

    def _mean_and_uncertainty(
        self, feats: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with (
            gpytorch.settings.fast_pred_var(),
            gpytorch.settings.max_root_decomposition_size(500),
        ):
            posterior = self.likelihood(self.model(feats))

        return posterior.mean, posterior.variance


class GPModel(gpytorch.models.ExactGP):
    """Gaussian Process model with RBF kernel."""

    def __init__(
        self, train_x, train_y, likelihood, kernel_type: str = "rbf", lengthscale=0.1
    ):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_type == "rbf":
            kernel = gpytorch.kernels.RBFKernel()
            kernel.lengthscale = lengthscale
        elif kernel_type == "linear":
            kernel = gpytorch.kernels.LinearKernel()
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        self.covar_module = kernel

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)
