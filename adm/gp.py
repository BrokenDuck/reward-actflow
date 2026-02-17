from typing import Any, Callable, Generic, Optional

import torch
import torch.nn as nn
import gpytorch
from diffusiongym import D, BaseModel, Reward


class FlowFeatureExtractor(nn.Module, Generic[D]):
    """Makes it possible to extract features from a specific layer of a flow model.

    Parameters
    ----------
    base_model : BaseModel[D]
        The base generative flow model from which to extract features.
    layer : str
        The name of the layer from which to extract features. Use dot notation for nested layers.
        Returns the input if "input" is specified.
    timestep : float
        The timestep at which to extract features during the diffusion process. Earlier timesteps
        will give more semantic features, whereas later ones will give more low-level features.
    postprocess : Callable[[D, Any], torch.Tensor]
        An optional post-processing function to apply to the extracted features.
    """

    def __init__(
        self,
        base_model: BaseModel[D],
        layer: str,
        timestep: float,
        postprocess: Callable[[D, Any], torch.Tensor],
    ):
        super().__init__()
        self.base_model = base_model
        self.layer = layer
        self.timestep = timestep
        self._features = None
        self.postprocess = postprocess

        if layer != "input":
            self._register_hook()

    def _get_module_by_name(self, module: nn.Module, name: str):
        parts = name.split(".")
        for p in parts:
            if p.isdigit():
                module = module[int(p)]  # type: ignore
            else:
                module = getattr(module, p)

        return module

    def _register_hook(self):
        target = self._get_module_by_name(self.base_model, self.layer)

        def hook_fn(_, __, output):
            self._features = output

        self._hook = target.register_forward_hook(hook_fn)

    def forward(self, x1: D, **kwargs: Any) -> torch.Tensor:
        if self.layer == "input":
            return self.postprocess(x1, x1)

        self._features = None

        x0 = x1.randn_like()
        t = self.timestep * torch.ones(len(x1), device=x1.device)
        alpha = self.base_model.scheduler.alpha(x1, t)
        beta = self.base_model.scheduler.beta(x1, t)
        xt = alpha * x1 + beta * x0

        # Forward pass through the base model to obtain features through hook
        _ = self.base_model(xt, t, **kwargs)

        if self._features is None:
            raise RuntimeError(f"No features captured from layer '{self.layer}'")

        feats = self.postprocess(x1, self._features)
        return feats / feats.norm(dim=-1, keepdim=True)

    def remove_hook(self):
        self._hook.remove()


class GPUncertaintyReward(Reward[D]):
    """Reward based on uncertainty estimates from a Gaussian Process.

    Parameters
    ----------
    feat_extractor : FlowFeatureExtractor[D]
        Feature extractor to obtain features from data points. The GP will operate in this feature
        space.
    feat_dim : int
        Dimensionality of the extracted features.
    kernel : "rbf" | "linear"
        Kernel type for the Gaussian process.
    lengthscale : float
        Lengthscale parameter for the RBF kernel of the GP.
    device : Optional[torch.device | str]
        Device on which to run the GP model.
    """

    def __init__(
        self,
        feat_extractor: FlowFeatureExtractor[D],
        feat_dim: int,
        valid_fn: Callable[[D, dict[str, Any]], torch.Tensor],
        kernel: str = "rbf",
        lengthscale: float = 0.1,
        mean_weight: float = 0.0,
        device: Optional[torch.device | str] = None,
    ):
        if device is None:
            device = torch.device("cpu")

        feats = torch.empty(0, feat_dim, device=device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        labels = torch.zeros(feats.shape[0], device=device)
        model = GPModel(feats, labels, likelihood, kernel_type=kernel, lengthscale=lengthscale).to(device)

        model.eval()
        likelihood.eval()

        self.feat_extractor = feat_extractor
        self.valid_fn = valid_fn
        self.likelihood = likelihood
        self.model = model
        self.mean_weight = mean_weight
        self.device = device

    @torch.no_grad()
    def set_data(
        self,
        latents: list[D],
        kwargs: list[dict[str, Any]],
        vals: list[torch.Tensor] | None = None,
    ):
        """Set data by computing features and storing."""
        feats_list = []
        for latent, kwarg in zip(latents, kwargs):
            feat = self.feat_extractor(latent.to(self.device), **kwarg).detach()
            feats_list.append(feat)

        feats = torch.cat(feats_list, dim=0)

        labels = torch.zeros(feats.shape[0], device=self.device)
        if vals is not None:
            labels = torch.cat(vals, dim=0).to(self.device)
            labels = (labels - labels.mean()) / (labels.std() + 1e-8)

        self.model.set_train_data(feats, labels, strict=False)

    def __call__(self, sample: D | None, latent: D, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        latent = latent.to(self.device)
        feats = self.feat_extractor(latent, **kwargs)

        with (
            gpytorch.settings.fast_pred_var(),
            gpytorch.settings.max_root_decomposition_size(500),
        ):
            posterior = self.likelihood(self.model(feats))

        # If we are doing DPS, we set sample to None because we do not use the verifier
        if sample is not None:
            valids = self.valid_fn(sample, kwargs).cpu()
        else:
            valids = torch.ones(len(latent), dtype=torch.bool)

        return self.mean_weight * posterior.mean + posterior.variance, valids


class GPModel(gpytorch.models.ExactGP):
    """Gaussian Process model with RBF kernel."""

    def __init__(self, train_x, train_y, likelihood, kernel_type: str = "rbf", lengthscale=0.1):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        if kernel_type == "rbf":
            kernel = gpytorch.kernels.RBFKernel()
            kernel.lengthscale = lengthscale  # type: ignore
        elif kernel_type == "linear":
            kernel = gpytorch.kernels.LinearKernel()
        else:
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        self.covar_module = kernel

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)  # type: ignore
