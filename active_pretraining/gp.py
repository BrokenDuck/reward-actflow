from typing import Any, Callable, Generic, Optional

import torch
import torch.nn as nn
import gpytorch
from flowgym import D, BaseModel, Reward


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

    postprocess : Optional[Callable[[Any], torch.Tensor]]
        An optional post-processing function to apply to the extracted features, defaults to
        identity.
    """

    def __init__(
        self,
        base_model: BaseModel[D],
        layer: str,
        timestep: float,
        postprocess: Optional[Callable[[D, Any], torch.Tensor]] = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.layer = layer
        self.timestep = timestep
        self._features = None

        if postprocess is None:
            postprocess = lambda x, feat: feat

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

    lengthscale : float
        Lengthscale parameter for the RBF kernel of the GP.

    device : Optional[torch.device | str]
        Device on which to run the GP model.
    """

    latent_space = True

    def __init__(
        self,
        feat_extractor: FlowFeatureExtractor[D],
        feat_dim: int,
        lengthscale: float = 0.1,
        device: Optional[torch.device | str] = None,
    ):
        if device is None:
            device = torch.device("cpu")

        feats = torch.empty(0, feat_dim, device=device)

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        labels = torch.zeros(feats.shape[0], device=device)
        model = GPModel(feats, labels, likelihood, lengthscale=lengthscale).to(device)

        model.eval()
        likelihood.eval()

        self.data = []
        self.feats = feats
        self.feat_extractor = feat_extractor
        self.likelihood = likelihood
        self.model = model
        self.device = device

    def add_data(self, new_data: D):
        new_data.to(self.device)
        self.data.append(new_data)
        with torch.no_grad():
            new_feats = self.feat_extractor(new_data)

        self.feats = torch.cat([self.feats, new_feats], dim=0)

        labels = torch.zeros(self.feats.shape[0], device=self.device)
        self.model.set_train_data(self.feats, labels, strict=False)

    def update_feats(self):
        """Re-compute all features."""
        feats = []
        with torch.no_grad():
            for data_points in self.data:
                feat = self.feat_extractor(data_points).detach()
                feats.append(feat)

        self.feats = torch.cat(feats, dim=0)

        labels = torch.zeros(self.feats.shape[0], device=self.device)
        self.model.set_train_data(self.feats, labels, strict=False)

    def __call__(self, x: D, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.to(self.device)
        feats = self.feat_extractor(x, **kwargs)

        with gpytorch.settings.fast_pred_var(), gpytorch.settings.max_root_decomposition_size(
            500
        ):
            posterior = self.likelihood(self.model(feats))

        uncertainty = posterior.variance
        return uncertainty, torch.ones_like(uncertainty)


class GPModel(gpytorch.models.ExactGP):
    """Gaussian Process model with RBF kernel."""

    def __init__(self, train_x, train_y, likelihood, lengthscale=0.1):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.RBFKernel()
        self.covar_module.lengthscale = lengthscale  # type: ignore

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)  # type: ignore
