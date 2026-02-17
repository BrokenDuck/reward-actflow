from typing import Any, Optional

import numpy as np
import torch
from diffusiongym import DDTensor, BaseModel, OptimalTransportScheduler, Scheduler, Environment, Reward, reward_registry
from diffusiongym.utils import train_base_model
from diffusiongym.base_models.one_dim_gmm import MLP
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pathlib import Path
import math

from .problem_setup import ProblemSetup
from adm.utils import Batch


class ToyProblemSetup(ProblemSetup[DDTensor]):
    def __init__(self, args: dict[str, Any], device: Optional[torch.device]=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._base_model = ToyBaseModel(device=device)

    @property
    def base_model(self) -> BaseModel[DDTensor]:
        return self._base_model

    def validity(self, samples: DDTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        x = samples.data[:, 0] + 0.5
        y = samples.data[:, 1]

        top = (x >= -2) & (x <= 1) & (y >= 1) & (y <= 2)
        middle = (x >= -2) & (x <= -1) & (y >= -1) & (y <= 1)
        bottom = (x >= -2) & (x <= 3) & (y >= -2) & (y <= -1)
        inside = top | middle | bottom
        return inside.bool()

    @property
    def feature_layer(self) -> str:
        return "input"

    def postprocess_features(self, latents: DDTensor, feats: DDTensor) -> torch.Tensor:
        return latents.data

    @torch.no_grad()
    def visualize_sample(self, env: Environment[DDTensor], batch: Batch[DDTensor]) -> Figure:
        xmin = -3.5
        xmax = 3.5
        delta = 0.05

        x = torch.linspace(xmin, xmax, 100)
        y = torch.linspace(xmin, xmax, 100)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        XY = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        Z, _ = env.reward(DDTensor(XY), DDTensor(XY), radius=math.sqrt(5)) 
        Z = Z.reshape(X.shape) 

        fig, axes = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
        im = axes[0].imshow(Z.T.cpu(), extent=(xmin, xmax, xmin, xmax), origin="lower", cmap="YlGn", aspect="equal") 
        axes[0].set_title("Uncertainty") 
        fig.colorbar(im, ax=axes[0], shrink=0.8)

        data = batch.samples.data
        valid = batch.valids

        axes[0].scatter(data[valid][:, 0].cpu(), data[valid][:, 1].cpu(), s=2, alpha=1.0)
        axes[0].scatter(data[~valid][:, 0].cpu(), data[~valid][:, 1].cpu(), s=2, alpha=1.0)

        axes[0].set_xlim(xmin, xmax)
        axes[0].set_ylim(xmin, xmax)

        many_samples = env.sample(50_000, pbar=False)[0].sample.data
        H, _, _ = np.histogram2d(
            many_samples[:, 0].cpu().numpy(),
            many_samples[:, 1].cpu().numpy(),
            bins=100,
            density=True,
            range=[[xmin, xmax], [xmin, xmax]],
        )
        H_flat = H.flatten()
        idx = np.argsort(H_flat)[::-1]
        cdf = np.cumsum(H_flat[idx])
        cdf /= cdf[-1]
        tau = H_flat[idx][np.searchsorted(cdf, 1 - delta)]
        support = H >= tau

        axes[1].imshow(
            support.T,
            extent=(xmin, xmax, xmin, xmax),
            origin="lower",
            cmap="binary",
        )
        axes[1].set_title(r"Model Support (1-$\delta$)")

        V = self.validity(DDTensor(XY), {}).reshape(X.shape)
        for i in range(2):
            axes[i].contour(X.cpu(), Y.cpu(), V.cpu(), levels=[0.5], colors="black", alpha=0.3)

        return fig

    def save_sample(self, sample: DDTensor, kwargs: dict, filename: Path):
        pass


class XReward(Reward[DDTensor]):
    def __call__(self, sample: DDTensor, latent: DDTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x = sample.data[:, 0]
        return x, torch.ones_like(x)



class ToyBaseModel(BaseModel[DDTensor]):
    output_type = "velocity"

    def __init__(self, device: Optional[torch.device]=None):
        super().__init__(device)
        device = self.device

        self._scheduler = OptimalTransportScheduler()
        self.model = MLP(2, 2).to(device)

        data_mean = torch.tensor([-1, 1.5])
        data = data_mean.unsqueeze(0) + 0.1 * torch.randn(512, 2)
        opt = torch.optim.Adam(self.parameters(), lr=1e-3)
        train_base_model(self, opt, [DDTensor(data)], steps=2500, batch_size=256, pbar=True)

    @property
    def scheduler(self) -> Scheduler[DDTensor]:
        return self._scheduler

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[DDTensor, dict[str, Any]]:
        return DDTensor(torch.randn(n, 2, device=self.device)), {}

    def forward(self, x: DDTensor, t: torch.Tensor, **kwargs: Any) -> DDTensor:
        return DDTensor(self.model(x.data, t))
