from typing import Any, Optional
from argparse import ArgumentParser

import numpy as np
import torch
from diffusiongym import DDTensor, BaseModel, OptimalTransportScheduler, Scheduler, Environment, Reward
from diffusiongym.utils import train_base_model
from diffusiongym.base_models.one_dim_gmm import MLP
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from pathlib import Path

from .problem_setup import ProblemSetup
from adm.uncertainty import UncertaintyEstimator
from adm.utils import Batch

from vendi_score.vendi import score_K, score_X

import gpytorch

class ToyProblemSetup(ProblemSetup[DDTensor]):
    def __init__(self, args: dict[str, Any], device: Optional[torch.device]=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._base_model = ToyBaseModel(device=device)
        self.device = device
        self.validity_mode = args.get("validity_mode", "original")

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument("--validity_mode", type=str, default="original",
                            choices=["original", "grid"],
                            help="Validity function: 'original' (C-shape) or 'grid' (9x9 checkerboard)")

    @property
    def base_model(self) -> BaseModel[DDTensor]:
        return self._base_model

    def validity(self, samples: DDTensor, kwargs: dict[str, Any]) -> torch.Tensor: # TODO move validity mode to kwargs
        if self.validity_mode == "grid":
            return self._grid_validity(samples)
        return self._original_validity(samples)

    def _original_validity(self, samples: DDTensor) -> torch.Tensor:
        x = samples.data[:, 0] + 0.5
        y = samples.data[:, 1]

        top = (x >= -2) & (x <= 1) & (y >= 1) & (y <= 2)
        middle = (x >= -2) & (x <= -1) & (y >= -1) & (y <= 1)
        bottom = (x >= -2) & (x <= 3) & (y >= -2) & (y <= -1)
        return (top | middle | bottom).bool()

    def _grid_validity(self, samples: DDTensor) -> torch.Tensor:
        xmin, xmax = -3.5, 3.5
        cell_size = (xmax - xmin) / 9
        x = samples.data[:, 0]
        y = samples.data[:, 1]

        in_bounds = (x >= xmin) & (x < xmax) & (y >= xmin) & (y < xmax)
        i = ((x - xmin) / cell_size).long().clamp(0, 8)
        j = ((y - xmin) / cell_size).long().clamp(0, 8)
        checkerboard = (i + j) % 2 == 0
        return (in_bounds & checkerboard).bool()

    def visualize_validity(self) -> Figure:
        xmin, xmax = -3.5, 3.5
        n = 300
        x = torch.linspace(xmin, xmax, n)
        y = torch.linspace(xmin, xmax, n)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        XY = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        V = self.validity(DDTensor(XY), {}).float().reshape(X.shape)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(V.T.cpu(), extent=(xmin, xmax, xmin, xmax), origin="lower",
                  cmap="RdYlGn", vmin=0, vmax=1, aspect="equal")
        cell_size = (xmax - xmin) / 9
        for k in range(10):
            pos = xmin + k * cell_size
            ax.axvline(pos, color="black", linewidth=0.5, alpha=0.4)
            ax.axhline(pos, color="black", linewidth=0.5, alpha=0.4)
        ax.set_title(f"Validity: {self.validity_mode}")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        return fig

    @property
    def feature_layer(self) -> str:
        return "input"

    def postprocess_features(self, latents: DDTensor, feats: DDTensor) -> torch.Tensor:
        return feats.data

    @torch.no_grad()
    def visualize_sample(
        self,
        env: Environment[DDTensor],
        uncertainty: UncertaintyEstimator[DDTensor],
        batch: Batch[DDTensor],
    ) -> Figure:
        xmin = -3.5
        xmax = 3.5

        x = torch.linspace(xmin, xmax, 100)
        y = torch.linspace(xmin, xmax, 100)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        XY = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        Z, _ = uncertainty(DDTensor(XY).to(self.device), DDTensor(XY).to(self.device)) 
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

        many_samples = env.sample(50_000, pbar=False).sample.data
        H, _, _ = np.histogram2d(
            many_samples[:, 0].cpu().numpy(),
            many_samples[:, 1].cpu().numpy(),
            bins=100,
            density=True,
            range=[[xmin, xmax], [xmin, xmax]],
        )
        support = H >= 0.01

        axes[1].imshow(
            support.T,
            extent=(xmin, xmax, xmin, xmax),
            origin="lower",
            cmap="binary",
        )
        axes[1].set_title(r"Model Support (p >= 0.01)")

        V = self.validity(DDTensor(XY), {}).reshape(X.shape)
        for i in range(2):
            axes[i].contour(X.cpu(), Y.cpu(), V.cpu(), levels=[0.5], colors="black", alpha=0.3)

        return fig

    def save_samples(self, samples: DDTensor, kwargs: dict, dir: Path) -> bool:
        pass

    def load_samples(self, dir: Path) -> tuple[DDTensor, dict]:  # type: ignore
        pass

    def compute_metrics(self, samples: DDTensor, kwargs: dict) -> dict[str, float]:
        valid_data = samples.data[self.validity(samples, kwargs)]
        vendi_linear = score_X(valid_data)
        
        length_scale = kwargs['vendi_lengthscale'] if 'vendi_lengthscale' in kwargs else 0.1

        kernel = gpytorch.kernels.RBFKernel()
        kernel.lengthscale = torch.tensor(length_scale)
        K = kernel(valid_data, valid_data).cpu().numpy()
        vendi_rbf = score_K(K)

        return {'vendi_rbf': float(vendi_rbf), 'vendi_linear': float(vendi_linear)}



class XReward(Reward[DDTensor]):
    def __call__(self, sample: DDTensor, latent: DDTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x = sample.data[:, 0]
        return x, torch.ones_like(x)


class ToyBaseModel(BaseModel[DDTensor]):
    output_type = "velocity"

    def __init__(self, device: Optional[torch.device]=None, val_mode: str = 'grid'):
        super().__init__(device)
        device = self.device

        self._scheduler = OptimalTransportScheduler()
        self.model = MLP(2, 2).to(device)

        data_mean = torch.zeros((2,)) if val_mode == 'grid' else torch.tensor([-1, 1.5])
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
