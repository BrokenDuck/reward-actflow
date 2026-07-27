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
    support_epsilon = 0.01

    def __init__(self, args: dict[str, Any], device: Optional[torch.device] = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        initial_model_invalid = args.get("initial_model_invalid", "false") == "true"
        data_mean = torch.tensor([-1.1, 0.0]) if initial_model_invalid else torch.zeros((2,))
        self._base_model = ToyBaseModel(device=device, data_mean=data_mean)
        self.device = device
        self.validity_mode = args.get("validity_mode", "grid3")
        if self.validity_mode == "grid3":
            self.N = 3
        elif self.validity_mode == "grid5":
            self.N = 5
        else:
            self.N = 3

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument("--validity_mode", type=str, default="grid3",
                            choices=["original", "grid3", "grid5"],
                            help="Validity function: 'original' (C-shape), 'grid3' (3x3 checkerboard), 'grid5' (5x5 checkerboard).")
        parser.add_argument("--initial_model_invalid", type=str, default="false",
                            choices=["true", "false"],
                            help="Set to 'true' to center initial model at (-1,0) instead of origin.")

    @property
    def base_model(self) -> BaseModel[DDTensor]:
        return self._base_model

    def validity(self, samples: DDTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        if self.validity_mode in ("grid3", "grid5"):
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
        N = self.N
        cell_size = (xmax - xmin) / N
        x = samples.data[:, 0]
        y = samples.data[:, 1]

        in_bounds = (x >= xmin) & (x < xmax) & (y >= xmin) & (y < xmax)
        i = ((x - xmin) / cell_size).long().clamp(0, N - 1)
        j = ((y - xmin) / cell_size).long().clamp(0, N - 1)
        checkerboard = (i + j) % 2 == 0
        return (in_bounds & checkerboard).bool()

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
        xmin, xmax = -3.5, 3.5
        bins = 100

        samples = env.sample(50_000, pbar=False).sample.data
        H, xedges, yedges = np.histogram2d(
            samples[:, 0].cpu().numpy(),
            samples[:, 1].cpu().numpy(),
            bins=bins,
            density=True,
            range=[[xmin, xmax], [xmin, xmax]],
        )
        support = H >= self.support_epsilon

        x = torch.linspace(xmin, xmax, bins)
        y = torch.linspace(xmin, xmax, bins)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        XY = torch.stack([X.flatten(), Y.flatten()], dim=-1)
        V = self.validity(DDTensor(XY), {}).reshape(X.shape)

        fig, ax = plt.subplots(figsize=(4, 4), constrained_layout=True)
        ax.imshow(
            support.T,
            extent=(xmin, xmax, xmin, xmax),
            origin="lower",
            cmap="binary",
        )
        ax.set_title(f"Generable Set (p >= {self.support_epsilon})")
        ax.contour(X.cpu(), Y.cpu(), V.cpu(), levels=[0.5], colors="black", alpha=0.3)

        return fig

    def save_samples(self, samples: DDTensor, kwargs: dict, dir: Path) -> bool:
        pass

    def load_samples(self, dir: Path) -> tuple[DDTensor, dict]:  # type: ignore
        pass

    def compute_metrics(self, samples: DDTensor, kwargs: dict) -> dict[str, float]:
        valid_data = samples.data[self.validity(samples, kwargs)]
        if len(valid_data) < 2:
            return {"vendi_rbf": 0.0, "vendi_linear": 0.0, "coverage": 0.0}

        vendi_linear = score_X(valid_data)

        length_scale = kwargs.get("vendi_lengthscale", 0.1)
        kernel = gpytorch.kernels.RBFKernel()
        kernel.lengthscale = torch.tensor(length_scale)
        K = kernel(valid_data, valid_data).cpu().numpy()
        vendi_rbf = score_K(K)

        coverage = self._compute_coverage(valid_data)

        return {
            "vendi_rbf": float(vendi_rbf),
            "vendi_linear": float(vendi_linear),
            "coverage": coverage,
        }

    def _compute_coverage(self, valid_data: torch.Tensor, subgrid: int = 100) -> float:
        """Fraction of valid sub-cells that contain at least one sample."""
        xmin, xmax = -3.5, 3.5
        n = self.N * subgrid

        coords = torch.linspace(xmin, xmax, n + 1)
        centers_x = (coords[:-1] + coords[1:]) / 2
        centers_y = centers_x.clone()
        CX, CY = torch.meshgrid(centers_x, centers_y, indexing="ij")
        grid_pts = torch.stack([CX.flatten(), CY.flatten()], dim=-1)
        valid_mask = self.validity(DDTensor(grid_pts), {}).reshape(n, n)
        n_valid = valid_mask.sum().item()
        if n_valid == 0:
            return 0.0

        cell_size = (xmax - xmin) / n
        ix = ((valid_data[:, 0] - xmin) / cell_size).long().clamp(0, n - 1)
        iy = ((valid_data[:, 1] - xmin) / cell_size).long().clamp(0, n - 1)
        occupied = torch.zeros(n, n, dtype=torch.bool)
        occupied[ix.cpu(), iy.cpu()] = True

        covered = (occupied & valid_mask.cpu()).sum().item()
        return covered / n_valid


class XReward(Reward[DDTensor]):
    def __call__(self, sample: DDTensor, latent: DDTensor, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x = sample.data[:, 0]
        return x, torch.ones_like(x)


class ToyBaseModel(BaseModel[DDTensor]):
    output_type = "velocity"

    def __init__(self, device: Optional[torch.device] = None, data_mean: Optional[torch.Tensor] = None):
        super().__init__(device)
        device = self.device

        self._scheduler = OptimalTransportScheduler()
        self.model = MLP(2, 2).to(device)

        if data_mean is None:
            data_mean = torch.zeros((2,))
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
