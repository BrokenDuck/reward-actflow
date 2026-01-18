from typing import Any, Optional

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid, save_image
from diffusers.models.unets.unet_2d import UNet2DModel
from flowgym import FlowTensor, BaseModel, Environment, Scheduler, OptimalTransportScheduler
from flowgym.utils import train_base_model
from PIL import Image
from matplotlib.figure import Figure
from pathlib import Path
import os

from active_pretraining.problem_setup import ProblemSetup
from active_pretraining.utils import add_valid_border


import torch.nn as nn
import torch.nn.functional as F


class AutoEncoder(nn.Module):
    def __init__(self, n=8):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 1024),
            nn.SiLU(),
            nn.Linear(1024, 512),
            nn.SiLU(),
            nn.Linear(512, n),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n, 512),
            nn.SiLU(),
            nn.Linear(512, 1024),
            nn.SiLU(),
            nn.Linear(1024, 28 * 28),
        )

    def forward(self, x):
        x = x.flatten(1)

        if self.training:
            x = x + 0.03 * torch.randn_like(x)
            x = x.clamp(0, 1)

        z = self.encoder(x)
        y = self.decoder(z)

        recon_error = (x - y).square()
        weight = 1 + 2 * x.float()

        loss = (weight * recon_error).mean(dim=1)
        score = torch.log(loss + 1e-6)

        return y.view(-1, 1, 28, 28), loss, score


class MNISTProblemSetup(ProblemSetup[FlowTensor]):
    def __init__(self, args: dict[str, Any], device: Optional[torch.device]=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        base_digits = args.get("mnist_base_digits", None)
        if base_digits is None:
            raise ValueError("mnist_base_digits must be specified in args")

        self.base_digits: tuple[int, ...] = tuple(base_digits)

        # Construct base model
        self._base_model = MNISTBaseModel(digits=self.base_digits, device=device)

        self.ae_loss = AutoEncoder().to(device)
        state_dict = torch.load(Path(__file__).resolve().parent / "auto_encoder.pth", map_location=device)
        self.ae_loss.load_state_dict(state_dict)
        self.ae_loss.eval()

    @property
    def base_model(self) -> BaseModel[FlowTensor]:
        return self._base_model

    def _to_pil_images(self, x: FlowTensor) -> list[Image.Image]:
        img_list = []
        for i in range(len(x)):
            img_list.append(to_pil_image(x.data[i].to(dtype=torch.float)))
        return img_list

    @torch.no_grad()
    def validity(self, x: FlowTensor, kwargs: dict) -> torch.Tensor:
        y = x.data
        _, _, scores = self.ae_loss(y)
        # All-black images are easy for the autoencoder, but should not count as valid
        black_prop = (y < 0.1).float().mean(dim=[1,2,3])
        return (scores < -3) & (black_prop < 0.9)

    @property
    def feature_layer(self) -> str:
        return "unet.mid_block"

    def latent_postprocess(self, latents: FlowTensor, valids: torch.Tensor, kwargs: dict) -> FlowTensor:
        latents = latents.clone()
        latents.data = latents.data.clamp(-1, 1)
        return latents

    def feature_postprocess(self, x: FlowTensor, feats: torch.Tensor) -> torch.Tensor:
        return feats.mean(dim=[-2, -1])

    def visualize_sample(
        self,
        env: Environment[FlowTensor],
        samples: list[FlowTensor],
        valids: list[torch.Tensor],
    ) -> Figure:
        x = samples[-1].data.cpu()
        v = valids[-1].cpu()
        grid = make_grid(add_valid_border(x, v, thickness=1), nrow=8)

        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.axis("off")

        return fig

    def save_sample(self, sample: FlowTensor, kwargs: dict, filename: os.PathLike | str):
        x = sample.data.cpu()
        save_image(x, f"{filename}.png")


class MNISTBaseModel(BaseModel[FlowTensor]):
    output_type = "velocity"

    def __init__(self, digits: tuple[int, ...], device: Optional[torch.device] = None):
        super().__init__(device)
        self.digits = digits
        self._scheduler = OptimalTransportScheduler()
        self.unet = UNet2DModel(
            sample_size=28,
            in_channels=1,
            out_channels=1,
            norm_num_groups=8,
            block_out_channels=(32, 64),  # type: ignore
            down_block_types=("DownBlock2D", "DownBlock2D"),  # type: ignore
            up_block_types=("UpBlock2D", "UpBlock2D"),  # type: ignore
        )
        self.unet = self.unet.to(self.device)  # type: ignore
        print(f"U-net parameters: {sum(p.numel() for p in self.unet.parameters()):,}")
        self._load_or_train_unet()

    @property
    def scheduler(self) -> Scheduler:
        return self._scheduler

    def _load_or_train_unet(self):
        # Derive cache directory and file path
        cache_dir = os.path.expanduser("~/.cache/mnist_unet")
        os.makedirs(cache_dir, exist_ok=True)
        digits_str = "_".join(map(str, self.digits))
        cache_path = os.path.join(cache_dir, f"{digits_str}.pt")

        # Load from cache if available, else train and save
        if os.path.exists(cache_path):
            print(f"Loading cached UNet weights from {cache_path}")
            self.unet.load_state_dict(torch.load(cache_path, map_location=self.device))
        else:
            print("No cached UNet found. Training from scratch...")
            self._train_unet()
            torch.save(self.unet.state_dict(), cache_path)
            print(f"Saved UNet weights to {cache_path}")

    def _train_unet(self, steps=10_000, batch_size=128, lr=1e-4):
        dataset = torchvision.datasets.MNIST(
            os.path.expanduser("~/.cache"),
            train=True,
            download=True,
        )

        # Filter only the specified digits
        targets = dataset.targets
        mask = torch.zeros_like(targets, dtype=torch.bool)
        for digit in self.digits:
            mask |= targets == digit

        # Normalize data
        transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        data = transform(dataset.data[mask]).unsqueeze(1)

        opt = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        train_base_model(
            self,
            opt,
            [FlowTensor(data)],
            steps=steps,
            batch_size=batch_size,
            pbar=True,
        )

        self.eval()

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[FlowTensor, dict[str, Any]]:
        return FlowTensor(torch.randn(n, 1, 28, 28, device=self.device)), kwargs

    def postprocess(self, x: FlowTensor) -> FlowTensor:
        return FlowTensor(((x.data + 1) / 2).clamp(0, 1))

    def forward(self, x: FlowTensor, t: torch.Tensor, **kwargs: Any) -> FlowTensor:
        return FlowTensor(self.unet(x.data, t * 1000, **kwargs).sample)
