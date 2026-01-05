from typing import Optional, Any

import torch
import torchvision
import torchvision.transforms as transforms
from diffusers.models.unets.unet_2d import UNet2DModel
from flowgym import BaseModel, FlowTensor, Scheduler, OptimalTransportScheduler
from flowgym.utils import train_base_model
import os


class MNISTBaseModel(BaseModel[FlowTensor]):
    output_type = "velocity"

    def __init__(self, digits: tuple[int, ...], device: Optional[torch.device] = None):
        super().__init__(device)
        self.digits = digits
        self._scheduler = OptimalTransportScheduler()
        self.unet = UNet2DModel(
            sample_size=32,
            in_channels=1,
            out_channels=1,
            freq_shift=1,
            norm_num_groups=8,
            block_out_channels=(8, 16, 32, 32, 32),  # type: ignore
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            ),
            downsample_padding=1,
            attention_head_dim=1,
            time_embedding_type="positional",
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
        cache_path = os.path.join(cache_dir, f"{digits_str}_TEST.pt")

        # Load from cache if available, else train and save
        if os.path.exists(cache_path):
            print(f"Loading cached UNet weights from {cache_path}")
            self.unet.load_state_dict(torch.load(cache_path, map_location=self.device))
        else:
            print("No cached UNet found. Training from scratch...")
            self._train_unet()
            torch.save(self.unet.state_dict(), cache_path)
            print(f"Saved UNet weights to {cache_path}")

    def _train_unet(self, epochs=100, batch_size=128, lr=1e-4):
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
                transforms.Resize((32, 32)),
                transforms.Lambda(lambda x: x.float() / 255.0),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        data = transform(dataset.data[mask]).unsqueeze(1)

        opt = torch.optim.AdamW(self.unet.parameters(), lr=lr)
        train_base_model(
            self,
            [FlowTensor(data)],
            epochs=epochs,
            batch_size=batch_size,
            opt=opt,
            pbar=True,
        )

        self.eval()

    def sample_p0(self, n: int, **kwargs: Any) -> tuple[FlowTensor, dict[str, Any]]:
        return FlowTensor(torch.randn(n, 1, 32, 32, device=self.device)), kwargs

    def postprocess(self, x: FlowTensor) -> FlowTensor:
        return FlowTensor(((x.data + 1) / 2).clamp(0, 1))

    def forward(self, x: FlowTensor, t: torch.Tensor, **kwargs: Any) -> FlowTensor:
        return FlowTensor(self.unet(x.data, t * 1000, **kwargs).sample)
