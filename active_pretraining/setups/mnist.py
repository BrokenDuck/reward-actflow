from typing import Any, Optional

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from diffusers.models.unets.unet_2d import UNet2DModel
from flowgym import FlowTensor, BaseModel, Environment, Scheduler, OptimalTransportScheduler
from flowgym.utils import train_base_model
from matplotlib.figure import Figure
from pathlib import Path
import os

from active_pretraining.problem_setup import ProblemSetup
from .mnist_classifier.lenet import LeNet5


BASE_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = BASE_DIR / "mnist_classifier" / "weights" / "lenet_epoch=12_test_acc=0.991.pth"


class MNISTProblemSetup(ProblemSetup[FlowTensor]):
    def __init__(self, args: dict[str, Any], device: Optional[torch.device]=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        valid_digits = args.get("mnist_valid_digits", None)
        if valid_digits is None:
            raise ValueError("mnist_valid_digits must be specified in args")

        base_digits = args.get("mnist_base_digits", None)
        if base_digits is None:
            raise ValueError("mnist_base_digits must be specified in args")

        self.valid_digits: tuple[int, ...] = tuple(valid_digits)
        self.base_digits: tuple[int, ...] = tuple(base_digits)

        # Construct classifier for validity function
        self.classifier = LeNet5().to(device)
        self.classifier.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        self.classifier.eval()

        # Construct base model
        self._base_model = MNISTBaseModel(digits=self.base_digits, device=device)

    @property
    def base_model(self) -> BaseModel[FlowTensor]:
        return self._base_model

    def validity(self, x: FlowTensor) -> torch.Tensor:
        y = x.data
        probs = self.classifier(y).softmax(dim=1)
        digit_probs = probs[:, self.valid_digits].max(dim=1)[0]
        return digit_probs > 0.99

    @property
    def feature_layer(self) -> str:
        return "unet.mid_block"

    def latent_postprocess(self, samples: FlowTensor) -> FlowTensor:
        samples = samples.clone()
        samples.data = samples.data.clamp(-1, 1)
        return samples

    def feature_postprocess(self, x: FlowTensor, feats: torch.Tensor) -> torch.Tensor:
        return feats.flatten(start_dim=1)

    def compute_metrics(self, samples: list[FlowTensor], valids: list[torch.Tensor]) -> dict[str, float]:
        # For each valid digit, compute the mean probability assigned by the classifier
        digit_probs = { digit: 0.0 for digit in self.valid_digits }
        for x, v in zip(samples, valids):
            probs = self.classifier(x.data[v]).softmax(dim=1)
            for digit in self.valid_digits:
                digit_probs[digit] += probs[:, digit].sum().item()

        total_valids = torch.cat(valids, dim=0).sum().item()
        return { f"digit_{digit}_prob": digit_probs[digit] / total_valids for digit in self.valid_digits }

    def visualize_sample(
        self,
        env: Environment[FlowTensor],
        samples: list[FlowTensor],
        valids: list[torch.Tensor],
    ) -> Figure:
        x = samples[-1].data.cpu()
        v = valids[-1].cpu()
        x_bordered = add_valid_border(x, v, thickness=1)

        fig = Figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(
            make_grid(x_bordered[:32], nrow=8, value_range=(0, 1), normalize=True).permute(1, 2, 0).numpy(),
            cmap="gray",
        )
        ax.axis("off")

        return fig

    def save_sample(self, sample: FlowTensor, filename: os.PathLike | str):
        x = sample.data.cpu()
        save_image(x, f"{filename}.png")


def add_valid_border(images: torch.Tensor, valids: torch.Tensor, thickness: int = 2) -> torch.Tensor:
    images = images.clone()
    if images.shape[1] == 1:
        images = torch.cat([images, images, images], dim=1)

    for i, valid in enumerate(valids):
        if valid:
            images[i, 1, :thickness, :] = 1.0
            images[i, 1, -thickness:, :] = 1.0
            images[i, 1, :, :thickness] = 1.0
            images[i, 1, :, -thickness:] = 1.0

    return images


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
