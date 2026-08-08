from argparse import ArgumentParser
from pathlib import Path
from typing import Any

import torch
from diffusiongym import FineTuningSetup
from diffusiongym.types import DDTensor
from matplotlib.figure import Figure

import reward_actflow.toy  # noqa: F401  (registers actflow/toy)
from reward_actflow.setups.problem_setup import ProblemSetup
from reward_actflow.toy.reward import TOY_REWARDS
from reward_actflow.toy.validity import base_training_data, staircase_validity
from reward_actflow.toy.visualize import (
    coverage_metrics,
    plot_actflow_r_iteration,
    plot_iteration,
    sample_model,
)
from reward_actflow.uncertainty import UncertaintyEstimator
from reward_actflow.utils import Batch


class ToyProblemSetup(ProblemSetup[DDTensor]):
    """2-D staircase region, explored from a tight blob inside its top slab."""

    def __init__(self, args: dict[str, Any], device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.device = device
        self.args = args

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument(
            "--toy_pretrain_steps",
            type=int,
            default=2500,
            help="Rectified-flow steps used to pretrain the base model.",
        )
        parser.add_argument(
            "--toy_checkpoint",
            type=Path,
            default=None,
            help="Cache path for the pretrained base model weights.",
        )
        parser.add_argument("--toy_width", type=int, default=128)
        parser.add_argument("--toy_depth", type=int, default=3)
        parser.add_argument(
            "--toy_reward",
            type=str,
            choices=tuple(TOY_REWARDS),
            default="linear",
            help=(
                "Black-box task reward for ActFlow-R. 'linear' increases in x "
                "and is maximised (within the valid set) on bottom's far right "
                "edge; 'bump' is a Gaussian centred there instead, non-zero "
                "only near the target."
            ),
        )

    @property
    def modality_id(self) -> str:
        return "actflow/toy"

    @property
    def modality_kwargs(self) -> dict[str, Any]:
        return {
            "pretrain_steps": self.args.get("toy_pretrain_steps", 2500),
            "checkpoint": self.args.get("toy_checkpoint"),
            "width": self.args.get("toy_width", 128),
            "depth": self.args.get("toy_depth", 3),
        }

    def validity(self, samples: DDTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        return staircase_validity(samples.data)

    @property
    def feature_layer(self) -> str:
        # The problem is 2-D, so the sample itself is the representation; there
        # is nothing a hidden layer could add.
        return "input"

    def postprocess_features(self, latents: DDTensor, feats: Any) -> torch.Tensor:
        return feats.data

    @torch.no_grad()
    def visualize_sample(
        self,
        setup: FineTuningSetup,
        uncertainty: UncertaintyEstimator[DDTensor],
        batch: Batch[DDTensor],
    ) -> tuple[Figure, dict[str, float]]:
        return plot_iteration(
            setup,
            uncertainty,
            batch.samples.data,
            batch.valids,
        )

    def save_samples(self, samples: DDTensor, kwargs: dict, dir: Path) -> bool:
        dir.mkdir(parents=True, exist_ok=True)
        torch.save(samples.data.detach().cpu(), dir / "samples.pt")
        return True

    def load_samples(self, dir: Path) -> tuple[DDTensor, dict]:
        return DDTensor(torch.load(dir / "samples.pt", map_location="cpu")), {}

    def compute_metrics(self, samples: DDTensor, kwargs: dict) -> dict[str, float]:
        return coverage_metrics(samples.data.detach().cpu())

    def task_reward(self, samples: DDTensor, kwargs: dict[str, Any]) -> torch.Tensor:
        reward_fn = TOY_REWARDS[self.args.get("toy_reward", "linear")]
        return reward_fn(samples.data)

    def anchor_latents(self, n: int, device: torch.device) -> DDTensor:
        # Sample space *is* latent space here (`IdentityCodec`), so this can
        # feed `base_training_data`'s raw coordinates straight through; a
        # setup with a real codec would need to encode first.
        return DDTensor(base_training_data(n, device=device))

    def diagnostic_coordinates(self, latents: DDTensor) -> torch.Tensor:
        # The problem is already 2-D, so the raw coordinates are the fixed
        # descriptor — same reasoning as feature_layer == "input".
        return latents.data

    @torch.no_grad()
    def visualize_reward_sample(
        self,
        setup: FineTuningSetup,
        uncertainty: UncertaintyEstimator[DDTensor],
        reward_uncertainty: UncertaintyEstimator[DDTensor],
        anchors: DDTensor,
        batch: Batch[DDTensor],
    ) -> tuple[Figure, dict[str, float]]:
        reward_fn = TOY_REWARDS[self.args.get("toy_reward", "linear")]
        return plot_actflow_r_iteration(
            setup,
            uncertainty,
            reward_fn,
            anchors.data,
            batch.samples.data,
            batch.valids,
        )


__all__ = ["ToyProblemSetup", "sample_model"]
