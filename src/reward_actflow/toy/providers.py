"""Registered `ModalityProvider` for the 2-D staircase toy problem.

Importing this module registers everything it defines.
"""

from pathlib import Path

import torch
from diffusiongym.core.codec import IdentityCodec
from diffusiongym.core.schedule import RectifiedFlowSchedule
from diffusiongym.core.space import TensorGeometry
from diffusiongym.registry import modality_registry

from reward_actflow.toy.model import (
    StandardNormalSampler,
    ToyFlowModel,
    VelocityMLP,
    pretrain_on_data,
)
from reward_actflow.toy.validity import base_training_data


@modality_registry.register("actflow/toy")
class ToyModality:
    """2-D staircase problem on dense `DDTensor` states.

    Parameters
    ----------
    checkpoint:
        Path to a saved `VelocityMLP` state dict. Loaded if it exists, and
        written after pretraining if it does not.
    pretrain_steps:
        Rectified-flow steps to run when no checkpoint is available. Zero leaves
        the model at its zero-output-head initialization, which exercises the
        wiring but produces no meaningful samples — useful in tests, useless in
        a run.
    width, depth:
        `VelocityMLP` size.
    """

    domain = "actflow"

    def __init__(
        self,
        *,
        checkpoint: str | Path | None = None,
        pretrain_steps: int = 2500,
        width: int = 128,
        depth: int = 3,
        num_base_points: int = 512,
    ) -> None:
        self.checkpoint = Path(checkpoint) if checkpoint is not None else None
        self.pretrain_steps = pretrain_steps
        self.width = width
        self.depth = depth
        self.num_base_points = num_base_points
        self._weights: dict | None = None

    def geometry(self) -> TensorGeometry:
        return TensorGeometry()

    def schedule(self) -> RectifiedFlowSchedule:
        return RectifiedFlowSchedule()

    def base_sampler(self) -> StandardNormalSampler:
        return StandardNormalSampler()

    def codec(self) -> IdentityCodec:
        return IdentityCodec()

    def model(self, *, device: torch.device) -> ToyFlowModel:
        """A fresh model carrying this provider's weights.

        Called once per policy. The weights are resolved once and cached, so the
        train, rollout and reference policies start identical — pretraining per
        call would leave them silently different, and every KL/EMA anchor in the
        framework would then be measuring against the wrong model.
        """
        network = VelocityMLP(width=self.width, depth=self.depth).to(device)
        network.load_state_dict(self._resolve_weights(device))
        return ToyFlowModel(network, device)

    def _resolve_weights(self, device: torch.device) -> dict:
        if self._weights is not None:
            return self._weights

        network = VelocityMLP(width=self.width, depth=self.depth).to(device)
        if self.checkpoint is not None and self.checkpoint.exists():
            network.load_state_dict(torch.load(self.checkpoint, map_location=device))
        elif self.pretrain_steps > 0:
            data = base_training_data(self.num_base_points, device=device)
            pretrain_on_data(
                ToyFlowModel(network, device),
                data,
                steps=self.pretrain_steps,
                device=device,
            )
            if self.checkpoint is not None:
                self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
                torch.save(network.state_dict(), self.checkpoint)

        self._weights = network.state_dict()
        return self._weights
