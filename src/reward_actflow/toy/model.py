"""Flow model and base sampler for the 2-D staircase toy problem."""

from collections.abc import Mapping
from typing import Any

import torch
from diffusiongym.core.model import PredictionKind
from diffusiongym.toy.gmm2d import VelocityMLP
from diffusiongym.types import DDTensor
from torch import Generator, Tensor

__all__ = [
    "StandardNormalSampler",
    "ToyFlowModel",
    "VelocityMLP",
    "pretrain_on_data",
]


class ToyFlowModel:
    """Wraps a `VelocityMLP` into diffusiongym's `FlowModel` protocol.

    `state_dict` / `load_state_dict` are not decorative: `make._replica` uses
    them to give the train, rollout and reference policies identical weights.
    """

    prediction_kind = PredictionKind.VELOCITY

    def __init__(self, mlp: VelocityMLP, device: torch.device) -> None:
        self._mlp = mlp
        self._device = device

    @property
    def device(self) -> torch.device:
        return self._device

    def parameters(self):
        return self._mlp.parameters()

    def __call__(
        self, x_t: DDTensor, t: Tensor, *, conditioning: Mapping[str, Any]
    ) -> DDTensor:
        return DDTensor(self._mlp(x_t.data, t))

    def state_dict(self):
        return self._mlp.state_dict()

    def load_state_dict(self, state_dict):
        return self._mlp.load_state_dict(state_dict)

    def train(self, mode: bool = True):
        self._mlp.train(mode)
        return self

    def eval(self):
        return self.train(False)


class StandardNormalSampler:
    """N(0, I_2) base distribution."""

    def sample(
        self,
        n: int,
        *,
        conditioning: Mapping[str, Any],
        device: torch.device,
        generator: Generator | None = None,
    ) -> tuple[DDTensor, Mapping[str, Any]]:
        noise = torch.randn(n, 2, device=device, generator=generator)
        return DDTensor(noise), conditioning

    def sample_like(
        self, x_data: DDTensor, *, generator: Generator | None = None
    ) -> DDTensor:
        return DDTensor(
            torch.randn(
                x_data.data.shape,
                dtype=x_data.data.dtype,
                device=x_data.data.device,
                generator=generator,
            )
        )


def pretrain_on_data(
    model: ToyFlowModel,
    data: Tensor,
    *,
    steps: int = 2500,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: torch.device | None = None,
    generator: Generator | None = None,
    verbose: bool = False,
) -> list[float]:
    """Fit the velocity field to `data` with the rectified-flow objective.

    Loss: `E[||u_theta(x_t, t) - (x_1 - z)||^2]` with `x_t = (1-t) z + t x_1`,
    matching `RectifiedFlowSchedule`. Replaces the deleted
    `diffusiongym.utils.train_base_model`.
    """
    if device is None:
        device = model.device

    data = data.to(device)
    model._mlp.to(device)
    model._mlp.train()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    log_every = max(1, steps // 10)
    for step in range(steps):
        idx = torch.randint(
            0, data.shape[0], (batch_size,), device=device, generator=generator
        )
        x1 = data[idx]
        z = torch.randn(batch_size, 2, device=device, generator=generator)
        t = torch.rand(batch_size, device=device, generator=generator)

        x_t = (1 - t).unsqueeze(1) * z + t.unsqueeze(1) * x1
        target = x1 - z

        loss = (model._mlp(x_t, t) - target).square().mean()

        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

        if verbose and (step + 1) % log_every == 0:
            recent = sum(losses[-log_every:]) / log_every
            print(f"  pretrain step {step + 1:5d}/{steps}  loss={recent:.4f}")

    model._mlp.eval()
    return losses
