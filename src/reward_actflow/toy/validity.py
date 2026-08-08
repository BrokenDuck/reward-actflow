"""The 2-D staircase verifier and the base distribution it is explored from.

The valid set is three axis-aligned rectangles forming a Z: a wide `top` slab, a
narrow `middle` corridor hanging off its left edge, and a wide `bottom` slab. The
flow model is pretrained on a tight blob that sits *inside* `top`, so a run that
merely stays where it started scores a perfect valid rate while discovering
nothing. Reaching `bottom` requires threading the one-unit-wide corridor, and
that is the behaviour the whole setup exists to measure.
"""

import torch
from torch import Tensor

#: Centre of the pretraining blob. Inside `top` once the x-offset is applied.
BASE_MEAN = (-1.0, 1.5)

#: Standard deviation of the pretraining blob.
BASE_STD = 0.1

#: Plot bounds that comfortably contain the whole valid set.
PLOT_LIMITS = (-3.5, 3.5)


def staircase_validity(x: Tensor) -> Tensor:
    """Black-box verifier `v(x)` for the staircase region.

    Parameters
    ----------
    x : Tensor
        Points of shape `(n, 2)`.

    Returns
    -------
    Tensor
        Boolean tensor of shape `(n,)`.
    """
    # The +0.5 shift is part of the problem definition: it puts the corridor
    # off-centre relative to the base blob.
    px = x[:, 0] + 0.5
    py = x[:, 1]

    top = (px >= -2) & (px <= 1) & (py >= 1) & (py <= 2)
    middle = (px >= -2) & (px <= -1) & (py >= -1) & (py <= 1)
    bottom = (px >= -2) & (px <= 3) & (py >= -2) & (py <= -1)

    return (top | middle | bottom).bool()


def valid_area() -> float:
    """Area of the valid set, for normalizing coverage metrics."""
    return 3.0 * 1.0 + 1.0 * 2.0 + 5.0 * 1.0


def base_training_data(
    n: int = 512,
    *,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Samples of the distribution the flow model is pretrained on."""
    mean = torch.tensor(BASE_MEAN, device=device)
    noise = torch.randn(n, 2, device=device, generator=generator)
    return mean.unsqueeze(0) + BASE_STD * noise
