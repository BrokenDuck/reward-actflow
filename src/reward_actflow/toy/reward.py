"""Black-box task rewards `r~(x)` for the 2-D staircase toy problem.

Both rewards are chosen so that maximising them *within the valid set* forces
the model through the one-unit corridor and onto the `bottom` slab — the
behaviour ActFlow-R is meant to produce, not just base ActFlow's coverage.

The geometry (`staircase_validity`, after its `+0.5` offset) is what makes this
work: `top` spans raw x in `[-2.5, 0.5]`, `bottom` spans raw x in `[-2.5, 2.5]`.
`top` tops out at x=0.5; `bottom` reaches x=2.5. So any reward that is strictly
increasing in x is maximised, over the whole valid set, only on `bottom`'s far
right edge — and the pretraining blob sits inside `top` (`BASE_MEAN=(-1.0,
1.5)`), so reaching it requires threading the corridor.
"""

from collections.abc import Callable

import torch
from torch import Tensor

from reward_actflow.toy.validity import PLOT_LIMITS

#: Default centre for the bump reward: well inside `bottom`, far right.
BUMP_CENTER = (2.0, -1.5)

#: Default spread for the bump reward.
BUMP_SCALE = 0.5


def linear_gradient_reward(
    x: Tensor,
    *,
    x_min: float = PLOT_LIMITS[0],
    x_max: float = PLOT_LIMITS[1],
) -> Tensor:
    """Reward increasing in x, normalised to `[0, 1]` over `PLOT_LIMITS`.

    Its unconstrained maximum (x = `PLOT_LIMITS[1]`) is unreachable inside the
    valid set — the constrained one, x = 2.5 on `bottom`'s right edge, is what
    a method that actually reaches the far side should find.
    """
    return ((x[:, 0] - x_min) / (x_max - x_min)).clamp(0.0, 1.0)


def gaussian_bump_reward(
    x: Tensor,
    *,
    center: tuple[float, float] = BUMP_CENTER,
    scale: float = BUMP_SCALE,
) -> Tensor:
    """Gaussian bump centred inside `bottom`, far from the pretraining blob.

    Unlike the linear gradient, this reward is zero almost everywhere — it
    gives the acquisition nothing to follow until the surrogate has already
    found the neighbourhood, which is a harder and more realistic test of
    "does exploration reach reward, not just reward reshape exploration".
    """
    c = torch.tensor(center, device=x.device, dtype=x.dtype)
    dist_sq = ((x - c) ** 2).sum(dim=-1)
    return torch.exp(-0.5 * dist_sq / scale**2)


#: Selected by `--toy_reward`. `"linear"` is the default (see module docstring
#: for why it is the primary test of ActFlow-R's claim).
TOY_REWARDS: dict[str, Callable[[Tensor], Tensor]] = {
    "linear": linear_gradient_reward,
    "bump": gaussian_bump_reward,
}
