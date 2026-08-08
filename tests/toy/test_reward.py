"""Tests for the ActFlow-R black-box task rewards on the staircase toy.

`test_linear_reward_is_maximized_on_the_lower_branch` pins the experimental
premise this whole toy problem exists to test: if the staircase geometry ever
changes, this should be the first thing to fail, not a downstream run that
"just doesn't work".
"""

import pytest
import torch
from diffusiongym.types import DDTensor

from reward_actflow.setups.toy import ToyProblemSetup
from reward_actflow.toy.reward import (
    TOY_REWARDS,
    gaussian_bump_reward,
    linear_gradient_reward,
)
from reward_actflow.toy.validity import BASE_MEAN, PLOT_LIMITS, staircase_validity

DEVICE = torch.device("cpu")


def _valid_grid(n: int = 400) -> torch.Tensor:
    lo, hi = PLOT_LIMITS
    axis = torch.linspace(lo, hi, n)
    xx, yy = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    return points[staircase_validity(points)]


def test_linear_reward_is_maximized_on_the_lower_branch():
    """`top` tops out at raw x=0.5 (px<=1 after the +0.5 offset); `bottom`
    reaches x=2.5 (px<=3). So maximising a +x reward *within the valid set*
    is only possible on `bottom`'s far right edge, which requires threading
    the corridor to get there from the pretraining blob inside `top`."""
    valid_points = _valid_grid()
    rewards = linear_gradient_reward(valid_points)
    best = valid_points[rewards.argmax()]

    assert best[0].item() == pytest.approx(2.5, abs=0.05)
    # On `bottom` (py in [-2,-1]), not `top` (py in [1,2]) or `middle`.
    assert -2.05 <= best[1].item() <= -0.95


def test_linear_reward_is_monotone_in_x():
    x = torch.tensor([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    r = linear_gradient_reward(x)
    assert r[0] < r[1] < r[2]


def test_bump_reward_is_unreachable_from_the_base_blob():
    """A geometric tilt of p^{theta_0} has ~no gradient toward a reward that
    is this close to zero at the pretraining blob — the point of including it
    as the harder alternative to the linear gradient."""
    base = torch.tensor([BASE_MEAN])
    assert gaussian_bump_reward(base).item() < 1e-3


def test_bump_reward_peaks_at_its_center():
    from reward_actflow.toy.reward import BUMP_CENTER

    center = torch.tensor([BUMP_CENTER])
    assert gaussian_bump_reward(center).item() == pytest.approx(1.0, abs=1e-6)


def test_toy_rewards_registry_defaults_to_linear():
    assert TOY_REWARDS["linear"] is linear_gradient_reward
    assert TOY_REWARDS["bump"] is gaussian_bump_reward


@pytest.mark.parametrize("choice", ["linear", "bump"])
def test_task_reward_is_wired_to_the_toy_reward_flag(choice):
    """`--toy_reward` actually reaches `ProblemSetup.task_reward`."""
    problem = ToyProblemSetup({"toy_reward": choice}, device=DEVICE)
    samples = DDTensor(torch.tensor([[2.0, -1.5]]))
    reward = problem.task_reward(samples, {})
    assert torch.equal(reward, TOY_REWARDS[choice](samples.data))


def test_task_reward_defaults_to_linear_when_unset():
    problem = ToyProblemSetup({}, device=DEVICE)
    samples = DDTensor(torch.tensor([[1.0, 0.0]]))
    assert torch.equal(
        problem.task_reward(samples, {}), linear_gradient_reward(samples.data)
    )


def test_anchor_latents_matches_the_base_training_distribution():
    """`anchor_latents` is what ActFlow-R freezes as `P`; on the toy it should
    be `base_training_data`'s blob, not something drawn from the policy."""
    problem = ToyProblemSetup({}, device=DEVICE)
    anchors = problem.anchor_latents(2048, DEVICE)
    assert isinstance(anchors, DDTensor)
    assert len(anchors) == 2048
    mean = anchors.data.mean(dim=0)
    assert mean[0].item() == pytest.approx(BASE_MEAN[0], abs=0.05)
    assert mean[1].item() == pytest.approx(BASE_MEAN[1], abs=0.05)
