"""The 2-D staircase toy problem. Importing this registers its providers."""

import reward_actflow.rewards  # noqa: F401  (registers actflow/uncertainty)
from reward_actflow.toy.model import (
    StandardNormalSampler,
    ToyFlowModel,
    VelocityMLP,
    pretrain_on_data,
)
from reward_actflow.toy.providers import ToyModality
from reward_actflow.toy.reward import (
    TOY_REWARDS,
    gaussian_bump_reward,
    linear_gradient_reward,
)
from reward_actflow.toy.validity import (
    BASE_MEAN,
    BASE_STD,
    PLOT_LIMITS,
    base_training_data,
    staircase_validity,
    valid_area,
)
from reward_actflow.toy.visualize import (
    coverage_metrics,
    plot_iteration,
    sample_model,
    support_mask,
)

__all__ = [
    "BASE_MEAN",
    "BASE_STD",
    "PLOT_LIMITS",
    "TOY_REWARDS",
    "StandardNormalSampler",
    "ToyFlowModel",
    "ToyModality",
    "VelocityMLP",
    "base_training_data",
    "coverage_metrics",
    "gaussian_bump_reward",
    "linear_gradient_reward",
    "plot_iteration",
    "pretrain_on_data",
    "sample_model",
    "staircase_validity",
    "support_mask",
    "valid_area",
]
