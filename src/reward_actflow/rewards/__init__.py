"""Acquisition rewards for ActFlow. Importing this registers them."""

from reward_actflow.rewards.acquisition import (
    AcquisitionRewardProvider,
    ActFlowRAcquisitionReward,
)
from reward_actflow.rewards.base import SoftGateTerminalCost, SurrogateReward
from reward_actflow.rewards.uncertainty import (
    DIFFERENTIABLE_GATES,
    GATES,
    ActFlowUncertaintyReward,
    UncertaintyRewardProvider,
)

__all__ = [
    "DIFFERENTIABLE_GATES",
    "GATES",
    "AcquisitionRewardProvider",
    "ActFlowRAcquisitionReward",
    "ActFlowUncertaintyReward",
    "SoftGateTerminalCost",
    "SurrogateReward",
    "UncertaintyRewardProvider",
]
