from typing import Generic, Any

import torch
import torch.nn as nn
from diffusiongym import Environment, Reward, D


class RewardGradient(nn.Module, Generic[D]):
    """DPS (https://arxiv.org/pdf/2209.14687) with a general reward function."""

    def __init__(self, env: Environment[D], reward: Reward[D] | None = None):
        super().__init__()
        if reward is None:
            reward = env.reward

        self.env = env
        self.reward = reward

    @torch.enable_grad()
    def forward(self, xt: D, t: torch.Tensor, **kwargs: Any) -> D:
        xt = xt.requires_grad()
        x1 = self.env.pred_final(xt, t, **kwargs)
        # For our purposes, we do not need a postprocessed sample
        r, _ = self.reward(None, x1, **kwargs)
        sigma = self.env.memoryless_schedule(xt, t)
        return sigma * self.env.reward_scale * xt.gradient(r)
