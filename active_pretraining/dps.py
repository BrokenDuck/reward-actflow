from typing import Generic, Any

import torch
import torch.nn as nn
from flowgym import Environment, D


class RewardGradient(nn.Module, Generic[D]):
    """DPS (https://arxiv.org/pdf/2209.14687) with a general reward function."""

    def __init__(self, env: Environment[D]):
        super().__init__()
        self.env = env

    @torch.enable_grad()
    def forward(self, xt: D, t: torch.Tensor, **kwargs: Any) -> D:
        xt = xt.requires_grad()
        x1 = self.env.pred_final(xt, t, **kwargs)
        # For our purposes, we do not need a postprocessed sample
        r, _ = self.env.reward(None, x1, **kwargs)  # type: ignore
        sigma = self.env.memoryless_schedule(xt, t)
        return sigma * self.env.reward_scale * xt.gradient(r)
