"""Particle guidance for joint diverse sampling (Corso et al., arXiv:2310.13102).

Repulsive pairwise interaction with an RBF-style kernel in the feature space
of the uncertainty estimator (FlowFeatureExtractor output). When the GP uses
an RBF kernel we set the bandwidth to match 2 * gp_lengthscale**2; for a
linear GP kernel we fall back to the median-distance bandwidth heuristic.
"""

from __future__ import annotations

import math
from typing import Any, Generic

import torch
import torch.nn as nn
from diffusiongym import Environment, D

from adm.uncertainty import UncertaintyEstimator


def _unwrap_sigma(sigma) -> torch.Tensor:
    """Extract a plain Tensor from whatever memoryless_schedule returns."""
    if hasattr(sigma, "data") and not isinstance(sigma, torch.Tensor):
        sigma = sigma.data
    if not isinstance(sigma, torch.Tensor):
        sigma = torch.as_tensor(sigma)
    return sigma


class ParticleGuidance(nn.Module, Generic[D]):
    """Batch-coupled diffusion guidance toward feature-space diversity."""

    def __init__(
        self,
        env: Environment[D],
        uncertainty: UncertaintyEstimator[D],
        coeff: float = 1.0,
        sigma_break: float = 1.0,
        power: int = 2,
    ):
        super().__init__()
        self.env = env
        self.feat_extractor = uncertainty.feat_extractor
        self.uncertainty = uncertainty
        self.coeff = coeff
        self.sigma_break = sigma_break
        self.power = power

    @torch.enable_grad()
    def forward(self, xt: D, t: torch.Tensor, **kwargs: Any) -> D:
        n = len(xt)
        xt = xt.requires_grad()
        x1 = self.env.pred_final(xt, t, **kwargs)
        feats = self.feat_extractor(x1, **kwargs)

        if n < 2:
            z = (feats * 0).sum()
            return xt.gradient(z)

        sigma_t = _unwrap_sigma(self.env.memoryless_schedule(xt, t))
        sigma_val = float(sigma_t.mean().detach().cpu())
        if sigma_val < self.sigma_break or self.coeff == 0.0:
            z = (feats * 0).sum()
            return xt.gradient(z)

        # Pairwise differences f_i - f_j, j != i  -> (N, N-1, d)
        latents_vec = feats
        diff_all = latents_vec.unsqueeze(1) - latents_vec.unsqueeze(0)
        mask = ~torch.eye(n, dtype=torch.bool, device=feats.device)
        diff = diff_all[mask].view(n, n - 1, feats.shape[-1])

        distance = torch.norm(diff, p=2, dim=-1, keepdim=True)

        args = getattr(self.uncertainty, "args", {}) or {}
        kernel = args.get("gp_kernel", "rbf")

        if kernel == "rbf":
            ell = float(args.get("gp_lengthscale", 0.1))
            h_t = torch.as_tensor(2.0 * ell**2, device=feats.device, dtype=feats.dtype)
            h_t = h_t.clamp(min=1e-12)
            weights = torch.exp(-(distance**self.power) / h_t)
        else:
            med = distance.median(dim=1, keepdim=True)[0].clamp(min=1e-8)
            h_t = med**2 / math.log(max(n - 1, 2))
            weights = torch.exp(-(distance**self.power / h_t))

        grad_phi_feat = 2.0 * weights * diff / h_t * sigma_val * self.coeff
        grad_phi_feat = grad_phi_feat.sum(dim=1)

        loss = (feats * grad_phi_feat.detach()).sum()
        return -xt.gradient(loss)
