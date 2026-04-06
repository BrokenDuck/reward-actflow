"""Particle guidance for joint diverse sampling (Corso et al., arXiv:2310.13102).

Adapted from the Stable Diffusion pipeline in
https://github.com/gcorso/particle-guidance — repulsive pairwise interaction
with an RBF-style kernel.  Here the interaction is defined on the **same
feature map** as the GP uncertainty estimator (``FlowFeatureExtractor`` output),
and when the GP uses an RBF kernel we set the bandwidth to match
``2 * gp_lengthscale**2`` (same exponent as ``gpytorch.kernels.RBFKernel`` with
that lengthscale).  For a linear GP kernel we fall back to the median-distance
bandwidth heuristic from the reference implementation.

At each denoising step, the correction is propagated to ``x_t`` by differentiating
``sum_i <f_i, \\psi_i>`` w.r.t. ``x_t``, where ``f_i`` are features of the
predicted ``x_1`` and ``\\psi_i`` is the repulsion direction in feature space
(detached), analogously to the DINO feature-space branch in the reference code.
"""

from __future__ import annotations

import math
from typing import Any, Generic

import torch
import torch.nn as nn
from diffusiongym import Environment, D

from adm.uncertainty import UncertaintyEstimator


def _unwrap_sigma(sigma) -> torch.Tensor:
    """Extract a plain Tensor from whatever memoryless_schedule returns (DDTensor, Tensor, or float)."""
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
            # Median heuristic (reference repo) in feature space
            med = distance.median(dim=1, keepdim=True)[0].clamp(min=1e-8)
            h_t = med**2 / math.log(max(n - 1, 2))
            weights = torch.exp(-(distance**self.power / h_t))

        grad_phi_feat = 2.0 * weights * diff / h_t * sigma_val * self.coeff
        grad_phi_feat = grad_phi_feat.sum(dim=1)

        # Same construction as DINO branch: backprop dot(f_i, psi_i) -> x_t
        loss = (feats * grad_phi_feat.detach()).sum()
        # Reference pipeline *subtracts* this from the score direction; control is *added* in diffusiongym.
        return -xt.gradient(loss)
