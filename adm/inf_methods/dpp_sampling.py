"""DPP-based diverse subset selection for data curation.

Sample a large candidate pool, then greedily select a diverse subset by
maximizing the determinant of a kernel matrix (greedy DPP approximation).

Supports linear and RBF kernels.  The feature space used here is either
the FlowFeatureExtractor output or raw data coordinates.  Selection is
purely diversity-driven (no quality weighting); validity is checked after
selection, identically to how other sampling methods work.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def dpp_select(
    embeddings: torch.Tensor,
    k: int,
    kernel: str = "linear",
    rbf_lengthscale: float = 1.0,
) -> torch.Tensor:
    """Greedy log-det maximization for diverse subset selection.

    Parameters
    ----------
    embeddings : (N, D) tensor
        Feature vectors for each candidate.
    k : int
        Number of items to select (k <= N).
    kernel : str
        ``"linear"`` for cosine-similarity kernel or ``"rbf"`` for
        Gaussian RBF kernel.
    rbf_lengthscale : float
        Lengthscale for the RBF kernel (ignored when kernel="linear").

    Returns
    -------
    (k,) long tensor of selected indices into the original pool.
    """
    N, D = embeddings.shape
    k = min(k, N)

    if kernel == "linear":
        return _select_linear(embeddings, k)
    elif kernel == "rbf":
        return _select_kernel_matrix(
            _rbf_kernel(embeddings, rbf_lengthscale), k
        )
    else:
        raise ValueError(f"Unknown DPP kernel: {kernel!r}")


def _rbf_kernel(X: torch.Tensor, lengthscale: float) -> torch.Tensor:
    """Build (N, N) RBF kernel matrix."""
    sq_dists = torch.cdist(X, X).square()
    return torch.exp(-sq_dists / (2.0 * lengthscale ** 2))


def _select_kernel_matrix(K: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy log-det maximization given a full kernel matrix.

    Uses incremental Cholesky updates: at each step picks the candidate
    that maximises the conditional variance (Schur complement diagonal),
    then updates the running Cholesky factor.

    Complexity: O(k^2 * N) time, O(k * N) memory.
    """
    N = K.shape[0]
    k = min(k, N)

    selected: list[int] = []
    cond_var = K.diagonal().clone()
    L = torch.zeros(k, N, device=K.device, dtype=K.dtype)

    for j in range(k):
        cond_var[selected] = -1.0
        winner = int(cond_var.argmax().item())
        selected.append(winner)

        if cond_var[winner] < 1e-12:
            break

        if j == 0:
            L[0] = K[winner]
        else:
            L[j] = K[winner]
            for m in range(j):
                L[j] -= L[m, winner] * L[m]
            L[j] /= cond_var[winner].sqrt()

        cond_var -= (L[j] ** 2)
        cond_var.clamp_(min=0.0)

    return torch.tensor(selected, dtype=torch.long, device=K.device)


def _select_linear(embeddings: torch.Tensor, k: int) -> torch.Tensor:
    """Greedy volume maximization with linear (cosine) kernel via Gram-Schmidt."""
    norms = embeddings.norm(dim=1, keepdim=True).clamp(min=1e-12)
    vecs = embeddings / norms
    dists_sq = (vecs * vecs).sum(dim=1)

    selected: list[int] = []
    for _ in range(k):
        if selected:
            dists_sq[selected] = -1.0

        winner = int(dists_sq.argmax().item())
        selected.append(winner)

        winner_vec = vecs[winner]
        winner_norm_sq = dists_sq[winner]
        if winner_norm_sq < 1e-12:
            break

        proj_coeffs = torch.mv(vecs, winner_vec) / winner_norm_sq
        dists_sq = dists_sq - proj_coeffs.square() * winner_norm_sq
        dists_sq.clamp_(min=0.0)

        vecs = vecs - torch.outer(proj_coeffs, winner_vec)

    return torch.tensor(selected, dtype=torch.long, device=embeddings.device)
