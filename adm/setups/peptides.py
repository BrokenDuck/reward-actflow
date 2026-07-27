"""Placeholder for therapeutic peptide design via discrete (CTMC) flow models.

The peptide experiments currently live on origin/peptides as a standalone
implementation using absorbing-state discrete diffusion (MDM) with a
RoFormer backbone and SMILES tokenization. This stub defines the
ProblemSetup interface to be implemented when porting that code to the
shared ActFlow framework.

See origin/peptides branch for the reference implementation.
"""
from typing import Any, Optional
import torch
from .problem_setup import ProblemSetup


class PeptideProblemSetup(ProblemSetup):
    """Therapeutic peptide design problem setup (not yet implemented).

    Will wrap a discrete CTMC flow model (e.g., PepTune) with:
    - SMILES2PEPTIDE verifier for validity
    - PeptideCLM embeddings for diversity/coverage metrics
    - Morgan fingerprint-based novelty and clustering
    """

    def __init__(self, args: dict[str, Any], device: Optional[torch.device] = None):
        raise NotImplementedError(
            "PeptideProblemSetup is a placeholder for future integration. "
            "See the origin/peptides branch for the standalone implementation."
        )
