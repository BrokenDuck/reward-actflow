"""QED reward for molecules."""

import torch
from rdkit.Chem import QED

from diffusiongym.registry import reward_registry
from diffusiongym.rewards import Reward
from reward_actflow.molecules.rewards.utils import graph_to_mols, is_not_fragmented, is_valid
from reward_actflow.molecules.types import DDGraph


@reward_registry.register("molecules/qed")
class QEDReward(Reward[DDGraph]):
    """QED reward for molecules."""

    def __call__(self, sample: DDGraph, latent: DDGraph, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        mols = graph_to_mols(sample)

        rewards = torch.zeros(len(sample), device=sample.device)
        valids = torch.zeros(len(sample), device=sample.device, dtype=torch.bool)
        for i, mol in enumerate(mols):
            if is_valid(mol) and is_not_fragmented(mol):
                rewards[i] = QED.qed(mol)
                valids[i] = True

        return rewards, valids
