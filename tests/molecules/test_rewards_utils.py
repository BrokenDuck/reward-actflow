"""Tests for graph_to_mols in rewards/utils.py."""

import pytest
import torch
import torch.nn.functional as F
from rdkit import Chem
from torch_geometric.data import Batch, Data

from reward_actflow.molecules.rewards.utils import ATOM_TYPE_MAP, graph_to_mols
from reward_actflow.molecules.types import DDGraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_postprocessed_ddgraph(n: int = 3) -> DDGraph:
    """Build a DDGraph as it would appear after postprocess().

    Attributes mirror what SampledMolecule expects: x_1, a_1, c_1, e_1, upper_edge_mask.
    """
    from flowmol.data_processing.utils import build_edge_idxs

    edge_index = build_edge_idxs(n)
    num_edges = edge_index.size(1)
    n_upper = n * (n - 1) // 2

    # All carbons, neutral charge, no bonds (safe defaults for a test)
    x_1 = torch.zeros(n, 3)
    a_1 = F.one_hot(torch.zeros(n, dtype=torch.long), num_classes=len(ATOM_TYPE_MAP)).float()
    c_1 = F.one_hot(torch.full((n,), 2, dtype=torch.long), num_classes=6).float()
    e_1 = F.one_hot(torch.zeros(num_edges, dtype=torch.long), num_classes=5).float()

    # Upper-edge mask: first n_upper edges are upper triangle
    upper_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
    upper_edge_mask[:n_upper] = True

    data = Data(
        edge_index=edge_index,
        num_nodes=n,
        x_1=x_1,
        a_1=a_1,
        c_1=c_1,
        e_1=e_1,
        upper_edge_mask=upper_edge_mask,
    )
    g = Batch.from_data_list([data])
    return DDGraph(g)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGraphToMols:
    def test_returns_list_of_correct_length(self) -> None:
        ddg = _make_postprocessed_ddgraph()
        mols = graph_to_mols(ddg)
        assert len(mols) == 1

    def test_returns_rdkit_mol_or_none(self) -> None:
        ddg = _make_postprocessed_ddgraph()
        mols = graph_to_mols(ddg)
        assert mols[0] is None or isinstance(mols[0], Chem.Mol)

    def test_batched_two_molecules(self) -> None:
        ddg1 = _make_postprocessed_ddgraph(n=3)
        ddg2 = _make_postprocessed_ddgraph(n=4)
        combined = DDGraph.collate([ddg1, ddg2])
        mols = graph_to_mols(combined)
        assert len(mols) == 2

    def test_single_atom_molecule(self) -> None:
        """Single atom — no edges, valid carbon atom."""
        data = Data(
            edge_index=torch.zeros(2, 0, dtype=torch.long),
            num_nodes=1,
            x_1=torch.zeros(1, 3),
            a_1=F.one_hot(torch.zeros(1, dtype=torch.long), num_classes=len(ATOM_TYPE_MAP)).float(),
            c_1=F.one_hot(torch.full((1,), 2, dtype=torch.long), num_classes=6).float(),
            e_1=torch.zeros(0, 5),
            upper_edge_mask=torch.zeros(0, dtype=torch.bool),
        )
        g = Batch.from_data_list([data])
        ddg = DDGraph(g)
        mols = graph_to_mols(ddg)
        assert len(mols) == 1

    @pytest.mark.parametrize("n_atoms", [2, 3, 4])
    def test_various_molecule_sizes(self, n_atoms: int) -> None:
        ddg = _make_postprocessed_ddgraph(n=n_atoms)
        mols = graph_to_mols(ddg)
        assert len(mols) == 1
