from typing import Any, Optional

import torch
import torch.nn.functional as F
import dgl
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem
from flowmol.analysis.molecule_builder import SampledMolecule
from flowgym import  BaseModel, Environment
from flowgym.molecules import FlowGraph, QM9BaseModel
from vendi_score import vendi
from vendi_score import molecule_utils
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import os

from active_pretraining.problem_setup import ProblemSetup


class QM9ProblemSetup(ProblemSetup[FlowGraph]):
    def __init__(self, args: dict[str, Any], device: Optional[torch.device]=None):
        RDLogger.DisableLog("rdApp.*")  # type: ignore

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._base_model = QM9BaseModel(device=device)

    @property
    def base_model(self) -> BaseModel[FlowGraph]:
        return self._base_model

    def validity(self, x: FlowGraph) -> torch.Tensor:
        valids = torch.ones(len(x), dtype=torch.bool)

        x.graph.edata["ue_mask"] = x.ue_mask
        batched_graph = x.graph.cpu()

        for i, g in enumerate(dgl.unbatch(batched_graph)):
            # Check if it passes rdkit validity rules
            mol = SampledMolecule(g, self.base_model.model.atom_type_map).rdkit_mol
            mol = validate_mol(mol)

            if mol is None:
                valids[i] = False
                continue
            
            # Check if it has only a single connected component
            if len(Chem.GetMolFrags(mol)) != 1:
                valids[i] = False

        return valids

    @property
    def feature_layer(self) -> str:
        return "model.vector_field.node_output_head.0"

    def feature_postprocess(self, x: FlowGraph, feats: torch.Tensor) -> torch.Tensor:
        # `feats` is a tensor of shape (num_nodes, feature_dim), which we want to take the mean over
        # for each graph in the batch. We have the indices that each node belongs to which graph in x.n_idx.
        device = x.device
        num_nodes, feature_dim = feats.shape
        num_graphs = int(x.n_idx.max().item()) + 1

        # Sum features per graph
        graph_sums = torch.zeros(num_graphs, feature_dim, device=device)
        graph_sums.index_add_(0, x.n_idx, feats)

        # Count nodes per graph
        graph_counts = torch.zeros(num_graphs, device=device)
        graph_counts.index_add_(0, x.n_idx, torch.ones(num_nodes, device=device))

        # Mean pooling
        return graph_sums / graph_counts.unsqueeze(1)

    def latent_postprocess(self, samples: FlowGraph) -> FlowGraph:
        graph = samples.graph.clone()

        def _discretize(x):
            # argmax finds the class index, one_hot creates the vector
            # type_as ensures we match the original float precision and device
            return F.one_hot(x.argmax(dim=-1), num_classes=x.shape[-1]).type_as(x)

        # Set categorical features to be one-hot vectors over dim -1
        graph.ndata["a_t"] = _discretize(graph.ndata["a_t"])
        graph.ndata["c_t"] = _discretize(graph.ndata["c_t"])
        graph.edata["e_t"] = _discretize(graph.edata["e_t"])

        # Relax the geometry, otherwise the structures will become increasingly distorted
        graph.edata["ue_mask"] = samples.ue_mask
        graphs = []
        for g in dgl.unbatch(graph):
            g = relax_positions(g, self.base_model.model.atom_type_map)
            graphs.append(g)

        graph = dgl.batch(graphs)

        # Remove center of mass
        init_coms = dgl.readout_nodes(graph, feat="x_t", op="mean")
        graph.ndata["x_t"] = graph.ndata["x_t"] - init_coms[samples.n_idx]

        return FlowGraph(graph, samples.ue_mask, samples.n_idx, samples.e_idx)

    def _to_mols(self, samples: FlowGraph) -> list[Chem.Mol]:
        mols = []
        for i, sample in enumerate(dgl.unbatch(samples.graph)):
            mol = SampledMolecule(sample.cpu(), self.base_model.model.atom_type_map).rdkit_mol
            mols.append(mol)
        return mols

    @torch.no_grad()
    def visualize_sample(
        self,
        env: Environment[FlowGraph],
        samples: list[FlowGraph],
        valids: list[torch.Tensor],
    ) -> Figure:
        mols = self._to_mols(samples[-1])

        fig = plt.figure(figsize=(12, 8))
        cols = 8
        rows = (len(mols) + cols - 1) // cols
        for i, mol in enumerate(mols):
            ax = fig.add_subplot(rows, cols, i + 1)
            if valids[-1][i]:
                ax.set_title("Valid", color="green", fontsize=8)
            else:
                ax.set_title("Invalid", color="red", fontsize=8)

            ax.axis("off")
            img = Draw.MolToImage(mol, size=(150, 150))
            ax.imshow(img)

        return fig

    def save_sample(self, sample: FlowGraph, filename: os.PathLike | str):
        if len(sample) != 1:
            raise ValueError("Can only save a single sample at a time.")

        mol = SampledMolecule(sample.graph.cpu(), self.base_model.model.atom_type_map).rdkit_mol
        mol = validate_mol(mol)

        if mol is None:
            return

        Chem.MolToMolFile(mol, f"{filename}.mol")

    def compute_metrics(self, samples: list[FlowGraph], valids: list[torch.Tensor]) -> dict[str, float]:
        n_samples = 0
        mols = []
        for d, v in zip(samples, valids):
            for j in range(len(d)):
                n_samples += 1

                if not v[j]:
                    continue

                mol = SampledMolecule(d[j].graph.cpu(), self.base_model.model.atom_type_map).rdkit_mol
                mol = validate_mol(mol)

                if mol is not None:
                    mols.append(mol)

        # Limit to only half the generated molecules, because there are some invalid ones, and we
        # need to keep the number of samples for computing diversity constant
        # Generally it does not go below 50% validity anyway
        mols = mols[:n_samples // 2]
        K = molecule_utils.get_tanimoto_K(mols)
        vendi_score = vendi.score_K(K)

        return { "vendi": float(vendi_score) }


def relax_positions(g: dgl.DGLGraph, atom_type_map: list[str]) -> dgl.DGLGraph:
    g_relaxed = g.clone()

    g_relaxed.ndata["x_1"] = g_relaxed.ndata["x_t"]
    g_relaxed.ndata["a_1"] = g_relaxed.ndata["a_t"]
    g_relaxed.ndata["c_1"] = g_relaxed.ndata["c_t"]
    g_relaxed.edata["e_1"] = g_relaxed.edata["e_t"]

    mol = SampledMolecule(g_relaxed, atom_type_map).rdkit_mol
    if mol is None:
        return g

    mol = validate_mol(mol)
    if mol is None:
        return g

    AllChem.MMFFOptimizeMolecule(mol)  # type: ignore
    positions = torch.from_numpy(mol.GetConformer().GetPositions())
    g.ndata["x_t"] = positions.to(g.device).type_as(g.ndata["x_t"])  # type: ignore

    return g


def validate_mol(mol: Chem.Mol) -> Optional[Chem.Mol]:
    """Validate molecules according to chemical validity.

    Source: https://arxiv.org/abs/2505.00518

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to validate.

    Returns
    -------
    validated_mol : Chem.Mol | None
        The sanitized molecule if valid, else None.
    """
    # sometimes it crashes randomly in C++, so guard it defensively just in case
    try:
        Chem.RemoveStereochemistry(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        Chem.AssignStereochemistryFrom3D(mol)

        for a in mol.GetAtoms():
            a.SetNoImplicit(True)  # type: ignore
            if a.HasProp("_MolFileHCount"):
                a.ClearProp("_MolFileHCount")

        flags = Chem.SanitizeFlags.SANITIZE_ALL & ~Chem.SanitizeFlags.SANITIZE_ADJUSTHS  # type: ignore

        # Full sanitization, minus ADJUSTHS
        err = Chem.SanitizeMol(
            mol,
            sanitizeOps=flags,
            catchErrors=True,
        )

        # Non-zero bitmask means some step failed
        if err:
            return None

        mol.UpdatePropertyCache(strict=True)
        return mol
    except:
        return None
