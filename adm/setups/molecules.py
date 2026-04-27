from typing import Optional, Any

import torch
import torch.nn.functional as F
import dgl
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem, Crippen, QED
from flowmol.data_processing.utils import build_edge_idxs
from flowmol.analysis.molecule_builder import SampledMolecule, bond_type_to_idx
from diffusiongym import  BaseModel, Environment
from diffusiongym.molecules import DDGraph, QM9BaseModel, GEOMBaseModel
from diffusiongym.molecules.rewards.xtb import parallel_xtb
from diffusiongym.molecules.rewards.utils import is_valid, is_not_fragmented
from diffusiongym.utils import temporary_workdir
from vendi_score import vendi, molecule_utils
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from argparse import ArgumentParser
import gzip
import os

import adm.sa_scorer.sascorer as sascorer
from adm.setups.problem_setup import ProblemSetup
from adm.uncertainty import UncertaintyEstimator
from adm.utils import Batch


class MoleculeProblemSetup(ProblemSetup[DDGraph]):
    def __init__(self, dataset: str, args: dict, device: Optional[torch.device] = None):
        RDLogger.DisableLog("rdApp.*")  # type: ignore

        self.geometry_opt: str = args["mol_geometry_opt"]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dataset == "qm9":
            self._base_model = QM9BaseModel(device=device)
        elif dataset == "geom_drugs":
            self._base_model = GEOMBaseModel(device=device)
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.atom_type_map = self.base_model.model.atom_type_map
        self.atom_type_to_idx = { atom_type: i for i, atom_type in enumerate(self.atom_type_map) }

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument("--mol_geometry_opt", type=str, choices=["none", "mmff", "uff", "gfn2"], default="mmff")

    @property
    def base_model(self) -> BaseModel[DDGraph]:
        return self._base_model

    def _is_mol_valid(self, mol: Chem.Mol) -> bool:
        return is_valid(mol) and is_not_fragmented(mol)

    def validity(self, samples: DDGraph, kwargs: dict) -> torch.Tensor:
        valids = torch.ones(len(samples), dtype=torch.bool)

        samples.graph.edata["ue_mask"] = samples.ue_mask
        batched_graph = samples.graph.cpu()

        for i, g in enumerate(dgl.unbatch(batched_graph)):
            # Check if it passes rdkit validity rules
            mol = SampledMolecule(g, self.base_model.model.atom_type_map).rdkit_mol
            valids[i] = self._is_mol_valid(mol)

        return valids

    @property
    def feature_layer(self) -> str:
        return "model.vector_field.node_output_head.0"

    def _postprocess_graph(self, samples: DDGraph) -> DDGraph:
        graph = samples.graph.clone()

        def _discretize(x):
            # argmax finds the class index, one_hot creates the vector
            # type_as ensures we match the original float precision and device
            return F.one_hot(x.argmax(dim=-1), num_classes=x.shape[-1]).type_as(x)

        x_key = "x_1" if "x_1" in graph.ndata.keys() else "x_t"
        a_key = "a_1" if "a_1" in graph.ndata.keys() else "a_t"
        c_key = "c_1" if "c_1" in graph.ndata.keys() else "c_t"
        e_key = "e_1" if "e_1" in graph.edata.keys() else "e_t"

        # Set categorical features to be one-hot encoded
        graph.ndata[a_key] = _discretize(graph.ndata[a_key])
        graph.ndata[c_key] = _discretize(graph.ndata[c_key])
        graph.edata[e_key] = _discretize(graph.edata[e_key])

        # Relax the geometry, otherwise the structures will become increasingly distorted
        graph.edata["ue_mask"] = samples.ue_mask
        if self.geometry_opt != "none":
            graphs = []
            for g in dgl.unbatch(graph):
                g = relax_positions(g, self.base_model.model.atom_type_map, self.geometry_opt)
                graphs.append(g)

            graph = dgl.batch(graphs)

        # Remove center of mass
        init_coms = dgl.readout_nodes(graph, feat=x_key, op="mean")
        graph.ndata[x_key] = graph.ndata[x_key] - init_coms[samples.n_idx]

        return DDGraph(graph, samples.ue_mask, samples.n_idx, samples.e_idx)

    def postprocess_latents(self, batch: Batch[DDGraph]) -> DDGraph:
        return self._postprocess_graph(batch.latents)

    def postprocess_features(self, latents: DDGraph, feats: torch.Tensor) -> torch.Tensor:
        # `feats` is a tensor of shape (num_nodes, feature_dim), which we want to take the mean over
        # for each graph in the batch. We have the indices that each node belongs to which graph in x.n_idx.
        device = latents.device
        num_nodes, feature_dim = feats.shape
        num_graphs = int(latents.n_idx.max().item()) + 1

        # Sum features per graph
        graph_sums = torch.zeros(num_graphs, feature_dim, device=device)
        graph_sums.index_add_(0, latents.n_idx, feats)
        # Count nodes per graph
        graph_counts = torch.zeros(num_graphs, device=device)
        graph_counts.index_add_(0, latents.n_idx, torch.ones(num_nodes, device=device))

        # Mean pooling
        return graph_sums / graph_counts.unsqueeze(1)

    def _graph_to_mols(self, samples: DDGraph) -> list[Chem.Mol | None]:
        samples.graph.edata["ue_mask"] = samples.ue_mask

        mols = []
        for g in dgl.unbatch(samples.graph):
            mol = SampledMolecule(g.cpu(), self.atom_type_map).rdkit_mol
            mols.append(mol)

        return mols

    def _mols_to_graph(self, mols: list[Chem.Mol]) -> dgl.DGLGraph:
        graphs = []
        for mol in mols:
            n = mol.GetNumAtoms()

            atom_positions = torch.from_numpy(mol.GetConformer().GetPositions())
            atom_types_idx = torch.zeros(n, dtype=torch.int64)
            atom_charges = torch.zeros(n, dtype=torch.int64)

            for i, atom in enumerate(mol.GetAtoms()):
                atom_types_idx[i] = self.atom_type_to_idx[atom.GetSymbol()]
                atom_charges[i] = atom.GetFormalCharge()

            src, dst = build_edge_idxs(n)
            g = dgl.graph((src, dst), num_nodes=n)

            e_idx = torch.empty(g.num_edges(), dtype=torch.int64)
            for k in range(g.num_edges()):
                a = int(src[k])
                b = int(dst[k])
                bond = mol.GetBondBetweenAtoms(a, b)
                bond_type = bond.GetBondType() if bond is not None else None
                e_idx[k] = bond_type_to_idx[bond_type]

            g.ndata["x_1"] = atom_positions
            g.ndata["a_1"] = F.one_hot(atom_types_idx, num_classes=len(self.atom_type_map)).float()
            g.ndata["c_1"] = F.one_hot(atom_charges + 2, num_classes=6).float()
            g.edata["e_1"] = F.one_hot(e_idx, num_classes=5).float()

            graphs.append(g)

        return dgl.batch(graphs)

    @torch.no_grad()
    def visualize_sample(
        self,
        env: Environment[DDGraph],
        uncertainty: UncertaintyEstimator[DDGraph],
        batch: Batch[DDGraph],
    ) -> Figure:
        mols = self._graph_to_mols(batch.samples)

        fig = plt.figure(figsize=(12, 8))
        cols = 8
        rows = (len(mols) + cols - 1) // cols
        for i, (mol, is_valid) in enumerate(zip(mols, batch.valids)):
            ax = fig.add_subplot(rows, cols, i + 1)
            if is_valid:
                ax.set_title("Valid", color="green", fontsize=8)
            else:
                ax.set_title("Invalid", color="red", fontsize=8)

            ax.axis("off")
            if mol is not None:
                try:
                    img = Draw.MolToImage(mol, size=(150, 150))
                    ax.imshow(img)
                except Exception:
                    pass

        return fig

    def eval_sampling_kwargs(self, n: int) -> dict:
        # Fix the atom sizes seen during evaluation
        probs = self.base_model.model.n_atoms_dist.probs

        expected = n * probs
        counts = expected.round().long()

        # Correct for rounding errors
        remainder = n - counts.sum()
        if remainder > 0:
            fractional = expected - counts
            extra = torch.topk(fractional, remainder).indices
            counts[extra] += 1

        indices = torch.arange(len(counts), device=counts.device)
        result = torch.repeat_interleave(indices, counts)

        # Randomly order result
        perm = torch.randperm(result.numel(), device=result.device)
        result = result[perm]

        return { "n_atoms": result }

    def save_samples(self, samples: DDGraph, kwargs: dict, dir: Path) -> bool:
        samples = self._postprocess_graph(samples)
        mols = self._graph_to_mols(samples)
        mols = [mol for mol in mols if mol is not None and self._is_mol_valid(mol)]

        sdf_path = dir / "samples.sdf.gz"
        with gzip.open(sdf_path, "wt") as f:
            w = Chem.SDWriter(f)
            for mol in mols:
                try:
                    w.write(mol)
                except Exception as e:
                    print(f"Failed to write molecule: {e}")
                    pass
            w.close()

        return True

    def load_samples(self, dir: Path) -> tuple[DDGraph, dict]:
        sdf_path = dir / "samples.sdf.gz"
        if not sdf_path.exists():
            raise FileNotFoundError(f"No samples found in {dir}")

        mols = []
        with gzip.open(sdf_path, "rb") as f:
            suppl = Chem.ForwardSDMolSupplier(f, sanitize=False, removeHs=False, strictParsing=False)
            for mol in suppl:
                if mol is not None:
                    mols.append(mol)

        graph = self._mols_to_graph(mols)
        return DDGraph(graph), {}

    def compute_metrics(self, samples: DDGraph, kwargs: dict) -> dict[str, float]:
        mols = self._graph_to_mols(samples)
        mols = [mol for mol in mols if mol is not None and self._is_mol_valid(mol)]

        K = molecule_utils.get_tanimoto_K(mols)
        vendi_score = vendi.score_K(K)

        n = len(mols)
        D = 1 - K
        avg_pairwise_dist = D.sum() / (n * (n - 1))

        return { "vendi": float(vendi_score), "avg_pairwise_dist": float(avg_pairwise_dist) }

    def compute_sample_metrics(self, samples: DDGraph, kwargs: dict) -> list[dict[str, Any]]:
        # In case these were not loaded from file, we need to relax geometry
        samples = self._postprocess_graph(samples)
        mols = self._graph_to_mols(samples)

        valid_indices = [i for i, mol in enumerate(mols) if mol is not None and self._is_mol_valid(mol)]
        valid_mols = [mols[i] for i in valid_indices]
        res = parallel_xtb(valid_mols)

        metrics: list[dict[str, Any]] = [{ "is_valid": True } for _ in mols]
        metric_names = ["energy", "homo", "lumo", "homo_lumo_gap", "dipole_moment", "polarizability", "heat_capacity"]
        for mol, idx, xtb_res in zip(valid_mols, valid_indices, res):
            if xtb_res is None:
                continue

            for name in metric_names:
                metrics[idx][name] = getattr(xtb_res, name)
            metrics[idx]["qed"] = QED.qed(mol)
            metrics[idx]["logp"] = Crippen.MolLogP(mol)  # type: ignore
            metrics[idx]["sa_score"] = sascorer.calculateScore(mol)

        return metrics


class QM9ProblemSetup(MoleculeProblemSetup):
    def __init__(self, args: dict, device: Optional[torch.device] = None):
        super().__init__(dataset="qm9", args=args, device=device)


class GEOMDrugsProblemSetup(MoleculeProblemSetup):
    def __init__(self, args: dict, device: Optional[torch.device] = None):
        super().__init__(dataset="geom_drugs", args=args, device=device)


def relax_positions(g: dgl.DGLGraph, atom_type_map: list[str], alg: str = "mmff") -> dgl.DGLGraph:
    g_relaxed = g.clone()

    if "x_1" not in g_relaxed.ndata.keys():
        g_relaxed.ndata["x_1"] = g_relaxed.ndata["x_t"]

    if "a_1" not in g_relaxed.ndata.keys():
        g_relaxed.ndata["a_1"] = g_relaxed.ndata["a_t"]

    if "c_1" not in g_relaxed.ndata.keys():
        g_relaxed.ndata["c_1"] = g_relaxed.ndata["c_t"]

    if "e_1" not in g_relaxed.edata.keys():
        g_relaxed.edata["e_1"] = g_relaxed.edata["e_t"]

    mol = SampledMolecule(g_relaxed, atom_type_map).rdkit_mol
    if mol is None or not is_valid(mol):
        return g

    if alg == "mmff":
        # Sometimes it crashes in the middle of a run, so we guard with try-except
        try:
            AllChem.MMFFOptimizeMolecule(mol)  # type: ignore
        except:
            pass
    elif alg == "uff":
        try:
            AllChem.UFFOptimizeMolecule(mol)  # type: ignore
        except:
            pass
    elif alg == "gfn2":
        mol = xtb_relax_geometry(mol)
    else:
        raise ValueError(f"Unknown geometry optimization algorithm: {alg}")

    if mol is None:
        return g

    positions = torch.from_numpy(mol.GetConformer().GetPositions())

    x_key = "x_1" if "x_1" in g.ndata.keys() else "x_t"
    g.ndata[x_key] = positions.to(g.device).type_as(g.ndata[x_key])  # type: ignore

    return g


def xtb_relax_geometry(mol: Chem.Mol) -> Chem.Mol | None:
    """Relax the geometry of a molecule using GFN2-xTB optimization.

    Parameters
    ----------
    mol : Chem.Mol
        The molecule to relax.

    Returns
    -------
    relaxed_mol : Chem.Mol | None
        The molecule with relaxed geometry. If any runtime errors, returns None.
    """
    with temporary_workdir():
        # Write molecule to XYZ file
        xyz_file = Path("input.xyz")
        output_file = Path("xtbopt.xyz")

        # Convert RDKit mol to XYZ format
        Chem.MolToXYZFile(mol, str(xyz_file))

        # Optimize geometry
        os.system(f"xtb {xyz_file} --opt --gfn 2 > /dev/null 2>&1")

        if not output_file.exists():
            return None

        # Load optimized structure back into RDKit
        opt_mol = Chem.MolFromXYZFile(str(output_file))  # type: ignore
        if opt_mol is None:
            return None

        # Copy the optimized coordinates to the original molecule
        opt_conf = opt_mol.GetConformer()
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, opt_conf.GetAtomPosition(i))  # type: ignore

    return mol
