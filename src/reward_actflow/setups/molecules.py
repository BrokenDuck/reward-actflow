from typing import Optional, Any

import torch
import torch.nn.functional as F
from torch_geometric.data import Batch as TGBatch, Data
from torch_geometric.nn import global_mean_pool
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem, Crippen, QED
from flowmol.data_processing.utils import build_edge_idxs
from flowmol.analysis.molecule_builder import SampledMolecule, bond_type_to_idx
from diffusiongym import BaseModel, Environment
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

import reward_actflow.sa_scorer.sascorer as sascorer
from reward_actflow.setups.problem_setup import ProblemSetup
from reward_actflow.uncertainty import UncertaintyEstimator
from reward_actflow.utils import Batch


class MoleculeProblemSetup(ProblemSetup[DDGraph]):
    def __init__(self, dataset: str, args: dict, device: Optional[torch.device] = None):
        RDLogger.DisableLog("rdApp.*")

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
        self.atom_type_to_idx = {
            atom_type: i for i, atom_type in enumerate(self.atom_type_map)
        }

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument(
            "--mol_geometry_opt",
            type=str,
            choices=["none", "mmff", "uff", "gfn2"],
            default="mmff",
        )

    @property
    def base_model(self) -> BaseModel[DDGraph]:
        return self._base_model

    def _is_mol_valid(self, mol: Chem.Mol) -> bool:
        return is_valid(mol) and is_not_fragmented(mol)

    def validity(self, samples: DDGraph, kwargs: dict) -> torch.Tensor:
        valids = torch.zeros(len(samples), dtype=torch.bool)

        g = samples.graph.clone()
        g.upper_edge_mask = samples.ue_mask

        for i, data in enumerate(g.to_data_list()):
            mol = SampledMolecule(data.cpu(), self.base_model.model.atom_type_map).rdkit_mol
            valids[i] = self._is_mol_valid(mol)

        return valids

    @property
    def feature_layer(self) -> str:
        return "model.vector_field.node_output_head.0"

    def _postprocess_graph(self, samples: DDGraph) -> DDGraph:
        g = samples.graph.clone()

        def _discretize(x):
            return F.one_hot(x.argmax(dim=-1), num_classes=x.shape[-1]).type_as(x)

        x_key = "x_1" if hasattr(g, "x_1") else "x_t"
        a_key = "a_1" if hasattr(g, "a_1") else "a_t"
        c_key = "c_1" if hasattr(g, "c_1") else "c_t"
        e_key = "e_1" if hasattr(g, "e_1") else "e_t"

        setattr(g, a_key, _discretize(getattr(g, a_key)))
        setattr(g, c_key, _discretize(getattr(g, c_key)))
        setattr(g, e_key, _discretize(getattr(g, e_key)))

        g.upper_edge_mask = samples.ue_mask

        if self.geometry_opt != "none":
            data_list = g.to_data_list()
            relaxed = [
                relax_positions(d, self.base_model.model.atom_type_map, self.geometry_opt)
                for d in data_list
            ]
            g = TGBatch.from_data_list(relaxed)

        x = getattr(g, x_key)
        coms = global_mean_pool(x, g.batch)
        setattr(g, x_key, x - coms[g.batch])

        return DDGraph(g)

    def postprocess_latents(self, batch: Batch[DDGraph]) -> DDGraph:
        return self._postprocess_graph(batch.latents)

    def postprocess_features(
        self, latents: DDGraph, feats: torch.Tensor
    ) -> torch.Tensor:
        device = latents.device
        num_nodes, feature_dim = feats.shape
        num_graphs = int(latents.n_idx.max().item()) + 1

        graph_sums = torch.zeros(num_graphs, feature_dim, device=device)
        graph_sums.index_add_(0, latents.n_idx, feats)
        graph_counts = torch.zeros(num_graphs, device=device)
        graph_counts.index_add_(0, latents.n_idx, torch.ones(num_nodes, device=device))

        return graph_sums / graph_counts.unsqueeze(1)

    def _graph_to_mols(self, samples: DDGraph) -> list[Chem.Mol | None]:
        g = samples.graph.clone()
        g.upper_edge_mask = samples.ue_mask

        mols = []
        for data in g.to_data_list():
            mol = SampledMolecule(data.cpu(), self.atom_type_map).rdkit_mol
            mols.append(mol)

        return mols

    def _mols_to_graph(self, mols: list[Chem.Mol]) -> DDGraph:
        data_list = []
        for mol in mols:
            n = mol.GetNumAtoms()

            atom_positions = torch.from_numpy(mol.GetConformer().GetPositions()).float()
            atom_types_idx = torch.zeros(n, dtype=torch.int64)
            atom_charges = torch.zeros(n, dtype=torch.int64)

            for i, atom in enumerate(mol.GetAtoms()):
                atom_types_idx[i] = self.atom_type_to_idx[atom.GetSymbol()]
                atom_charges[i] = atom.GetFormalCharge()

            edge_index = build_edge_idxs(n)
            num_edges = edge_index.size(1)

            e_idx = torch.zeros(num_edges, dtype=torch.int64)
            for k in range(num_edges):
                a = int(edge_index[0, k])
                b = int(edge_index[1, k])
                bond = mol.GetBondBetweenAtoms(a, b)
                bond_type = bond.GetBondType() if bond is not None else None
                e_idx[k] = bond_type_to_idx[bond_type]

            num_upper = n * (n - 1) // 2
            upper_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
            upper_edge_mask[:num_upper] = True

            data = Data(
                edge_index=edge_index,
                num_nodes=n,
                x_1=atom_positions,
                a_1=F.one_hot(atom_types_idx, num_classes=len(self.atom_type_map)).float(),
                c_1=F.one_hot(atom_charges + 2, num_classes=6).float(),
                e_1=F.one_hot(e_idx, num_classes=5).float(),
                upper_edge_mask=upper_edge_mask,
            )
            data_list.append(data)

        return DDGraph(TGBatch.from_data_list(data_list))

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
        for i, (mol, is_valid_mol) in enumerate(zip(mols, batch.valids)):
            ax = fig.add_subplot(rows, cols, i + 1)
            if is_valid_mol:
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
        probs = self.base_model.model.n_atoms_dist.probs

        expected = n * probs
        counts = expected.round().long()

        remainder = n - counts.sum()
        if remainder > 0:
            fractional = expected - counts
            extra = torch.topk(fractional, remainder).indices
            counts[extra] += 1

        indices = torch.arange(len(counts), device=counts.device)
        result = torch.repeat_interleave(indices, counts)

        perm = torch.randperm(result.numel(), device=result.device)
        result = result[perm]

        return {"n_atoms": result}

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
            suppl = Chem.ForwardSDMolSupplier(
                f, sanitize=False, removeHs=False, strictParsing=False
            )
            for mol in suppl:
                if mol is not None:
                    mols.append(mol)

        return self._mols_to_graph(mols), {}

    def compute_metrics(self, samples: DDGraph, kwargs: dict) -> dict[str, float]:
        mols = self._graph_to_mols(samples)
        mols = [mol for mol in mols if mol is not None and self._is_mol_valid(mol)]

        K = molecule_utils.get_tanimoto_K(mols)
        vendi_score = vendi.score_K(K)

        n = len(mols)
        D = 1 - K
        avg_pairwise_dist = D.sum() / (n * (n - 1))

        return {
            "vendi": float(vendi_score),
            "avg_pairwise_dist": float(avg_pairwise_dist),
        }

    def compute_sample_metrics(
        self, samples: DDGraph, kwargs: dict
    ) -> list[dict[str, Any]]:
        samples = self._postprocess_graph(samples)
        mols = self._graph_to_mols(samples)

        valid_indices = [
            i
            for i, mol in enumerate(mols)
            if mol is not None and self._is_mol_valid(mol)
        ]
        valid_mols = [mols[i] for i in valid_indices]
        res = parallel_xtb(valid_mols)

        metrics: list[dict[str, Any]] = [{"is_valid": True} for _ in mols]
        metric_names = [
            "energy",
            "homo",
            "lumo",
            "homo_lumo_gap",
            "dipole_moment",
            "polarizability",
            "heat_capacity",
        ]
        for mol, idx, xtb_res in zip(valid_mols, valid_indices, res):
            if xtb_res is None:
                continue

            for name in metric_names:
                metrics[idx][name] = getattr(xtb_res, name)
            metrics[idx]["qed"] = QED.qed(mol)
            metrics[idx]["logp"] = Crippen.MolLogP(mol)
            metrics[idx]["sa_score"] = sascorer.calculateScore(mol)

        return metrics


class QM9ProblemSetup(MoleculeProblemSetup):
    def __init__(self, args: dict, device: Optional[torch.device] = None):
        super().__init__(dataset="qm9", args=args, device=device)


class GEOMDrugsProblemSetup(MoleculeProblemSetup):
    def __init__(self, args: dict, device: Optional[torch.device] = None):
        super().__init__(dataset="geom_drugs", args=args, device=device)


def relax_positions(data: Data, atom_type_map: list[str], alg: str = "mmff") -> Data:
    x_key = "x_1" if hasattr(data, "x_1") else "x_t"

    mol = SampledMolecule(data.cpu(), atom_type_map).rdkit_mol
    if mol is None or not is_valid(mol):
        return data

    if alg == "mmff":
        try:
            AllChem.MMFFOptimizeMolecule(mol)
        except Exception:
            pass
    elif alg == "uff":
        try:
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass
    elif alg == "gfn2":
        mol = xtb_relax_geometry(mol)
    else:
        raise ValueError(f"Unknown geometry optimization algorithm: {alg}")

    if mol is None:
        return data

    data = data.clone()
    positions = torch.from_numpy(mol.GetConformer().GetPositions())
    setattr(data, x_key, positions.to(data.edge_index.device).type_as(getattr(data, x_key)))
    return data


def xtb_relax_geometry(mol: Chem.Mol) -> Chem.Mol | None:
    with temporary_workdir():
        xyz_file = Path("input.xyz")
        output_file = Path("xtbopt.xyz")

        Chem.MolToXYZFile(mol, str(xyz_file))

        os.system(f"xtb {xyz_file} --opt --gfn 2 > /dev/null 2>&1")

        if not output_file.exists():
            return None

        opt_mol = Chem.MolFromXYZFile(str(output_file))
        if opt_mol is None:
            return None

        opt_conf = opt_mol.GetConformer()
        conf = mol.GetConformer()
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(i, opt_conf.GetAtomPosition(i))

    return mol
