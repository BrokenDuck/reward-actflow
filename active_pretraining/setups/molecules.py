from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import dgl
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw, AllChem
from flowmol.analysis.molecule_builder import SampledMolecule
from flowgym import  BaseModel, Environment
from flowgym.molecules import FlowGraph, QM9BaseModel, GEOMBaseModel
from flowgym.utils import temporary_workdir
from vendi_score import vendi
from vendi_score import molecule_utils
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path
from argparse import ArgumentParser
import json
from json import JSONDecodeError
import re
import os

from active_pretraining.problem_setup import ProblemSetup, SampleFile
from active_pretraining.utils import Batch


class MoleculeProblemSetup(ProblemSetup[FlowGraph]):
    def __init__(self, dataset: str, args: dict, device: Optional[torch.device] = None):
        RDLogger.DisableLog("rdApp.*")  # type: ignore

        self.geometry_opt: str = args["mol_geometry_opt"]

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dataset == "qm9":
            self._base_model = QM9BaseModel(device=device)
            self.div_valid = 2
        elif dataset == "geom_drugs":
            self._base_model = GEOMBaseModel(device=device)
            self.div_valid = 4
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument("--mol_geometry_opt", type=str, choices=["none", "mmff", "uff", "gfn2"], default="mmff")

    @property
    def base_model(self) -> BaseModel[FlowGraph]:
        return self._base_model

    def validity(self, samples: FlowGraph, kwargs: dict) -> torch.Tensor:
        valids = torch.ones(len(samples), dtype=torch.bool)

        samples.graph.edata["ue_mask"] = samples.ue_mask
        batched_graph = samples.graph.cpu()

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

    def postprocess_features(self, latents: FlowGraph, feats: torch.Tensor) -> torch.Tensor:
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

    def postprocess_latents(self, batch: Batch[FlowGraph]) -> FlowGraph:
        latents = batch.latents
        graph = latents.graph.clone()

        def _discretize(x):
            # argmax finds the class index, one_hot creates the vector
            # type_as ensures we match the original float precision and device
            return F.one_hot(x.argmax(dim=-1), num_classes=x.shape[-1]).type_as(x)

        # Set categorical features to be one-hot vectors over dim -1
        graph.ndata["a_t"] = _discretize(graph.ndata["a_t"])
        graph.ndata["c_t"] = _discretize(graph.ndata["c_t"])
        graph.edata["e_t"] = _discretize(graph.edata["e_t"])

        # Relax the geometry, otherwise the structures will become increasingly distorted
        graph.edata["ue_mask"] = latents.ue_mask
        if self.geometry_opt != "none":
            graphs = []
            for i, g in enumerate(dgl.unbatch(graph)):
                if batch.valids[i]:
                    g = relax_positions(g, self.base_model.model.atom_type_map, self.geometry_opt)

                graphs.append(g)

            graph = dgl.batch(graphs)

        # Remove center of mass
        init_coms = dgl.readout_nodes(graph, feat="x_t", op="mean")
        graph.ndata["x_t"] = graph.ndata["x_t"] - init_coms[latents.n_idx]

        return FlowGraph(graph, latents.ue_mask, latents.n_idx, latents.e_idx)

    def _to_mols(self, samples: FlowGraph) -> list[Chem.Mol]:
        mols = []
        for sample in dgl.unbatch(samples.graph):
            mol = SampledMolecule(sample.cpu(), self.base_model.model.atom_type_map).rdkit_mol
            mols.append(mol)

        return mols

    @torch.no_grad()
    def visualize_batch(self, env: Environment[FlowGraph], batch: Batch[FlowGraph]) -> Figure:
        mols = self._to_mols(batch.samples)

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
            img = Draw.MolToImage(mol, size=(150, 150))
            ax.imshow(img)

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

        return {"n_atoms": result}

    def save_sample(self, sample: FlowGraph, kwargs: dict, filename: Path):
        if len(sample) != 1:
            raise ValueError("Can only save a single sample at a time.")

        mol = SampledMolecule(sample.graph.cpu(), self.base_model.model.atom_type_map).rdkit_mol
        mol = validate_mol(mol)

        if mol is None:
            return

        Chem.MolToMolFile(mol, f"{filename}.mol")

    def compute_metrics(
        self,
        batches: list[Batch[FlowGraph]],
    ) -> dict[str, float]:
        n_samples = 0
        mols = []
        for batch in batches:
            for j in range(len(batch)):
                n_samples += 1

                if not batch.valids[j]:
                    continue

                mol = SampledMolecule(batch.samples[j].graph.cpu(), self.base_model.model.atom_type_map).rdkit_mol
                mol = validate_mol(mol)

                if mol is not None:
                    mols.append(mol)

        # We want to make sure the number of samples we compute the diversity on is constant between iterations
        mols = mols[:n_samples // self.div_valid]
        K = molecule_utils.get_tanimoto_K(mols)
        vendi_score = vendi.score_K(K)

        return { "vendi": float(vendi_score) }

    def compute_sample_metrics(self, sample_files: list[SampleFile]) -> dict[str, dict[str, float]]:
        # Load molecule samples
        valid_mols: list[Chem.Mol] = []
        valid_files: list[Path] = []
        for sample_file in sample_files:
            if sample_file.is_valid:
                mol = Chem.MolFromMolFile(str(sample_file.file), sanitize=False, removeHs=False, strictParsing=False)
                if mol is not None:
                    valid_files.append(sample_file.file)
                    valid_mols.append(mol)

        # Compute quantum chemistry properties
        res = parallel_xtb(valid_mols)

        # Return in nice format
        metric_names = ["energy", "homo", "lumo", "homo_lumo_gap", "dipole_moment", "polarizability", "heat_capacity"]
        out = dict()
        for file_path, xtb_res in zip(valid_files, res):
            if xtb_res is None:
                continue

            sample_name = file_path.name.replace(".mol", "")
            metrics = { name: getattr(xtb_res, name) for name in metric_names }
            out[sample_name] = metrics

        return out


class QM9ProblemSetup(MoleculeProblemSetup):
    def __init__(self, args: dict, device: Optional[torch.device] = None):
        super().__init__(dataset="qm9", args=args, device=device)


class GEOMDrugsProblemSetup(MoleculeProblemSetup):
    def __init__(self, args: dict, device: Optional[torch.device] = None):
        super().__init__(dataset="geom_drugs", args=args, device=device)


def relax_positions(g: dgl.DGLGraph, atom_type_map: list[str], alg: str = "mmff") -> dgl.DGLGraph:
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

    if alg == "mmff":
        # Sometimes it crashes in the middle of a run, so we guard with try-except
        try:
            AllChem.MMFFOptimizeMolecule(mol)  # type: ignore
        except RuntimeError:
            pass
    elif alg == "uff":
        try:
            AllChem.UFFOptimizeMolecule(mol)  # type: ignore
        except RuntimeError:
            pass
    elif alg == "gfn2":
        mol = xtb_relax_geometry(mol)
    else:
        raise ValueError(f"Unknown geometry optimization algorithm: {alg}")

    if mol is None:
        return g

    positions = torch.from_numpy(mol.GetConformer().GetPositions())
    g.ndata["x_t"] = positions.to(g.device).type_as(g.ndata["x_t"])  # type: ignore

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


class XTBResult:
    """Class to parse the output of GFN2-xTB."""

    def __init__(self, filename: Path):
        assert filename.suffix == ".json", f"Filename ({filename}) must end with .json"
        
        # Load JSON data
        with open(filename, "r") as f:
            self.data = json.load(f)

        # Load Log data (assumes .out file exists next to .json)
        # The parallel_xtb function saves JSON as *.xtbout.json and log as *.out
        log_filename = filename.with_suffix(".out").with_suffix(".out")
        
        if log_filename.exists():
            with open(log_filename, "r") as f:
                self.log_content = f.read()
        else:
            self.log_content = ""

    @property
    def energy(self) -> float:
        """Energy (Hartree)."""
        return float(self.data["total energy"])

    @property
    def homo(self) -> float:
        """Highest occupied molecular orbital (eV)."""
        occupation = np.asarray(self.data["fractional occupation"])
        energies = np.asarray(self.data["orbital energies/eV"])

        occupied_indices = np.where(occupation > 0)[0]
        if len(occupied_indices) == 0:
            raise ValueError("No occupied orbitals found.")

        highest_occupied_orbital = occupied_indices[-1]
        return float(energies[highest_occupied_orbital])

    @property
    def lumo(self) -> float:
        """Lowest unoccupied molecular orbital (eV)."""
        occupation = np.asarray(self.data["fractional occupation"])
        energies = np.asarray(self.data["orbital energies/eV"])

        unoccupied_indices = np.where(occupation == 0)[0]
        if len(unoccupied_indices) == 0:
            raise ValueError("No unoccupied orbitals found.")

        lowest_unoccupied_orbital = unoccupied_indices[0]
        return float(energies[lowest_unoccupied_orbital])

    @property
    def homo_lumo_gap(self) -> float:
        """HOMO-LUMO gap (eV)."""
        return self.lumo - self.homo

    @property
    def dipole_moment(self) -> float:
        """Dipole moment (Debye)."""
        return 2.5417 * float(np.linalg.norm(self.data["dipole"]))

    @property
    def polarizability(self) -> float:
        """Polarizability (Bohr^3)."""
        # Regex to find: Mol. α(0) /au      :         <val>
        match = re.search(r"Mol\.\s+α\(0\)\s+/au\s+:\s+([\d\.]+)", self.log_content)
        if not match:
            raise ValueError("Polarizability not found in log output.")

        return float(match.group(1))

    @property
    def heat_capacity(self) -> float:
        """Heat Capacity (cal/K/mol) at 298.15K."""
        # Look for the TOT line in the thermodynamic table.
        # Format: TOT     enthalpy    heat_capacity    entropy    entropy(J)
        # Example: TOT    5299.6013   30.4586          85.7819    358.9115
        # We search for TOT, skipping the enthalpy (first number), capturing the second number.
        # [+-]?\d+ matches integers, floats, scientific notation
        number_pattern = r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?"
        pattern = fr"TOT\s+{number_pattern}\s+({number_pattern})"
        
        match = re.search(pattern, self.log_content)
        if not match:
            raise ValueError("Heat capacity (TOT row) not found in log output. Did you run with --ohess?")
            
        return float(match.group(1))


def parallel_xtb(mols: list[Chem.Mol]):
    """Run GFN2-xTB in parallel for molecules in the graph."""
    results = []
    with temporary_workdir():
        i = 0
        for mol in mols:
            i += 1
            Chem.MolToXYZFile(mol, f"{i}.xyz")

        ncpus = len(os.sched_getaffinity(0))

        # Compute properties using GFN2-xTB
        # Added --ohess to calculate Hessian (needed for Heat Capacity)
        os.system(
            f"parallel -j {ncpus} "
            f"'xtb {{}} --ohess --parallel 1 --namespace {{/.}} --json > {{/.}}.out 2>&1' "
            "::: *.xyz"
        )

        # Read results
        for i in range(1, len(mols) + 1):
            path = Path(f"{i}.xtbout.json")

            try:
                res = XTBResult(path) if path.exists() else None
            except JSONDecodeError:
                res = None

            results.append(res)

    return results
