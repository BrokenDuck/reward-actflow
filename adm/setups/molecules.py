from typing import Optional, Any

import torch
import torch.nn.functional as F
import dgl
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Draw, AllChem, Crippen, QED, rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import gaussian_kde
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

    def compute_metrics(self, samples: DDGraph, kwargs: dict, n_valid: int = 0, compute_vendi: bool = False) -> dict[str, float]:
        mols = self._graph_to_mols(samples)
        mols = [mol for mol in mols if mol is not None and self._is_mol_valid(mol)]

        if n_valid > 0 and len(mols) > n_valid:
            mols = mols[:n_valid]

        n = len(mols)
        result: dict[str, float] = {"n_valid_for_metrics": n}
        if compute_vendi:
            K = molecule_utils.get_tanimoto_K(mols)
            result["vendi"] = float(vendi.score_K(K))

        if n >= 2:
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
            fps = [fpgen.GetFingerprint(mol) for mol in mols]
            picks = rdSimDivPickers.LeaderPicker().LazyBitVectorPick(fps, len(fps), 0.85)
            result["sphere_exclusion_diversity"] = len(picks) / n
            result["n_clusters"] = float(len(picks))

        return result

    def get_morgan_fingerprints(self, samples: DDGraph, kwargs: dict, n_valid: int = 0) -> np.ndarray:
        """Extract Morgan fingerprint bit vectors for valid molecules."""
        mols = self._graph_to_mols(samples)
        mols = [mol for mol in mols if mol is not None and self._is_mol_valid(mol)]
        if n_valid > 0 and len(mols) > n_valid:
            mols = mols[:n_valid]

        fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in mols]
        arr = np.zeros((len(fps), 1024), dtype=np.float32)
        for i, fp in enumerate(fps):
            DataStructs.ConvertToNumpyArray(fp, arr[i])
        return arr

    @staticmethod
    def compute_novelty(current_fps: np.ndarray, reference_fps: np.ndarray) -> float:
        """Novelty = 1 - mean nearest-neighbour Tanimoto similarity to reference set."""
        if len(current_fps) == 0 or len(reference_fps) == 0:
            return 0.0
        cur = current_fps.astype(np.int32)
        ref = reference_fps.astype(np.int32)
        intersection = cur @ ref.T
        cur_bits = cur.sum(axis=1, keepdims=True)
        ref_bits = ref.sum(axis=1, keepdims=True)
        union = cur_bits + ref_bits.T - intersection
        tanimoto = np.divide(
            intersection.astype(float), union.astype(float),
            out=np.zeros_like(intersection, dtype=float), where=union > 0,
        )
        max_sim = tanimoto.max(axis=1)
        return float(1.0 - max_sim.mean())

    @staticmethod
    def compute_fid(current_fps: np.ndarray, reference_fps: np.ndarray) -> float:
        """Fréchet Inception Distance in Morgan fingerprint space.

        Compares the multivariate Gaussian fit (mean, covariance) of
        current_fps vs reference_fps using the Fréchet distance.
        """
        from scipy.linalg import sqrtm

        if len(current_fps) < 2 or len(reference_fps) < 2:
            return float("nan")

        mu1 = current_fps.mean(axis=0)
        mu2 = reference_fps.mean(axis=0)
        sigma1 = np.cov(current_fps, rowvar=False)
        sigma2 = np.cov(reference_fps, rowvar=False)

        diff = mu1 - mu2
        covmean = sqrtm(sigma1 @ sigma2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real

        return float(diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean))

    @staticmethod
    def plot_fingerprint_pca(
        current_fps: np.ndarray,
        pretrained_fps: np.ndarray,
    ) -> Figure | None:
        """PCA + KDE density plot of current vs pretrained fingerprints.

        PCA is fit on the concatenation of both sets each time so the axes
        adapt to capture directions where the distributions differ most.
        """
        if len(current_fps) < 3 or len(pretrained_fps) < 3:
            return None

        combined = np.vstack([pretrained_fps, current_fps])
        reducer = make_pipeline(StandardScaler(), PCA(n_components=2)).fit(combined)
        emb_pre = reducer.transform(pretrained_fps)
        emb_cur = reducer.transform(current_fps)

        return MoleculeProblemSetup._kde_contour_plot(emb_pre, emb_cur)

    @staticmethod
    def plot_fingerprint_fixed_projection(
        current_fps: np.ndarray,
        pretrained_fps: np.ndarray,
        projection: np.ndarray,
    ) -> Figure | None:
        """KDE density plot using a fixed random projection (stable across iterations).

        Axis limits are derived from the pretrained embeddings with generous
        padding so the viewport stays constant across all iterations.
        """
        if len(current_fps) < 3 or len(pretrained_fps) < 3:
            return None
        emb_pre = pretrained_fps @ projection
        emb_cur = current_fps @ projection

        margin = 0.5
        x_range = emb_pre[:, 0].max() - emb_pre[:, 0].min()
        y_range = emb_pre[:, 1].max() - emb_pre[:, 1].min()
        xlim = (emb_pre[:, 0].min() - margin * x_range, emb_pre[:, 0].max() + margin * x_range)
        ylim = (emb_pre[:, 1].min() - margin * y_range, emb_pre[:, 1].max() + margin * y_range)
        return MoleculeProblemSetup._kde_contour_plot(emb_pre, emb_cur, xlim=xlim, ylim=ylim)

    @staticmethod
    def _kde_contour_plot(
        emb_pre: np.ndarray,
        emb_cur: np.ndarray,
        xlim: tuple[float, float] | None = None,
        ylim: tuple[float, float] | None = None,
    ) -> Figure:
        if xlim is None or ylim is None:
            all_emb = np.vstack([emb_pre, emb_cur])
            x_margin = max((all_emb[:, 0].max() - all_emb[:, 0].min()) * 0.15, 1e-3)
            y_margin = max((all_emb[:, 1].max() - all_emb[:, 1].min()) * 0.15, 1e-3)
            xlim = (all_emb[:, 0].min() - x_margin, all_emb[:, 0].max() + x_margin)
            ylim = (all_emb[:, 1].min() - y_margin, all_emb[:, 1].max() + y_margin)

        gridsize = 150
        xs = np.linspace(xlim[0], xlim[1], gridsize)
        ys = np.linspace(ylim[0], ylim[1], gridsize)
        xx, yy = np.meshgrid(xs, ys)
        grid_pts = np.vstack([xx.ravel(), yy.ravel()])

        def _kde_level(emb, percentile=0.1):
            kde = gaussian_kde(emb.T)
            zz = kde(grid_pts).reshape(xx.shape)
            densities_at_pts = kde(emb.T)
            level = float(np.percentile(densities_at_pts, 100 * percentile))
            return zz, level

        fig, ax = plt.subplots(figsize=(7, 7))

        try:
            zz_pre, lvl_pre = _kde_level(emb_pre)
            ax.contourf(xx, yy, zz_pre, levels=[lvl_pre, zz_pre.max() * 10],
                        colors=["gray"], alpha=0.25)
            ax.contour(xx, yy, zz_pre, levels=[lvl_pre],
                       colors=["gray"], linewidths=1.5, alpha=0.6)
        except np.linalg.LinAlgError:
            ax.scatter(emb_pre[:, 0], emb_pre[:, 1], c="gray", alpha=0.3, s=10)

        try:
            zz_cur, lvl_cur = _kde_level(emb_cur)
            ax.contourf(xx, yy, zz_cur, levels=[lvl_cur, zz_cur.max() * 10],
                        colors=["#7e57c2"], alpha=0.25)
            ax.contour(xx, yy, zz_cur, levels=[lvl_cur],
                       colors=["#7e57c2"], linewidths=1.5, alpha=0.6)
        except np.linalg.LinAlgError:
            ax.scatter(emb_cur[:, 0], emb_cur[:, 1], c="purple", alpha=0.3, s=10)

        ax.plot([], [], color="gray", linewidth=6, alpha=0.5, label="Pretrained")
        ax.plot([], [], color="#7e57c2", linewidth=6, alpha=0.5, label="Current")
        ax.legend(fontsize=12)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        plt.tight_layout()
        return fig

    @staticmethod
    def compute_cumulative_cluster_metrics(
        current_fps: np.ndarray,
        cumulative_centers: np.ndarray | None,
        threshold: float = 0.85,
    ) -> tuple[dict[str, float], np.ndarray]:
        """Track cluster expansion across iterations using Tanimoto sphere exclusion.

        Returns metrics dict and the updated cumulative_centers array.
        """
        cur_int = current_fps.astype(np.int32)

        cur_bits = cur_int.sum(axis=1, keepdims=True)
        cur_self_inter = cur_int @ cur_int.T
        cur_union = cur_bits + cur_bits.T - cur_self_inter
        cur_tanimoto = np.divide(
            cur_self_inter.astype(float), cur_union.astype(float),
            out=np.zeros_like(cur_self_inter, dtype=float), where=cur_union > 0,
        )
        n = len(current_fps)
        picked = [0]
        for i in range(1, n):
            if all(cur_tanimoto[i, j] < threshold for j in picked):
                picked.append(i)
        current_centers = current_fps[picked]

        if cumulative_centers is None:
            return {
                "cumulative_clusters": float(len(current_centers)),
                "coverage_of_cumulative": 1.0,
                "new_clusters": float(len(current_centers)),
                "clusters_lost": 0.0,
            }, current_centers.copy()

        def _max_tanimoto(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
            q = query.astype(np.int32)
            r = reference.astype(np.int32)
            inter = q @ r.T
            q_bits = q.sum(axis=1, keepdims=True)
            r_bits = r.sum(axis=1, keepdims=True)
            union = q_bits + r_bits.T - inter
            sim = np.divide(
                inter.astype(float), union.astype(float),
                out=np.zeros_like(inter, dtype=float), where=union > 0,
            )
            return sim.max(axis=1)

        coverage_sim = _max_tanimoto(cumulative_centers, current_fps)
        covered = (coverage_sim >= threshold).sum()
        n_prev = len(cumulative_centers)
        clusters_lost = n_prev - int(covered)

        novelty_sim = _max_tanimoto(current_centers, cumulative_centers)
        new_mask = novelty_sim < threshold
        new_centers = current_centers[new_mask]

        updated = np.vstack([cumulative_centers, new_centers]) if len(new_centers) > 0 else cumulative_centers.copy()

        return {
            "cumulative_clusters": float(len(updated)),
            "coverage_of_cumulative": float(covered) / n_prev if n_prev > 0 else 1.0,
            "new_clusters": float(len(new_centers)),
            "clusters_lost": float(clusters_lost),
        }, updated

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
