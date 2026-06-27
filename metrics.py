"""
Metrics for evaluating generated peptide sequences.

Provides:
  - **Novelty**: 1 − mean(max Tanimoto similarity to reference set).
  - **#Clusters / Sphere-exclusion diversity**: fraction of sequences
    that are cluster centres under a Tanimoto distance threshold.

Both metrics operate on Morgan fingerprints derived from peptide
sequences converted to SMILES via ``Chem.MolFromSequence``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdFingerprintGenerator
from rdkit.SimDivFilters import rdSimDivPickers
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def peptide_to_mol(sequence: str) -> Optional[Chem.Mol]:
    """Convert a one-letter amino-acid string to an RDKit Mol."""
    return Chem.MolFromSequence(sequence)


def peptide_to_smiles(sequence: str) -> Optional[str]:
    """Convert a peptide sequence to a canonical SMILES string."""
    mol = peptide_to_mol(sequence)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def peptide_to_morgan_fp(
    sequence: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[DataStructs.ExplicitBitVect]:
    """Return a Morgan fingerprint bit-vector for a peptide sequence."""
    mol = peptide_to_mol(sequence)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def sequences_to_fps(
    sequences: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[List[DataStructs.ExplicitBitVect], List[int]]:
    """Convert peptide sequences to fingerprints, skipping failures.

    Returns
    -------
    fps : list[ExplicitBitVect]
        Valid fingerprints.
    valid_idx : list[int]
        Indices into *sequences* that produced a valid fingerprint.
    """
    fps, valid_idx = [], []
    for i, seq in enumerate(sequences):
        fp = peptide_to_morgan_fp(seq, radius=radius, n_bits=n_bits)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    return fps, valid_idx


def smiles_to_morgan_fp(
    smi: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> Optional[DataStructs.ExplicitBitVect]:
    """Morgan FP directly from a SMILES string (handles non-canonical residues
    that `Chem.MolFromSequence` would silently drop)."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def smiles_to_fps(
    smiles: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[List[DataStructs.ExplicitBitVect], List[int]]:
    """Vectorised `smiles_to_morgan_fp` — skips SMILES RDKit cannot parse."""
    fps, valid_idx = [], []
    for i, smi in enumerate(smiles):
        fp = smiles_to_morgan_fp(smi, radius=radius, n_bits=n_bits)
        if fp is not None:
            fps.append(fp)
            valid_idx.append(i)
    return fps, valid_idx


# ---------------------------------------------------------------------------
# Novelty
# ---------------------------------------------------------------------------

def compute_novelty(
    sequences: Sequence[str],
    reference_sequences: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
    n_threads: int = 1,
) -> float:
    """Compute novelty of *sequences* w.r.t. a set of *reference_sequences*.

    Novelty = 1 − mean_i( max_j  Tanimoto(fp_i, ref_fp_j) )

    A value of 1.0 means every generated sequence is maximally dissimilar
    to every reference; 0.0 means every generated sequence has an exact
    match in the reference set.

    Parameters
    ----------
    sequences : list[str]
        Generated peptide sequences to evaluate.
    reference_sequences : list[str]
        Reference (e.g. training-set) peptide sequences.
    radius, n_bits : int
        Morgan fingerprint parameters.
    n_threads : int
        Number of threads for parallel Tanimoto lookups.

    Returns
    -------
    float
        Novelty score in [0, 1].
    """
    gen_fps, _ = sequences_to_fps(sequences, radius=radius, n_bits=n_bits)
    ref_fps, _ = sequences_to_fps(reference_sequences, radius=radius, n_bits=n_bits)

    if not gen_fps or not ref_fps:
        return 0.0

    def _max_sim(fp):
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        return float(max(sims))

    if n_threads > 1:
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            max_sims = list(tqdm(pool.map(_max_sim, gen_fps), total=len(gen_fps), desc="novelty"))
    else:
        max_sims = [_max_sim(fp) for fp in gen_fps]

    return 1.0 - float(np.mean(max_sims))


def compute_fraction_novel(
    sequences: Sequence[str],
    reference_sequences: Sequence[str],
    sim_threshold: float = 0.8,
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """Fraction of generated sequences with max Tanimoto sim < threshold.

    A generated sequence is "novel" if its nearest neighbour in the
    reference set has Tanimoto similarity strictly below *sim_threshold*.

    Returns 0.0 when there are no valid fingerprints.
    """
    gen_fps, _ = sequences_to_fps(sequences, radius=radius, n_bits=n_bits)
    ref_fps, _ = sequences_to_fps(reference_sequences, radius=radius, n_bits=n_bits)

    if not gen_fps or not ref_fps:
        return 0.0

    novel_count = 0
    for fp in gen_fps:
        max_sim = max(DataStructs.BulkTanimotoSimilarity(fp, ref_fps))
        if max_sim < sim_threshold:
            novel_count += 1

    return novel_count / len(gen_fps)


def compute_nn_novelty(
    sequences: Sequence[str],
    reference_sequences: Sequence[str],
    k: int = 5,
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """Nearest-neighbour novelty using the k-th nearest reference distance.

    For each generated fingerprint, compute Tanimoto similarity to all
    reference fingerprints, take the k-th largest similarity (1-indexed),
    and return ``1 − mean(k-th similarity)``.

    Falls back to max similarity when the reference set has fewer than
    *k* members.
    """
    gen_fps, _ = sequences_to_fps(sequences, radius=radius, n_bits=n_bits)
    ref_fps, _ = sequences_to_fps(reference_sequences, radius=radius, n_bits=n_bits)

    if not gen_fps or not ref_fps:
        return 0.0

    effective_k = min(k, len(ref_fps))

    kth_sims = []
    for fp in gen_fps:
        sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
        # k-th largest similarity (0-indexed partial sort)
        top_k = np.partition(sims, -effective_k)[-effective_k]
        kth_sims.append(float(top_k))

    return 1.0 - float(np.mean(kth_sims))


def _fps_to_numpy(fps: List[DataStructs.ExplicitBitVect]) -> np.ndarray:
    """Convert a list of RDKit bit vectors to a float numpy array."""
    arr = np.zeros((len(fps), fps[0].GetNumBits()), dtype=np.float32)
    for i, fp in enumerate(fps):
        DataStructs.ConvertToNumpyArray(fp, arr[i])
    return arr


def compute_fid_fingerprint(
    sequences: Sequence[str],
    reference_sequences: Sequence[str],
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """Fréchet distance between Morgan fingerprint distributions.

    Treats each n_bits-dimensional fingerprint as a point, projects to
    a PCA subspace (to avoid singular covariance matrices when
    n_samples < n_bits), then estimates Gaussian statistics and computes
    the Fréchet distance.

    Lower is more similar.  Returns 0.0 on degenerate inputs.
    """
    from scipy.linalg import sqrtm

    gen_fps, _ = sequences_to_fps(sequences, radius=radius, n_bits=n_bits)
    ref_fps, _ = sequences_to_fps(reference_sequences, radius=radius, n_bits=n_bits)

    if len(gen_fps) < 2 or len(ref_fps) < 2:
        return 0.0

    gen_arr = _fps_to_numpy(gen_fps)
    ref_arr = _fps_to_numpy(ref_fps)

    # PCA projection to avoid singular covariance matrices
    # Use min(n_samples - 1, n_bits) components on the combined data
    combined = np.vstack([gen_arr, ref_arr])
    n_components = min(combined.shape[0] - 1, combined.shape[1], 128)
    mean_all = combined.mean(axis=0)
    centered = combined - mean_all
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = Vt[:n_components]  # (n_components, n_bits)
    gen_proj = (gen_arr - mean_all) @ proj.T
    ref_proj = (ref_arr - mean_all) @ proj.T

    mu_g, mu_r = gen_proj.mean(axis=0), ref_proj.mean(axis=0)
    sigma_g = np.cov(gen_proj, rowvar=False)
    sigma_r = np.cov(ref_proj, rowvar=False)

    # Small regularisation for numerical stability
    eps = 1e-6
    sigma_g += np.eye(n_components) * eps
    sigma_r += np.eye(n_components) * eps

    diff = mu_g - mu_r
    mean_term = float(diff @ diff)

    product = sigma_g @ sigma_r
    sqrt_product = sqrtm(product)
    if np.iscomplexobj(sqrt_product):
        sqrt_product = sqrt_product.real

    fid = mean_term + np.trace(sigma_g + sigma_r - 2.0 * sqrt_product)
    return max(float(fid), 0.0)


def compute_novelty_fpsim2(
    sequences: Sequence[str],
    ref_fp_path: str,
    n_threads: int = 4,
) -> float:
    """Compute novelty using a pre-built FPSim2 fingerprint database.

    Requires the ``FPSim2`` package and a ``.h5`` reference database
    built with ``FPSim2.FPSim2Engine.create_db_file()``.

    Parameters
    ----------
    sequences : list[str]
        Generated peptide sequences.
    ref_fp_path : str
        Path to the FPSim2 ``.h5`` fingerprint database.
    n_threads : int
        Worker threads for parallel similarity search.

    Returns
    -------
    float
        Novelty score in [0, 1].
    """
    from FPSim2 import FPSim2Engine  # lazy import

    eng = FPSim2Engine(ref_fp_path)

    smiles_list = [peptide_to_smiles(seq) for seq in sequences]
    smiles_list = [s for s in smiles_list if s is not None]

    if not smiles_list:
        return 0.0

    def _one(smi):
        res = eng.top_k(smi, threshold=0.0, k=1, metric="tanimoto", n_workers=1)
        return float(res[0][1]) if len(res) > 0 else 0.0

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        sims = list(tqdm(pool.map(_one, smiles_list), total=len(smiles_list), desc="novelty-fpsim2"))

    return 1.0 - float(np.mean(sims)) if sims else 0.0


# ---------------------------------------------------------------------------
# Clustering / sphere-exclusion diversity
# ---------------------------------------------------------------------------

def compute_num_clusters(
    sequences: Sequence[str],
    threshold: float = 0.65, # change this
    radius: int = 2,
    n_bits: int = 2048,
) -> Tuple[int, float, List[int]]:
    """Cluster peptide sequences using sphere-exclusion (LeaderPicker).

    The LeaderPicker greedily picks cluster centres such that no two
    centres have Tanimoto similarity above *threshold*.  Each remaining
    sequence is assigned to the nearest centre.

    Parameters
    ----------
    sequences : list[str]
        Peptide sequences to cluster.
    threshold : float
        Tanimoto *distance* threshold for the LeaderPicker.  A value of
        0.65 means two sequences must differ by at least 0.65 to both
        be cluster centres (i.e. max similarity within a cluster ≈ 0.35).
    radius, n_bits : int
        Morgan fingerprint parameters.

    Returns
    -------
    n_clusters : int
        Number of clusters (= number of picked centres).
    diversity : float
        ``n_clusters / n_valid`` — fraction of sequences that are unique
        cluster centres.  Higher → more diverse.
    picks : list[int]
        Indices (into the *valid* fingerprint list) of cluster centres.
    """
    fps, _ = sequences_to_fps(sequences, radius=radius, n_bits=n_bits)

    if not fps:
        return 0, 0.0, []

    picker = rdSimDivPickers.LeaderPicker()
    picks = list(picker.LazyBitVectorPick(fps, len(fps), threshold))
    n_clusters = len(picks)
    diversity = n_clusters / len(fps)

    return n_clusters, diversity, picks
