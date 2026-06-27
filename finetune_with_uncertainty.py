"""
Fine-tuning with GP Uncertainty Reward for Active Learning

This script runs uncertainty-based exploration: the policy MDM is
fine-tuned on its own samples, tilted toward high-uncertainty regions of
sequence space estimated by a GP over backbone embeddings.
"""

import json
import numpy as np
import torch
import argparse
import wandb
import os
import datetime
from pathlib import Path
from vendi_score import vendi
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for cluster
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# resolve repo-internal imports relative to this file: src/ holds the model
# code, the repo root holds the orchestration modules
import sys
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT / 'src'))

from diffusion import Diffusion
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from finetune_peptides import finetune, finetune_continued_pretraining, _sample_n_valid
from utils.utils import str2bool, set_seed

# Import uncertainty reward options
sys.path.insert(0, str(_REPO_ROOT))
from gaussian_process import GPUncertaintyReward
from metrics import compute_novelty, compute_num_clusters


def _seq_to_morgan_fp(sequence: str, radius: int = 2, n_bits: int = 2048):
    """Convert an amino-acid sequence to a Morgan fingerprint bit vector.

    The sequence is first converted to a SMILES string via RDKit's
    ``Chem.MolFromSequence``, then a Morgan (circular) fingerprint is
    computed.

    Args:
        sequence: One-letter amino-acid string.
        radius: Morgan fingerprint radius (default 2 ≈ ECFP4).
        n_bits: Length of the folded bit vector.

    Returns:
        An ``rdkit.DataStructs.ExplicitBitVect``, or *None* if the
        sequence cannot be parsed.
    """
    mol = Chem.MolFromSequence(sequence)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_vendi_score_tanimoto(
    sequences: list[str],
    radius: int = 2,
    n_bits: int = 2048,
    q: float = 1.0,
    threshold: float | None = None,
) -> float:
    """Compute the Vendi Score using a Tanimoto kernel over Morgan fingerprints.

    Peptide sequences are converted to SMILES via ``Chem.MolFromSequence``,
    then Morgan fingerprints are generated and all-pairs Tanimoto
    similarity is used as the kernel matrix for the Vendi Score.

    Sequences that fail SMILES conversion are silently skipped.

    Args:
        sequences: List of amino-acid sequence strings.
        radius: Morgan fingerprint radius (2 = ECFP4).
        n_bits: Bit-vector length for the folded fingerprint.

    Returns:
        Scalar Vendi Score (float).  Returns 1.0 if fewer than 2
        valid fingerprints are obtained.
    """
    fps = []
    for seq in sequences:
        fp = _seq_to_morgan_fp(seq, radius=radius, n_bits=n_bits)
        if fp is not None:
            fps.append(fp)

    if len(fps) < 2:
        return 1.0

    n = len(fps)
    K = np.ones((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            K[i, j] = sim
            K[j, i] = sim

    if threshold is not None:
        # Hard-threshold kernel: 1 if sim >= threshold else 0. PSD is not
        # guaranteed but works well in practice as a sharp diversity signal.
        K = (K >= threshold).astype(np.float64)

    return _vendi_score_K_q(K, q=q)


def _vendi_score_K_q(K: np.ndarray, q: float = 1.0) -> float:
    """Vendi Score of order q from a kernel matrix.

    Vendi^q(K) = exp(H_q(lambda_hat)) where lambda_hat are the eigenvalues
    of K/n (which sum to 1 when diag(K)=1) and H_q is the Renyi entropy of
    order q.

      q = 0  -> effective number of nonzero modes (richness; most sensitive
               to new regions)
      q = 1  -> exp(Shannon entropy)  [the standard Vendi Score]
      q = 2  -> 1 / sum(lambda_hat**2)
      q = inf-> 1 / max(lambda_hat)
    """
    n = K.shape[0]
    if n <= 1:
        return 1.0
    # Symmetrize and get eigenvalues of K/n
    Ksym = 0.5 * (K + K.T)
    w = np.linalg.eigvalsh(Ksym) / n
    # Numerical floor; eigenvalues should be >= 0 for a PSD kernel
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        return 1.0
    w = w / s  # renormalize against tiny numerical drift

    eps = 1e-12
    if q == 1.0:
        nz = w > eps
        H = -np.sum(w[nz] * np.log(w[nz]))
        return float(np.exp(H))
    if np.isinf(q):
        return float(1.0 / max(w.max(), eps))
    if q == 0.0:
        return float(np.sum(w > eps))
    # General Renyi: H_q = log(sum w^q) / (1 - q)
    Sq = np.sum(np.power(w[w > eps], q))
    return float(Sq ** (1.0 / (1.0 - q)))


def compute_vendi_score_from_embeddings(
    embeddings: np.ndarray,
    q: float = 1.0,
    kernel: str = 'rbf_median',
    sigma: float | None = None,
) -> float:
    """Compute the Vendi Score from a matrix of embeddings.

    Args:
        embeddings: (B, D) numpy array of embedding vectors.
        q: Order of the Vendi Score (Renyi entropy order). q=0 gives the
            effective number of nonzero modes (most sensitive to discovering
            new regions); q=1 (default) is the standard Shannon-entropy Vendi.
        kernel: One of:
            - 'rbf_median': Gaussian RBF on L2-normalized embeddings with
              bandwidth set to the median pairwise distance (median heuristic).
              Sharp diversity signal: K_ij ~ 1 for near-duplicates, ~0 for
              distinct points.
            - 'cosine_shift': legacy `(cos_sim + 1) / 2` kernel (weaker signal,
              kept for backward compatibility).
        sigma: Optional fixed RBF bandwidth (overrides median heuristic).

    Returns:
        Scalar Vendi Score (float).  Returns 1.0 for B <= 1.
    """
    if embeddings.shape[0] <= 1:
        return 1.0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    normed = embeddings / norms

    if kernel == 'cosine_shift':
        K = (normed @ normed.T + 1.0) / 2.0
    elif kernel == 'rbf_median':
        # Squared L2 distance on unit-norm vectors: ||x-y||^2 = 2(1 - cos)
        cos = normed @ normed.T
        d2 = np.clip(2.0 * (1.0 - cos), 0.0, None)
        if sigma is None:
            iu = np.triu_indices_from(d2, k=1)
            med = np.median(np.sqrt(d2[iu]))
            sigma = float(med) if med > 1e-9 else 1.0
        K = np.exp(-d2 / (2.0 * sigma * sigma))
    else:
        raise ValueError(f'Unknown kernel: {kernel}')

    return _vendi_score_K_q(K, q=q)


def compute_vendi_score_from_sequences(
    sequences: list,
    uncertainty_reward: 'GPUncertaintyReward',
    q: float = 1.0,
    kernel: str = 'rbf_median',
    sigma: float | None = None,
) -> float:
    """Compute the Vendi Score for a list of sequences using model embeddings.

    Extracts embeddings from the diffusion model backbone (via the GP uncertainty
    reward's `extract_embeddings`), then computes a cosine-similarity
    kernel Vendi Score.

    Args:
        sequences: List of peptide sequence strings.
        uncertainty_reward: GP uncertainty reward that exposes
            ``extract_embeddings(sequences) -> (B, D) Tensor``.

    Returns:
        Scalar Vendi Score (float).
    """
    if len(sequences) <= 1:
        return 1.0
    with torch.no_grad():
        emb = uncertainty_reward.extract_embeddings(sequences)  # (B, D) tensor
    return compute_vendi_score_from_embeddings(
        emb.cpu().numpy(), q=q, kernel=kernel, sigma=sigma
    )


def plot_exploration_pca(
    embeddings_per_iter: list[np.ndarray],
    save_path: str,
    title: str = 'Exploration in PCA Space',
    max_points_per_iter: int = 500,
) -> plt.Figure:
    """Plot the progression of generated sequences in PCA space.

    Each active-learning iteration is shown in a different colour, with
    earlier iterations rendered at lower opacity so the temporal
    progression is easy to read.

    Args:
        embeddings_per_iter: List of (B_i, D) arrays, one per AL iteration.
        save_path: Directory to save the PNG figure.
        title: Plot title.
        max_points_per_iter: Subsample each iteration to at most this many
            points to keep the plot readable.

    Returns:
        The matplotlib Figure object (also saved to *save_path*/exploration_pca.png
        and logged to wandb).
    """
    # Subsample large iterations
    subsampled, labels = [], []
    for i, emb in enumerate(embeddings_per_iter):
        if emb.shape[0] > max_points_per_iter:
            idx = np.random.choice(emb.shape[0], max_points_per_iter, replace=False)
            emb = emb[idx]
        subsampled.append(emb)
        labels.append(np.full(emb.shape[0], i))

    all_emb = np.concatenate(subsampled, axis=0)
    all_labels = np.concatenate(labels, axis=0)

    # Fit PCA on all collected embeddings
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(all_emb)  # (N, 2)

    n_iters = len(embeddings_per_iter)
    cmap = plt.cm.get_cmap('viridis', max(n_iters, 2))

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot each iteration; use increasing alpha so later iterations stand out
    for i in range(n_iters):
        mask = all_labels == i
        alpha = 0.3 + 0.7 * (i / max(n_iters - 1, 1))  # 0.3 -> 1.0
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            c=[cmap(i)],
            label=f'Iter {i + 1}',
            alpha=alpha,
            s=20,
            edgecolors='none',
        )

    ax.set_xlabel(f'PC 1  ({pca.explained_variance_ratio_[0]:.1%} var)')
    ax.set_ylabel(f'PC 2  ({pca.explained_variance_ratio_[1]:.1%} var)')
    ax.set_title(title)

    # Shrink legend for many iterations
    if n_iters <= 15:
        ax.legend(fontsize=8, markerscale=1.5, framealpha=0.8)
    else:
        # Summarise with a colorbar instead
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(1, n_iters))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, label='AL Iteration')

    fig.tight_layout()
    out_path = os.path.join(save_path, 'exploration_pca.png')
    fig.savefig(out_path, dpi=200)
    print(f'Saved PCA exploration plot to {out_path}')

    # Log to wandb
    try:
        wandb.log({'al/exploration_pca': wandb.Image(fig)})
    except Exception:
        pass  # wandb may not be initialised in tests

    plt.close(fig)
    return fig


class ContinuedPretrainingRewardShim:
    """Returns constant (uniform) rewards so WDCE importance weights stay flat.

    This is equivalent to continued pretraining on the model's own samples
    without any reward tilting.  The model still sees self-generated data
    and learns from it, but the reward signal carries no directional bias.
    """
    def __init__(self):
        self.alpha = 0.0

    def __call__(self, input_seqs: list) -> np.ndarray:
        # Return a constant reward of 1.0 for every sequence so that
        # WDCE importance weights remain uniform (no tilting).
        return np.ones((len(input_seqs), 1), dtype=np.float32)

    @property
    def score_func_names(self):
        return ['uniform']


class UncertaintyOnlyRewardShim:
    """Uses only the GP uncertainty reward.

    Pure exploration: the reward is the GP uncertainty, driving the policy
    toward novel, underexplored regions of sequence space.
    """
    def __init__(self, uncertainty_reward: 'GPUncertaintyReward',
                 conditional: bool = False):
        self.uncertainty_reward = uncertainty_reward
        self.alpha = 1.0
        self.conditional = conditional

    def __call__(self, input_seqs: list) -> np.ndarray:
        u = self.uncertainty_reward(input_seqs, conditional=self.conditional)  # (B,)
        return u[:, np.newaxis]                  # (B, 1)

    @property
    def score_func_names(self):
        return ['uncertainty']


def active_learning_loop(
    args,
    cfg,
    policy_model,
    pretrained,
    reward_func,
    uncertainty_reward,
    filename,
    prot_name,
):
    """
    Active learning loop that alternates between:
    1. Fine-tuning on current data
    2. Sampling with uncertainty guidance
    3. Evaluating samples and updating uncertainty model
    """
    
    # Track all generated sequences
    all_sequences = []
    all_embeddings = []        # embeddings per iteration (for PCA plot)
    vendi_score_log = []       # model-diversity Vendi Scores per iteration
    cluster_log = []           # list of {iteration, n_valid, n_clusters_0005, ...}
    cluster_thresholds = [0.005, 0.01, 0.05]
    cluster_n_valid = 1000

    skip_eval = getattr(args, 'skip_eval', False)

    # Always build the fixed pretrained-backbone embedder — used for the
    # once-per-AL-iteration Vendi computed on the exploration set (cheap).
    _pretrained_device = next(pretrained.parameters()).device
    _fixed_emb_extractor = GPUncertaintyReward(
        diffusion_model=pretrained,
        device=_pretrained_device,
        normalize_uncertainty=False,
    )
    print(f"Using fixed pretrained model for Vendi embeddings (device={_pretrained_device})")

    if not skip_eval:
        # ── Generate pretrained reference set for novelty computation ────
        _ref_n = getattr(args, 'vendi_n_valid', 50) * 2
        print(f"Generating {_ref_n} pretrained reference sequences for novelty...")
        pretrained_ref_seqs = []
        with torch.no_grad():
            for _ in range(20):  # max rounds
                if len(pretrained_ref_seqs) >= _ref_n:
                    break
                _ref_gen = pretrained.restore_model_and_sample(
                    batch_size=min(200, _ref_n),
                    seq_length=args.seq_length, eps=1e-3,
                )
                _ref_decoded = [pretrained.tokenizer.decode(s) for s in _ref_gen]
                for s in _ref_decoded:
                    if pretrained.analyzer.is_peptide(s) and len(pretrained_ref_seqs) < _ref_n:
                        pretrained_ref_seqs.append(s)
        print(f"  Pretrained reference set: {len(pretrained_ref_seqs)} valid sequences")
    else:
        print("[skip_eval] Skipping novelty / clustering / PCA — keeping the "
              "cheap Vendi compute on the exploration sample set.")
        pretrained_ref_seqs = []

    for iteration in range(args.num_al_iterations):
        print(f"\n{'='*60}")
        print(f"Active Learning Iteration {iteration + 1}/{args.num_al_iterations}")
        print(f"{'='*60}\n")
        
        # Gradually decrease exploration (alpha annealing)
        if args.alpha_anneal:
            current_alpha = args.alpha * (1 - iteration / args.num_al_iterations)
            reward_func.alpha = current_alpha
            print(f"Exploration weight (alpha): {current_alpha:.3f}")
        
        # Fine-tune the model
        print("Fine-tuning model...")
        if args.mode == 'continued_pretraining':
            finetune_continued_pretraining(
                args,
                cfg,
                policy_model,
                reward_model=reward_func,
                pretrained=pretrained,
                filename=f"{filename}_iter{iteration}",
                prot_name=prot_name,
            )
        else:
            finetune(
                args,
                cfg,
                policy_model,
                reward_model=reward_func,
                mcts=None,
                pretrained=pretrained,
                filename=f"{filename}_iter{iteration}",
                prot_name=prot_name,
            )
        
        # Generate samples for exploration
        print(f"\nGenerating {args.num_exploration_samples} sequences...")
        policy_model.eval()
        with torch.no_grad():
            # Sample sequences
            generated = policy_model.restore_model_and_sample(
                batch_size=args.num_exploration_samples,
                seq_length=args.seq_length,
                eps=1e-3,
            )
            sequences = [policy_model.tokenizer.decode(seq) for seq in generated]
        
        # ── Validity rate for exploration samples ──
        valid_exploration = [policy_model.analyzer.is_peptide(seq) for seq in sequences]
        valid_fraction_exploration = sum(valid_exploration) / len(sequences)
        print(f"  Exploration validity: {valid_fraction_exploration:.3f} ({sum(valid_exploration)}/{len(sequences)})")

        # ── Once-per-AL-iteration Vendi on the EXPLORATION samples ──
        # No extra sampling: reuses the same sequences fed to the reward / GP
        # update, so cost is just one backbone forward pass + eigendecomp.
        # Uses --vendi_kernel (default tanimoto) and --vendi_sigma for the
        # embedding-RBF branch (RBF kernel on the diffusion backbone
        # embeddings when --vendi_kernel != tanimoto).
        valid_exploration_seqs = [s for s, ok in zip(sequences, valid_exploration) if ok]
        if len(valid_exploration_seqs) >= 2:
            if getattr(args, 'vendi_kernel', 'cosine') == 'tanimoto':
                vs_model = compute_vendi_score_tanimoto(
                    valid_exploration_seqs,
                    radius=getattr(args, 'morgan_radius', 2),
                    n_bits=getattr(args, 'morgan_n_bits', 2048),
                )
            else:
                # <=0 sentinel -> fall back to median heuristic
                _vsig = getattr(args, 'vendi_sigma', 0.25)
                _vsig = None if (_vsig is None or _vsig <= 0) else float(_vsig)
                vs_model = compute_vendi_score_from_sequences(
                    valid_exploration_seqs, _fixed_emb_extractor, sigma=_vsig)
        else:
            vs_model = float('nan')
        vendi_score_log.append(vs_model)

        # Aliases for downstream code (PCA, embeddings dump) — point at the
        # exploration valid set so the eval block doesn't re-sample.
        vendi_valid_seqs = valid_exploration_seqs
        vendi_sequences = valid_exploration_seqs
        valid_fraction_vendi = valid_fraction_exploration

        if not skip_eval:
            # ── Novelty & clustering on the exploration valid set ─────────
            if len(vendi_valid_seqs) >= 2 and len(pretrained_ref_seqs) >= 2:
                novelty = compute_novelty(vendi_valid_seqs, pretrained_ref_seqs)
                n_clusters, cluster_diversity, _ = compute_num_clusters(
                    vendi_valid_seqs, threshold=0.65)
            else:
                novelty = float('nan')
                n_clusters, cluster_diversity = 0, float('nan')
            print(f"  Novelty: {novelty:.4f}  "
                  f"Clusters: {n_clusters} ({cluster_diversity:.3f})")

            # ── Multi-threshold cluster evaluation ─────────────────────────
            print(f'  [cluster eval] Sampling {cluster_n_valid} valid seqs...')
            _, cluster_seqs, _ = _sample_n_valid(
                policy_model, n_target=cluster_n_valid,
                num_steps=args.total_num_steps,
                seq_length=args.seq_length,
                batch_size=200, eps=1e-5, max_rounds=50,
            )
            cluster_info = {'iteration': iteration, 'n_valid': len(cluster_seqs)}
            for thr in cluster_thresholds:
                key = str(thr).replace('.', '')
                if len(cluster_seqs) >= 2:
                    try:
                        nc, cd, _ = compute_num_clusters(
                            cluster_seqs, threshold=thr)
                    except Exception as e:
                        print(f'    Warning: clustering thr={thr} failed: {e}')
                        nc, cd = 0, 0.0
                else:
                    nc, cd = 0, 0.0
                cluster_info[f'n_clusters_{key}'] = nc
                cluster_info[f'cluster_diversity_{key}'] = cd
            cluster_log.append(cluster_info)
            c_parts = []
            for t in cluster_thresholds:
                k = str(t).replace('.', '')
                c_parts.append(
                    f'thr={t}: {cluster_info[f"n_clusters_{k}"]}'
                    f'/{cluster_info[f"cluster_diversity_{k}"]:.3f}')
            print(f'    Clusters ({len(cluster_seqs)} seqs): {"  ".join(c_parts)}')

            # Extract embeddings from the fixed pretrained model for PCA / disk
            with torch.no_grad():
                iter_emb = _fixed_emb_extractor.extract_embeddings(sequences).cpu().numpy()
                if len(vendi_sequences) > 0:
                    vendi_emb = _fixed_emb_extractor.extract_embeddings(vendi_sequences).cpu().numpy()
                else:
                    vendi_emb = np.empty((0, iter_emb.shape[1]))
            all_embeddings.append(iter_emb)

            # Persist embeddings and sequences to disk so PCA plots survive crashes
            emb_dir = os.path.join(args.save_path, 'embeddings')
            os.makedirs(emb_dir, exist_ok=True)
            np.save(os.path.join(emb_dir, f'embeddings_iter{iteration}.npy'), iter_emb)
            np.save(os.path.join(emb_dir, f'vendi_embeddings_iter{iteration}.npy'), vendi_emb)
            with open(os.path.join(emb_dir, f'sequences_iter{iteration}.txt'), 'w') as f:
                for seq in sequences:
                    f.write(seq + '\n')
            with open(os.path.join(emb_dir, f'vendi_sequences_iter{iteration}.txt'), 'w') as f:
                for seq in vendi_sequences:
                    f.write(seq + '\n')
        else:
            # Eval skipped — vs_model and valid_fraction_vendi already set
            # above from the exploration-set Vendi compute.
            novelty = float('nan')
            n_clusters, cluster_diversity = 0, float('nan')
            cluster_info = {'iteration': iteration, 'n_valid': 0}

        # Evaluate sequences
        print("Evaluating sequences...")
        # Uncertainty scores for logging
        if uncertainty_reward is not None:
            uncertainty_scores = uncertainty_reward(sequences)
            # Update uncertainty model with new sequences.
            print("Updating uncertainty model...")
            # Apply --gp_filter_invalid before mutating the GP buffer so it
            # fits only on-manifold sequences.
            gp_seqs = sequences
            if getattr(args, 'gp_filter_invalid', False):
                valid_mask = [policy_model.analyzer.is_peptide(s) for s in sequences]
                kept = [s for s, ok in zip(sequences, valid_mask) if ok]
                if kept:
                    print(f"  [GP filter_invalid] kept {len(kept)}/{len(sequences)} valid")
                    gp_seqs = kept
                else:
                    print(f"  [GP filter_invalid] WARNING 0/{len(sequences)} valid — using unfiltered")
            buffer_mode = getattr(args, 'gp_buffer_mode', 'append')
            if buffer_mode == 'replace':
                uncertainty_reward.replace_data(gp_seqs)
                print(f"  [GP] replace: buffer now {len(uncertainty_reward.data)} pts")
            elif buffer_mode == 'window':
                cap = getattr(args, 'gp_buffer_window', 2000)
                uncertainty_reward.add_data(gp_seqs, max_buffer_size=cap)
                print(f"  [GP] window(cap={cap}): buffer now "
                      f"{len(uncertainty_reward.data)} pts")
            else:
                uncertainty_reward.add_data(gp_seqs)
            mean_uncertainty = float(uncertainty_scores.mean())
        else:
            mean_uncertainty = 0.0
        
        # Track statistics
        all_sequences.extend(sequences)

        # Log statistics — use commit=False so these don't advance the
        # global step, then commit with an explicit al_step so the AL
        # metrics get their own clean x-axis in wandb.
        log_dict = {
            f'al/iteration': iteration,
            f'al/mean_uncertainty': mean_uncertainty,
            f'al/exploration_alpha': getattr(reward_func, 'alpha', 0.0),
            f'al/num_total_sequences': len(all_sequences),
            f'al/valid_fraction_exploration': valid_fraction_exploration,
        }
        if not skip_eval:
            log_dict.update({
                f'al/novelty': novelty,
                f'al/num_clusters': n_clusters,
                f'al/cluster_diversity': cluster_diversity,
                f'al/valid_fraction_vendi': valid_fraction_vendi,
            })
            for k, v in cluster_info.items():
                if k not in ('iteration', 'n_valid'):
                    log_dict[f'al/{k}'] = v
        wandb.log(log_dict)

        print(f"\nIteration {iteration + 1} Summary:")
        if not skip_eval:
            print(f"  Novelty vs pretrained:        {novelty:.4f}")
            print(f"  Clusters: {n_clusters}  diversity: {cluster_diversity:.4f}")
        print(f"  Mean uncertainty: {mean_uncertainty:.4f}")
        print(f"  Valid fraction (exploration): {valid_fraction_exploration:.4f}")
        if not skip_eval:
            print(f"  Valid fraction (vendi eval):  {valid_fraction_vendi:.4f}")

        # Save best sequences so far
        if iteration % args.save_every_n_iters == 0:
            save_path = Path(args.save_path) / f"sequences_iter{iteration}.txt"
            with open(save_path, 'w') as f:
                for seq in sequences[:100]:  # Save top 100
                    f.write(seq + '\n')

        # Periodically save PCA exploration plot (and always on last iteration)
        if not skip_eval:
            is_last = (iteration == args.num_al_iterations - 1)
            if is_last or (iteration > 0 and iteration % args.save_every_n_iters == 0):
                plot_exploration_pca(
                    embeddings_per_iter=all_embeddings,
                    save_path=args.save_path,
                    title=f'Exploration in PCA Space  (iter 1–{iteration + 1})',
                )

    # Save cluster evaluation log
    if not skip_eval and cluster_log:
        cluster_path = os.path.join(args.save_path, f'cluster_log_{filename}.json')
        with open(cluster_path, 'w') as f:
            json.dump(cluster_log, f, indent=2)
        print(f'Saved cluster log to {cluster_path}')

    return all_sequences, vendi_score_log


def main():
    parser = argparse.ArgumentParser()
    
    # Model args
    parser.add_argument('--base_path', type=str, default=str(_REPO_ROOT))
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--save_path_dir', type=str, default=str(_REPO_ROOT / 'checkpoints'))
    parser.add_argument('--run_name', type=str, default=None,
                        help='Override auto-generated run name (subfolder under --save_path_dir).')
    
    # Fine-tuning args
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--training_mini_batch_size', type=int, default=0,
                        help='If >0, randomly sample this many sequences from '
                             'the buffer at each training step instead of '
                             'using the full buffer. 0 = use full buffer.')
    parser.add_argument('--seq_length', type=int, default=200)
    parser.add_argument('--gradnorm_clip', type=float, default=1.0)
    parser.add_argument('--save_every_n_epochs', type=int, default=10)
    parser.add_argument('--total_num_steps', type=int, default=128)
    parser.add_argument('--no_mcts', action='store_true', default=True,
                       help='Disable MCTS (use direct sampling)')
    parser.add_argument('--num_accum_steps', type=int, default=4)
    parser.add_argument('--truncate_steps', type=int, default=50)
    parser.add_argument('--wdce_num_replicates', type=int, default=16)
    parser.add_argument('--centering', action='store_true', default=False,
                       help='Use reward centering in WDCE loss')
    parser.add_argument('--grad_clip', action='store_true', default=False,
                       help='Enable gradient clipping')
    parser.add_argument('--truncate_kl', action='store_true', default=False,
                       help='Truncate KL divergence')
    parser.add_argument('--gumbel_temp', type=float, default=1.0,
                       help='Gumbel temperature for sampling')
    parser.add_argument('--copy_flag_temp', type=float, default=None,
                       help='Copy flag temperature')
    parser.add_argument('--alpha_schedule_warmup', type=int, default=0,
                       help='Alpha schedule warmup steps')
    parser.add_argument('--time_conditioning', action='store_true', default=False,
                       help='Enable time conditioning')
    parser.add_argument('--noise_removal', action='store_true', default=False,
                       help='Enable noise removal')
    parser.add_argument('--num_obj', type=int, default=1,
                       help='Number of objectives (uncertainty only)')
    parser.add_argument('--resample_every_n_step', type=int, default=10,
                       help='Resample trajectories every N steps')
    parser.add_argument('--reset_every_n_step', type=int, default=100,
                       help='Reset MCTS tree every N steps')
    parser.add_argument('--exploration', type=float, default=0.1,
                       help='MCTS exploration parameter')
    parser.add_argument('--num_sequences', type=int, default=10,
                       help='Number of sequences for MCTS')
    parser.add_argument('--num_children', type=int, default=50,
                       help='Number of children per MCTS node')
    parser.add_argument('--num_iter', type=int, default=30,
                       help='Number of MCTS iterations')
    parser.add_argument('--mcts_sampling', type=int, default=0,
                       help='MCTS sampling mode')
    parser.add_argument('--buffer_size', type=int, default=100,
                       help='MCTS buffer size')
    
    # Active learning args
    parser.add_argument('--num_al_iterations', type=int, default=10,
                       help='Number of active learning iterations')
    parser.add_argument('--num_exploration_samples', type=int, default=100,
                       help='Number of sequences to generate per iteration')
    parser.add_argument('--alpha', type=float, default=0.3,
                       help='Temperature scaling the uncertainty reward tilt in '
                            'the WDCE loss (higher = stronger exploration push)')
    parser.add_argument('--alpha_anneal', action='store_true',
                       help='Gradually decrease alpha over iterations')
    parser.add_argument('--gp_lengthscale', type=float, default=0.1,
                       help='GP kernel lengthscale (used by RBF kernel)')
    parser.add_argument('--gp_kernel', type=str, default='cosine',
                       choices=['cosine', 'rbf', 'linear'],
                       help='GP kernel function: cosine (direction-based), '
                            'rbf (Euclidean distance), linear (dot-product)')
    parser.add_argument('--vendi_eval_samples', type=int, default=200,
                       help='Number of sequences to sample from model per Vendi round')
    parser.add_argument('--vendi_n_valid', type=int, default=100,
                       help='Fixed number of *valid* sequences to collect for Vendi Score')
    parser.add_argument('--vendi_kernel', type=str, default='tanimoto',
                       choices=['cosine', 'tanimoto'],
                       help='kernel for Vendi Score. "cosine" routes to '
                            'compute_vendi_score_from_sequences which builds an '
                            'RBF kernel on the diffusion backbone embedding — use '
                            'with --vendi_sigma to match the eval script.')
    parser.add_argument('--vendi_sigma', type=float, default=0.25,
                       help='Fixed RBF bandwidth for the embedding-based Vendi '
                            'Score (only used when --vendi_kernel is not tanimoto). '
                            'Defaults to 0.25. '
                            'Pass <= 0 to fall back to the adaptive median-distance '
                            'heuristic.')
    parser.add_argument('--morgan_radius', type=int, default=2,
                       help='Morgan fingerprint radius (only used with --vendi_kernel tanimoto)')
    parser.add_argument('--morgan_n_bits', type=int, default=2048,
                       help='Morgan fingerprint bit-vector length (only used with --vendi_kernel tanimoto)')
    parser.add_argument('--skip_eval', action='store_true', default=False,
                       help='Skip all costly in-loop validation/eval steps '
                            '(Vendi sampling, novelty/clustering, embeddings, '
                            'PCA plots, pretrained reference set). Training only.')
    parser.add_argument('--save_every_n_iters', type=int, default=2)
    parser.add_argument('--initial_pool_size', type=int, default=1000,
                       help='initial finetuning pool of sequences (0 = start with empty pool, all exploration)')
    parser.add_argument('--pool_refresh_fraction', type=float, default=0.5,
                       help='fraction of pool to refresh with new samples each iteration (0 = keep all, 1 = replace all)')
    parser.add_argument('--noise_timestep', type=float, default=0.0,
                       help='Diffusion timestep for noising inputs before embedding '
                            'extraction in the GP (0 = clean, >0 = partial masking)')
    parser.add_argument('--n_noise_samples', type=int, default=5,
                       help='Number of independent noise draws to average when '
                            'noise_timestep > 0 (reduces masking variance)')
    parser.add_argument('--gp_buffer_mode', type=str, default='append',
                       choices=['append', 'replace', 'window'],
                       help='How the GP buffer is updated each AL iteration. '
                            'append: cumulative (default, current behavior). '
                            'replace: discard buffer, refit on this iter\'s samples '
                            '(stops the stale-buffer feedback loop that lets '
                            'collapsed-region samples keep scoring high). '
                            'window: rolling FIFO buffer of size '
                            '--gp_buffer_window.')
    parser.add_argument('--gp_filter_invalid', action='store_true',
                       help='Drop sequences that fail analyzer.is_peptide before '
                            'adding them to the GP uncertainty buffer. '
                            'Applies to both bootstrap and per-iteration updates. '
                            'Independent of --filter_invalid, which only affects '
                            'the WDCE trajectory buffer.')
    parser.add_argument('--gp_buffer_window', type=int, default=2000,
                       help='Buffer size for --gp_buffer_mode=window. '
                            'Ignored otherwise.')
    parser.add_argument('--gp_conditional_reward', action='store_true', default=False,
                       help='Use conditional GP posterior variance as the '
                            'uncertainty reward (Schur complement over the '
                            'in-batch sample set). Two near-duplicate samples '
                            'in the same WDCE batch jointly inflate each '
                            'other\'s denominator and both score low → '
                            'collapse self-penalizes.')
    
    # Task args
    parser.add_argument('--prot_seq', type=str, default=None)
    parser.add_argument('--prot_name', type=str, default='tfr')
    parser.add_argument('--scalarization', type=str, default='sum',
                       choices=['sum', 'product', 'min'])
    
    # Experiment mode
    parser.add_argument('--mode', type=str, default='uncertainty_only',
                       choices=['uncertainty_only', 'continued_pretraining'],
                       help='Reward mode: uncertainty_only (GP uncertainty '
                            'tilting) or continued_pretraining '
                            '(uniform reward, no tilting)')
    
    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--filter_invalid', action='store_true',
                       help='(continued_pretraining only) Filter out invalid peptides '
                            'from self-generated samples before denoising loss')
    
    args = parser.parse_args()

    # Setup
    set_seed(args.seed, use_cuda=True)
    device = torch.device(args.device)
    
    curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_tag = args.mode
    if getattr(args, 'filter_invalid', False):
        mode_tag += '_filtered'
    # Include noise timestep in run name when non-zero
    noise_tag = ""
    if getattr(args, 'noise_timestep', 0.0) > 0:
        noise_tag = f"_nt{args.noise_timestep}"
    if not args.run_name:
        args.run_name = f"{args.prot_name}_{mode_tag}{noise_tag}_alpha{args.alpha}_{curr_time}"
    args.save_path = os.path.join(args.save_path_dir, args.run_name)
    os.makedirs(args.save_path, exist_ok=True)
    
    # Initialize wandb
    wandb.init(project='actflow_pep', name=args.run_name, config=args, dir=args.save_path)
    # Give al/ metrics their own x-axis so they don't get buried among
    # the per-epoch finetune logs (which advance the global step ~1000x
    # per AL iteration).
    wandb.define_metric('al/iteration')
    wandb.define_metric('al/*', step_metric='al/iteration')
    
    # Load pretrained model
    ckpt_path = f'{_REPO_ROOT}/src/pretrained/peptune-pretrained.ckpt'
    
    GlobalHydra.instance().clear()
    initialize(
        version_base="1.1",
        config_path="src/configs",
        job_name="load_model"
    )
    cfg = compose(config_name="peptune_config.yaml")
    
    print("Loading models...")
    policy_model = Diffusion.load_from_checkpoint(
        ckpt_path, config=cfg, mode="train", device=device, map_location=device
    )
    pretrained = Diffusion.load_from_checkpoint(
        ckpt_path, config=cfg, mode="eval", device=device, map_location=device
    )
    
    # ---- Build the uncertainty reward based on --mode ----
    uncertainty_reward = None  # remains None for continued_pretraining

    if args.mode == 'uncertainty_only':
        print("Initializing uncertainty reward...")
        uncertainty_reward = GPUncertaintyReward(
            diffusion_model=policy_model,
            lengthscale=args.gp_lengthscale,
            device=device,
            normalize_uncertainty=True,
            kernel=args.gp_kernel,
            noise_timestep=args.noise_timestep,
            n_noise_samples=args.n_noise_samples,
        )
        # Bootstrap uncertainty model
        print("Bootstrapping uncertainty model with initial samples...")
        with torch.no_grad():
            initial_samples = policy_model.restore_model_and_sample(
                batch_size=50, seq_length=args.seq_length, eps=1e-3)
            initial_seqs = [policy_model.tokenizer.decode(seq) for seq in initial_samples]
        if getattr(args, 'gp_filter_invalid', False):
            valid_mask = [policy_model.analyzer.is_peptide(s) for s in initial_seqs]
            kept = [s for s, ok in zip(initial_seqs, valid_mask) if ok]
            if kept:
                print(f"  [GP filter_invalid] bootstrap: kept {len(kept)}/{len(initial_seqs)} valid")
                initial_seqs = kept
            else:
                print(f"  [GP filter_invalid] bootstrap: WARNING 0/{len(initial_seqs)} valid — using unfiltered")
        uncertainty_reward.add_data(initial_seqs)
    
    if args.mode == 'continued_pretraining':
        # Uniform rewards → no tilting.  Model does continued self-training.
        combined_reward = ContinuedPretrainingRewardShim()
        args.num_obj = 1
    else:  # uncertainty_only
        combined_reward = UncertaintyOnlyRewardShim(
            uncertainty_reward,
            conditional=args.gp_conditional_reward,
        )
        args.num_obj = 1

    # Run active learning loop
    print(f"\nStarting active learning with {args.num_al_iterations} iterations...")
    all_sequences, vendi_log = active_learning_loop(
        args=args,
        cfg=cfg,
        policy_model=policy_model,
        pretrained=pretrained,
        reward_func=combined_reward,
        uncertainty_reward=uncertainty_reward,
        filename=args.prot_name,
        prot_name=args.prot_name,
    )
    
    print("\nActive learning complete!")
    wandb.finish()


if __name__ == '__main__':
    main()
