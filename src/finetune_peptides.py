# direct reward backpropagation
import numpy as np
import torch
import wandb
import os
from finetune_utils import loss_wdce
from tqdm import tqdm
import pandas as pd
from vendi_score import vendi
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs


def compute_vendi_score(x_tokens: torch.Tensor) -> float:
    """Compute the Vendi Score for a batch of discrete token sequences.

    Uses a normalized Hamming similarity kernel:
        K_ij = 1 - hamming_distance(x_i, x_j) / seq_length

    A higher Vendi Score indicates greater diversity in the generated batch.

    Args:
        x_tokens: (B, L) integer tensor of token IDs.

    Returns:
        Scalar Vendi Score (float).  Returns 1.0 for batches of size <= 1.
    """
    if x_tokens.ndim != 2 or x_tokens.size(0) <= 1:
        return 1.0
    x = x_tokens.long().cpu()
    B, L = x.shape
    # pairwise Hamming similarity: K_ij = 1 - (# mismatches) / L
    # expand for broadcasting: (B, 1, L) vs (1, B, L)
    matches = (x.unsqueeze(1) == x.unsqueeze(0)).float()  # (B, B, L)
    K = matches.mean(dim=-1).numpy()  # (B, B) in [0, 1]
    return float(vendi.score_K(K))


@torch.no_grad()
def extract_pretrained_embeddings(pretrained_model, sequences: list) -> torch.Tensor:
    """Extract L2-normalised mean-pooled embeddings from the pretrained backbone.

    Uses the same extraction logic as :class:`GPUncertaintyReward` so
    that Vendi Scores are computed in the same space across the codebase.

    Args:
        pretrained_model: Pretrained :class:`Diffusion` model (kept frozen).
        sequences: List of peptide sequence strings.

    Returns:
        (B, D) float tensor of unit-norm embeddings on the model's device.
    """
    device = next(pretrained_model.parameters()).device
    encoded_list = []
    for seq in sequences:
        tokens = pretrained_model.tokenizer._tokenize(seq)
        encoded = pretrained_model.tokenizer.encode(tokens)
        encoded_list.append(encoded)

    max_len = max(enc['input_ids'].shape[1] for enc in encoded_list)
    batch_ids, batch_mask = [], []
    for enc in encoded_list:
        ids = enc['input_ids'].squeeze(0)
        mask = enc['attention_mask'].squeeze(0)
        pad = max_len - len(ids)
        if pad > 0:
            ids = torch.cat([ids, torch.zeros(pad, dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(pad, dtype=torch.long)])
        batch_ids.append(ids)
        batch_mask.append(mask)

    input_ids = torch.stack(batch_ids).to(device)
    attn_mask = torch.stack(batch_mask).to(device)

    pretrained_model.backbone.eval()
    outputs = pretrained_model.backbone.model.roformer(
        input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True,
    )
    hidden = outputs.hidden_states[-1]
    mask_exp = attn_mask.unsqueeze(-1).expand(hidden.size()).float()
    emb = torch.sum(hidden * mask_exp, dim=1) / mask_exp.sum(dim=1).clamp(min=1e-9)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-9)
    return emb


def compute_vendi_score_cosine(embeddings: np.ndarray) -> float:
    """Vendi Score with cosine similarity kernel on (already L2-normed) embeddings.

    Kernel:  K_ij = (cos_sim + 1) / 2  ∈ [0, 1].

    Args:
        embeddings: (B, D) numpy array — rows should be L2-normalised.

    Returns:
        Scalar Vendi Score (float).  Returns 1.0 for B <= 1.
    """
    if embeddings.shape[0] <= 1:
        return 1.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    normed = embeddings / norms
    K = (normed @ normed.T + 1.0) / 2.0
    return float(vendi.score_K(K))


def _seq_to_morgan_fp(sequence: str, radius: int = 2, n_bits: int = 2048):
    """Convert an amino-acid sequence to a Morgan fingerprint bit vector.

    Args:
        sequence: One-letter amino-acid string.
        radius: Morgan fingerprint radius (default 2 ≈ ECFP4).
        n_bits: Length of the folded bit vector.

    Returns:
        An rdkit ExplicitBitVect, or None if the sequence cannot be parsed.
    """
    mol = Chem.MolFromSequence(sequence)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def compute_vendi_score_tanimoto(
    sequences: list,
    radius: int = 2,
    n_bits: int = 2048,
) -> float:
    """Compute the Vendi Score using a Tanimoto kernel over Morgan fingerprints.

    Peptide sequences are converted to SMILES via Chem.MolFromSequence,
    then Morgan fingerprints are generated and all-pairs Tanimoto
    similarity is used as the kernel matrix for the Vendi Score.

    Sequences that fail SMILES conversion are silently skipped.

    Args:
        sequences: List of amino-acid sequence strings.
        radius: Morgan fingerprint radius (2 = ECFP4).
        n_bits: Bit-vector length for the folded fingerprint.

    Returns:
        Scalar Vendi Score (float). Returns 1.0 if fewer than 2
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
    return float(vendi.score_K(K))


def _sample_n_valid(policy_model, n_target, num_steps, seq_length,
                    batch_size=50, eps=1e-5, max_rounds=10):
    """Sample from the policy until *n_target* valid peptides are collected.

    Args:
        policy_model: Diffusion model (switched to eval internally).
        n_target: Number of valid sequences to collect.
        num_steps: Diffusion sampling steps.
        seq_length: Sequence length.
        batch_size: Sequences generated per round.
        eps: Diffusion epsilon.
        max_rounds: Maximum sampling rounds before giving up.

    Returns:
        valid_tokens: (N, L) tensor of valid sequence tokens  (N <= n_target).
        valid_seqs:   list[str] of decoded valid SMILES / peptide strings.
        total_sampled: Total number of sequences generated across all rounds.
    """
    valid_tokens_list = []
    valid_seqs_list = []
    total_sampled = 0

    policy_model.eval()
    with torch.no_grad():
        for _ in range(max_rounds):
            if len(valid_seqs_list) >= n_target:
                break
            x_raw = policy_model.restore_model_and_sample(
                num_steps=num_steps,
                batch_size=batch_size,
                seq_length=seq_length,
                eps=eps,
            )
            seqs = policy_model.tokenizer.batch_decode(x_raw)
            total_sampled += len(seqs)
            for i, s in enumerate(seqs):
                if policy_model.analyzer.is_peptide(s) and len(valid_seqs_list) < n_target:
                    valid_tokens_list.append(x_raw[i])
                    valid_seqs_list.append(s)

    if valid_tokens_list:
        valid_tokens = torch.stack(valid_tokens_list)
    else:
        valid_tokens = torch.empty(
            0, seq_length, dtype=torch.long,
            device=next(policy_model.parameters()).device,
        )

    return valid_tokens, valid_seqs_list, total_sampled


def finetune_continued_pretraining(
    args, cfg, policy_model, reward_model=None, pretrained=None,
    filename=None, prot_name=None, eps=1e-5,
):
    """Continued pretraining: standard denoising loss on self-generated samples.

    Instead of WDCE importance-weighted training, this function:
      1. Samples sequences from the current policy (no validity filtering).
      2. Trains using the model's own denoising diffusion loss (_loss).
      3. Optionally evaluates with ``reward_model`` for logging only.

    This avoids the mode-collapse failure of uniform-reward WDCE because
    there is no filtering step that drops invalid sequences.
    """
    base_path = args.base_path

    policy_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)

    batch_losses = []
    valid_fraction_log = []
    scores_logs = {}
    vendi_score_log = []
    vendi_morgan_log = []  # Vendi score using Morgan fingerprints

    filter_invalid = getattr(args, 'filter_invalid', False)
    buffer_size = getattr(args, 'buffer_size', args.batch_size)

    # Buffer for self-generated samples
    x_buffer = None

    # --- Generate large initial pool for mode coverage ---
    # A large upfront pool ensures all modes of the pretrained distribution
    # are represented.  Each resample step draws a random subset from this
    # pool instead of regenerating a tiny batch, which prevents the model
    # from losing validity as training progresses.
    #
    # Set --initial_pool_size 0 to disable the pool entirely and fall back
    # to the original per-step regeneration behaviour.
    #
    # pool_refresh_fraction (0–1): at each resample step, this fraction of
    # the pool is replaced with fresh samples from the *current* policy,
    # allowing continual learning while retaining mode coverage from the
    # original generation.  0 = static pool, 1 = fully replace each step.
    initial_pool_size = getattr(args, 'initial_pool_size', 0)
    pool_refresh_fraction = getattr(args, 'pool_refresh_fraction', 0.2)
    x_pool = None
    if initial_pool_size > 0 and initial_pool_size > buffer_size:
        print(f"Generating initial sequence pool ({initial_pool_size} sequences) for mode coverage...")
        policy_model.eval()
        pool_chunks = []
        with torch.no_grad():
            while sum(c.shape[0] for c in pool_chunks) < initial_pool_size:
                sample_size = args.batch_size
                x_raw = policy_model.restore_model_and_sample(
                    num_steps=args.total_num_steps,
                    batch_size=sample_size,
                    seq_length=args.seq_length,
                    eps=eps,
                )
                if filter_invalid:
                    seqs = policy_model.tokenizer.batch_decode(x_raw)
                    valid_mask = [policy_model.analyzer.is_peptide(s) for s in seqs]
                    valid_idx = [i for i, v in enumerate(valid_mask) if v]
                    if len(valid_idx) > 0:
                        x_raw = x_raw[valid_idx]
                pool_chunks.append(x_raw)
            x_pool = torch.cat(pool_chunks, dim=0)[:initial_pool_size]
        policy_model.train()
        _pool_seqs = policy_model.tokenizer.batch_decode(x_pool)
        _pool_valid = sum(policy_model.analyzer.is_peptide(s) for s in _pool_seqs)
        print(f"  Initial pool: {x_pool.shape[0]} sequences "
              f"({_pool_valid} valid, {_pool_valid/x_pool.shape[0]:.1%})")
        del _pool_seqs  # free memory

    pbar = tqdm(range(args.num_epochs))
    for epoch in pbar:
        # --- Generate self-play data (token sequences) ---
        with torch.no_grad():
            if x_buffer is None or epoch % args.resample_every_n_step == 0:
                if x_pool is not None:
                    # -- Rolling refresh: replace a fraction of the pool --
                    n_refresh = int(x_pool.shape[0] * pool_refresh_fraction)
                    if n_refresh > 0:
                        policy_model.eval()
                        refresh_chunks = []
                        _refresh_attempts = 0
                        _max_refresh_attempts = 50  # prevent infinite loop
                        while sum(c.shape[0] for c in refresh_chunks) < n_refresh and _refresh_attempts < _max_refresh_attempts:
                            _refresh_attempts += 1
                            sample_size = args.batch_size
                            x_raw = policy_model.restore_model_and_sample(
                                num_steps=args.total_num_steps,
                                batch_size=sample_size,
                                seq_length=args.seq_length,
                                eps=eps,
                            )
                            if filter_invalid:
                                seqs = policy_model.tokenizer.batch_decode(x_raw)
                                valid_idx = [i for i, v in enumerate(
                                    policy_model.analyzer.is_peptide(s) for s in seqs) if v]
                                if len(valid_idx) > 0:
                                    x_raw = x_raw[valid_idx]
                                    print(f"  [pool refresh] kept {len(valid_idx)}/{len(seqs)} valid")
                                else:
                                    print(f"  [pool refresh] WARNING: 0/{len(seqs)} valid — skipping batch")
                                    continue  # don't add invalid sequences to pool
                            refresh_chunks.append(x_raw)
                        n_actually_refreshed = sum(c.shape[0] for c in refresh_chunks)
                        if n_actually_refreshed == 0:
                            print(f"  [pool refresh] Could not generate any valid sequences — keeping pool unchanged")
                        else:
                            x_new = torch.cat(refresh_chunks, dim=0)[:n_refresh]
                            n_refresh = min(n_refresh, x_new.shape[0])  # may be fewer than requested
                            # Replace random positions in the pool
                            replace_idx = torch.randperm(x_pool.shape[0])[:n_refresh]
                            x_pool[replace_idx] = x_new.to(x_pool.device)
                            print(f"  Pool refreshed: {n_refresh}/{x_pool.shape[0]} sequences replaced")
                        policy_model.train()

                    # Subsample from the (possibly refreshed) pool
                    indices = torch.randperm(x_pool.shape[0])[:buffer_size]
                    x_buffer = x_pool[indices]
                    print(f"  Buffer subsampled from pool: {x_buffer.shape[0]}/{buffer_size}")
                else:
                    # Fallback: original behaviour — generate fresh samples
                    policy_model.eval()
                    chunks = []
                    while sum(c.shape[0] for c in chunks) < buffer_size:
                        sample_size = args.batch_size
                        x_raw = policy_model.restore_model_and_sample(
                            num_steps=args.total_num_steps,
                            batch_size=sample_size,
                            seq_length=args.seq_length,
                            eps=eps,
                        )
                        if filter_invalid:
                            seqs = policy_model.tokenizer.batch_decode(x_raw)
                            valid_mask = [policy_model.analyzer.is_peptide(s) for s in seqs]
                            valid_idx = [i for i, v in enumerate(valid_mask) if v]
                            if len(valid_idx) > 0:
                                x_raw = x_raw[valid_idx]
                                print(f"  [filter_invalid] kept {len(valid_idx)}/{len(seqs)} valid")
                            else:
                                print(f"  [filter_invalid] WARNING: 0/{len(seqs)} valid — using unfiltered")
                        chunks.append(x_raw)
                    x_buffer = torch.cat(chunks, dim=0)[:buffer_size]
                    print(f"  Buffer filled: {x_buffer.shape[0]}/{buffer_size} sequences")
                    policy_model.train()

        x0 = x_buffer.to(policy_model.device)

        # --- Mini-batch sampling from buffer ---
        mini_batch_size = getattr(args, 'training_mini_batch_size', 0)
        if mini_batch_size > 0 and mini_batch_size < x0.shape[0]:
            idx = torch.randperm(x0.shape[0])[:mini_batch_size]
            x0 = x0[idx]

        # --- WDCE loss with uniform weights (no importance weighting) ---
        dummy_log_rnd = torch.zeros(x0.shape[0], device=x0.device)
        loss = loss_wdce(policy_model, dummy_log_rnd, x0,
                         num_replicates=args.wdce_num_replicates,
                         uniform_weights=True)

        loss.backward()

        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(), args.gradnorm_clip,
            )

        optim.step()
        optim.zero_grad()

        pbar.set_postfix(loss=loss.item())
        batch_losses.append(loss.item())

        # --- Periodic evaluation (for logging only) ---
        with torch.no_grad():
            policy_model.eval()
            x_eval = policy_model.restore_model_and_sample(
                num_steps=args.total_num_steps,
                batch_size=50,
                seq_length=args.seq_length,
                eps=eps,
            )
            eval_seqs = policy_model.tokenizer.batch_decode(x_eval)
            valid = [policy_model.analyzer.is_peptide(s) for s in eval_seqs]
            valid_fraction = sum(valid) / len(eval_seqs)
            valid_fraction_log.append(valid_fraction)

            # Per-epoch Vendi removed — computed once per AL iteration in
            # finetune_with_uncertainty.py using the exploration sample set.
            vs = float('nan')
            vs_morgan = float('nan')

            scores_dict = {}
            if reward_model is not None and hasattr(reward_model, 'task_reward') and reward_model.task_reward is not None:
                task_reward = reward_model.task_reward
                valid_seqs = [s for s, v in zip(eval_seqs, valid) if v]
                if len(valid_seqs) > 0:
                    raw_scores = task_reward(input_seqs=valid_seqs)
                    for i, name in enumerate(task_reward.score_func_names):
                        scores_dict[name] = raw_scores[:, i].tolist()
                else:
                    for name in task_reward.score_func_names:
                        scores_dict[name] = [0.0]

            for name, vals in scores_dict.items():
                if name not in scores_logs:
                    scores_logs[name] = []
                scores_logs[name].append(vals)

            policy_model.train()

        score_summary = " ".join(
            f"{name} {np.mean(scores_dict[name]):f}" for name in scores_dict
        )
        print(
            f"epoch {epoch} valid_frac {valid_fraction:f} "
            f"{score_summary} loss {loss.item():f}"
        )

        log_dict = {
            "epoch": epoch, "mean_loss": loss.item(),
            "valid_fraction": valid_fraction,
        }
        for name in scores_dict:
            log_dict[name] = np.mean(scores_dict[name])
        wandb.log(log_dict)

        if (epoch + 1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(args.save_path, f'model_{filename}_epoch{epoch}.ckpt')
            torch.save(policy_model.state_dict(), model_path)
            print(f"model saved at epoch {epoch}")

    # --- Save logs ---
    os.makedirs(f'{base_path}/results/{args.run_name}', exist_ok=True)
    output_log_path = f'{base_path}/results/{args.run_name}/log_{filename}.csv'
    save_logs_to_file(valid_fraction_log, scores_logs, vendi_score_log, output_log_path, vendi_morgan_log)

    # Final evaluation with dataframe
    if reward_model is not None and hasattr(reward_model, 'task_reward') and reward_model.task_reward is not None:
        x_eval, scores_dict_final, valid_fraction, df = policy_model.sample_finetuned(
            args, reward_model.task_reward, batch_size=200, dataframe=True,
        )
        df.to_csv(
            f'{base_path}/results/{args.run_name}/{prot_name}_generation_results.csv',
            index=False,
        )

    return batch_losses


def finetune(args, cfg, policy_model, reward_model, mcts=None, pretrained=None, filename=None, prot_name=None, eps=1e-5):
    """
    Finetuning with WDCE loss
    """
    base_path = args.base_path
    dt = (1 - eps) / args.total_num_steps
    
    if args.no_mcts:
        assert pretrained is not None, "pretrained model is required for no mcts"
    else:
        assert mcts is not None, "mcts is required for mcts"
        
    # set model to train mode
    policy_model.train()
    torch.set_grad_enabled(True)
    optim = torch.optim.AdamW(policy_model.parameters(), lr=args.learning_rate)
    
    # record metrics
    batch_losses = []
    #batch_rewards = []
    
    # Buffer for trajectories, filled to buffer_size at each resample step (--no_mcts)
    x_buffer, log_rnd_buffer, rewards_buffer = None, None, None
    buffer_size = getattr(args, 'buffer_size', args.batch_size)
    filter_invalid = getattr(args, 'filter_invalid', False)
    
    valid_fraction_log = []
    scores_logs = {}  # {obj_name: [per-epoch values]}
    vendi_score_log = []
    vendi_morgan_log = []  # Vendi score using Morgan fingerprints

    # --- Generate large initial pool for mode coverage (WDCE) ---
    # A large upfront pool ensures all modes of the current distribution
    # are represented.  Each resample step draws a random subset from this
    # pool instead of regenerating a tiny batch, preventing mode collapse
    # and validity degradation.
    #
    # Set --initial_pool_size 0 to disable the pool and use original behaviour.
    #
    # pool_refresh_fraction (0–1): at each resample step, this fraction of
    # the pool is replaced with fresh trajectories from the *current* policy,
    # enabling continual learning while retaining earlier mode coverage.
    initial_pool_size = getattr(args, 'initial_pool_size', 0)
    pool_refresh_fraction = getattr(args, 'pool_refresh_fraction', 0.2)
    x_pool, lr_pool, rw_pool = None, None, None
    if args.no_mcts and initial_pool_size > 0 and initial_pool_size > buffer_size:
        print(f"Generating initial WDCE pool ({initial_pool_size} valid sequences) for mode coverage...")
        _x_chunks, _lr_chunks, _rw_chunks = [], [], []
        _pool_skip = 0
        with torch.no_grad():
            while sum(c.shape[0] for c in _x_chunks) < initial_pool_size:
                x_new, log_rnd_new, rewards_new = policy_model.sample_finetuned_with_rnd(
                    args, reward_model, pretrained)
                if not isinstance(rewards_new, torch.Tensor):
                    rewards_new = torch.as_tensor(
                        rewards_new, dtype=torch.float32, device=x_new.device)
                # Skip all-invalid fallback batches when filtering
                if filter_invalid and rewards_new.abs().sum() == 0:
                    _pool_skip += 1
                    if _pool_skip > 50:
                        print("  [filter_invalid] Too many empty batches — using what we have")
                        break
                    continue
                _x_chunks.append(x_new)
                _lr_chunks.append(log_rnd_new)
                _rw_chunks.append(rewards_new)
            x_pool = torch.cat(_x_chunks, dim=0)[:initial_pool_size]
            lr_pool = torch.cat(_lr_chunks, dim=0)[:initial_pool_size]
            rw_pool = torch.cat(_rw_chunks, dim=0)[:initial_pool_size]
        print(f"  Initial pool: {x_pool.shape[0]} sequences "
              f"(mean reward {rw_pool.mean().item():.4f})")

     ### End of Fine-Tuning Loop ###
    pbar = tqdm(range(args.num_epochs))
    
    for epoch in pbar:
        # store metrics
        rewards = []
        losses = []
        
        policy_model.train()
        
        with torch.no_grad():
            if x_buffer is None or epoch % args.resample_every_n_step == 0:
                if args.no_mcts and x_pool is not None:
                    # -- Rolling refresh: replace a fraction of the pool --
                    n_refresh = int(x_pool.shape[0] * pool_refresh_fraction)
                    if n_refresh > 0:
                        _rx, _rl, _rr = [], [], []
                        _refresh_skip = 0
                        while sum(c.shape[0] for c in _rx) < n_refresh:
                            x_new, lr_new, rw_new = policy_model.sample_finetuned_with_rnd(
                                args, reward_model, pretrained)
                            if not isinstance(rw_new, torch.Tensor):
                                rw_new = torch.as_tensor(rw_new, dtype=torch.float32, device=x_new.device)
                            if filter_invalid and rw_new.abs().sum() == 0:
                                _refresh_skip += 1
                                if _refresh_skip > 50:
                                    break
                                continue
                            _rx.append(x_new); _rl.append(lr_new); _rr.append(rw_new)
                        if len(_rx) == 0:
                            print(f"  [filter_invalid] Pool refresh: no valid batches — keeping pool unchanged")
                        else:
                            x_ref = torch.cat(_rx, dim=0)[:n_refresh]
                            lr_ref = torch.cat(_rl, dim=0)[:n_refresh]
                            rw_ref = torch.cat(_rr, dim=0)[:n_refresh]
                            n_refresh = min(n_refresh, x_ref.shape[0])
                            replace_idx = torch.randperm(x_pool.shape[0])[:n_refresh]
                            x_pool[replace_idx] = x_ref.to(x_pool.device)
                            lr_pool[replace_idx] = lr_ref.to(lr_pool.device)
                            rw_pool[replace_idx] = rw_ref.to(rw_pool.device)
                            print(f"  Pool refreshed: {n_refresh}/{x_pool.shape[0]} sequences replaced "
                                  f"(mean reward {rw_pool.mean().item():.4f})")

                    # Subsample from the (possibly refreshed) pool
                    indices = torch.randperm(x_pool.shape[0])[:buffer_size]
                    x_buffer = x_pool[indices]
                    log_rnd_buffer = lr_pool[indices]
                    rewards_buffer = rw_pool[indices]
                    print(f"  Buffer subsampled from pool: {x_buffer.shape[0]}/{buffer_size}")
                elif args.no_mcts:
                    # Fallback: original behaviour — generate fresh samples
                    x_chunks, lr_chunks, rw_chunks = [], [], []
                    _buf_skip = 0
                    while sum(c.shape[0] for c in x_chunks) < buffer_size:
                        x_new, log_rnd_new, rewards_new = policy_model.sample_finetuned_with_rnd(args, reward_model, pretrained)
                        if not isinstance(rewards_new, torch.Tensor):
                            rewards_new = torch.as_tensor(rewards_new, dtype=torch.float32, device=x_new.device)
                        if filter_invalid and rewards_new.abs().sum() == 0:
                            _buf_skip += 1
                            if _buf_skip > 50:
                                print("  [filter_invalid] Too many empty batches — using what we have")
                                break
                            continue
                        _buf_skip = 0  # reset on success
                        x_chunks.append(x_new)
                        lr_chunks.append(log_rnd_new)
                        rw_chunks.append(rewards_new)
                    x_buffer = torch.cat(x_chunks, dim=0)[:buffer_size]
                    log_rnd_buffer = torch.cat(lr_chunks, dim=0)[:buffer_size]
                    rewards_buffer = torch.cat(rw_chunks, dim=0)[:buffer_size]
                    print(f"  Buffer filled: {x_buffer.shape[0]}/{buffer_size} sequences")
                else:
                    # MCTS has its own buffer management
                    if (epoch) % args.reset_every_n_step == 0:
                        x_buffer, log_rnd_buffer, rewards_buffer, _, _ = mcts.forward(resetTree=True)
                    else:
                        x_buffer, log_rnd_buffer, rewards_buffer, _, _ = mcts.forward(resetTree=False)

        x_final, log_rnd, final_rewards = x_buffer, log_rnd_buffer, rewards_buffer

        # --- Mini-batch sampling from buffer ---
        mini_batch_size = getattr(args, 'training_mini_batch_size', 0)
        if mini_batch_size > 0 and mini_batch_size < x_final.shape[0]:
            idx = torch.randperm(x_final.shape[0])[:mini_batch_size]
            x_final = x_final[idx]
            log_rnd = log_rnd[idx]
            final_rewards = final_rewards[idx]
                
        # compute wdce loss
        loss = loss_wdce(policy_model, log_rnd, x_final, num_replicates=args.wdce_num_replicates, centering=args.centering)
        
        # gradient descent
        loss.backward()
        
        # optimizer
        if args.grad_clip:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), args.gradnorm_clip)
        
        optim.step()
        optim.zero_grad()
        
        pbar.set_postfix(loss=loss.item())
                
        # sample a eval batch with updated policy to evaluate rewards
        x_eval, scores_dict, valid_fraction = policy_model.sample_finetuned(args, reward_model, batch_size=50, dataframe=False)

        # Per-epoch Vendi removed — computed once per AL iteration in
        # finetune_with_uncertainty.py using the exploration sample set.
        vs = float('nan')
        vs_morgan = float('nan')

        # append per-objective scores to logs
        for name, vals in scores_dict.items():
            if name not in scores_logs:
                scores_logs[name] = []
            scores_logs[name].append(vals)
        valid_fraction_log.append(valid_fraction)
        
        batch_losses.append(loss.cpu().detach().numpy())
        
        losses.append(loss.cpu().detach().numpy())
        losses = np.array(losses)
        
        if args.no_mcts:
            mean_reward_search = final_rewards.mean().item()
            min_reward_search = final_rewards.min().item()
            max_reward_search = final_rewards.max().item()
            median_reward_search = final_rewards.median().item()
        else:
            mean_reward_search = np.mean(final_rewards)
            min_reward_search = np.min(final_rewards)
            max_reward_search = np.max(final_rewards)
            median_reward_search = np.median(final_rewards)
        
        score_summary = " ".join(f"{name} {np.mean(scores_dict[name]):f}" for name in scores_dict)
        print(f"epoch {epoch} valid_frac {valid_fraction:.6f} {score_summary} mean loss {np.mean(losses):f}")

        log_dict = {"epoch": epoch, "valid_fraction": valid_fraction,
                    "mean_loss": np.mean(losses),
                    "mean_reward_search": mean_reward_search, "min_reward_search": min_reward_search,
                    "max_reward_search": max_reward_search, "median_reward_search": median_reward_search}
        for name in scores_dict:
            log_dict[name] = np.mean(scores_dict[name])
        # Also log raw (un-normalized) task scores if the reward model caches them
        if hasattr(reward_model, '_last_raw_task_scores') and reward_model._last_raw_task_scores is not None:
            raw_names = reward_model.task_reward.score_func_names if hasattr(reward_model, 'task_reward') else []
            raw_means = reward_model._last_raw_task_scores.mean(axis=0)
            for rname, rval in zip(raw_names, raw_means):
                log_dict[f'{rname}_raw'] = rval
        wandb.log(log_dict)
        
        if (epoch+1) % args.save_every_n_epochs == 0:
            model_path = os.path.join(args.save_path, f'model_{filename}_epoch{epoch}.ckpt')
            torch.save(policy_model.state_dict(), model_path)
            print(f"model saved at epoch {epoch}")
    
    ### End of Fine-Tuning Loop ###
    
    # NOTE: do NOT call wandb.finish() here — the caller (e.g. the active
    # learning loop in finetune_with_uncertainty.py) is responsible for
    # finishing the wandb run.  Calling it here kills the session mid-loop.
    
    # save logs
    os.makedirs(f'{base_path}/results/{args.run_name}', exist_ok=True)
    output_log_path = f'{base_path}/results/{args.run_name}/log_{filename}.csv'
    save_logs_to_file(valid_fraction_log, scores_logs, vendi_score_log, output_log_path, vendi_morgan_log)

    x_eval, scores_dict_final, valid_fraction, df = policy_model.sample_finetuned(args, reward_model, batch_size=200, dataframe=True)
    df.to_csv(f'{base_path}/results/{args.run_name}/{prot_name}_generation_results.csv', index=False)

    return batch_losses

def save_logs_to_file(valid_fraction_log, scores_logs, vendi_score_log, output_path, vendi_morgan_log=None):
    """
    Saves the logs to a CSV file.  Works with any number of objectives.

    Parameters:
        valid_fraction_log (list): Log of valid fractions over iterations.
        scores_logs (dict): {obj_name: [per-epoch values]}.
        vendi_score_log (list): Log of Vendi Score over iterations.
        output_path (str): Path to save the log CSV file.
        vendi_morgan_log (list, optional): Log of Morgan fingerprint Vendi Score over iterations.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Combine logs into a DataFrame. Vendi columns may be empty now that
    # per-epoch Vendi is gone — only include them if their length matches.
    n_rows = len(valid_fraction_log)
    log_data = {
        "Iteration": list(range(1, n_rows + 1)),
        "Valid Fraction": valid_fraction_log,
    }
    for name, vals in scores_logs.items():
        if len(vals) == n_rows:
            log_data[name] = vals
    if len(vendi_score_log) == n_rows:
        log_data["Vendi Score"] = vendi_score_log
    if vendi_morgan_log is not None and len(vendi_morgan_log) == n_rows:
        log_data["Vendi Morgan"] = vendi_morgan_log

    df = pd.DataFrame(log_data)

    # Save to CSV
    df.to_csv(output_path, index=False)