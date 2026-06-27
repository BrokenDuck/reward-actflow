#!/usr/bin/env python
"""Generate 1000 VALID sequences per variant and compute diversity metrics:
  - Vendi (PeptideCLM RBF, sigma=0.25 and sigma=0.50) — valid sequences only
  - FID on Morgan fingerprints (vs pretrained reference)
  - FID on PeptideCLM embeddings (vs pretrained reference)
  - Number of clusters via sphere-exclusion on PeptideCLM embeddings (thr 0.1)

Variants compared: pretrained + each model_99.ckpt under
  checkpoints/tfr_{uncertainty_only,continued_pretraining,continued_pretraining_filtered}_alpha0.005_*
"""

import argparse
import datetime
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd
import torch

# resolve repo-internal imports relative to this file: src/ holds the model
# code, the repo root holds the orchestration modules
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE, 'src'))
sys.path.insert(0, BASE)

from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from diffusion import Diffusion
from utils.app import PeptideAnalyzer
from utils.utils import set_seed
from transformers import AutoModelForMaskedLM
from tokenizer.my_tokenizers import SMILES_SPE_Tokenizer

from metrics import (
    compute_fid_fingerprint,
    sequences_to_fps,
)


# ---------------------------------------------------------------------------
# Model loading + sampling
# ---------------------------------------------------------------------------

def load_model_lightning(ckpt_path, cfg, device):
    model = Diffusion.load_from_checkpoint(
        ckpt_path, config=cfg, strict=False, map_location=device)
    return model.eval().to(device)


def load_model_plain(ckpt_path, cfg, device, pretrained_ckpt):
    model = Diffusion.load_from_checkpoint(
        pretrained_ckpt, config=cfg, strict=False, map_location=device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    for prefix in ('module.', 'model.'):
        if all(k.startswith(prefix) for k in state.keys()):
            state = {k[len(prefix):]: v for k, v in state.items()}
            break
    model.load_state_dict(state, strict=False)
    return model.eval().to(device)


@torch.no_grad()
def _sample_batch(model, bs, seq_length, total_num_steps, noise_removal, device):
    eps = 1e-5
    x = model.sample_prior(bs, seq_length).to(device, dtype=torch.long)
    timesteps = torch.linspace(1, eps, total_num_steps + 1, device=device)
    dt = torch.tensor((1 - eps) / total_num_steps, device=device)
    for i in range(total_num_steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
        _, x = model.single_reverse_step(x, t=t, dt=dt)
        x = x.to(device)
    if noise_removal:
        if (x == model.mask_index).any().item():
            t = timesteps[-2] * torch.ones(x.shape[0], 1, device=device)
            _, x = model.single_noise_removal(x, t=t, dt=dt)
            x = x.to(device)
    return model.tokenizer.batch_decode(x)


@torch.no_grad()
def generate_n_valid(model, n_valid_target, seq_length, total_num_steps,
                     noise_removal, device, batch_size, max_attempts):
    """Keep sampling batches until we have n_valid_target valid peptide SMILES.

    Returns
    -------
    valid_smiles : list[str]
    valid_aa     : list[str]
    n_total_drawn : int
    """
    analyzer = PeptideAnalyzer()
    valid_smiles, valid_aa = [], []
    n_total = 0
    batch_valid_rates = []  # per-batch validity rates
    first_batch_raw = None  # debug: keep the first batch verbatim
    first_batch_is_peptide = None
    first_batch_aa_not_none = None
    for attempt in range(max_attempts):
        if len(valid_smiles) >= n_valid_target:
            break
        smis = _sample_batch(model, batch_size, seq_length, total_num_steps,
                             noise_removal, device)
        batch_valid = 0
        # Pass 1: classify each smi (debug: track is_peptide and analyze_structure separately)
        is_p_flags, aa_ok_flags, aa_strs = [], [], []
        for smi in smis:
            n_total += 1
            is_p = analyzer.is_peptide(smi)
            try:
                aa, _ = analyzer.analyze_structure(smi)
            except Exception:
                aa = None
            is_p_flags.append(is_p)
            aa_ok_flags.append(aa is not None)
            aa_strs.append(aa)
            if is_p and aa is not None:
                valid_smiles.append(smi)
                valid_aa.append(aa)
                batch_valid += 1
        batch_valid_rates.append(batch_valid / max(len(smis), 1))

        # DEBUG: snapshot the very first batch
        if attempt == 0:
            first_batch_raw = list(smis)
            first_batch_is_peptide = list(is_p_flags)
            first_batch_aa_not_none = list(aa_ok_flags)
            n_isp = sum(is_p_flags)
            n_aa = sum(aa_ok_flags)
            print(f'    [DEBUG attempt 1] drawn={len(smis)}  '
                  f'is_peptide=True: {n_isp}  '
                  f'analyze_structure!=None: {n_aa}  '
                  f'BOTH (kept): {batch_valid}')
            print(f'    [DEBUG] first 5 raw SMILES (truncated to 120 chars):')
            for k, smi in enumerate(smis[:5]):
                print(f'      [{k}] is_p={is_p_flags[k]}  aa_ok={aa_ok_flags[k]}  '
                      f'smi={smi[:120]}{"..." if len(smi) > 120 else ""}')

        if (attempt + 1) % 5 == 0 or len(valid_smiles) >= n_valid_target:
            print(f'    attempt {attempt + 1}: {len(valid_smiles)}/{n_valid_target} valid '
                  f'({n_total} drawn, batch_val_rate={batch_valid_rates[-1]:.3f}, '
                  f'mean_val_rate={float(np.mean(batch_valid_rates)):.3f})')

        # Early exit: if the *first* batch produced zero valid peptides AND
        # zero is_peptide=True (i.e. truly degenerate), bail out — variants
        # like continued_pretraining (unfiltered) have near-zero validity.
        if attempt == 0 and len(valid_smiles) == 0:
            print(f'    attempt 1: 0/{n_valid_target} valid '
                  f'({n_total} drawn) — initial validity is 0, aborting early')
            break
    valid_smiles = valid_smiles[:n_valid_target]
    valid_aa = valid_aa[:n_valid_target]
    mean_batch_valid_rate = float(np.mean(batch_valid_rates)) if batch_valid_rates else 0.0
    debug = {
        'first_batch_raw': first_batch_raw,
        'first_batch_is_peptide': first_batch_is_peptide,
        'first_batch_aa_not_none': first_batch_aa_not_none,
    }
    return valid_smiles, valid_aa, n_total, batch_valid_rates, mean_batch_valid_rate, debug


# ---------------------------------------------------------------------------
# PeptideCLM
# ---------------------------------------------------------------------------

def load_peptideclm(device):
    roformer = AutoModelForMaskedLM.from_pretrained(
        'aaronfeller/PeptideCLM-23M-all').roformer.eval().to(device)
    tok_base = os.path.join(BASE, 'src', 'tokenizer')
    tokenizer = SMILES_SPE_Tokenizer(
        os.path.join(tok_base, 'new_vocab.txt'),
        os.path.join(tok_base, 'new_splits.txt'),
    )
    return roformer, tokenizer


@torch.no_grad()
def extract_pclm_embeddings(roformer, tokenizer, smiles_list, device, batch_size=32):
    all_emb = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        encs = [tokenizer(smi, return_tensors='pt') for smi in batch]
        max_len = max(e['input_ids'].shape[1] for e in encs)
        ids_list, mask_list = [], []
        for enc in encs:
            ids = enc['input_ids'].squeeze(0)
            mask = enc['attention_mask'].squeeze(0)
            pad = max_len - ids.shape[0]
            if pad > 0:
                ids = torch.cat([ids, torch.zeros(pad, dtype=torch.long)])
                mask = torch.cat([mask, torch.zeros(pad, dtype=torch.long)])
            ids_list.append(ids)
            mask_list.append(mask)
        input_ids = torch.stack(ids_list).to(device)
        attn_mask = torch.stack(mask_list).to(device)
        out = roformer(input_ids=input_ids, attention_mask=attn_mask)
        h = out.last_hidden_state
        m = attn_mask.unsqueeze(-1).float()
        emb = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        all_emb.append(emb.cpu().numpy())
    return np.concatenate(all_emb, axis=0)


# ---------------------------------------------------------------------------
# Diversity metrics
# ---------------------------------------------------------------------------

def _vendi_score_K(K):
    n = K.shape[0]
    if n <= 1:
        return 1.0
    Ksym = 0.5 * (K + K.T)
    w = np.linalg.eigvalsh(Ksym) / n
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        return 1.0
    w = w / s
    eps = 1e-12
    nz = w > eps
    H = -np.sum(w[nz] * np.log(w[nz]))
    return float(np.exp(H))


def vendi_pclm_rbf(embeddings, sigmas):
    if embeddings.shape[0] <= 1:
        return {f'vendi_pclm_rbf_sigma{s}': 1.0 for s in sigmas}
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-9, None)
    normed = embeddings / norms
    cos = normed @ normed.T
    d2 = np.clip(2.0 * (1.0 - cos), 0.0, None)
    out = {}
    for s in sigmas:
        K = np.exp(-d2 / (2.0 * s * s))
        out[f'vendi_pclm_rbf_sigma{s}'] = _vendi_score_K(K)
    return out


def n_clusters_pclm(embeddings, threshold):
    """Greedy sphere-exclusion (Leader algorithm) on L2-normalised cosine
    distance over PCLM embeddings. Returns the number of cluster centres.
    Mirrors `_compute_pclm_clusters`/`_sphere_exclusion` in the plot scripts."""
    n = embeddings.shape[0]
    if n == 0:
        return 0
    norms = np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-9, None)
    emb = embeddings / norms
    dists = 1.0 - (emb @ emb.T)
    available = np.ones(n, dtype=bool)
    n_centers = 0
    for i in range(n):
        if not available[i]:
            continue
        n_centers += 1
        available &= ~(dists[i] < threshold)
        available[i] = False
    return n_centers


def fid_pclm(gen_emb, ref_emb):
    """Fréchet distance in PeptideCLM embedding space (no PCA — D=768)."""
    from scipy.linalg import sqrtm
    if gen_emb.shape[0] < 2 or ref_emb.shape[0] < 2:
        return 0.0
    mu_g, mu_r = gen_emb.mean(axis=0), ref_emb.mean(axis=0)
    sigma_g = np.cov(gen_emb, rowvar=False) + np.eye(gen_emb.shape[1]) * 1e-6
    sigma_r = np.cov(ref_emb, rowvar=False) + np.eye(ref_emb.shape[1]) * 1e-6
    diff = mu_g - mu_r
    mean_term = float(diff @ diff)
    sqrt_prod = sqrtm(sigma_g @ sigma_r)
    if np.iscomplexobj(sqrt_prod):
        sqrt_prod = sqrt_prod.real
    return float(mean_term + np.trace(sigma_g + sigma_r - 2.0 * sqrt_prod))


# ---------------------------------------------------------------------------
# Per-variant evaluation
# ---------------------------------------------------------------------------

def sample_variant(name, ckpt_path, loader, cfg, pretrained_ckpt, args, device, output_dir):
    """Load model + sample valid sequences. PeptideCLM must NOT be loaded yet —
    `from_pretrained('aaronfeller/PeptideCLM-23M-all')` overwrites
    SMILES_SPE_Tokenizer.batch_decode/decode on the class itself, breaking
    decoding for every Diffusion model loaded in the same process."""
    print(f"\n{'=' * 70}\n  Variant: {name}\n  Ckpt: {ckpt_path}\n{'=' * 70}")
    t0 = time.perf_counter()
    if loader == 'lightning':
        model = load_model_lightning(ckpt_path, cfg, device)
    else:
        model = load_model_plain(ckpt_path, cfg, device, pretrained_ckpt)
    print(f'  Model loaded in {time.perf_counter() - t0:.1f}s')

    print(f'  Generating until {args.n_valid} VALID sequences (batch={args.batch_size})...')
    t0 = time.perf_counter()
    valid_smiles, valid_aa, n_drawn, batch_valid_rates, mean_batch_valid_rate, debug = generate_n_valid(
        model, args.n_valid, args.seq_length, args.total_num_steps,
        args.noise_removal, device, args.batch_size, args.max_attempts,
    )
    gen_time = time.perf_counter() - t0
    print(f'  Got {len(valid_smiles)}/{args.n_valid} valid in {gen_time:.1f}s '
          f'({n_drawn} drawn, mean_batch_val_rate={mean_batch_valid_rate:.3f}, '
          f'cumulative_val_rate={len(valid_smiles) / max(n_drawn, 1):.3f})')

    if debug['first_batch_raw'] is not None:
        debug_path = os.path.join(output_dir, f'{name}_first_batch_raw.csv')
        pd.DataFrame({
            'smiles': debug['first_batch_raw'],
            'is_peptide': debug['first_batch_is_peptide'],
            'analyze_structure_not_none': debug['first_batch_aa_not_none'],
        }).to_csv(debug_path, index=False)
        print(f'  [DEBUG] First-batch raw SMILES saved → {debug_path}')

    del model
    torch.cuda.empty_cache()

    return {
        'name': name,
        'valid_smiles': valid_smiles,
        'valid_aa': valid_aa,
        'n_drawn': n_drawn,
        'batch_valid_rates': batch_valid_rates,
        'mean_batch_valid_rate': mean_batch_valid_rate,
        'gen_time_sec': gen_time,
    }


def compute_variant_metrics(sample, args, device, pclm_roformer, pclm_tokenizer, output_dir):
    """Compute PCLM embeddings + diversity metrics for a sampled variant."""
    name = sample['name']
    valid_smiles = sample['valid_smiles']
    valid_aa = sample['valid_aa']
    n_drawn = sample['n_drawn']
    batch_valid_rates = sample['batch_valid_rates']
    mean_batch_valid_rate = sample['mean_batch_valid_rate']
    gen_time = sample['gen_time_sec']

    print(f"\n--- Metrics: {name} ---")

    if len(valid_smiles) < 2:
        print('  Not enough valid sequences — returning 0 for all metrics')
        zero_metrics = {
            'name': name,
            'n_valid': len(valid_smiles),
            'n_drawn': n_drawn,
            'mean_batch_valid_rate': mean_batch_valid_rate,
            'cumulative_valid_rate': len(valid_smiles) / max(n_drawn, 1),
            'batch_valid_rates': batch_valid_rates,
            'n_batches': len(batch_valid_rates),
            'gen_time_sec': gen_time,
        }
        for s in args.sigmas:
            zero_metrics[f'vendi_pclm_rbf_sigma{s}'] = 0.0
        for thr in getattr(args, 'pclm_cluster_thresholds', []):
            zero_metrics[f'n_clusters_pclm_thr{thr}'] = 0
        zero_metrics['fid_morgan_vs_pretrained'] = 0.0
        zero_metrics['fid_pclm_vs_pretrained'] = 0.0
        return zero_metrics, None

    print('  PeptideCLM embeddings...')
    pclm_emb = extract_pclm_embeddings(
        pclm_roformer, pclm_tokenizer, valid_smiles, device,
        batch_size=args.pclm_batch_size)
    print(f'  pclm_emb: {pclm_emb.shape}')

    metrics = {
        'name': name,
        'n_valid': len(valid_smiles),
        'n_drawn': n_drawn,
        'mean_batch_valid_rate': mean_batch_valid_rate,
        'cumulative_valid_rate': len(valid_smiles) / max(n_drawn, 1),
        'batch_valid_rates': batch_valid_rates,
        'n_batches': len(batch_valid_rates),
        'gen_time_sec': gen_time,
    }

    vendi = vendi_pclm_rbf(pclm_emb, sigmas=args.sigmas)
    metrics.update(vendi)
    for k, v in vendi.items():
        print(f'    {k} = {v:.4f}')

    for thr in getattr(args, 'pclm_cluster_thresholds', []):
        n_clust = n_clusters_pclm(pclm_emb, threshold=thr)
        metrics[f'n_clusters_pclm_thr{thr}'] = n_clust
        print(f'    n_clusters_pclm@thr={thr}: {n_clust}')

    out_csv = os.path.join(output_dir, f'{name}_valid_sequences.csv')
    pd.DataFrame({'smiles': valid_smiles, 'aa_sequence': valid_aa}).to_csv(out_csv, index=False)
    np.save(os.path.join(output_dir, f'{name}_pclm_embeddings.npy'), pclm_emb)
    print(f'  Saved → {out_csv}')

    return metrics, pclm_emb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--n_valid', type=int, default=1000,
                   help='Target number of VALID sequences per variant')
    p.add_argument('--seq_length', type=int, default=200)
    p.add_argument('--total_num_steps', type=int, default=128)
    p.add_argument('--noise_removal', action='store_true', default=True)
    p.add_argument('--batch_size', type=int, default=100,
                   help='Sampling batch size')
    p.add_argument('--pclm_batch_size', type=int, default=16)
    p.add_argument('--max_attempts', type=int, default=200,
                   help='Max sampling rounds per variant')
    p.add_argument('--sigmas', type=float, nargs='+', default=[0.25, 0.5])
    p.add_argument('--pclm_cluster_thresholds', type=float, nargs='*',
                   default=[0.1],
                   help='Sphere-exclusion thresholds (cosine distance) for '
                        'clustering on PCLM embeddings -> n_clusters_pclm_thr{thr}. '
                        'Pass an empty list to disable.')
    p.add_argument('--seed', type=int, default=42,
                   help='(legacy) single seed; ignored if --seeds is provided')
    p.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45, 46],
                   help='Seeds to run the full generate+evaluate pipeline with. '
                        '95% CIs are computed across these replicates.')
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--pretrained_ckpt', type=str,
                   default=os.path.join(BASE, 'src/pretrained/peptune-pretrained.ckpt'))
    p.add_argument('--uncertainty_ckpt', type=str,
                   default=os.path.join(BASE, 'checkpoints/for_eval/tfr_uncertainty_only_alpha0.005_20260502_065059/model_99.ckpt'))
    p.add_argument('--continued_ckpt', type=str,
                   default=os.path.join(BASE, 'checkpoints/for_eval/tfr_continued_pretraining_alpha0.005_20260502_065058/model_99.ckpt'))
    p.add_argument('--continued_filtered_ckpt', type=str,
                   default=os.path.join(BASE, 'checkpoints/for_eval/tfr_continued_pretraining_filtered_alpha0.005_20260502_065059/model_99.ckpt'))
    p.add_argument('--variant_names', type=str, nargs='*', default=None,
                   help='Override default variant set: list of names matching --variant_ckpts. '
                        'When provided, replaces the (uncertainty_only, continued_pretraining, '
                        'continued_pretraining_filtered) trio. Pretrained is always included. '
                        'Pass with no values (e.g. --variant_names --variant_ckpts) to evaluate '
                        'ONLY the pretrained model.')
    p.add_argument('--variant_ckpts', type=str, nargs='*', default=None,
                   help='Checkpoint paths matching --variant_names (parallel arrays).')
    p.add_argument('--zero_validity_variants', type=str, nargs='*', default=[],
                   help='Variant names to skip generation for and record as 0-validity '
                        '(all metrics emitted as 0). Avoids spending hours sampling from '
                        'models known to have near-zero validity.')
    p.add_argument('--output_dir', type=str,
                   default=os.path.join(BASE, 'results/diversity_eval'))
    p.add_argument('--tag', type=str, default='')
    return p.parse_args()


def run_one_seed(seed, seed_output_dir, cfg, variants, args, device):
    """Run a full generate+evaluate pass for a single seed. Returns
    {variant_name: metrics_dict}. metrics_dict values are scalars suitable
    for aggregation across seeds."""
    set_seed(seed, use_cuda=True)

    # Phase 1: sample all variants BEFORE loading PeptideCLM. Loading PCLM via
    # AutoModelForMaskedLM.from_pretrained replaces SMILES_SPE_Tokenizer.batch_decode
    # on the class itself, which breaks Diffusion sample decoding.
    zero_validity = set(args.zero_validity_variants or [])
    samples = {}
    for name, ckpt, loader in variants:
        if name in zero_validity:
            print(f'\n[zero-validity] {name} — skipping generation, emitting 0-validity row')
            samples[name] = {
                'name': name,
                'valid_smiles': [],
                'valid_aa': [],
                'n_drawn': 0,
                'batch_valid_rates': [],
                'mean_batch_valid_rate': 0.0,
                'gen_time_sec': 0.0,
            }
            continue
        if not os.path.exists(ckpt):
            print(f'\n[skip] {name} — checkpoint missing: {ckpt}')
            continue
        samples[name] = sample_variant(
            name, ckpt, loader, cfg, args.pretrained_ckpt, args, device, seed_output_dir)

    # Phase 2: load PeptideCLM and compute embedding-based metrics.
    print('\nLoading PeptideCLM...')
    pclm_roformer, pclm_tokenizer = load_peptideclm(device)

    all_metrics = {}
    valid_aa_by_variant = {}
    pclm_by_variant = {}

    for name, sample in samples.items():
        m, pclm = compute_variant_metrics(
            sample, args, device, pclm_roformer, pclm_tokenizer, seed_output_dir)
        all_metrics[name] = m
        valid_aa_by_variant[name] = sample['valid_aa']
        pclm_by_variant[name] = pclm

    # ---- FID vs pretrained reference ----
    if 'pretrained' in valid_aa_by_variant and len(valid_aa_by_variant['pretrained']) >= 2:
        ref_aa = valid_aa_by_variant['pretrained']
        ref_pclm = pclm_by_variant['pretrained']
        print(f'\n--- FID vs pretrained (n_ref={len(ref_aa)}) ---')
        for name in all_metrics:
            if name == 'pretrained':
                all_metrics[name]['fid_morgan_vs_pretrained'] = 0.0
                all_metrics[name]['fid_pclm_vs_pretrained'] = 0.0
                continue
            gen_aa = valid_aa_by_variant.get(name, [])
            gen_pclm = pclm_by_variant.get(name)
            if len(gen_aa) >= 2:
                t0 = time.perf_counter()
                fid_morgan = compute_fid_fingerprint(gen_aa, ref_aa)
                fid_pclm_v = fid_pclm(gen_pclm, ref_pclm) if gen_pclm is not None else 0.0
                print(f'  {name}: FID_morgan={fid_morgan:.4f}  '
                      f'FID_pclm={fid_pclm_v:.4f}  ({time.perf_counter() - t0:.1f}s)')
                all_metrics[name]['fid_morgan_vs_pretrained'] = fid_morgan
                all_metrics[name]['fid_pclm_vs_pretrained'] = fid_pclm_v
            else:
                print(f'  {name}: 0 valid → FID = 0')
                all_metrics[name]['fid_morgan_vs_pretrained'] = 0.0
                all_metrics[name]['fid_pclm_vs_pretrained'] = 0.0
    else:
        print('\n[warn] Pretrained variant unavailable — skipping FID computation')

    # Free PeptideCLM before next seed
    del pclm_roformer, pclm_tokenizer, pclm_by_variant
    torch.cuda.empty_cache()

    # Persist this seed's per-variant metrics
    with open(os.path.join(seed_output_dir, 'diversity_results.json'), 'w') as f:
        json.dump(all_metrics, f, indent=2)
    pd.DataFrame(list(all_metrics.values())).to_csv(
        os.path.join(seed_output_dir, 'diversity_summary.csv'), index=False)

    return all_metrics


def aggregate_across_seeds(per_seed_metrics, seeds):
    """per_seed_metrics: list of {variant_name: metrics_dict}, one per seed.
    Returns {variant_name: {metric_key: {mean, std, ci95_low, ci95_high, n, values}}}.
    Only numeric scalar metrics are aggregated."""
    skip_keys = {'name', 'batch_valid_rates'}
    variant_names = []
    for d in per_seed_metrics:
        for name in d:
            if name not in variant_names:
                variant_names.append(name)

    agg = {}
    for name in variant_names:
        per_metric_values = {}
        for d in per_seed_metrics:
            if name not in d:
                continue
            for k, v in d[name].items():
                if k in skip_keys:
                    continue
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                per_metric_values.setdefault(k, []).append(float(v))

        agg[name] = {}
        for k, vals in per_metric_values.items():
            arr = np.asarray(vals, dtype=float)
            n = arr.size
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            # 95% CI: percentile bootstrap when n>=5 else normal-approx (t-ish)
            if n >= 5:
                lo = float(np.percentile(arr, 2.5))
                hi = float(np.percentile(arr, 97.5))
            elif n > 1:
                # 1.96 * SE — coarse but fine as a fallback
                se = std / np.sqrt(n)
                lo, hi = mean - 1.96 * se, mean + 1.96 * se
            else:
                lo, hi = mean, mean
            agg[name][k] = {
                'mean': mean, 'std': std,
                'ci95_low': lo, 'ci95_high': hi,
                'n_seeds': n, 'values': vals, 'seeds': list(seeds[:n]),
            }
    return agg


_VARIANT_RE = re.compile(r'^seed(\d+)_(.+)$')


def aggregate_by_training_seed(agg):
    """Group variants named `seed{N}_{group}` by `{group}` and compute mean +
    95% CI across the training seeds in each group. Pulls the per-variant
    scalar from `agg[name][k]['mean']` (which is itself the mean across
    generation seeds — usually a single value here).

    Returns {group: {'seeds': [...], 'metrics': {k: {mean, std, ci95_low,
                                                      ci95_high, n_seeds, values}}}}.
    """
    grouped = {}
    for name, metric_map in agg.items():
        m = _VARIANT_RE.match(name)
        if not m:
            continue  # skip 'pretrained' and any non-seed-prefixed entries
        seed_num, group = int(m.group(1)), m.group(2)
        bucket = grouped.setdefault(group, {'seeds': [], 'per_metric': {}})
        if seed_num not in bucket['seeds']:
            bucket['seeds'].append(seed_num)
        for k, stats in metric_map.items():
            bucket['per_metric'].setdefault(k, []).append(float(stats['mean']))

    out = {}
    for group, info in grouped.items():
        out[group] = {'seeds': sorted(info['seeds']), 'metrics': {}}
        for k, vals in info['per_metric'].items():
            arr = np.asarray(vals, dtype=float)
            n = arr.size
            mean = float(arr.mean())
            std = float(arr.std(ddof=1)) if n > 1 else 0.0
            if n >= 5:
                lo = float(np.percentile(arr, 2.5))
                hi = float(np.percentile(arr, 97.5))
            elif n > 1:
                se = std / np.sqrt(n)
                lo, hi = mean - 1.96 * se, mean + 1.96 * se
            else:
                lo, hi = mean, mean
            out[group]['metrics'][k] = {
                'mean': mean, 'std': std,
                'ci95_low': lo, 'ci95_high': hi,
                'n_seeds': n, 'values': vals,
            }
    return out


def main():
    args = parse_args()
    seeds = args.seeds if args.seeds else [args.seed]
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f'_{args.tag}' if args.tag else ''
    output_dir = os.path.join(args.output_dir, f'run_{timestamp}{tag}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'Output: {output_dir}')
    print(f'Seeds:  {seeds}  (n={len(seeds)} replicates for 95% CI)')

    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    # Hydra config (shared across loaders + seeds)
    GlobalHydra.instance().clear()
    initialize(version_base='1.1',
               config_path='src/configs',
               job_name='evaluate_diversity')
    cfg = compose(config_name='peptune_config.yaml')

    if args.variant_names is not None or args.variant_ckpts is not None:
        if args.variant_names is None or args.variant_ckpts is None:
            raise ValueError('--variant_names and --variant_ckpts must be provided together')
        if len(args.variant_names) != len(args.variant_ckpts):
            raise ValueError(f'--variant_names ({len(args.variant_names)}) and '
                             f'--variant_ckpts ({len(args.variant_ckpts)}) must have equal length')
        variants = [('pretrained', args.pretrained_ckpt, 'lightning')]
        for name, ckpt in zip(args.variant_names, args.variant_ckpts):
            variants.append((name, ckpt, 'plain'))
    else:
        variants = [
            ('pretrained',                      args.pretrained_ckpt,        'lightning'),
            ('uncertainty_only',                args.uncertainty_ckpt,        'plain'),
            ('continued_pretraining',           args.continued_ckpt,          'plain'),
            ('continued_pretraining_filtered',  args.continued_filtered_ckpt, 'plain'),
        ]

    per_seed_metrics = []
    for i, seed in enumerate(seeds):
        print('\n' + '#' * 80)
        print(f'#  SEED {seed}  ({i + 1}/{len(seeds)})')
        print('#' * 80)
        seed_dir = os.path.join(output_dir, f'seed_{seed}')
        os.makedirs(seed_dir, exist_ok=True)
        m = run_one_seed(seed, seed_dir, cfg, variants, args, device)
        per_seed_metrics.append(m)

    # ---- Aggregate across seeds: mean / std / 95% CI per (variant, metric) ----
    agg = aggregate_across_seeds(per_seed_metrics, seeds)

    with open(os.path.join(output_dir, 'diversity_results_aggregated.json'), 'w') as f:
        json.dump(agg, f, indent=2)

    # Flat CSV: one row per (variant, metric)
    rows = []
    for name, metric_map in agg.items():
        for k, stats in metric_map.items():
            rows.append({
                'variant': name, 'metric': k,
                'mean': stats['mean'], 'std': stats['std'],
                'ci95_low': stats['ci95_low'], 'ci95_high': stats['ci95_high'],
                'n_seeds': stats['n_seeds'],
                'values': stats['values'],
            })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_dir, 'diversity_summary_ci.csv'), index=False)

    print('\n' + '=' * 80)
    print(f'SUMMARY — per variant (mean ± half-95%-CI across {len(seeds)} gen seeds)')
    print('=' * 80)
    for name, metric_map in agg.items():
        print(f'\n[{name}]')
        for k, s in metric_map.items():
            half = 0.5 * (s['ci95_high'] - s['ci95_low'])
            print(f'  {k:42s}  {s["mean"]:.4f} ± {half:.4f}  '
                  f'(95% CI: [{s["ci95_low"]:.4f}, {s["ci95_high"]:.4f}], n={s["n_seeds"]})')

    # ---- Aggregate across TRAINING seeds (seed50/51/52/...) per variant group ----
    by_type = aggregate_by_training_seed(agg)
    if by_type:
        type_rows = []
        for group, info in by_type.items():
            for k, s in info['metrics'].items():
                half = 0.5 * (s['ci95_high'] - s['ci95_low'])
                type_rows.append({
                    'variant_type': group,
                    'metric': k,
                    'mean': s['mean'],
                    'std': s['std'],
                    'ci95_low': s['ci95_low'],
                    'ci95_high': s['ci95_high'],
                    'half_ci95': half,
                    'n_seeds': s['n_seeds'],
                    'training_seeds': info['seeds'],
                    'values': s['values'],
                })
        df_type = pd.DataFrame(type_rows)
        df_type.to_csv(os.path.join(output_dir, 'diversity_summary_by_type.csv'), index=False)
        with open(os.path.join(output_dir, 'diversity_results_by_type.json'), 'w') as f:
            json.dump(by_type, f, indent=2)

        print('\n' + '=' * 80)
        print('SUMMARY — by variant type (mean ± half-95%-CI across training seeds)')
        print('=' * 80)
        for group, info in by_type.items():
            print(f'\n[{group}]  training_seeds={info["seeds"]}  (n={len(info["seeds"])})')
            for k, s in info['metrics'].items():
                half = 0.5 * (s['ci95_high'] - s['ci95_low'])
                print(f'  {k:42s}  {s["mean"]:.4f} ± {half:.4f}  '
                      f'(95% CI: [{s["ci95_low"]:.4f}, {s["ci95_high"]:.4f}], n={s["n_seeds"]})')

        # Compact wide table: rows = metric, columns = variant_type with "mean ± half_CI"
        wide_rows = []
        all_metrics = []
        for info in by_type.values():
            for k in info['metrics']:
                if k not in all_metrics:
                    all_metrics.append(k)
        for k in all_metrics:
            row = {'metric': k}
            for group, info in by_type.items():
                s = info['metrics'].get(k)
                if s is None:
                    row[group] = ''
                else:
                    half = 0.5 * (s['ci95_high'] - s['ci95_low'])
                    row[group] = f'{s["mean"]:.4f} ± {half:.4f}'
            wide_rows.append(row)
        df_wide = pd.DataFrame(wide_rows)
        df_wide.to_csv(os.path.join(output_dir, 'diversity_summary_by_type_wide.csv'), index=False)

        print('\n' + '-' * 80)
        print('TABLE — by variant type (mean ± half-95%-CI)')
        print('-' * 80)
        print(df_wide.to_string(index=False))

    print(f'\nResults saved to {output_dir}')


if __name__ == '__main__':
    main()
