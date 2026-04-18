from typing import Generic, Literal, Optional
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import torch
from diffusiongym import construct_env, D, Reward
from diffusiongym.utils import train_base_model, DDDataset, dict_to_device
from torch.utils.data import DataLoader, Subset
import torch.nn as _nn
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import logging
import yaml
import time
import csv
import os
import signal
import wandb

from .inf_methods.dps import RewardGradient
from .uncertainty import UncertaintyEstimator, FlowFeatureExtractor, uncertainty_estimators
from .setups import setups as problem_setups
from .setups.problem_setup import ProblemSetup
from .utils import write_video, filter_out_invalids, filter_out_valids, Batch, serialize_args, setup_logger


@dataclass(frozen=True)
class ExploreConfig:
    # Experiment directory
    folder: Path

    # Exploration sampling
    num_iters: int = 1000
    samples_per_iter: int = 64
    sample_batch_size: int = 64
    num_steps: int = 100

    # Feature extraction
    feat_timestep: float = 0.9
    dps_weight: float = 100.0

    # Uncertainty reward weight
    mean_weight: float = 0.0
    reward_opt: Literal["max", "min"] = "max"

    # Fine-tuning
    ft_min_dataset_size: int = 64
    ft_batch_size: int = 64
    ft_accumulate_steps: int = 1
    ft_steps: int = 500
    ft_lr: float = 1e-4
    ft_weight_decay: float = 0.0

    # Checkpointing
    ckpt_every: int = 100

    # Sampling and evaluation
    eval_samples: int = 0
    eval_valid_samples: int = 0
    eval_batch_size: int = 64
    eval_every: int = 10
    video_fps: int = 4

    # Flags
    no_uncertainty: bool = False
    no_verifier: bool = False
    compute_vendi: bool = False

    # Warmup caching
    warmup_cache_dir: Optional[Path] = None

    # Regularization data from base model distribution
    reg_data: bool = False
    alpha_reg: float = 1.0
    n_reg_samples: int = 10000

    # Combined pos+neg finetuning with gradient norm scaling
    neg_grad_scale: float = 0.0
    max_neg_samples: int = 5000

    def __post_init__(self):
        # Create experiment directory if it doesn't exist
        self.folder.mkdir(parents=True, exist_ok=True)

        # Validation
        if not (0 <= self.feat_timestep <= 1):
            raise ValueError(f"feat_timestep must be in [0, 1], got {self.feat_timestep}")

        if self.dps_weight < 0:
            raise ValueError(f"dps_weight cannot be negative, got {self.dps_weight}")

    @staticmethod
    def construct_from_args(args: argparse.Namespace | dict) -> "ExploreConfig":
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        # Map argparse flags to config names
        name_mapping = { "dir": "folder" }

        # We only take keys that exist in the dataclass fields
        config_fields = {f.name for f in ExploreConfig.__dataclass_fields__.values()}
        clean_kwargs = {}

        for key, value in args.items():
            # Check if the key needs a rename
            target_key = name_mapping.get(key, key)

            # Only add if it's a valid field in our config
            if target_key in config_fields:
                clean_kwargs[target_key] = value

        return ExploreConfig(**clean_kwargs)


class ExploreLoop(Generic[D]):
    def __init__(
        self,
        problem_setup: ProblemSetup[D],
        uncertainty: UncertaintyEstimator[D],
        reward: Reward,
        config: ExploreConfig,
        logger: logging.Logger,
    ):
        self.problem = problem_setup
        self.config = config
        self.logger = logger

        env = construct_env(
            problem_setup.base_model,
            reward,
            discretization_steps=config.num_steps,
            reward_scale=config.dps_weight,
        )

        self.base_model = problem_setup.base_model
        self.uncertainty = uncertainty
        self.env = env

        self.logger.info(f"base model parameters: {sum(p.numel() for p in self.base_model.parameters() if p.requires_grad):,}")

        self._timings = {}
        self._pretrained_fps = None
        self._cumulative_centers = None
        self._fixed_projection = None

        # Create file and write headers
        self.timing_heads = ["iteration", "eval", "sampling", "visualize", "finetune", "uncertainty_update", "checkpoint"]
        self.timing_path = self.config.folder / "timings.csv"
        with open(self.timing_path, "w", newline="") as f:
            csv.writer(f).writerow(self.timing_heads)

    @contextmanager
    def _timer(self, name: str):
        """Context manager to measure execution time of a block."""
        t0 = time.perf_counter()
        yield
        self._timings[name] = time.perf_counter() - t0

    def _flush_timings(self, iteration: int):
        """Writes the stored timings to the CSV file."""
        row = [iteration] + [self._timings.get(head, 0.0) for head in self.timing_heads[1:]]
        with open(self.timing_path, "a", newline="") as f:
            csv.writer(f).writerow(row)

        self._timings = {}

    def finetune_base_model(self, batches: list[Batch[D]], reg_batch: Batch[D] | None = None, pbar: bool = False):
        """Fine-tune the base model with obtained valid samples.

        Parameters
        ----------
        batches : list[Batch[D]]
            Data batches obtained so far, where each entry is from a separate iteration.
        reg_batch : Batch[D] | None
            Regularization data sampled from the base model at init, weighted by alpha_reg.
        pbar : bool, default: False
            Progress bar or not.
        """

        if not self.config.no_verifier:
            valid_batch = filter_out_invalids(batches)
            if len(valid_batch) == 0:
                self.logger.warning("No valid data to fine-tune")
                return
        else:
            valid_batch = Batch.concat(batches)

        device = self.env.base_model.device
        data = [valid_batch.latents.to(device)]
        kw = [valid_batch.kwargs]
        weights = [torch.ones(len(valid_batch))]

        if reg_batch is not None:
            data.append(reg_batch.latents.to(device))
            kw.append(reg_batch.kwargs)
            weights.append(self.config.alpha_reg * torch.ones(len(reg_batch)))

        opt = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.config.ft_lr,
            weight_decay=self.config.ft_weight_decay,
        )
        train_base_model(
            self.base_model,
            opt,
            data,
            kw,
            weights=weights,
            steps=self.config.ft_steps,
            batch_size=self.config.ft_batch_size,
            accumulate_steps=self.config.ft_accumulate_steps,
            pbar=pbar,
        )

    def combined_finetune_base_model(self, batches: list[Batch[D]], reg_batch: Batch[D] | None = None, pbar: bool = False):
        """Fine-tune with combined pos+neg signal using gradient norm scaling.

        Separate dataloaders for valid and invalid data:
        - Positive update sees a full batch of valid samples (identical to finetune_base_model).
        - Negative gradient is rescaled so ||grad_neg|| = neg_grad_scale * ||grad_pos||.

        When neg_grad_scale=0 or there are no invalid samples, this falls back
        to finetune_base_model (identical computational path).
        """
        if self.config.no_verifier:
            self.finetune_base_model(batches, reg_batch=reg_batch, pbar=pbar)
            return

        valid_batch = filter_out_invalids(batches)
        invalid_batch = filter_out_valids(batches)

        m = len(valid_batch) if valid_batch is not None else 0
        k = len(invalid_batch) if invalid_batch is not None else 0

        if m == 0:
            self.logger.warning("Combined FT: no valid data, skipping")
            return

        scale = self.config.neg_grad_scale
        use_neg = scale > 0 and k > 0

        if not use_neg:
            self.finetune_base_model(batches, reg_batch=reg_batch, pbar=pbar)
            return

        max_neg = self.config.max_neg_samples
        k_used = min(k, max_neg)

        self.logger.info(
            f"Combined FT (grad-norm): m={m} k_total={k} k_used={k_used}, "
            f"neg_grad_scale={scale}"
        )

        device = self.env.base_model.device
        bs = self.config.ft_batch_size

        pos_data = [valid_batch.latents.to(device)]
        pos_kw = [valid_batch.kwargs]
        pos_weights = None

        if reg_batch is not None:
            pos_data.append(reg_batch.latents.to(device))
            pos_kw.append(reg_batch.kwargs)
            pos_weights = [
                torch.ones(len(valid_batch)),
                self.config.alpha_reg * torch.ones(len(reg_batch)),
            ]

        pos_dataset = DDDataset(pos_data, pos_kw, pos_weights)
        pos_loader = DataLoader(
            pos_dataset, bs, shuffle=True,
            collate_fn=pos_dataset.collate, num_workers=0, pin_memory=False,
        )

        neg_dataset_full = DDDataset(
            [invalid_batch.latents.to(device)],
            [invalid_batch.kwargs],
            None,
        )
        if k > max_neg:
            subset_idx = torch.randperm(k)[:max_neg].tolist()
            neg_dataset = Subset(neg_dataset_full, subset_idx)
        else:
            neg_dataset = neg_dataset_full
        neg_loader = DataLoader(
            neg_dataset, bs, shuffle=True,
            collate_fn=neg_dataset_full.collate, num_workers=0, pin_memory=False,
        )

        opt = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.config.ft_lr,
            weight_decay=self.config.ft_weight_decay,
        )

        accum = self.config.ft_accumulate_steps

        self.base_model.train()
        opt.zero_grad()
        pos_iter = iter(pos_loader)
        neg_iter = iter(neg_loader)

        accum_grad_pos = {}
        accum_grad_neg = {}

        for step in range(self.config.ft_steps):
            # --- positive micro-batch ---
            try:
                x1_pos, kw_pos, w_pos = next(pos_iter)
            except StopIteration:
                pos_iter = iter(pos_loader)
                x1_pos, kw_pos, w_pos = next(pos_iter)

            x1_pos = x1_pos.to(device)
            kw_pos = dict_to_device(kw_pos, device)

            loss_pos = (w_pos.to(device) * self.base_model.train_loss(x1_pos, **kw_pos)).mean() / accum
            loss_pos.backward()

            for name, p in self.base_model.named_parameters():
                if p.grad is not None:
                    if name in accum_grad_pos:
                        accum_grad_pos[name].add_(p.grad)
                    else:
                        accum_grad_pos[name] = p.grad.clone()

            # --- negative micro-batch ---
            opt.zero_grad()

            try:
                x1_neg, kw_neg, _ = next(neg_iter)
            except StopIteration:
                neg_iter = iter(neg_loader)
                x1_neg, kw_neg, _ = next(neg_iter)

            x1_neg = x1_neg.to(device)
            kw_neg = dict_to_device(kw_neg, device)

            loss_neg = self.base_model.train_loss(x1_neg, **kw_neg).mean() / accum
            loss_neg.backward()

            for name, p in self.base_model.named_parameters():
                if p.grad is not None:
                    if name in accum_grad_neg:
                        accum_grad_neg[name].add_(p.grad)
                    else:
                        accum_grad_neg[name] = p.grad.clone()

            opt.zero_grad()

            if (step + 1) % accum != 0:
                continue

            # --- optimizer step with accumulated gradients ---
            norm_pos = torch.sqrt(sum(g.pow(2).sum() for g in accum_grad_pos.values()))
            norm_neg_val = torch.sqrt(sum(g.pow(2).sum() for g in accum_grad_neg.values())) if accum_grad_neg else torch.tensor(0.0, device=device)

            rescale = (scale * norm_pos) / norm_neg_val.clamp(min=1e-8)
            rescale = rescale.clamp(max=1.0)

            for name, p in self.base_model.named_parameters():
                has_pos = name in accum_grad_pos
                has_neg = name in accum_grad_neg
                if has_pos and has_neg:
                    p.grad = accum_grad_pos[name] - rescale * accum_grad_neg[name]
                elif has_pos:
                    p.grad = accum_grad_pos[name]
                elif has_neg:
                    p.grad = -rescale * accum_grad_neg[name]

            has_nan = any(
                p.grad is not None and torch.isnan(p.grad).any()
                for p in self.base_model.parameters()
            )
            if has_nan:
                self.logger.warning(f"NaN gradient detected at step {step}, skipping optimizer step")
                opt.zero_grad()
                accum_grad_pos = {}
                accum_grad_neg = {}
                continue

            _nn.utils.clip_grad_norm_(self.base_model.parameters(), 0.1)
            opt.step()
            opt.zero_grad()
            accum_grad_pos = {}
            accum_grad_neg = {}

        self.base_model.eval()

    def update_uncertainty_estimator(self, batches: list[Batch[D]]):
        """Update the uncertainty estimator with obtained samples.

        Parameters
        ----------
        batches : list[Batch[D]]
            Data batches obtained so far, where one batch comes from one iteration in the order.
        """
        if self.config.mean_weight > 0:
            scale = -1.0 if self.config.reward_opt == "min" else 1.0
            targets = [scale * b.rewards for b in batches]
        else:
            targets = [b.valids.float() for b in batches]

        self.uncertainty.set_data(
            [b.latents for b in batches],
            targets,
            [b.kwargs for b in batches],
        )

    def visualize_iter(self, batch: Batch[D], iteration: int):
        # Visualizing an iteration is problem-dependent
        fig = self.problem.visualize_sample(self.env, self.uncertainty, batch)

        # Save frame
        directory = self.config.folder / "frames"
        directory.mkdir(parents=True, exist_ok=True)
        fig_path = directory / f"{iteration:04d}.png"
        fig.savefig(fig_path, dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def _generate_or_load_pretrained_fps(self):
        """Generate fingerprints from the pretrained model (before any fine-tuning).

        Cached in warmup_cache_dir so all runs share the same reference.
        Used for FID computation and PCA plots.
        """
        if not hasattr(self.problem, "get_morgan_fingerprints"):
            return
        if self.config.eval_valid_samples <= 0 and self.config.eval_samples <= 0:
            return

        cache_dir = self.config.warmup_cache_dir or self.config.folder
        n = max(self.config.eval_valid_samples, 200)

        self._load_or_generate_fps_cache(cache_dir, "pretrained_fps.npy", n, "_pretrained_fps")

    @torch.no_grad()
    def _load_or_generate_fps_cache(self, cache_dir: Path, filename: str, n_valid_target: int, attr_name: str):
        cache_path = cache_dir / filename

        if cache_path.exists():
            fps = np.load(cache_path)
            setattr(self, attr_name, fps)
            self.logger.info(f"Loaded {attr_name}: {fps.shape[0]} mols from {cache_path}")
            return

        bs = self.config.eval_batch_size
        chunk = max(self.config.eval_samples, 64)
        all_fps = []
        n_valid_so_far = 0

        self.logger.info(f"Generating {attr_name} ({n_valid_target} valid target)...")
        while n_valid_so_far < n_valid_target:
            eval_kwargs = self.problem.eval_sampling_kwargs(chunk)
            sample = self.env.batch_sample(chunk, bs, **eval_kwargs)
            batch = Batch.from_sample(sample)
            batch.valids = self.problem.validity(batch.samples, batch.kwargs)
            chunk_fps = self.problem.get_morgan_fingerprints(batch.samples, batch.kwargs)
            all_fps.append(chunk_fps)
            n_valid_so_far += len(chunk_fps)
            self.logger.info(f"  {attr_name}: {n_valid_so_far}/{n_valid_target} valid mols collected")

        fps = np.vstack(all_fps)[:n_valid_target]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, fps)
        setattr(self, attr_name, fps)
        self.logger.info(f"Saved {attr_name}: {fps.shape[0]} mols to {cache_path}")

    def _get_fixed_projection(self) -> np.ndarray | None:
        """Get or create a fixed random projection matrix for stable PCA plots.

        The projection is a (fp_dim, 2) matrix seeded from the pretrained fps
        so it's deterministic and identical across runs sharing the same cache.
        """
        if self._fixed_projection is not None:
            return self._fixed_projection
        if self._pretrained_fps is None:
            return None

        cache_dir = self.config.warmup_cache_dir or self.config.folder
        proj_path = cache_dir / "fixed_projection.npy"

        if proj_path.exists():
            self._fixed_projection = np.load(proj_path)
            self.logger.info(f"Loaded fixed projection from {proj_path}")
            return self._fixed_projection

        fp_dim = self._pretrained_fps.shape[1]
        rng = np.random.RandomState(42)
        raw = rng.randn(fp_dim, 2).astype(np.float32)
        self._fixed_projection = raw / np.linalg.norm(raw, axis=0, keepdims=True)
        proj_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(proj_path, self._fixed_projection)
        self.logger.info(f"Saved fixed projection ({fp_dim}, 2) to {proj_path}")
        return self._fixed_projection

    @torch.no_grad()
    def eval_model(self, iteration: int) -> dict[str, float]:
        n = self.config.eval_samples
        bs = self.config.eval_batch_size
        n_valid_target = self.config.eval_valid_samples

        if n <= 0 and n_valid_target <= 0:
            return dict()

        n = max(n, bs)
        eval_kwargs = self.problem.eval_sampling_kwargs(n)
        sample = self.env.batch_sample(n, bs, **eval_kwargs)
        batch = Batch.from_sample(sample)
        batch.valids = self.problem.validity(batch.samples, batch.kwargs)
        total_sampled = n

        while n_valid_target > 0 and batch.valids.sum().item() < n_valid_target:
            extra_kwargs = self.problem.eval_sampling_kwargs(bs)
            extra_sample = self.env.batch_sample(bs, bs, **extra_kwargs)
            extra_batch = Batch.from_sample(extra_sample)
            extra_batch.valids = self.problem.validity(extra_batch.samples, extra_batch.kwargs)
            batch = Batch.concat([batch, extra_batch])
            total_sampled += bs

        # Save samples
        directory = self.config.folder / "eval" / f"{iteration:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.problem.save_samples(batch.samples, batch.kwargs, directory)
        torch.save(batch.valids, directory / "valids.pt")

        # Compute and save evaluation metrics
        metrics = self.problem.compute_metrics(
            batch.samples, batch.kwargs,
            n_valid=n_valid_target,
            compute_vendi=self.config.compute_vendi,
        )
        metrics["model_valid"] = batch.valids.float().mean().item()
        metrics["n_eval_sampled"] = total_sampled

        # Fingerprint-based metrics
        current_fps = None
        if self._pretrained_fps is not None and hasattr(self.problem, "get_morgan_fingerprints"):
            current_fps = self.problem.get_morgan_fingerprints(
                batch.samples, batch.kwargs, n_valid=n_valid_target
            )
            metrics["fid"] = self.problem.compute_fid(current_fps, self._pretrained_fps)

            # Cumulative cluster coverage tracking (disabled — kept for future use)
            # if hasattr(self.problem, "compute_cumulative_cluster_metrics"):
            #     cluster_metrics, self._cumulative_centers = self.problem.compute_cumulative_cluster_metrics(
            #         current_fps, self._cumulative_centers, threshold=0.85,
            #     )
            #     metrics.update(cluster_metrics)

        with open(directory / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

        # Fixed random projection PCA plot (disabled — kept for future use)
        # if current_fps is not None:
        #     proj = self._get_fixed_projection()
        #     if proj is not None:
        #         fig = self.problem.plot_fingerprint_fixed_projection(
        #             current_fps, self._pretrained_fps, proj
        #         )
        #         if fig is not None:
        #             fig.savefig(directory / "fingerprint_pca.png", dpi=100, bbox_inches="tight")
        #             wandb.log({"fingerprint_fixed_proj": wandb.Image(fig)}, commit=False)
        #             plt.close(fig)

        return metrics

    def _warmup_cache_path(self) -> Path | None:
        if self.config.warmup_cache_dir is None:
            return None
        return self.config.warmup_cache_dir / "warmup_batches.pt"

    def _save_warmup_cache(self, batches: list[Batch[D]], total_valid_samples: int, warmup_iters: int):
        cache_path = self._warmup_cache_path()
        if cache_path is None or cache_path.exists():
            return
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = cache_path.with_suffix(".tmp")
        torch.save({
            "batches": [b.cpu() for b in batches],
            "total_valid_samples": total_valid_samples,
            "warmup_iters": warmup_iters,
        }, tmp_path)
        tmp_path.rename(cache_path)
        self.logger.info(f"Saved warmup cache ({warmup_iters} iters, {total_valid_samples} valid samples) to {cache_path}")

    def _load_warmup_cache(self) -> tuple[list[Batch[D]], int, int] | None:
        cache_path = self._warmup_cache_path()
        if cache_path is None or not cache_path.exists():
            return None
        cache = torch.load(cache_path, weights_only=False)
        batches = cache["batches"]
        total_valid = cache["total_valid_samples"]
        warmup_iters = cache["warmup_iters"]
        self.logger.info(f"Loaded warmup cache: {warmup_iters} iters, {len(batches)} batches, {total_valid} valid samples")
        return batches, total_valid, warmup_iters

    def _reg_cache_path(self) -> Path | None:
        if self.config.warmup_cache_dir is not None:
            return self.config.warmup_cache_dir / "reg_data.pt"
        return self.config.folder / "reg_data.pt"

    @torch.no_grad()
    def _generate_or_load_reg_data(self) -> Batch[D] | None:
        if not self.config.reg_data:
            return None

        cache_path = self._reg_cache_path()
        if cache_path is not None and cache_path.exists():
            reg_batch = torch.load(cache_path, weights_only=False)
            self.logger.info(f"Loaded reg data: {len(reg_batch)} samples from {cache_path}")
            return reg_batch

        n = self.config.n_reg_samples
        bs = self.config.sample_batch_size
        chunk = self.config.samples_per_iter
        self.logger.info(f"Generating {n} regularization samples from base model...")
        batches: list[Batch[D]] = []
        for i in range(0, n, chunk):
            current_n = min(chunk, n - i)
            sample = self.env.batch_sample(current_n, bs)
            b = Batch.from_sample(sample)
            b.latents = self.problem.postprocess_latents(b)
            batches.append(b.cpu())
            self.logger.info(f"  generated {i + current_n}/{n} reg samples")
        reg_batch = Batch.concat(batches)

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_suffix(".tmp")
            torch.save(reg_batch, tmp_path)
            tmp_path.rename(cache_path)
            self.logger.info(f"Saved reg data ({n} samples) to {cache_path}")

        return reg_batch

    def explore_loop(self, num_iterations: int, samples_per_iter: int):
        metrics = dict()
        batches: list[Batch[D]] = []
        total_valid_samples = 0
        ft_start_iter: int | None = None
        start_iter = 0

        reg_batch = self._generate_or_load_reg_data()
        self._generate_or_load_pretrained_fps()

        cached = self._load_warmup_cache()
        if cached is not None:
            batches, total_valid_samples, start_iter = cached
            ft_start_iter = start_iter
            self.logger.info(f"Resuming from warmup cache at iter={start_iter}, ft_iter=1")

            with self._timer("eval"):
                metrics = self.eval_model(start_iter)
            self.logger.info(f"(iter={start_iter}, ft_iter=00000) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}")
            wandb.log({
                "iter": start_iter, "ft_iter": 0,
                **{f"by_iter/{k}": v for k, v in metrics.items()},
                **{f"by_ft_iter/{k}": v for k, v in metrics.items()},
            })

        for i in range(start_iter, num_iterations):
            # Determine if we should use uncertainty guidance
            has_enough_data = total_valid_samples > self.config.ft_min_dataset_size
            use_guidance = has_enough_data and not self.config.no_uncertainty

            if has_enough_data and ft_start_iter is None:
                ft_start_iter = i
                self.logger.info(f"fine-tuning starts at iter={i}, ft_iter=1")
                self._save_warmup_cache(batches, total_valid_samples, i)

            ft_iter = i - (ft_start_iter - 1) if ft_start_iter is not None else i - num_iterations

            # Evaluate current model state every eval_every ft_iters
            with self._timer("eval"):
                metrics = {}
                if ft_start_iter is not None:
                    should_eval = ft_iter > 0 and ft_iter % self.config.eval_every == 0
                else:
                    should_eval = (i == 0) or (i % self.config.eval_every == 0)
                if should_eval:
                    metrics = self.eval_model(i)

            # Collect new samples
            with self._timer("sampling"):
                if use_guidance:
                    self.env.control_policy = RewardGradient(self.env, self.uncertainty)

                sample = self.env.batch_sample(samples_per_iter, self.config.sample_batch_size)
                batch = Batch.from_sample(sample)
                batch.latents = self.problem.postprocess_latents(batch)
                batch.valids = self.problem.validity(batch.samples, batch.kwargs)
                total_valid_samples += batch.valids.int().sum().item()

                self.env.control_policy = None

            # Store data
            batches.append(batch)

            # Logging and visualization
            with self._timer("visualize"):
                self.visualize_iter(batch, i)

                with torch.no_grad():
                    mean, uncert = self.uncertainty.mean_and_uncertainty(batch.latents, **batch.kwargs)

                if self.config.mean_weight > 0:
                    metrics["pred_mean"] = mean.mean().item()
                    metrics["biased_reward"] = batch.rewards.mean().item()
                metrics["uncertainty"] = uncert.mean().item()
                metrics["biased_valid"] = batch.valids.float().mean().item()
                metrics["max_vram"] = torch.cuda.max_memory_allocated() * 1e-9
                self.logger.info(f"(iter={i}, ft_iter={ft_iter:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}")
                log_dict = {
                    "iter": i, "ft_iter": ft_iter,
                    **{f"by_iter/{k}": v for k, v in metrics.items()},
                }
                if ft_iter >= 0:
                    log_dict.update({f"by_ft_iter/{k}": v for k, v in metrics.items()})
                wandb.log(log_dict)

            # Update models with full buffer
            with self._timer("finetune"):
                if has_enough_data:
                    if self.config.neg_grad_scale > 0:
                        self.combined_finetune_base_model(batches, reg_batch=reg_batch, pbar=False)
                    else:
                        self.finetune_base_model(batches, reg_batch=reg_batch, pbar=False)

            with self._timer("uncertainty_update"):
                if not self.config.no_uncertainty:
                    self.update_uncertainty_estimator(batches)

            # Save checkpoint
            with self._timer("checkpoint"):
                ckpt_dir = self.config.folder / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                state_dict = self.base_model.state_dict()
                torch.save(state_dict, ckpt_dir / "last.pt")

                if should_eval:
                    torch.save(state_dict, ckpt_dir / f"ckpt_{i}.pt")
                elif self.config.ckpt_every > 0 and i % self.config.ckpt_every == 0:
                    torch.save(state_dict, ckpt_dir / f"ckpt_{i}.pt")

            # Write timings to CSV
            self._flush_timings(i)

        write_video(
            sorted(self.config.folder.glob("frames/*.png")),
            self.config.folder / "video.mp4",
            fps=self.config.video_fps,
        )


def setup_and_run(args: argparse.Namespace, reward: Reward, mean_weight: float):
    """Shared initialization and run logic for task-agnostic and task-directed."""
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ExploreConfig.construct_from_args(args)
    problem_setup = problem_setups[args.problem_setup](vars(args), device=device)

    # Construct feature extractor for uncertainty quantification
    feat_extractor = FlowFeatureExtractor(
        problem_setup.base_model,
        layer=problem_setup.feature_layer,
        timestep=config.feat_timestep,
        postprocess=problem_setup.postprocess_features,
    )

    # Probe the feature extractor for testing and obtaining the dimensionality of the features
    x, kwargs = problem_setup.base_model.sample_p0(1)
    x, kwargs = problem_setup.base_model.preprocess(x, **kwargs)
    feat = feat_extractor(x, **kwargs)

    if not isinstance(feat, torch.Tensor):
        raise TypeError(f"Feature extractor output must be a torch.Tensor, got {type(feat)}")

    if feat.ndim != 2:
        raise ValueError(f"Feature extractor output must be a 2D tensor, got {feat.ndim}D tensor")

    uncertainty = uncertainty_estimators[args.uncertainty_estimator](
        feat_extractor,
        feat_dim=feat.shape[1],
        mean_weight=mean_weight,
        device=device,
        args=vars(args),
    )

    # Save arguments
    with open(config.folder / "args.yaml", "w") as f:
        yaml.safe_dump(serialize_args(args), f)

    # Set up logging
    logger = setup_logger(config.folder, args.verbose)
    logger.info("starting...")

    use_wandb = not getattr(args, "no_wandb", False)

    if use_wandb:
        tags = []
        if config.no_uncertainty:
            tags.append("no_uncertainty")
        if config.no_verifier:
            tags.append("no_verifier")
        if config.reg_data:
            tags.append(f"reg_a{config.alpha_reg}")
        if config.neg_grad_scale > 0:
            tags.append(f"neg{config.neg_grad_scale}")
        prefix = f"{args.problem_setup}_" if args.problem_setup != "geom_drugs" else ""
        wandb_name = prefix + config.folder.name + ("_" + "_".join(tags) if tags else "")

        wandb.init(
            entity="riccardodesanti",
            project="active_flow_expansion",
            name=wandb_name,
            tags=tags,
            config=serialize_args(args),
        )
        wandb.define_metric("iter")
        wandb.define_metric("ft_iter")
        wandb.define_metric("by_iter/*", step_metric="iter")
        wandb.define_metric("by_ft_iter/*", step_metric="ft_iter")

        def _handle_sigterm(signum, frame):
            wandb.finish(exit_code=1)
            raise SystemExit(1)

        signal.signal(signal.SIGTERM, _handle_sigterm)
    else:
        wandb.init(mode="disabled")

    loop = ExploreLoop(problem_setup=problem_setup, uncertainty=uncertainty, reward=reward, config=config, logger=logger)
    loop.explore_loop(config.num_iters, config.samples_per_iter)

    if use_wandb:
        wandb.finish()


def build_parser(add_extra_args):
    """Build the nested argparse parser structure.

    Parameters
    ----------
    add_extra_args : callable
        Function that adds mode-specific arguments to a parser.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="problem_setup", required=True)

    for name, setup_cls in problem_setups.items():
        sub = subparsers.add_parser(name)

        # Create nested uncertainty subparser first
        uncertainty_subparsers = sub.add_subparsers(dest="uncertainty_estimator", required=True)

        for u_name, u_cls in uncertainty_estimators.items():
            u_sub = uncertainty_subparsers.add_parser(u_name)

            # Attach all arguments to the final nested subparser
            add_global_args(u_sub)
            add_extra_args(u_sub)
            setup_cls.add_args(u_sub)
            u_cls.add_args(u_sub)

    return parser


def add_global_args(parser):
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    # Exploration sampling
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--samples_per_iter", type=int, default=64)
    parser.add_argument("--sample_batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=100)

    # Uncertainty estimator
    parser.add_argument("--feat_timestep", type=float, default=0.9)
    parser.add_argument("--dps_weight", type=float, default=100)

    # Fine-tuning
    parser.add_argument("--ft_min_dataset_size", type=int, default=64)
    parser.add_argument("--ft_steps", type=int, default=500)
    parser.add_argument("--ft_batch_size", type=int, default=64)
    parser.add_argument("--ft_accumulate_steps", type=int, default=1)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument("--ft_weight_decay", type=float, default=0.0)

    # Checkpointing
    parser.add_argument("--ckpt_every", type=int, default=100)

    # Sampling
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--eval_valid_samples", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=10)

    # Logging
    parser.add_argument("--video_fps", type=int, default=4)

    # Warmup caching
    parser.add_argument("--warmup_cache_dir", type=Path, default=None)

    # Regularization data
    parser.add_argument("--reg_data", action="store_true")
    parser.add_argument("--alpha_reg", type=float, default=1.0)
    parser.add_argument("--n_reg_samples", type=int, default=10000)

    # Combined pos+neg finetuning
    parser.add_argument("--neg_grad_scale", type=float, default=0.0)
    parser.add_argument("--max_neg_samples", type=int, default=5000)
