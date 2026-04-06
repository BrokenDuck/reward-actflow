from typing import Generic
from dataclasses import dataclass
from contextlib import contextmanager
import math

import numpy as np
import torch
from diffusiongym import construct_env, D, DummyReward
from diffusiongym.utils import train_base_model, DDDataset, dict_to_device
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import nn as _nn
from tqdm import tqdm as _tqdm
from pathlib import Path
import argparse
import logging
import yaml
import time
import csv

from .inf_methods.dps import RewardGradient
from .uncertainty import UncertaintyEstimator, FlowFeatureExtractor, uncertainty_estimators
from .setups import setups as problem_setups
from .setups.problem_setup import ProblemSetup
from .utils import write_video, filter_out_invalids, filter_out_valids, Batch, serialize_args, setup_logger


class RandomGPRewards:
    """Random reward functions sampled from a GP prior (Matern-5/2) via Random Fourier Features."""

    def __init__(self, dim: int, n_functions: int = 20, n_features: int = 256,
                 lengthscale: float = 1.0, output_scale: float = 1.0, seed: int = 123):
        rng = torch.Generator()
        rng.manual_seed(seed)

        nu = 2.5
        df = int(2 * nu)  # 5 for Matern-5/2
        z = torch.randn(n_functions, n_features, dim, generator=rng)
        chi2 = sum(torch.randn(n_functions, n_features, generator=rng) ** 2 for _ in range(df))
        self.omega = z / (lengthscale * torch.sqrt(chi2 / df).unsqueeze(-1))
        self.bias = torch.rand(n_functions, n_features, generator=rng) * 2 * math.pi
        self.weights = (torch.randn(n_functions, n_features, generator=rng)
                        * output_scale * math.sqrt(2.0 / n_features))

    def to(self, device: torch.device) -> "RandomGPRewards":
        self.omega = self.omega.to(device)
        self.bias = self.bias.to(device)
        self.weights = self.weights.to(device)
        return self

    @torch.no_grad()
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate all reward functions at points x.

        Args:
            x: (N, D) tensor of sample locations.
        Returns:
            (n_functions, N) tensor of reward values.
        """
        proj = torch.einsum('fmd,nd->fmn', self.omega, x) + self.bias.unsqueeze(-1)
        return torch.einsum('fm,fmn->fn', self.weights, torch.cos(proj))


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = TaskAgnosticConfig.construct_from_args(args)
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
        mean_weight=0,
        device=device,
        args=vars(args),
    )

    # Save arguments
    with open(config.folder / "args.yaml", "w") as f:
        yaml.safe_dump(serialize_args(args), f)

    # Set up logging
    logger = setup_logger(config.folder, args.verbose)
    logger.info("starting...")

    apt = TaskAgnostic(problem_setup=problem_setup, uncertainty=uncertainty, config=config, logger=logger)
    apt.explore_loop(config.num_iters, config.samples_per_iter)


@dataclass(frozen=True)
class TaskAgnosticConfig:
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
    
    # Fine-tuning
    ft_min_dataset_size: int = 64
    ft_batch_size: int = 64
    ft_accumulate_steps: int = 1
    ft_steps: int = 500
    ft_lr: float = 1e-4
    ft_weight_decay: float = 0.0

    # Sampling and evaluation
    eval_samples: int = 0
    eval_samples_curves: int = 0
    eval_batch_size: int = 64
    eval_every: int = 10
    video_fps: int = 4
    
    # Particle guidance (Corso et al.; feature-space repulsion, GP-aligned RBF bandwidth when gp_kernel=rbf)
    particle_guidance_coeff: float = 0.0
    particle_guidance_sigma_break: float = 1.0

    # DPP sampling: sample a large pool, greedily select diverse subset
    dpp_pool_size: int = 512
    dpp_space: str = "features"  # "features" or "data"
    dpp_kernel: str = "linear"   # "linear" or "rbf"
    dpp_rbf_lengthscale: float = 1.0

    # Negative score matching (gradient ascent on flow matching loss for invalid samples)
    nsm_enabled: bool = False
    nsm_weight: float = 1.0
    nsm_steps: int = 0       # 0 → use ft_steps
    nsm_lr: float = 0.0      # 0 → use ft_lr

    # Combined pos+neg finetuning (SISS No-IS) with gradient norm scaling:
    # two backward passes per step; the negative gradient is rescaled so its
    # L2 norm equals neg_grad_scale × ||grad_pos||.
    combined_finetuning: bool = False
    neg_grad_scale: float = 0.1  # target ||grad_neg|| as fraction of ||grad_pos||

    # Fine-tune on new samples only (current iteration) instead of full buffer
    ft_new_only: bool = False

    # Best-of-N evaluation with GP reward functions
    bon_n: int = 100
    bon_k: int = 10
    bon_n_functions: int = 20
    bon_lengthscale: float = 1.0

    # Flags
    guidance_method: str = "uncertainty_tilting"  # "uncertainty_tilting", "particle_guidance", "dpp_sampling", "none"
    no_verifier: bool = False
    plot_uncertainty: bool = False
    save_frames: bool = False

    def __post_init__(self):
        # Create experiment directory if it doesn't exist
        self.folder.mkdir(parents=True, exist_ok=True)

        # Validation
        if not (0 <= self.feat_timestep <= 1):
            raise ValueError(f"feat_timestep must be in [0, 1], got {self.feat_timestep}")

        if self.dps_weight < 0:
            raise ValueError(f"dps_weight cannot be negative, got {self.dps_weight}")

    @staticmethod
    def construct_from_args(args: argparse.Namespace | dict) -> "TaskAgnosticConfig":
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        # Map argparse flags to config names
        name_mapping = { "dir": "folder" }
        
        # We only take keys that exist in the dataclass fields
        config_fields = {f.name for f in TaskAgnosticConfig.__dataclass_fields__.values()}
        clean_kwargs = {}

        for key, value in args.items():
            # Check if the key needs a rename
            target_key = name_mapping.get(key, key)
            
            # Only add if it's a valid field in our config
            if target_key in config_fields:
                clean_kwargs[target_key] = value

        return TaskAgnosticConfig(**clean_kwargs)


class TaskAgnostic(Generic[D]):
    def __init__(
        self,
        problem_setup: ProblemSetup[D],
        uncertainty: UncertaintyEstimator[D],
        config: TaskAgnosticConfig,
        logger: logging.Logger,
    ):
        self.problem = problem_setup
        self.config = config
        self.logger = logger

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

        env = construct_env(
            problem_setup.base_model,
            DummyReward(),
            discretization_steps=config.num_steps,
            reward_scale=config.dps_weight,
        )

        self.base_model = problem_setup.base_model
        self.uncertainty = uncertainty
        self.env = env

        self.logger.info(f"base model parameters: {sum(p.numel() for p in self.base_model.parameters() if p.requires_grad):,}")

        self._timings = {}

        # Create file and write headers
        self.timing_heads = ["iteration", "eval", "sampling", "visualize", "finetune", "negative_finetune", "uncertainty_update", "checkpoint"]
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

    def finetune_base_model(self, batches: list[Batch[D]], pbar: bool = False):
        """Fine-tune the base model with obtained valid samples.

        Parameters
        ----------
        batches : list[Batch[D]]
            Data batches obtained so far, where each entry is from a separate iteration.
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

        # Fine-tune base model
        opt = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.config.ft_lr,
            weight_decay=self.config.ft_weight_decay,
        )
        train_base_model(
            self.base_model,
            opt,
            [valid_batch.latents.to(self.env.base_model.device)],
            [valid_batch.kwargs],
            steps=self.config.ft_steps,
            batch_size=self.config.ft_batch_size,
            accumulate_steps=self.config.ft_accumulate_steps,
            pbar=pbar,
        )

    def negative_finetune_base_model(self, batches: list[Batch[D]], pbar: bool = False):
        """Push the model away from invalid samples via negative score matching.

        Uses the same flow matching machinery as positive finetuning, but with
        negated per-sample weights so that gradient descent becomes gradient ascent
        on the flow matching loss for invalid data.  This implements a first-order
        approximation of Eq. 8 in the negative score matching paper.
        """
        invalid_batch = filter_out_valids(batches)
        if invalid_batch is None:
            self.logger.info("NSM: no invalid samples, skipping")
            return

        nsm_steps = self.config.nsm_steps if self.config.nsm_steps > 0 else self.config.ft_steps
        nsm_lr = self.config.nsm_lr if self.config.nsm_lr > 0 else self.config.ft_lr

        opt = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=nsm_lr,
            weight_decay=self.config.ft_weight_decay,
        )

        neg_weights = torch.full(
            (len(invalid_batch),), -self.config.nsm_weight, dtype=torch.float32,
        )

        train_base_model(
            self.base_model,
            opt,
            [invalid_batch.latents.to(self.env.base_model.device)],
            [invalid_batch.kwargs],
            weights=[neg_weights],
            steps=nsm_steps,
            batch_size=self.config.ft_batch_size,
            accumulate_steps=self.config.ft_accumulate_steps,
            pbar=pbar,
        )

    def combined_finetune_base_model(self, batches: list[Batch[D]], pbar: bool = False):
        """Fine-tune with combined pos+neg signal (SISS No-IS, gradient norm scaling).

        Uses **separate dataloaders** for valid and invalid data so that:
        - The positive update always sees a full batch of valid samples
          (identical to finetune_base_model).
        - The negative gradient is computed from a full batch of invalid
          samples and rescaled so ||grad_neg|| = neg_grad_scale × ||grad_pos||.

        When neg_grad_scale=0 or there are no invalid samples the method
        is exactly equivalent to finetune_base_model.
        """
        if self.config.no_verifier:
            self.finetune_base_model(batches, pbar=pbar)
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
        self.logger.info(
            f"Combined FT (grad-norm): m={m} k={k}, "
            f"neg_grad_scale={scale}, use_neg={use_neg}"
        )

        device = self.env.base_model.device
        bs = self.config.ft_batch_size

        pos_dataset = DDDataset(
            [valid_batch.latents.to(device)],
            [valid_batch.kwargs],
            None,
        )
        pos_loader = DataLoader(
            pos_dataset, bs, shuffle=True,
            collate_fn=pos_dataset.collate, num_workers=0, pin_memory=False,
        )

        neg_loader = None
        if use_neg:
            neg_dataset = DDDataset(
                [invalid_batch.latents.to(device)],
                [invalid_batch.kwargs],
                None,
            )
            neg_loader = DataLoader(
                neg_dataset, bs, shuffle=True,
                collate_fn=neg_dataset.collate, num_workers=0, pin_memory=False,
            )

        opt = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.config.ft_lr,
            weight_decay=self.config.ft_weight_decay,
        )

        self.base_model.train()
        opt.zero_grad()
        pos_iter = iter(pos_loader)
        neg_iter = iter(neg_loader) if neg_loader is not None else None

        iterator = range(self.config.ft_steps)
        if pbar:
            iterator = _tqdm(iterator)

        for _ in iterator:
            # --- positive batch (full batch of valid samples) ---
            try:
                x1_pos, kw_pos, _ = next(pos_iter)
            except StopIteration:
                pos_iter = iter(pos_loader)
                x1_pos, kw_pos, _ = next(pos_iter)

            x1_pos = x1_pos.to(self.base_model.device)
            kw_pos = dict_to_device(kw_pos, self.base_model.device)

            loss_pos = self.base_model.train_loss(x1_pos, **kw_pos).mean()
            loss_pos.backward()

            if neg_iter is None:
                _nn.utils.clip_grad_norm_(self.base_model.parameters(), 0.1)
                opt.step()
                opt.zero_grad()
                continue

            # Store positive gradients
            grad_pos = {
                name: p.grad.clone()
                for name, p in self.base_model.named_parameters()
                if p.grad is not None
            }
            norm_pos = torch.sqrt(sum(
                g.pow(2).sum() for g in grad_pos.values()
            ))

            # --- negative batch (full batch of invalid samples) ---
            opt.zero_grad()

            try:
                x1_neg, kw_neg, _ = next(neg_iter)
            except StopIteration:
                neg_iter = iter(neg_loader)
                x1_neg, kw_neg, _ = next(neg_iter)

            x1_neg = x1_neg.to(self.base_model.device)
            kw_neg = dict_to_device(kw_neg, self.base_model.device)

            loss_neg = self.base_model.train_loss(x1_neg, **kw_neg).mean()
            loss_neg.backward()

            norm_neg = torch.sqrt(sum(
                p.grad.pow(2).sum()
                for p in self.base_model.parameters()
                if p.grad is not None
            ))

            if norm_neg > 0:
                rescale = (scale * norm_pos) / norm_neg
            else:
                rescale = 0.0

            for name, p in self.base_model.named_parameters():
                if p.grad is not None and name in grad_pos:
                    p.grad.data = grad_pos[name] - rescale * p.grad.data
                elif name in grad_pos:
                    p.grad = -grad_pos[name]

            _nn.utils.clip_grad_norm_(self.base_model.parameters(), 0.1)
            opt.step()
            opt.zero_grad()

        self.base_model.eval()

    def update_uncertainty_estimator(self, batches: list[Batch[D]]):
        """Update the uncertainty estimator with obtained samples.

        Parameters
        ----------
        samples : list[Batch[D]]
            Data batches obtained so far, where one batch comes from one iteration in the order.
        """
        self.uncertainty.set_data(
            [b.latents for b in batches],
            [b.valids.float() for b in batches],
            [b.kwargs for b in batches],
        )

    def visualize_iter(self, batch: Batch[D], iteration: int) -> dict[str, float]:
        save_dir = self.config.folder / "eval" / f"{iteration:04d}"
        save_dir.mkdir(parents=True, exist_ok=True)

        result = self.problem.visualize_sample(
            self.env, self.uncertainty, batch,
            n_samples=self.config.eval_samples,
            save_dir=save_dir if self.config.save_frames else None,
        )
        if isinstance(result, tuple):
            fig, vis_metrics = result
        else:
            fig, vis_metrics = result, {}

        if self.config.save_frames:
            directory = self.config.folder / "frames"
            directory.mkdir(parents=True, exist_ok=True)
            fig.savefig(directory / f"{iteration:04d}.png", dpi=300)

            if self.config.plot_uncertainty and hasattr(self.problem, 'visualize_uncertainty'):
                fig_unc = self.problem.visualize_uncertainty(self.uncertainty, batch)
                unc_directory = self.config.folder / "uncertainty_frames"
                unc_directory.mkdir(parents=True, exist_ok=True)
                fig_unc.savefig(unc_directory / f"{iteration:04d}.png", dpi=300)
                plt.close(fig_unc)

        plt.close(fig)
        return vis_metrics

    @torch.no_grad()
    def _dpp_sample(self, k: int) -> Batch[D]:
        """Sample a large pool, then select k diverse points via greedy DPP.

        Diversity selection is verifier-agnostic: validity is checked
        *after* selection, identically to how other methods work.
        """
        from adm.inf_methods.dpp_sampling import dpp_select

        pool_size = self.config.dpp_pool_size
        sample = self.env.batch_sample(pool_size, self.config.sample_batch_size)
        pool = Batch.from_sample(sample)
        pool.latents = self.problem.postprocess_latents(pool)

        if self.config.dpp_space == "data":
            embeddings = pool.samples.data
        else:
            embeddings = self.uncertainty.feat_extractor(pool.samples, **pool.kwargs)
            if not isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.data

        sel = dpp_select(
            embeddings, k,
            kernel=self.config.dpp_kernel,
            rbf_lengthscale=self.config.dpp_rbf_lengthscale,
        )

        from diffusiongym import DDTensor
        selected_data = pool.samples.data[sel]
        selected_latents = pool.latents.data[sel]

        batch = Batch(
            samples=DDTensor(selected_data),
            latents=DDTensor(selected_latents),
            rewards=pool.rewards[sel],
            valids=torch.zeros(len(sel), dtype=torch.bool),
            kwargs={key: val[sel] if isinstance(val, torch.Tensor) else val
                    for key, val in pool.kwargs.items()},
        )
        batch.valids = self.problem.validity(batch.samples, batch.kwargs)
        return batch

    @torch.no_grad()
    def _evaluate_bon(self, gp_rewards: RandomGPRewards) -> dict[str, torch.Tensor]:
        """Sample valid points from the current model and compute Best-of-N metrics."""
        old_policy = self.env.control_policy
        self.env.control_policy = None

        n_target = self.config.bon_n
        valid_chunks: list[torch.Tensor] = []
        n_collected = 0
        max_rounds = 200
        for _ in range(max_rounds):
            sample = self.env.sample(n_target, pbar=False)
            valids = self.problem.validity(sample.sample, {})
            valid_data = sample.sample.data[valids]
            if len(valid_data) > 0:
                valid_chunks.append(valid_data)
                n_collected += len(valid_data)
            if n_collected >= n_target:
                break
        else:
            raise RuntimeError(
                f"BoN evaluation failed: only collected {n_collected}/{n_target} valid samples "
                f"after {max_rounds} rounds. The model's validity is too low."
            )

        self.env.control_policy = old_policy

        samples = torch.cat(valid_chunks, dim=0)[:n_target]
        device = gp_rewards.omega.device
        rewards = gp_rewards.evaluate(samples.to(device))  # (F, N)

        top1 = rewards.max(dim=1).values  # (F,)
        topk = rewards.topk(min(self.config.bon_k, rewards.shape[1]), dim=1).values.mean(dim=1)

        return {"top1": top1, "topk": topk}

    def _plot_bon(self, initial_bon: dict, current_bon: dict, iteration: int,
                  save_dir: Path):
        """Bar chart comparing initial vs expanded model on BoN metrics."""
        save_dir.mkdir(parents=True, exist_ok=True)
        k = self.config.bon_k
        n = self.config.bon_n

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)

        for ax, key, title in [
            (ax1, "top1", f"Best-of-{n}"),
            (ax2, "topk", f"Avg Top-{k}-of-{n}"),
        ]:
            init_vals = initial_bon[key]
            curr_vals = current_bon[key]
            means = [init_vals.mean().item(), curr_vals.mean().item()]
            n_fn = len(init_vals)
            ci95 = [1.96 * init_vals.std().item() / math.sqrt(n_fn),
                    1.96 * curr_vals.std().item() / math.sqrt(n_fn)]
            ax.bar(["Initial", "Expanded"], means, yerr=ci95, capsize=5,
                   color=["#888888", "#4CAF50"])
            ax.set_title(title)
            ax.set_ylabel("Reward")

        fig.suptitle(f"Best-of-N Evaluation (iter {iteration})")
        fig.savefig(save_dir / f"bon_{iteration:04d}.png", dpi=300)
        plt.close(fig)

    @torch.no_grad()
    def eval_model(self, iteration: int) -> dict[str, float]:
        n = self.config.eval_samples_curves if self.config.eval_samples_curves > 0 else self.config.eval_samples
        bs = self.config.eval_batch_size

        if n <= 0:
            return dict()

        eval_kwargs = self.problem.eval_sampling_kwargs(n)
        sample = self.env.batch_sample(n, bs, **eval_kwargs)
        batch = Batch.from_sample(sample)
        batch.valids = self.problem.validity(batch.samples, batch.kwargs)

        # Save samples
        directory = self.config.folder / "eval" / f"{iteration:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.problem.save_samples(batch.samples, batch.kwargs, directory)
        torch.save(batch.valids, directory / "valids.pt")

        # Compute and save evaluation metrics
        metrics = self.problem.compute_metrics(batch.samples, batch.kwargs)
        metrics["model_valid"] = batch.valids.float().mean().item()

        with open(directory / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

        return metrics

    def explore_loop(self, num_iterations: int, samples_per_iter: int):
        metrics = dict()
        batches: list[Batch[D]] = []
        total_valid_samples = 0
        eval_history: list[dict] = []

        # Set up GP reward functions for Best-of-N evaluation
        sample_dim = self.env.sample(1, pbar=False).sample.data.shape[1]
        gp_rewards = RandomGPRewards(
            dim=sample_dim,
            n_functions=self.config.bon_n_functions,
            n_features=256,
            lengthscale=self.config.bon_lengthscale,
            seed=123,
        ).to(self.base_model.device)
        initial_bon = self._evaluate_bon(gp_rewards)

        for i in range(num_iterations):
            # Evaluate current model state
            with self._timer("eval"):
                metrics = {}
                if i % self.config.eval_every == 0:
                    metrics = self.eval_model(i)

            # Determine if we should use guidance
            has_enough_data = total_valid_samples > self.config.ft_min_dataset_size
            method = self.config.guidance_method

            # Collect new samples
            with self._timer("sampling"):
                if has_enough_data and method == "uncertainty_tilting":
                    self.env.control_policy = RewardGradient(self.env, self.uncertainty)
                elif has_enough_data and method == "particle_guidance":
                    from adm.inf_methods.particle_guidance import ParticleGuidance
                    self.env.control_policy = ParticleGuidance(
                        self.env,
                        self.uncertainty,
                        coeff=self.config.particle_guidance_coeff,
                        sigma_break=self.config.particle_guidance_sigma_break,
                    )

                if method == "dpp_sampling":
                    batch = self._dpp_sample(samples_per_iter)
                else:
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
                if i % self.config.eval_every == 0:
                    vis_metrics = self.visualize_iter(batch, i)

                    current_bon = self._evaluate_bon(gp_rewards)

                    if self.config.save_frames:
                        bon_dir = self.config.folder / "bon_frames"
                        self._plot_bon(initial_bon, current_bon, i, bon_dir)

                    eval_dir = self.config.folder / "eval" / f"{i:04d}"
                    eval_dir.mkdir(parents=True, exist_ok=True)
                    np.savez(
                        eval_dir / "bon_results.npz",
                        initial_top1=initial_bon["top1"].cpu().numpy(),
                        initial_topk=initial_bon["topk"].cpu().numpy(),
                        current_top1=current_bon["top1"].cpu().numpy(),
                        current_topk=current_bon["topk"].cpu().numpy(),
                    )

                    vis_metrics["bon_avg_top1"] = current_bon["top1"].mean().item()
                    vis_metrics["bon_avg_topk"] = current_bon["topk"].mean().item()
                    metrics.update(vis_metrics)
                    eval_history.append({"iteration": i, **metrics})

                with torch.no_grad():
                    _, uncert = self.uncertainty.mean_and_uncertainty(batch.latents, **batch.kwargs)

                metrics["uncertainty"] = uncert.mean().item()
                metrics["biased_valid"] = batch.valids.float().mean().item()
                metrics["max_vram"] = torch.cuda.max_memory_allocated() * 1e-9
                self.logger.info(f"(iter={i:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}")

            # Update models
            ft_data = [batch] if self.config.ft_new_only else batches
            with self._timer("finetune"):
                if has_enough_data:
                    if self.config.combined_finetuning:
                        self.combined_finetune_base_model(ft_data, pbar=False)
                    else:
                        self.finetune_base_model(ft_data, pbar=False)

            with self._timer("negative_finetune"):
                if has_enough_data and self.config.nsm_enabled and not self.config.no_verifier and not self.config.combined_finetuning:
                    self.negative_finetune_base_model(batches, pbar=False)

            with self._timer("uncertainty_update"):
                if method != "none":
                    self.update_uncertainty_estimator(batches)

            # Save checkpoint
            with self._timer("checkpoint"):
                torch.save(self.base_model.state_dict(), self.config.folder / "base_model.pt")

            # Write timings to CSV
            self._flush_timings(i)
        
        if self.config.save_frames:
            write_video(
                sorted(self.config.folder.glob("frames/*.png")),
                self.config.folder / "video.mp4",
                fps=self.config.video_fps,
            )

            if self.config.plot_uncertainty:
                write_video(
                    sorted(self.config.folder.glob("uncertainty_frames/*.png")),
                    self.config.folder / "uncertainty_video.mp4",
                    fps=self.config.video_fps,
                )

        self._save_eval_history(eval_history)
        self._plot_eval_curves(eval_history)

    def _save_eval_history(self, eval_history: list[dict]):
        if not eval_history:
            return
        path = self.config.folder / "eval_history.csv"
        keys = list(eval_history[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(eval_history)

    def _plot_eval_curves(self, eval_history: list[dict]):
        if not eval_history:
            return

        iters = [h["iteration"] for h in eval_history]

        if "model_valid" in eval_history[0]:
            validity = [h["model_valid"] * 100 for h in eval_history]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(iters, validity, marker="o", color="steelblue")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Validity (%)")
            ax.set_ylim(0, 105)
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(self.config.folder / "validity_curve.png", dpi=150)
            plt.close(fig)

        if "generable_coverage" in eval_history[0]:
            coverage = [h["generable_coverage"] * 100 for h in eval_history]
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(iters, coverage, marker="o", color="seagreen")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Generable Coverage (%)")
            ax.set_ylim(0, 105)
            ax.grid(True)
            fig.tight_layout()
            fig.savefig(self.config.folder / "coverage_curve.png", dpi=150)
            plt.close(fig)


def build_parser():
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
            setup_cls.add_args(u_sub)
            u_cls.add_args(u_sub)

    return parser


def add_global_args(parser):
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    # Baselines
    parser.add_argument("--guidance_method", type=str, default="uncertainty_tilting",
                        choices=["uncertainty_tilting", "particle_guidance", "dpp_sampling", "none"])
    parser.add_argument("--particle_guidance_coeff", type=float, default=0.0,
                        help="Particle guidance strength (0 disables term; try 0.1–5 as in reference --coeff)")
    parser.add_argument("--particle_guidance_sigma_break", type=float, default=1.0,
                        help="Apply particle guidance only when memoryless sigma exceeds this (cf. reference code)")
    parser.add_argument("--dpp_pool_size", type=int, default=512,
                        help="Candidate pool size N for DPP sampling (selects samples_per_iter from N)")
    parser.add_argument("--dpp_space", type=str, default="features", choices=["features", "data"],
                        help="Space for DPP diversity: 'features' (learned representation) or 'data' (raw x,y)")
    parser.add_argument("--dpp_kernel", type=str, default="linear", choices=["linear", "rbf"],
                        help="Kernel for DPP selection: 'linear' (cosine) or 'rbf' (Gaussian)")
    parser.add_argument("--dpp_rbf_lengthscale", type=float, default=1.0,
                        help="Lengthscale for the RBF kernel (only used when --dpp_kernel rbf)")
    parser.add_argument("--no_verifier", action="store_true")
    parser.add_argument("--plot_uncertainty", action="store_true")
    parser.add_argument("--save_frames", action="store_true",
                        help="Save generable set frames, videos, and generable_set.npz")

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

    # Negative score matching (gradient ascent on flow matching loss for invalid samples)
    parser.add_argument("--nsm_enabled", action="store_true",
                        help="Enable negative score matching: push model away from invalid samples")
    parser.add_argument("--nsm_weight", type=float, default=1.0,
                        help="Weight (alpha) for the negative loss; controls push-away strength")
    parser.add_argument("--nsm_steps", type=int, default=0,
                        help="Gradient steps for NSM per iteration (0 = same as --ft_steps)")
    parser.add_argument("--nsm_lr", type=float, default=0.0,
                        help="Learning rate for NSM steps (0 = same as --ft_lr)")

    # Combined positive+negative finetuning (SISS No-IS)
    parser.add_argument("--combined_finetuning", action="store_true",
                        help="Use combined pos+neg SISS finetuning instead of separate steps")
    parser.add_argument("--neg_grad_scale", type=float, default=0.1,
                        help="Target ||grad_neg|| as fraction of ||grad_pos|| (0 = positive-only)")
    parser.add_argument("--ft_new_only", action="store_true",
                        help="Fine-tune on new samples only (current iteration) instead of full buffer")

    # Sampling
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--eval_samples_curves", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=10)

    # Best-of-N evaluation
    parser.add_argument("--bon_n", type=int, default=100, help="N for Best-of-N sampling")
    parser.add_argument("--bon_k", type=int, default=10, help="K for Top-K-of-N")
    parser.add_argument("--bon_n_functions", type=int, default=20, help="Number of GP reward functions")
    parser.add_argument("--bon_lengthscale", type=float, default=1.0, help="Matern-5/2 lengthscale for GP rewards")

    # Logging
    parser.add_argument("--video_fps", type=int, default=4)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
