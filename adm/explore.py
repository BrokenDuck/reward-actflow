from typing import Generic, Literal, Optional
from dataclasses import dataclass
from contextlib import contextmanager

import numpy as np
import torch
from diffusiongym import construct_env, D, Reward
from diffusiongym.utils import train_base_model, DDDataset, dict_to_device
from torch.utils.data import DataLoader, Subset
import torch.nn as _nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
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

    # Shared offline buffer
    buffer_dir: Optional[Path] = None

    # Flags
    no_uncertainty: bool = False
    no_verifier: bool = False
    verbose: bool = False

    # Negative sample finetuning
    neg_grad_scale: float = 0.0
    max_neg_samples: int = 5000

    def __post_init__(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        if self.buffer_dir is not None:
            (self.buffer_dir / self.folder.name).mkdir(parents=True, exist_ok=True)

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

    def _buffer_run_dir(self) -> Path:
        return self.config.buffer_dir / self.config.folder.name

    def _write_to_buffer(self, batch: Batch, idx: int, run_valid_count: int):
        run_dir = self._buffer_run_dir()
        torch.save(
            {"samples": batch.samples, "latents": batch.latents, "valids": batch.valids, "kwargs": batch.kwargs},
            run_dir / f"batch_{idx:06d}.pt",
        )
        (run_dir / "count.txt").write_text(str(run_valid_count))

    def _total_buffer_valid_count(self) -> int:
        total = 0
        for f in self.config.buffer_dir.glob("*/count.txt"):
            try:
                total += int(f.read_text())
            except Exception:
                pass
        return total

    def _load_buffer(self, max_valid: int | None = None) -> list[Batch]:
        batches = []
        valid_count = 0
        for f in sorted(self.config.buffer_dir.glob("*/batch_*.pt")):
            if max_valid is not None and valid_count >= max_valid:
                break
            data = torch.load(f, map_location="cpu")
            batches.append(Batch(
                samples=data["samples"],
                latents=data["latents"],
                rewards=torch.zeros(len(data["valids"])),
                valids=data["valids"],
                kwargs=data["kwargs"],
            ))
            valid_count += int(data["valids"].sum().item())
        return batches

    def _save_loop_state(
        self,
        iteration: int,
        offline_batches: list,
        guided_batches: list,
        offline_valid_count: int,
        guided_valid_count: int,
        saved_offline_buffer: bool,
    ):
        def batch_to_dict(b: Batch) -> dict:
            return {"samples": b.samples, "latents": b.latents, "rewards": b.rewards, "valids": b.valids, "kwargs": b.kwargs}

        torch.save(
            {
                "iteration": iteration,
                "offline_valid_count": offline_valid_count,
                "guided_valid_count": guided_valid_count,
                "saved_offline_buffer": saved_offline_buffer,
                "offline_batches": [batch_to_dict(b) for b in offline_batches],
                "guided_batches": [batch_to_dict(b) for b in guided_batches],
            },
            self.config.folder / "loop_state.pt",
        )

    def _load_loop_state(self) -> dict | None:
        path = self.config.folder / "loop_state.pt"
        if not path.exists():
            return None
        state = torch.load(path, map_location="cpu")

        def dict_to_batch(d: dict) -> Batch:
            return Batch(samples=d["samples"], latents=d["latents"], rewards=d["rewards"], valids=d["valids"], kwargs=d["kwargs"])

        return {
            "iteration": state["iteration"],
            "offline_valid_count": state["offline_valid_count"],
            "guided_valid_count": state["guided_valid_count"],
            "saved_offline_buffer": state["saved_offline_buffer"],
            "offline_batches": [dict_to_batch(d) for d in state["offline_batches"]],
            "guided_batches": [dict_to_batch(d) for d in state["guided_batches"]],
        }

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
            [valid_batch.latents],
            [valid_batch.kwargs],
            steps=self.config.ft_steps,
            batch_size=self.config.ft_batch_size,
            accumulate_steps=self.config.ft_accumulate_steps,
            pbar=pbar,
        )

    def combined_finetune_base_model(self, batches: list[Batch[D]], pbar: bool = False):
        """Fine-tune with combined pos+neg signal using gradient norm scaling.

        - Positive update sees a full batch of valid samples.
        - Negative gradient is rescaled so ||grad_neg|| = neg_grad_scale * ||grad_pos||.

        Falls back to finetune_base_model when neg_grad_scale=0 or there are no invalid samples.
        """
        if self.config.no_verifier:
            self.finetune_base_model(batches, pbar=pbar)
            return

        valid_batch = filter_out_invalids(batches)
        invalid_batch = filter_out_valids(batches)

        m = len(valid_batch)
        k = len(invalid_batch) if invalid_batch is not None else 0

        if m == 0:
            self.logger.warning("Combined FT: no valid data, skipping")
            return

        scale = self.config.neg_grad_scale
        use_neg = scale > 0 and k > 0

        if not use_neg:
            self.finetune_base_model(batches, pbar=pbar)
            return

        max_neg = self.config.max_neg_samples
        k_used = min(k, max_neg)

        self.logger.info(
            f"Combined FT (grad-norm): m={m} k_total={k} k_used={k_used}, neg_grad_scale={scale}"
        )

        device = self.env.base_model.device
        bs = self.config.ft_batch_size

        pos_dataset = DDDataset([valid_batch.latents.to(device)], [valid_batch.kwargs], None)
        pos_loader = DataLoader(
            pos_dataset, bs, shuffle=True,
            collate_fn=pos_dataset.collate, num_workers=0, pin_memory=False,
        )

        neg_dataset_full = DDDataset([invalid_batch.latents.to(device)], [invalid_batch.kwargs], None)
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
            try:
                x1_pos, kw_pos, _ = next(pos_iter)
            except StopIteration:
                pos_iter = iter(pos_loader)
                x1_pos, kw_pos, _ = next(pos_iter)

            x1_pos = x1_pos.to(device)
            kw_pos = dict_to_device(kw_pos, device)

            loss_pos = self.base_model.train_loss(x1_pos, **kw_pos).mean() / accum
            loss_pos.backward()

            for name, p in self.base_model.named_parameters():
                if p.grad is not None:
                    if name in accum_grad_pos:
                        accum_grad_pos[name].add_(p.grad)
                    else:
                        accum_grad_pos[name] = p.grad.clone()

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
    def eval_model(self, iteration: int) -> dict[str, float]:
        n = self.config.eval_samples
        bs = self.config.eval_batch_size
        n_valid_target = self.config.eval_valid_samples

        if n <= 0 and n_valid_target <= 0:
            return dict()

        verbose = self.config.verbose
        n = max(n, bs)
        eval_kwargs = self.problem.eval_sampling_kwargs(n)
        sample = self.env.batch_sample(n, bs, pbar=verbose, **eval_kwargs)
        batch = Batch.from_sample(sample)
        del sample
        torch.cuda.empty_cache()
        batch.valids = self.problem.validity(batch.samples, batch.kwargs)

        while n_valid_target > 0 and batch.valids.sum().item() < n_valid_target:
            extra_kwargs = self.problem.eval_sampling_kwargs(bs)
            extra_sample = self.env.batch_sample(bs, bs, pbar=verbose, **extra_kwargs)
            extra_batch = Batch.from_sample(extra_sample)
            del extra_sample
            torch.cuda.empty_cache()
            extra_batch.valids = self.problem.validity(extra_batch.samples, extra_batch.kwargs)
            batch = Batch.concat([batch, extra_batch])

        # Save samples
        directory = self.config.folder / "eval" / f"{iteration:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        self.problem.save_samples(batch.samples, batch.kwargs, directory)
        torch.save(batch.valids, directory / "valids.pt")

        # Compute metrics on exactly n_valid_target valid samples (or all valids if target is 0)
        valid_batches = [batch[i] for i in range(len(batch)) if batch.valids[i]]
        if n_valid_target > 0:
            valid_batches = valid_batches[:n_valid_target]
        valid_batch = Batch.concat(valid_batches)

        metrics = self.problem.compute_metrics(valid_batch.samples, valid_batch.kwargs)
        metrics["model_valid"] = batch.valids.float().mean().item()
        with open(directory / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

        return metrics

    def explore_loop(self, num_iterations: int, samples_per_iter: int):
        metrics = dict()
        use_buffer = self.config.buffer_dir is not None

        offline_batches: list[Batch[D]] = []
        guided_batches: list[Batch[D]] = []
        offline_valid_count = 0
        guided_valid_count = 0

        saved_offline_buffer = False
        shared_offline_batches: list[Batch[D]] = []
        shared_offline_loaded = False

        start_iter = 0
        resume_state = self._load_loop_state()
        if resume_state is not None:
            offline_batches = resume_state["offline_batches"]
            guided_batches = resume_state["guided_batches"]
            offline_valid_count = resume_state["offline_valid_count"]
            guided_valid_count = resume_state["guided_valid_count"]
            saved_offline_buffer = resume_state["saved_offline_buffer"]
            start_iter = resume_state["iteration"] + 1

            ckpt_path = self.config.folder / "checkpoints" / "last.pt"
            if ckpt_path.exists():
                device = next(self.base_model.parameters()).device
                self.base_model.load_state_dict(torch.load(ckpt_path, map_location=device))

            all_batches = offline_batches + guided_batches
            if all_batches and not self.config.no_uncertainty:
                self.update_uncertainty_estimator(all_batches)

            self.logger.info(f"Resuming from iteration {start_iter}")

        verbose = self.config.verbose
        iter_range = tqdm(range(start_iter, num_iterations), desc=f"iter {start_iter}/? | idle") if verbose else range(start_iter, num_iterations)

        for i in iter_range:
            def step(label: str):
                if verbose:
                    iter_range.set_description(f"iter {i}/{num_iterations} | {label}")

            with self._timer("eval"):
                metrics = {}
                if i % self.config.eval_every == 0:
                    step("eval")
                    metrics = self.eval_model(i)

            offline_done = False
            if use_buffer:
                total_offline_shared = self._total_buffer_valid_count()
                offline_done = total_offline_shared >= self.config.ft_min_dataset_size // 2
                use_guidance = offline_done and not self.config.no_uncertainty
                has_enough_data = offline_done and guided_valid_count >= self.config.ft_min_dataset_size // 2
            else:
                total_valid = offline_valid_count + guided_valid_count
                use_guidance = total_valid > self.config.ft_min_dataset_size / 2 and not self.config.no_uncertainty
                has_enough_data = total_valid > self.config.ft_min_dataset_size

            if not use_buffer and has_enough_data and not saved_offline_buffer:
                all_batches = Batch.concat(offline_batches + guided_batches)
                self.problem.save_samples(all_batches.samples, all_batches.kwargs, self.config.folder)
                torch.save(all_batches.valids, self.config.folder / "valids.pt")
                saved_offline_buffer = True

            with self._timer("sampling"):
                step("sampling")
                if use_guidance:
                    self.env.control_policy = RewardGradient(self.env, self.uncertainty)

                sample = self.env.batch_sample(samples_per_iter, self.config.sample_batch_size, pbar=verbose)
                batch = Batch.from_sample(sample)
                batch.latents = self.problem.postprocess_latents(batch)
                step("validity")
                batch.valids = self.problem.validity(batch.samples, batch.kwargs)

                self.env.control_policy = None

            batch = batch.cpu()
            n_valid = batch.valids.int().sum().item()

            if not use_guidance:
                offline_batches.append(batch)
                offline_valid_count += n_valid
                if use_buffer:
                    self._write_to_buffer(batch, i, offline_valid_count)
            else:
                guided_batches.append(batch)
                guided_valid_count += n_valid

            if use_buffer and offline_done and not shared_offline_loaded:
                shared_offline_batches = self._load_buffer(max_valid=self.config.ft_min_dataset_size // 2)
                shared_offline_loaded = True

            all_local_batches = offline_batches + guided_batches

            with self._timer("visualize"):
                step("visualize")
                self.visualize_iter(batch, i)

                with torch.no_grad():
                    mean, uncert = self.uncertainty.mean_and_uncertainty(batch.latents, **batch.kwargs)

                if self.config.mean_weight > 0:
                    metrics["pred_mean"] = mean.mean().item()
                    metrics["biased_reward"] = batch.rewards.mean().item()
                metrics["uncertainty"] = uncert.mean().item()
                metrics["biased_valid"] = batch.valids.float().mean().item()
                metrics["max_vram"] = torch.cuda.max_memory_allocated() * 1e-9
                self.logger.info(f"(iter={i:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}")

            with self._timer("finetune"):
                if has_enough_data:
                    step("finetune")
                    ft_batches = shared_offline_batches + guided_batches if use_buffer else all_local_batches
                    self.combined_finetune_base_model(ft_batches, pbar=verbose)

            with self._timer("uncertainty_update"):
                if not self.config.no_uncertainty:
                    step("uncertainty update")
                    if use_buffer:
                        uncertainty_batches = shared_offline_batches + guided_batches if shared_offline_loaded else offline_batches
                    else:
                        uncertainty_batches = all_local_batches
                    self.update_uncertainty_estimator(uncertainty_batches)

            with self._timer("checkpoint"):
                step("checkpoint")
                ckpt_dir = self.config.folder / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)

                state_dict = self.base_model.state_dict()
                torch.save(state_dict, ckpt_dir / "last.pt")

                if self.config.ckpt_every > 0 and i % self.config.ckpt_every == 0:
                    torch.save(state_dict, ckpt_dir / f"ckpt_{i}.pt")

                self._save_loop_state(i, offline_batches, guided_batches, offline_valid_count, guided_valid_count, saved_offline_buffer)

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

    loop = ExploreLoop(problem_setup=problem_setup, uncertainty=uncertainty, reward=reward, config=config, logger=logger)
    loop.explore_loop(config.num_iters, config.samples_per_iter)


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
    parser.add_argument("--buffer_dir", type=Path, default=None)
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

    # Negative sample finetuning
    parser.add_argument("--neg_grad_scale", type=float, default=0.0)
    parser.add_argument("--max_neg_samples", type=int, default=5000)

    # Logging
    parser.add_argument("--video_fps", type=int, default=4)
