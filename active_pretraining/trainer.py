from typing import Generic
from contextlib import contextmanager

import torch
from flowgym import construct_env, D
from flowgym.utils import train_base_model, index_dict
import matplotlib.pyplot as plt
import logging
import yaml
import shutil
import time
import csv

from .gp import GPUncertaintyReward, FlowFeatureExtractor
from .dps import RewardGradient
from .problem_setup import ProblemSetup
from .utils import write_video, filter_out_invalids, Batch
from .config import ActivePretrainingConfig


class ActivePretraining(Generic[D]):
    def __init__(
        self,
        problem_setup: ProblemSetup[D],
        config: ActivePretrainingConfig,
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

        reward = GPUncertaintyReward(
            feat_extractor=feat_extractor,
            feat_dim=feat.shape[1],
            valid_fn=self.problem.validity,
            kernel=config.gp_kernel,
            lengthscale=config.gp_lengthscale,
            device=x.device,
        )
        env = construct_env(
            problem_setup.base_model,
            reward,
            discretization_steps=config.num_steps,
            reward_scale=config.uncertainty_weight,
        )

        self.base_model = problem_setup.base_model
        self.reward = reward
        self.env = env

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

    def finetune_base_model(self, batches: list[Batch[D]], pbar: bool = False):
        """Fine-tune the base model with obtained valid samples.

        Parameters
        ----------
        batches : list[Batch[D]]
            Data batches obtained so far, where each entry is from a separate iteration.

        pbar : bool, default: False
            Progress bar or not.
        """
        valid_batch = filter_out_invalids(batches)
        if len(valid_batch) == 0:
            self.logger.warning("No valid data to fine-tune")
            return

        # Fine-tune base model
        opt = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=self.config.ft_lr,
            weight_decay=self.config.ft_weight_decay,
        )
        train_base_model(
            self.env.base_model,
            opt,
            [valid_batch.latents.to(self.env.base_model.device)],
            [valid_batch.kwargs],
            steps=self.config.ft_steps,
            batch_size=self.config.ft_batch_size,
            accumulate_steps=self.config.ft_accumulate_steps,
            pbar=True,
        )

    def update_uncertainty_estimator(self, batches: list[Batch[D]]):
        """Update the uncertainty estimator with obtained samples.

        Parameters
        ----------
        samples : list[Batch[D]]
            Data batches obtained so far, where one batch comes from one iteration in the order.
        """
        self.reward.set_data([b.latents for b in batches], [b.kwargs for b in batches])

    def visualize_iter(self, batch: Batch[D], iteration: int):
        # Visualizing an iteration is problem-dependent
        fig = self.problem.visualize_sample(self.env, batch)

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

        if n <= 0:
            return dict()

        eval_kwargs = self.problem.eval_sampling_kwargs(n)
        sample = self.env.batch_sample(n, bs, **eval_kwargs)
        batch = Batch.from_sample(sample)

        # Save samples
        directory = self.config.folder / "eval" / f"{iteration:04d}"
        samples_dir = directory / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        for i in range(len(sample)):
            self.problem.save_sample(
                batch.samples[i],
                index_dict(batch.kwargs, i),
                samples_dir / f"{i:04d}",
            )

        torch.save(sample.valids, directory / "valids.pt")

        # Compute and save evaluation metrics
        metrics = self.problem.compute_metrics(batch)
        metrics["model_valid"] = batch.valids.float().mean().item()
        with open(directory / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

        # Zip and delete folder
        shutil.make_archive(str(samples_dir), "zip", samples_dir)
        shutil.rmtree(samples_dir)

        return metrics

    def explore_loop(self, num_iterations: int, samples_per_iter: int):
        metrics = dict()
        batches: list[Batch[D]] = []
        total_valid_samples = 0

        for i in range(num_iterations):
            # Evaluate current model state
            with self._timer("eval"):
                metrics = {}
                if i % self.config.eval_every == 0:
                    metrics = self.eval_model(i)

            # Determine if we should use uncertainty guidance
            has_enough_data = total_valid_samples > self.config.ft_min_dataset_size
            use_guidance = has_enough_data and not self.config.no_uncertainty

            # Collect new samples
            with self._timer("sampling"):
                if use_guidance:
                    self.env.control_policy = RewardGradient(self.env)

                sample = self.env.batch_sample(samples_per_iter, self.config.sample_batch_size)
                batch = Batch.from_sample(sample)
                batch.latents = self.problem.postprocess_latents(batch)
                total_valid_samples += batch.valids.int().sum().item()

                self.env.control_policy = None

            # Store data
            batches.append(batch)

            # Logging and visualization
            with self._timer("visualize"):
                self.visualize_iter(batch, i)
                metrics["biased_valid"] = batch.valids.float().mean().item()
                metrics["max_vram"] = torch.cuda.max_memory_allocated() * 1e-9
                self.logger.info(f"(iter={i:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}")

            # Update models with full buffer
            with self._timer("finetune"):
                if has_enough_data:
                    self.finetune_base_model(batches, pbar=False)

            with self._timer("uncertainty_update"):
                if not self.config.no_uncertainty:
                    self.update_uncertainty_estimator(batches)

            # Save checkpoint
            with self._timer("checkpoint"):
                torch.save(self.base_model.state_dict(), self.config.folder / "base_model.pt")

            # Write timings to CSV
            self._flush_timings(i)
        
        write_video(
            sorted(self.config.folder.glob("frames/*.png")),
            self.config.folder / "video.mp4",
            fps=self.config.video_fps,
        )
