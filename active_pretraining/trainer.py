from typing import Generic, Optional
from contextlib import contextmanager

import torch
from flowgym import construct_env, D
from flowgym.utils import train_base_model
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import logging
import yaml
import shutil
import time
import csv

from .gp import GPUncertaintyReward, FlowFeatureExtractor
from .dps import RewardGradient
from .svdd import sample_svdd_pm
from .problem_setup import ProblemSetup
from .utils import index_dict, Batch
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
            Data batches obtained so far, where one batch comes from one iteration in the order.

        pbar : bool, default: False
            Progress bar or not.
        """
        # Extract valid samples
        valid_latents = []
        valid_kwargs = []
        for batch in batches:
            for j in range(len(batch)):
                if batch.valids[j] or self.config.no_verifier:
                    valid_latents.append(batch.latents[j])
                    valid_kwargs.append(index_dict(batch.kwargs, j))

        if len(valid_latents) == 0:
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
            valid_latents,
            valid_kwargs,
            steps=self.config.ft_steps,
            batch_size=self.config.ft_batch_size,
            accumulate_steps=self.config.ft_accumulate_steps,
            pbar=pbar,
        )

    def update_uncertainty_estimator(self, batches: list[Batch[D]]):
        """Update the uncertainty estimator with obtained samples.

        Parameters
        ----------
        batches : list[Batch[D]]
            Data batches obtained so far, where one batch comes from one iteration in the order.
        """
        self.reward.set_data(batches)

    @torch.no_grad()
    def get_samples(self, n: int, kwargs: Optional[dict] = None, guided: bool = True) -> Batch[D]:
        """Obtain samples from the environment, optionally using uncertainty guidance.

        Parameters
        ----------
        n : int
            Number of samples to obtain.

        kwargs : Optional[dict], default=None
            Keyword arguments for sampling.

        guided : bool, default=True
            Whether to use uncertainty guidance when obtaining samples.

        Returns
        -------
        batch : Batch[D]
            Obtained samples, latents, and kwargs.
        """
        if kwargs is None:
            kwargs = {}

        # Use uncertainty gradients to guide the base model
        if guided:
            if self.config.reward_opt_algo == "svdd":
                output = sample_svdd_pm(self.env, n, m=4, alpha=1 / self.env.reward_scale, pbar=False, **kwargs)
            elif self.config.reward_opt_algo == "dps":
                self.env.control_policy = RewardGradient(self.env)
                output = self.env.sample(n, pbar=False, **kwargs)
                self.env.control_policy = None
            else:
                raise ValueError
        else:
            output = self.env.sample(n, pbar=False, **kwargs)

        # Obtain the final samples (converted to data space) and latents (terminator of SDE). We use
        # samples for validity and latents for updating the models.
        samples = output[0]
        latents = output[1][-1]
        kwargs = output[-1]
        valids = self.problem.validity(samples, kwargs).cpu()

        # Postprocess latents
        batch = Batch(samples, latents, valids, kwargs)
        batch.latents = self.problem.postprocess_latents(batch)
        return batch

    def get_many_batches(
        self,
        n: int,
        batch_size: int,
        kwargs: Optional[dict] = None,
        guided: bool = True,
        pbar: bool = False,
    ) -> list[Batch[D]]:
        """Obtain multiple batches of samples.

        Parameters
        ----------
        n : int
            Total number of samples to obtain.

        batch_size : int
            Number of samples per batch.

        kwargs : Optional[dict], default=None
            Keyword arguments for sampling.

        guided : bool, default=True
            Whether to use uncertainty guidance when obtaining samples.

        pbar : bool, default=True
            Progress bar or not.

        Returns
        -------
        batches : list[Batch[D]]
            Obtained batches of samples.
        """
        batches: list[Batch[D]] = []
        for i in range(0, n, batch_size):
            bsz = min(batch_size, n - i)
            batch_kwargs = {}
            if kwargs is not None:
                batch_kwargs = index_dict(kwargs, i, i + bsz)
            batch = self.get_samples(bsz, batch_kwargs, guided=guided)
            batches.append(batch)

        return batches

    def _write_video(self) -> None:
        frame_paths = sorted(self.config.folder.glob("frames/*.png"))
        if len(frame_paths) == 0:
            self.logger.warning("No frames found; skipping video creation.")
            return

        video_path = self.config.folder / "video.mp4"
        with imageio.get_writer(video_path, fps=self.config.video_fps, codec="libx264") as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))  # type: ignore

        self.logger.info(f"Wrote video to {video_path}")

    def visualize_iter(self, batch: Batch[D], iteration: int) -> None:
        # Visualizing an iteration is problem-dependent
        fig = self.problem.visualize_batch(self.env, batch)

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

        batches = self.get_many_batches(n, bs, guided=False)

        # Save samples
        directory = self.config.folder / "eval" / f"{iteration:04d}"
        samples_dir = directory / "samples"
        samples_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for i, batch in enumerate(batches):
            for j in range(len(batch)):
                self.problem.save_sample(
                    batch.samples[j],
                    index_dict(batch.kwargs, j),
                    samples_dir / f"{count:04d}",
                )
                count += 1

        all_valids = torch.cat([batch.valids for batch in batches], dim=0)
        torch.save(all_valids, directory / "valids.pt")

        # Compute and save evaluation metrics
        metrics = self.problem.compute_metrics(batches)
        metrics["model_valid"] = all_valids.float().mean().item()
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
                new_batches = self.get_many_batches(samples_per_iter, self.config.sample_batch_size, guided=use_guidance)
                batch = Batch.concat(new_batches)
                total_valid_samples += batch.valids.int().sum().item()

            # Store data
            batches.append(batch)

            # Logging and visualization
            with self._timer("visualize"):
                self.visualize_iter(batch, i)
                metrics["biased_valid"] = batch.valids.float().mean().item()
                metrics["max_vram"] = torch.cuda.max_memory_allocated() * 1e-9
                self.logger.info(f"(iter={i:05d}) {f', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}")

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

        self._write_video()
