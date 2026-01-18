from typing import Generic, Any, Optional
from typing_extensions import Self
from dataclasses import dataclass

import torch
from flowgym import construct_env, D
from flowgym.utils import train_base_model
import matplotlib.pyplot as plt
import argparse
import imageio.v2 as imageio
from pathlib import Path
import logging
import yaml
import glob
import shutil
import os

from .gp import GPUncertaintyReward, FlowFeatureExtractor
from .dps import RewardGradient
from .svdd import sample_svdd_pm
from .problem_setup import ProblemSetup
from .setups import setups as problem_setups
from .utils import index_dict


logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ActivePretrainingConfig:
    # Experiment directory
    folder: Path

    # Feature extraction and Gaussian Process
    feat_timestep: float = 0.9
    gp_kernel: str = "rbf"
    gp_lengthscale: float = 0.1

    # Uncertainty reward and uncertainty sampling algorithm
    reward_scale: float = 100.0
    reward_opt_algo: str = "dps"
    
    # Fine-tuning
    ft_min_dataset_size: int = 64
    ft_batch_size: int = 64
    ft_accumulate_steps: int = 1
    ft_steps: int = 500
    ft_lr: float = 1e-4
    ft_weight_decay: float = 0.0

    # Sampling and evaluation
    num_steps: int = 100
    eval_samples: int = 0
    eval_batch_size: int = 64
    eval_every: int = 10
    video_fps: int = 4
    
    # Flags
    no_uncertainty: bool = False
    no_verifier: bool = False

    def __post_init__(self):
        # Create experiment directory if it doesn't exist
        self.folder.mkdir(parents=True, exist_ok=True)

        # Validation
        if not (0 <= self.feat_timestep <= 1):
            raise ValueError(f"feat_timestep must be in [0, 1], got {self.feat_timestep}")

        if self.reward_scale < 0:
            raise ValueError(f"reward_scale cannot be negative, got {self.reward_scale}")

        if self.gp_lengthscale < 0:
            raise ValueError(f"gp_lengthscale cannot be negative, got {self.gp_lengthscale}")

        allowed_algos = { "dps", "svdd" }
        if self.reward_opt_algo not in allowed_algos:
            raise ValueError(f"reward_opt_algo must be one of {allowed_algos}")

        allowed_kernels = { "rbf", "linear" }
        if self.gp_kernel not in allowed_kernels:
            raise ValueError(f"gp_kernel must be one of {allowed_kernels}")

    @staticmethod
    def construct_from_args(args: argparse.Namespace | dict) -> "ActivePretrainingConfig":
        if isinstance(args, argparse.Namespace):
            args = vars(args)

        # Map argparse flags to config names
        name_mapping = {
            "dir": "folder",
            "feature_timestep": "feat_timestep",
            "uncertainty_weight": "reward_scale",
        }
        
        # We only take keys that exist in the dataclass fields
        config_fields = {f.name for f in ActivePretrainingConfig.__dataclass_fields__.values()}
        clean_kwargs = {}

        for key, value in args.items():
            # Check if the key needs a rename
            target_key = name_mapping.get(key, key)
            
            # Only add if it's a valid field in our config
            if target_key in config_fields:
                clean_kwargs[target_key] = value

        return ActivePretrainingConfig(**clean_kwargs)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ActivePretrainingConfig.construct_from_args(args)
    problem_setup = problem_setups[args.problem_setup](vars(args), device=device)

    # Save arguments
    with open(config.folder / "args.yaml", "w") as f:
        yaml.dump(vars(args), f)

    # Set up logging
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] (%(levelname)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(config.folder / "log.txt")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    apt = ActivePretraining(problem_setup=problem_setup, config=config)
    apt.explore_loop(args.num_iters, args.samples_per_iter)


class ActivePretraining(Generic[D]):
    def __init__(self, problem_setup: ProblemSetup[D], config: ActivePretrainingConfig):
        self.problem = problem_setup
        self.config = config

        feat_extractor = FlowFeatureExtractor(
            problem_setup.base_model,
            layer=problem_setup.feature_layer,
            timestep=config.feat_timestep,
            postprocess=problem_setup.feature_postprocess,
        )

        # Probe the feature extractor for testing and obtaining the dimensionality of the features
        x, kwargs = problem_setup.base_model.sample_p0(1)
        x, kwargs = problem_setup.base_model.preprocess(x, **kwargs)
        feat = feat_extractor(x, **kwargs)

        assert isinstance(
            feat, torch.Tensor
        ), "Feature extractor output must be a torch.Tensor"
        assert feat.ndim == 2, "Feature extractor output must be a 2D tensor"

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
            reward_scale=config.reward_scale,
        )

        self.base_model = problem_setup.base_model
        self.reward = reward
        self.env = env

    def update_models(
        self,
        latents: list[D],
        valids: list[torch.Tensor],
        kwargs: list[dict[str, Any]],
        update_base_model: bool = True,
        update_uncertainty: bool = True,
    ) -> None:
        """Update the uncertainty estimator and fine-tune the base model on valid samples.

        Parameters
        ----------
        latents : list[D]
            List of latent samples obtained so far, where each element corresponds to the latents
            from one iteration.

        valids : list[torch.Tensor]
            List of boolean tensors indicating the validity of samples obtained so far, where each
            element corresponds to the valids from one iteration.

        kwargs : list[dict[str, Any]]
            Keyword arguments used for sampling.

        update_base_model : bool, default: True
            Whether to fine-tune the base model on valid samples.

        update_uncertainty : bool, default: True
            Whether to update the uncertainty estimator.
        """
        if update_base_model:
            # Extract valid samples
            valid_latents = []
            valid_kwargs = []
            for d, v, k in zip(latents, valids, kwargs):
                for j in range(len(d)):
                    if v[j]:
                        valid_latents.append(d[j])
                        valid_kwargs.append(index_dict(k, j))

            if len(valid_latents) == 0:
                logger.warning("No valid data to fine-tune")
                return

            if self.config.no_verifier:
                valid_latents = latents

            logger.debug(f"fine-tuning the base model on {len(valid_latents)} samples")

            # Fine-tune base model
            opt = torch.optim.Adam(
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
                pbar=False,
            )

        if update_uncertainty:
            # Update uncertainty model
            logger.debug("updating uncertainty model")
            self.reward.set_data(latents, valids, kwargs)

    @torch.no_grad()
    def get_samples(
        self,
        num_samples: int,
        kwargs: Optional[dict] = None,
        guided: bool = True,
    ) -> tuple[D, D, dict[str, Any]]:
        """Obtain samples from the environment, optionally using uncertainty guidance.

        Parameters
        ----------
        num_samples : int
            Number of samples to obtain.

        kwargs : Optional[dict], default=None
            Keyword arguments for sampling.

        guided : bool, default=True
            Whether to use uncertainty guidance when obtaining samples.

        Returns
        -------
        samples : D
            The obtained samples in data space (e.g., pixel space for images).

        latents : D
            The obtained samples in latent space.

        kwargs : dict[str, Any]
            Keyword arguments output.
        """
        if kwargs is None:
            kwargs = {}

        logger.debug(f"obtaining {num_samples} samples (guided={guided})")

        # Use uncertainty gradients to guide the base model
        if guided:
            if self.config.reward_opt_algo == "svdd":
                output = sample_svdd_pm(
                    self.env,
                    num_samples,
                    m=4,
                    alpha=1 / self.env.reward_scale,
                    pbar=False,
                    **kwargs,
                )
            elif self.config.reward_opt_algo == "dps":
                self.env.control_policy = RewardGradient(self.env)
                output = self.env.sample(num_samples, pbar=False, **kwargs)
                self.env.control_policy = None
            else:
                raise ValueError
        else:
            output = self.env.sample(num_samples, pbar=False, **kwargs)

        # Obtain the final samples (converted to data space) and latents (terminator of SDE). For
        # validity, the samples are more important, but the latents are used for updating the models.
        samples = output[0]
        latents = output[1][-1]
        kwargs = output[-1]

        return samples, latents, kwargs

    def _write_video(self) -> None:
        logger.debug("writing video")

        frame_paths = sorted(glob.glob(str(self.config.folder / "frames" / "*.png")))
        if len(frame_paths) == 0:
            logging.warning("No frames found; skipping video creation.")
            return

        video_path = self.config.folder / "video.mp4"
        with imageio.get_writer(
            video_path, fps=self.config.video_fps, codec="libx264"
        ) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))  # type: ignore

        logger.info(f"Wrote video to {video_path}")

    def visualize_iter(self, samples: list[D], valids: list[torch.Tensor], iteration: int) -> None:
        logger.debug(f"visualizing iteration {iteration}")

        # Problem-dependent visualization
        fig = self.problem.visualize_sample(self.env, samples, valids)

        # Save frame
        directory = self.config.folder / "frames"
        os.makedirs(directory, exist_ok=True)
        fig_path = directory / f"{iteration:04d}.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

    @torch.no_grad()
    def eval_model(self, iteration: int) -> dict[str, float]:
        n = self.config.eval_samples
        bs = self.config.eval_batch_size

        logger.debug(f"evaluating model at iteration {iteration} (n={n}, bs={bs})")

        if n <= 0:
            return dict()

        samples: list[D] = []
        valids: list[torch.Tensor] = []
        kwargs: list[dict] = []

        # Obtain samples in batches
        eval_kwargs = self.problem.eval_sampling_kwargs(n)
        for i in range(0, n, bs):
            bsz = min(bs, n - i)
            batch_kwargs = index_dict(eval_kwargs, i, i + bsz)
            batch_samples, _, batch_kwargs = self.get_samples(bsz, batch_kwargs, guided=False)
            batch_valids = self.problem.validity(batch_samples, batch_kwargs).cpu()

            samples.append(batch_samples)
            valids.append(batch_valids)
            kwargs.append(batch_kwargs)

        # Save samples
        directory = self.config.folder / "eval" / f"{iteration:04d}"
        samples_dir = directory / "samples"
        os.makedirs(samples_dir, exist_ok=True)

        count = 0
        for i, (sample, kwarg) in enumerate(zip(samples, kwargs)):
            for j in range(len(sample)):
                self.problem.save_sample(
                    sample[j],
                    index_dict(kwarg, j),
                    samples_dir / f"{count:04d}",
                )
                count += 1

        torch.save(torch.cat(valids, dim=0), directory / "valids.pt")

        # Compute and save evaluation metrics
        metrics = self.problem.compute_metrics(samples, valids, kwargs)
        metrics["model_valid"] = torch.cat(valids, dim=0).float().mean().item()
        with open(directory / "metrics.yaml", "w") as f:
            yaml.dump(metrics, f)

        # Zip and delete folder
        shutil.make_archive(str(samples_dir), "zip", samples_dir)
        shutil.rmtree(samples_dir)

        return metrics

    def explore_loop(self, num_iterations: int, samples_per_iter: int):
        metrics = dict()
        all_samples: list[D] = []
        all_latents: list[D] = []
        all_valids: list[torch.Tensor] = []
        all_kwargs: list[dict[str, Any]] = []
        n_valids = 0

        for i in range(num_iterations):
            # Evaluate current model state
            if i % self.config.eval_every == 0:
                metrics = self.eval_model(i)
            else:
                metrics = dict()

            # Fetch data and evaluate their validity
            samples, latents, kwargs = self.get_samples(
                samples_per_iter,
                guided=(n_valids > self.config.ft_min_dataset_size) and (not self.config.no_uncertainty),
            )
            valids = self.problem.validity(samples, kwargs).cpu()
            n_valids += valids.float().sum().item()

            # Postprocess latents
            latents = self.problem.latent_postprocess(latents, valids, kwargs)

            # Add to data buffer
            all_samples.append(samples)
            all_latents.append(latents)
            all_valids.append(valids)
            all_kwargs.append(kwargs)

            # Visualize current state
            self.visualize_iter(all_samples, all_valids, i)

            # Logging
            metrics["biased_valid"] = valids.float().mean().item()
            metrics["max_vram"] = torch.cuda.max_memory_allocated() * 1e-9

            logger.info(f"(iter={i:05d}) {f', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}")

            # Update models with full buffer
            self.update_models(
                all_latents,
                all_valids,
                all_kwargs,
                update_base_model=n_valids > self.config.ft_min_dataset_size,
                update_uncertainty=(not self.config.no_uncertainty),
            )

            # Save checkpoint
            torch.save(self.base_model.state_dict(), self.config.folder / "base_model.pt")

        self._write_video()
        return all_samples, all_valids, all_kwargs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--problem_setup",
        type=str,
        choices=list(problem_setups.keys()),
        required=True,
        help="Problem setup.",
    )
    parser.add_argument(
        "--dir", type=Path, required=True, help="Directory to save outputs."
    )
    parser.add_argument(
        "--no_uncertainty",
        action="store_true",
        help="Whether to disable uncertainty-guided sampling.",
    )
    parser.add_argument(
        "--no_verifier",
        action="store_true",
        help="Whether to filter out invalid samples when fine-tuning.",
    )
    parser.add_argument("--verbose", action="store_true", help="Whether to enable verbose logging.")
    parser.add_argument(
        "--feature_timestep",
        type=float,
        default=0.9,
        help="Which timestep to use for obtaining features. Earlier timesteps give more high-level semantic information whereas later timesteps give more texture information.",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=50,
        help="Number of active exploration iterations.",
    )
    parser.add_argument(
        "--samples_per_iter",
        type=int,
        default=64,
        help="Number of 'informative' samples to generate per iteration.",
    )
    parser.add_argument(
        "--gp_kernel",
        type=str,
        choices=["rbf", "linear"],
        default="rbf",
        help="Kernel type for the Gaussian process.",
    )
    parser.add_argument(
        "--gp_lengthscale",
        type=float,
        default=0.1,
        help="Lengthscale for the GP uncertainty reward.",
    )
    parser.add_argument(
        "--uncertainty_weight",
        type=float,
        default=100,
        help="Weight of the uncertainty reward for finding informative samples.",
    )
    parser.add_argument(
        "--ft_min_dataset_size",
        type=int,
        default=64,
        help="Minimum number of samples required to fine-tune the base model.",
    )
    parser.add_argument(
        "--ft_steps",
        type=int,
        default=500,
        help="Number of optimization steps for fine-tuning the base model on informative samples.",
    )
    parser.add_argument(
        "--ft_batch_size",
        type=int,
        default=64,
        help="Batch size for fine-tuning the base model on informative samples.",
    )
    parser.add_argument(
        "--ft_accumulate_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps for fine-tuning the base model on informative samples.",
    )
    parser.add_argument(
        "--ft_lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning the base model on informative samples.",
    )
    parser.add_argument(
        "--ft_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay for fine-tuning the base model on informative samples.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of diffusion ODE/SDE discretization steps.",
    )
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=0,
        help="Number of samples for computing metrics.",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=64,
        help="Batch size for evaluation sampling.",
    )
    parser.add_argument(
        "--eval_every",
        type=int,
        default=10,
        help="Frequency (in iterations) of computing metrics.",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=4,
        help="Frames per second for the output video.",
    )
    parser.add_argument(
        "--reward_opt_algo",
        type=str,
        choices=["dps", "svdd"],
        default="dps",
        help="Optimization algorithm for obtaining samples with uncertainty reward.",
    )

    # MNIST-specific arguments
    parser.add_argument(
        "--mnist_base_digits",
        type=int,
        nargs="+",
        default=[3, 5],
        help="Digits to train the MNIST base model on.",
    )
    parser.add_argument(
        "--mnist_valid_digits",
        type=int,
        nargs="+",
        default=[3, 5, 8],
        help="Digits considered valid for the MNIST validity function.",
    )

    # Stable Diffusion-specific arguments
    parser.add_argument(
        "--sd_prompts",
        type=str,
        nargs="+",
        default=None,
        help="Text prompts for Stable Diffusion sampling.",
    )
    parser.add_argument(
        "--sd_cfg_scale",
        type=float,
        default=0.0,
        help="Classifier-free guidance scale for Stable Diffusion sampling.",
    )
    parser.add_argument(
        "--sd_score_threshold",
        type=float,
        default=20.0,
        help="Threshold for the Stable Diffusion validity function.",
    )

    # Molecule-specific arguments
    parser.add_argument(
        "--mol_geometry_opt",
        type=str,
        choices=["none", "mmff", "uff", "gfn2"],
        default="mmff",
        help="Geometry optimization method for molecules.",
    )

    args = parser.parse_args()
    main(args)
