from typing import Generic

import torch
from flowgym import construct_env, D
from flowgym.utils import train_base_model
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import argparse
from tqdm import trange
import imageio.v2 as imageio
import yaml
import glob
import shutil
import os

from .gp import GPUncertaintyReward, FlowFeatureExtractor
from .guidance import RewardGradient
from .problem_setup import ProblemSetup
from .setups import setups as problem_setups


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.dir, exist_ok=True)
    with open(os.path.join(args.dir, "args.yaml"), "w") as f:
        yaml.dump(vars(args), f)

    problem_setup = problem_setups[args.problem_setup](vars(args), device=device)

    active_pre = ActivePretraining(
        problem_setup,
        args.dir,
        feat_timestep=args.feature_timestep,
        gp_lengthscale=args.gp_lengthscale,
        discretization_steps=args.num_steps,
        reward_scale=args.uncertainty_weight,
        ft_batch_size=args.ft_batch_size,
        ft_epochs=args.ft_epochs,
        ft_lr=args.ft_lr,
        eval_samples=args.eval_samples,
        eval_every=args.eval_every,
        video_fps=args.video_fps,
    )

    active_pre.explore_loop(
        num_iterations=args.num_iters,
        samples_per_iter=args.samples_per_iter,
    )


class ActivePretraining(Generic[D]):
    def __init__(
        self,
        problem_setup: ProblemSetup[D],
        dir: os.PathLike,
        feat_timestep: float = 0.5,
        gp_lengthscale: float = 0.1,
        discretization_steps: int = 100,
        reward_scale: float = 1.0,
        ft_batch_size: int = 32,
        ft_epochs: int = 10,
        ft_lr: float = 1e-4,
        eval_samples: int = 0,
        eval_every: int = 5,
        video_fps: int = 10,
    ):
        assert (
            feat_timestep > 0 and feat_timestep <= 1
        ), "feat_timestep must be in (0, 1]"

        self.problem = problem_setup

        self.dir = dir
        self.ft_batch_size = ft_batch_size
        self.ft_epochs = ft_epochs
        self.ft_lr = ft_lr
        self.eval_samples = eval_samples
        self.eval_every = eval_every
        self.video_fps = video_fps

        feat_extractor = FlowFeatureExtractor(
            problem_setup.base_model,
            layer=problem_setup.feature_layer,
            timestep=feat_timestep,
            postprocess=problem_setup.feature_postprocess,
        )

        # Probe the feature extractor for testing and obtaining the dimensionality of the features
        x, kwargs = problem_setup.base_model.sample_p0(1)
        feat = feat_extractor(x, **kwargs)

        assert isinstance(
            feat, torch.Tensor
        ), "Feature extractor output must be a torch.Tensor"
        assert feat.ndim == 2, "Feature extractor output must be a 2D tensor"

        reward = GPUncertaintyReward(
            feat_extractor=feat_extractor,
            feat_dim=feat.shape[1],
            lengthscale=gp_lengthscale,
            device=x.device,
        )
        env = construct_env(
            problem_setup.base_model,
            reward,
            discretization_steps=discretization_steps,
            reward_scale=reward_scale,
        )

        self.base_model = problem_setup.base_model
        self.reward = reward
        self.env = env

    def update_models(self, latents: list[D], valids: list[torch.Tensor]) -> None:
        """Update the uncertainty estimator and fine-tune the base model on valid samples.

        Parameters
        ----------
        latents : list[D]
            List of latent samples obtained so far, where each element corresponds to the latents
            from one iteration.

        valids : list[torch.Tensor]
            List of boolean tensors indicating the validity of samples obtained so far, where each
            element corresponds to the valids from one iteration.
        """
        # Update uncertainty model
        self.reward.add_data(latents[-1].to(self.env.device))

        # Do not fine-tune the base model on the first iteration's samples. Those are only used for
        # initializing the uncertainty model.
        if len(latents) <= 1:
            return

        # Extract valid samples
        valid_latents = []
        for d, v in zip(latents, valids):
            for j in range(len(d)):
                if v[j]:
                    valid_latents.append(d[j])

        if len(valid_latents) == 0:
            print("No valid data to fine-tune")
            return

        # Fine-tune base model
        opt = torch.optim.AdamW(self.env.base_model.parameters(), lr=self.ft_lr)
        train_base_model(
            self.env.base_model,
            valid_latents,
            epochs=self.ft_epochs,
            batch_size=self.ft_batch_size,
            opt=opt,
            pbar=False,
        )

        # Update features of the uncertainty model after fine-tuning the base model
        self.reward.update_feats()

    @torch.no_grad()
    def get_samples(self, num_samples: int, guided: bool = True) -> tuple[D, D]:
        """Obtain samples from the environment, optionally using uncertainty guidance.

        Parameters
        ----------
        num_samples : int
            Number of samples to obtain.

        guided : bool, default=True
            Whether to use uncertainty guidance when obtaining samples.

        Returns
        -------
        samples : D
            The obtained samples in data space (e.g., pixel space for images).

        latents : D
            The obtained samples in latent space.
        """
        # Use uncertainty gradients to guide the base model
        if guided:
            self.env.control_policy = RewardGradient(self.env)

        # Obtain the final samples (converted to data space) and latents (terminator of SDE)
        # For validity, the samples are more important, but the latents are used for updating the
        # models
        output = self.env.sample(num_samples, pbar=False)
        samples = output[0]
        latents = output[1][-1]

        latents = self.problem.latent_postprocess(latents)

        # Reset control policy
        self.env.control_policy = None
        return samples, latents

    def _write_video(self) -> None:
        frame_dir = os.path.join(self.dir, "frames")
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
        if len(frame_paths) == 0:
            print("No frames found; skipping video creation.")
            return

        video_path = os.path.join(self.dir, "video.mp4")
        with imageio.get_writer(
            video_path, fps=self.video_fps, codec="libx264"
        ) as writer:
            for frame_path in frame_paths:
                writer.append_data(imageio.imread(frame_path))  # type: ignore

        print(f"Wrote video to {video_path}")

    def visualize_iter(self, samples: list[D], valids: list[torch.Tensor], iteration: int) -> None:
        # Problem-dependent visualization
        fig = self.problem.visualize_sample(self.env, samples, valids)

        # Save frame
        directory = os.path.join(self.dir, "frames")
        os.makedirs(directory, exist_ok=True)
        fig_path = os.path.join(directory, f"{iteration:04d}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)

    @torch.no_grad()
    def eval_model(self, iteration: int) -> dict[str, float]:
        n = self.eval_samples
        bs = self.ft_batch_size

        if n <= 0:
            return dict()

        samples: list[D] = []
        valids: list[torch.Tensor] = []

        # Obtain samples in batches
        for i in range(0, n, bs):
            bsz = min(bs, n - i)
            batch_samples, _ = self.get_samples(bsz, guided=False)
            batch_valids = self.problem.validity(batch_samples.to(self.env.device)).cpu()

            samples.append(batch_samples)
            valids.append(batch_valids)

        # Save samples
        directory = os.path.join(self.dir, "eval", f"{iteration:04d}")
        os.makedirs(directory, exist_ok=True)

        count = 0
        for i, sample in enumerate(samples):
            for j in range(len(sample)):
                self.problem.save_sample(sample[j], os.path.join(directory, f"{count:04d}"))
                count += 1

        torch.save(
            torch.cat(valids, dim=0),
            os.path.join(directory, "valids.pt"),
        )

        # Compute and save evaluation metrics
        metrics = self.problem.compute_metrics(samples, valids)
        metrics["model_valid"] = torch.cat(valids, dim=0).float().mean().item()
        with open(os.path.join(directory, "metrics.yaml"), "w") as f:
            yaml.dump(metrics, f)

        # Zip and delete folder
        shutil.make_archive(directory, "zip", directory)
        shutil.rmtree(directory)

        return metrics

    def explore_loop(self, num_iterations: int, samples_per_iter: int):
        metrics = dict()
        all_samples: list[D] = []
        all_latents: list[D] = []
        all_valids: list[torch.Tensor] = []

        pbar = trange(num_iterations)
        for i in pbar:
            # Fetch data and evaluate their validity
            samples, latents = self.get_samples(samples_per_iter, guided=i > 0)
            valids = self.problem.validity(samples.to(self.env.device)).cpu()

            # Add to data buffer
            all_samples.append(samples)
            all_latents.append(latents)
            all_valids.append(valids)

            # Update models with full buffer
            self.update_models(all_latents, all_valids)

            # Visualize and evaluate current model state
            self.visualize_iter(all_samples, all_valids, i)
            if i % self.eval_every == 0:
                metrics = self.eval_model(i)

            # Logging
            pbar.set_postfix(
                {
                    **metrics,
                    "biased_valid": valids.float().mean().item(),
                    "vram (gb)": torch.cuda.memory_allocated() * 1e-9,
                    "max_vram (gb)": torch.cuda.max_memory_allocated() * 1e-9,
                }
            )

        self._write_video()
        return all_samples, all_valids


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
        "--dir", type=str, required=True, help="Directory to save outputs."
    )
    parser.add_argument(
        "--feature_timestep",
        type=float,
        default=0.5,
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
        "--gp_lengthscale",
        type=float,
        default=0.1,
        help="Lengthscale for the GP uncertainty reward.",
    )
    parser.add_argument(
        "--uncertainty_weight",
        type=float,
        default=10.0,
        help="Weight of the uncertainty reward for finding informative samples.",
    )
    parser.add_argument(
        "--ft_epochs",
        type=int,
        default=100,
        help="Number of optimization steps for fine-tuning the base model on informative samples.",
    )
    parser.add_argument(
        "--ft_batch_size",
        type=int,
        default=256,
        help="Batch size for fine-tuning the base model on informative samples.",
    )
    parser.add_argument(
        "--ft_lr",
        type=float,
        default=1e-4,
        help="Learning rate for fine-tuning the base model on informative samples.",
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
        "--eval_every",
        type=int,
        default=5,
        help="Frequency (in iterations) of computing metrics.",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=4,
        help="Frames per second for the output video.",
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

    args = parser.parse_args()
    main(args)
