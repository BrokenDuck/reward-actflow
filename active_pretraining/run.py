import os
import numpy as np
import torch
import argparse
from pathlib import Path
import logging
import yaml

from .setups import setups as problem_setups
from .config import ActivePretrainingConfig
from .trainer import ActivePretraining


def main(args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # If running inside a wandb sweep, append the run ID as a subdirectory
    if os.environ.get("WANDB_SWEEP_ID"):
        import wandb
        wandb.init()
        args.dir = args.dir / wandb.run.id

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = ActivePretrainingConfig.construct_from_args(args)
    problem_setup = problem_setups[args.problem_setup](vars(args), device=device)

    # Save arguments
    with open(config.folder / "args.yaml", "w") as f:
        yaml.safe_dump(serialize_args(args), f)

    # Set up logging
    logger = setup_logger(config, args.verbose)

    apt = ActivePretraining(problem_setup=problem_setup, config=config, logger=logger)
    apt.explore_loop(config.num_iters, config.samples_per_iter)


def setup_logger(config, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("active_pretraining")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] (%(levelname)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    fh = logging.FileHandler(config.folder / "log.txt")
    fh.setFormatter(formatter)

    logger.handlers.clear()
    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger


def serialize_args(args: argparse.Namespace) -> dict:
    out = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            out[k] = str(v)
        else:
            out[k] = v
    return out


def build_parser():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="problem_setup", required=True)
    for name, setup_cls in problem_setups.items():
        sub = subparsers.add_parser(name)
        add_global_args(sub)
        setup_cls.add_args(sub)

    return parser


def add_global_args(parser):
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--seed", type=int, default=None)

    # Baselines
    parser.add_argument("--no_uncertainty", action="store_true")
    parser.add_argument("--no_verifier", action="store_true")

    # Exploration sampling
    parser.add_argument("--num_iters", type=int, default=1000)
    parser.add_argument("--samples_per_iter", type=int, default=64)
    parser.add_argument("--sample_batch_size", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=100)

    # Uncertainty estimator
    parser.add_argument("--gp_kernel", type=str, choices=["rbf", "linear"], default="rbf")
    parser.add_argument("--gp_lengthscale", type=float, default=0.1)
    parser.add_argument("--feat_timestep", type=float, default=0.9)
    parser.add_argument("--uncertainty_weight", type=float, default=100)

    # Fine-tuning
    parser.add_argument("--ft_min_dataset_size", type=int, default=64)
    parser.add_argument("--ft_steps", type=int, default=500)
    parser.add_argument("--ft_batch_size", type=int, default=64)
    parser.add_argument("--ft_accumulate_steps", type=int, default=1)
    parser.add_argument("--ft_lr", type=float, default=1e-4)
    parser.add_argument("--ft_weight_decay", type=float, default=0.0)

    # Sampling
    parser.add_argument("--eval_samples", type=int, default=0)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--eval_every", type=int, default=10)

    # Logging
    parser.add_argument("--video_fps", type=int, default=4)


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)
