from typing import Optional, Callable

import torch
from flowgym import reward_registry, construct_env, D, Environment
from flowgym.utils import train_base_model
import argparse
import yaml
import logging
from pathlib import Path
import json

from .setups import setups as problem_setups
from .problem_setup import ProblemSetup
from .utils import Batch, filter_out_invalids


def main(args):
    setup_logging()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir: Path = args.dir

    with open(exp_dir / "args.yaml", "r") as f:
        exp_args = yaml.safe_load(f)

    problem_setup: ProblemSetup = problem_setups[exp_args["problem_setup"]](exp_args, device=device)
    reward = reward_registry.get(args.reward).instantiate(**args.reward_args)
    env = construct_env(problem_setup.base_model, reward, exp_args["num_steps"], args.reward_scale)

    if args.use_last_ckpt:
        ckpt_path = exp_dir / "base_model.pt"
        state_dict = torch.load(ckpt_path, map_location=device)
        env.base_model.load_state_dict(state_dict)

    orw_cfm(
        env,
        samples_per_iter=args.samples_per_iter,
        batch_size=args.batch_size,
        num_iterations=args.num_iterations,
        lr=args.lr,
        postprocess_latents=problem_setup.postprocess_latents,
        log_every=args.log_every,
        save_path=exp_dir / "fine_tuned" / f"{args.reward.split('/')[-1]}.pt"
    )


def orw_cfm(
    env: Environment[D],
    samples_per_iter: int,
    batch_size: int,
    num_iterations: int = 100,
    lr: float = 1e-5,
    postprocess_latents: Optional[Callable[[Batch[D]], D]] = None,
    log_every: Optional[int] = None,
    save_path: Optional[Path] = None,
):
    opt = torch.optim.AdamW(env.base_model.parameters(), lr=lr)

    if log_every is None:
        log_every = max(1, num_iterations // 100)

    for it in range(1, num_iterations + 1):
        sample = env.batch_sample(samples_per_iter, batch_size)

        metrics = {
            "r_mean": sample.rewards[sample.valids].mean(),
            "r_std": sample.rewards[sample.valids].std(),
            "r_max": sample.rewards[sample.valids].max(),
            "valid": sample.valids.float().mean(),
        }
        logging.info(f"(iter={it:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}")

        batch = Batch.from_sample(sample)
        batch = filter_out_invalids([batch])

        if postprocess_latents is not None:
            batch.latents = postprocess_latents(batch)

        weights = torch.exp(env.reward_scale * batch.rewards)

        train_base_model(
            env.base_model,
            opt,
            [batch.latents.to(env.base_model.device)],
            [batch.kwargs],
            weights=[weights],
            steps=100,
            pbar=False,
        )

        if save_path is not None:
            torch.save(env.base_model.state_dict(), save_path)


def setup_logging():
    formatter = logging.Formatter(
        "[%(asctime)s] (%(levelname)s) %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    root.addHandler(ch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument(
        "--reward_args",
        type=json.loads,
        default={},
        help="JSON string, e.g. '{\"alpha\": 0.1, \"beta\": 2}'"
    )
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--samples_per_iter", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_last_ckpt", action="store_true")
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--log_every", type=int, default=None)
    args = parser.parse_args()
    main(args)
