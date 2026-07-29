from typing import Optional, Callable

import torch
from diffusiongym import reward_registry, construct_env, D, Environment
from diffusiongym.utils import train_base_model
import argparse
import yaml
import logging
from pathlib import Path
import json

from reward_actflow.setups import setups as problem_setups
from reward_actflow.setups.problem_setup import ProblemSetup
from reward_actflow.utils import Batch, filter_out_invalids, serialize_args


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir: Path = args.dir

    with open(exp_dir / "args.yaml", "r") as f:
        exp_args = yaml.safe_load(f)

    problem_setup: ProblemSetup = problem_setups[exp_args["problem_setup"]](
        exp_args, device=device
    )
    reward = reward_registry.get(args.reward).instantiate(**args.reward_args)
    env = construct_env(
        problem_setup.base_model, reward, exp_args["num_steps"], args.reward_scale
    )

    ft_dir = (
        args.dir
        / "fine_tuned"
        / f"{args.reward.split('/')[-1]}_{args.reward_scale}_{'last' if args.use_last_ckpt else 'base'}"
    )
    ft_dir.mkdir(parents=True, exist_ok=True)

    with open(ft_dir / "args.yaml", "w") as f:
        yaml.safe_dump(serialize_args(args), f)

    if args.use_last_ckpt:
        ckpt_path = exp_dir / "base_model.pt"
        state_dict = torch.load(ckpt_path, map_location=device)
        env.base_model.load_state_dict(state_dict)

    orw_cfm(
        env,
        samples_per_iter=args.samples_per_iter,
        batch_size=args.batch_size,
        steps_per_iter=args.steps_per_iter,
        num_iterations=args.num_iterations,
        lr=args.lr,
        postprocess_latents=problem_setup.postprocess_latents,
        log_every=args.log_every,
        exp_dir=ft_dir,
    )


def orw_cfm(
    env: Environment[D],
    samples_per_iter: int,
    batch_size: int,
    steps_per_iter: int,
    num_iterations: int = 100,
    lr: float = 1e-5,
    postprocess_latents: Optional[Callable[[Batch[D]], D]] = None,
    log_every: Optional[int] = None,
    exp_dir: Optional[Path] = None,
):
    if exp_dir is not None:
        exp_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(exp_dir)

    if log_every is None:
        log_every = max(1, num_iterations // 100)

    logging.info(f"Reward scale: {env.reward_scale}")

    opt = torch.optim.AdamW(env.base_model.parameters(), lr=lr)

    # Track first and second moment
    r_ema_m1 = None
    r_ema_m2 = None

    # Determine beta based on halflife 10% of iterations
    halflife = 0.1 * num_iterations
    beta = 1 - 2 ** (-1 / halflife)

    for it in range(1, num_iterations + 1):
        sample = env.batch_sample(samples_per_iter, batch_size)
        batch = Batch.from_sample(sample)
        batch = filter_out_invalids([batch])

        if postprocess_latents is not None:
            batch.latents = postprocess_latents(batch)

        r = batch.rewards

        with torch.no_grad():
            if r_ema_m1 is None or r_ema_m2 is None:
                r_ema_m1 = r.mean()
                r_ema_m2 = (r**2).mean()
            else:
                r_ema_m1 = (1 - beta) * r_ema_m1 + beta * r.mean()
                r_ema_m2 = (1 - beta) * r_ema_m2 + beta * (r**2).mean()

            r_ema_var = (r_ema_m2 - r_ema_m1**2).clamp_min(1e-6)

        r = (r - r_ema_m1) / (r_ema_var.sqrt() + 1e-8)
        r = r.clamp(-5, 5)
        weights = torch.exp(env.reward_scale * r)
        weights = weights / weights.mean()

        metrics = {
            "r_mean": sample.rewards[sample.valids].mean(),
            "r_std": sample.rewards[sample.valids].std(),
            "r_min": sample.rewards[sample.valids].min(),
            "r_max": sample.rewards[sample.valids].max(),
            "valid": sample.valids.float().mean(),
            "ess": (weights.sum() ** 2)
            / (weights.pow(2).sum() + 1e-8)
            / sample.valids.int().sum(),
        }
        logging.info(
            f"(iter={it:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}"
        )

        train_base_model(
            env.base_model,
            opt,
            [batch.latents.to(env.base_model.device)],
            [batch.kwargs],
            weights=[weights],
            steps=steps_per_iter,
            pbar=False,
        )

        if exp_dir is not None:
            torch.save(env.base_model.state_dict(), exp_dir / "last.pt")


def setup_logging(log_dir: Optional[Path] = None) -> None:
    formatter = logging.Formatter(
        "[%(asctime)s] (%(levelname)s) %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers = []

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    handlers.append(ch)

    # File handler
    if log_dir is not None:
        fh = logging.FileHandler(log_dir / "log.txt")
        fh.setFormatter(formatter)
        handlers.append(fh)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Clear existing handlers (important for notebooks / re-runs)
    root.handlers.clear()
    for h in handlers:
        root.addHandler(h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--reward", type=str, required=True)
    parser.add_argument(
        "--reward_args",
        type=json.loads,
        default={},
        help='JSON string, e.g. \'{"alpha": 0.1, "beta": 2}\'',
    )
    parser.add_argument("--reward_scale", type=float, default=1.0)
    parser.add_argument("--samples_per_iter", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps_per_iter", type=int, default=500)
    parser.add_argument("--use_last_ckpt", action="store_true")
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()
    main(args)
