from typing import Optional

import torch
import torch.nn as nn
from diffusiongym import (
    reward_registry,
    construct_env,
    D,
    Environment,
    MemorylessNoiseSchedule,
)
from diffusiongym.molecules import DDGraph
import argparse
import yaml
import logging
from pathlib import Path
import json

from reward_actflow.setups import setups as problem_setups
from reward_actflow.setups.problem_setup import ProblemSetup
from reward_actflow.utils import serialize_args


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
        / "fine_tuned_rw_mle_test"
        / f"{args.reward.split('/')[-1]}_{args.reward_scale}_{'last' if args.use_last_ckpt else 'base'}"
    )
    ft_dir.mkdir(parents=True, exist_ok=True)

    with open(ft_dir / "args.yaml", "w") as f:
        yaml.safe_dump(serialize_args(args), f)

    if args.use_last_ckpt:
        ckpt_path = exp_dir / "base_model.pt"
        state_dict = torch.load(ckpt_path, map_location=device)
        env.base_model.load_state_dict(state_dict)

    reward_weighted_mle(
        env,
        samples_per_iter=args.samples_per_iter,
        num_iterations=args.num_iterations,
        lr=args.lr,
        log_every=args.log_every,
        exp_dir=ft_dir,
    )


def reward_weighted_mle(
    env: Environment[D],
    samples_per_iter: int,
    num_iterations: int = 100,
    lr: float = 1e-5,
    log_every: Optional[int] = None,
    exp_dir: Optional[Path] = None,
):
    env.base_model.scheduler.noise_schedule = MoleculeMemorylessNoiseSchedule(
        env.base_model.scheduler
    )

    if exp_dir is not None:
        exp_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(exp_dir)

    if log_every is None:
        log_every = max(1, num_iterations // 100)

    opt = torch.optim.AdamW(env.base_model.parameters(), lr=lr)

    # Track first and second moment
    r_ema_m1 = None
    r_ema_m2 = None

    # Determine beta based on halflife 10% of iterations
    halflife = 0.1 * num_iterations
    beta = 1 - 2 ** (-1 / halflife)

    for it in range(1, num_iterations + 1):
        s = env.sample(samples_per_iter, pbar=False)
        r = s.rewards.clone()
        v = s.valids.clone()

        with torch.no_grad():
            if r_ema_m1 is None or r_ema_m2 is None:
                r_ema_m1 = r[v].mean()
                r_ema_m2 = (r[v] ** 2).mean()
            else:
                r_ema_m1 = (1 - beta) * r_ema_m1 + beta * r[v].mean()
                r_ema_m2 = (1 - beta) * r_ema_m2 + beta * (r[v] ** 2).mean()

            r_ema_var = (r_ema_m2 - r_ema_m1**2).clamp_min(1e-6)

        r[v] = (r[v] - r_ema_m1) / (r_ema_var.sqrt() + 1e-8)
        r[v] = r[v].clamp(-5, 5)
        r[~v] = -20

        r = r.to(env.device)

        total_loss = 0.0

        opt.zero_grad()

        for x_t, x_t_next, t0, t1, diffusion_t in zip(
            s.trajectory[:-1],
            s.trajectory[1:],
            s.timesteps[:-1],
            s.timesteps[1:],
            s.diffusions,
        ):
            x_t = x_t.to(env.device)
            x_t_next = x_t_next.to(env.device)
            diffusion_t = diffusion_t.to(env.device)

            dt = t1 - t0
            t = t0 * torch.ones(len(x_t), device=x_t.device)
            new_drift_t, _ = env.drift(x_t, t, **s.kwargs)
            new_mean_t_next = x_t + dt * new_drift_t

            weight = torch.exp(env.reward_scale * r)
            loss = weight * (
                ((x_t_next - new_mean_t_next) / diffusion_t) ** 2
            ).aggregate("mean")
            loss = loss.mean()
            loss /= s.num_steps

            if loss.isnan().any() or loss.isinf().any():
                raise ValueError("Loss is NaN or Inf")

            loss.backward()
            total_loss += loss.item()

        grad_norm = nn.utils.clip_grad_norm_(env.base_model.parameters(), 1.0)
        opt.step()

        metrics = {
            "loss": total_loss,
            "grad_norm": grad_norm,
            "r_mean": s.rewards[s.valids].mean(),
            "r_std": s.rewards[s.valids].std(),
            "r_min": s.rewards[s.valids].min(),
            "r_max": s.rewards[s.valids].max(),
            "valid": s.valids.float().mean(),
        }
        logging.info(
            f"(iter={it:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}"
        )

        if exp_dir is not None:
            torch.save(env.base_model.state_dict(), exp_dir / "last.pt")


class MoleculeMemorylessNoiseSchedule(MemorylessNoiseSchedule[DDGraph]):
    def __call__(self, x: DDGraph, t: torch.Tensor) -> DDGraph:
        out = super().__call__(x, t)
        out.graph.ndata["x_t"] = 0.1 * out.graph.ndata["x_t"]
        return out


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
    parser.add_argument("--samples_per_iter", type=int, default=64)
    parser.add_argument("--use_last_ckpt", action="store_true")
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()
    main(args)
