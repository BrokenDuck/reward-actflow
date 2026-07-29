from typing import Optional, Callable

import torch
from torch.utils.data import DataLoader
from diffusiongym import reward_registry, construct_env, D, Environment
from diffusiongym.utils import DDDataset
import argparse
import yaml
import logging
from pathlib import Path
import copy
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

    diffusion_nft(
        env,
        samples_per_iter=args.samples_per_iter,
        sample_batch_size=args.sample_batch_size,
        ft_batch_size=args.ft_batch_size,
        num_iterations=args.num_iterations,
        beta=args.beta,
        lr=args.lr,
        postprocess_latents=problem_setup.postprocess_latents,
        log_every=args.log_every,
        exp_dir=ft_dir,
    )


def diffusion_nft(
    env: Environment[D],
    samples_per_iter: int,
    sample_batch_size: int,
    ft_batch_size: int,
    num_iterations: int = 100,
    beta: float = 1.0,
    eta_schedule: Callable[[int], float] = lambda i: min(0.001 * i, 0.5),
    lr: float = 1e-4,
    postprocess_latents: Optional[Callable[[Batch[D]], D]] = None,
    log_every: Optional[int] = None,
    exp_dir: Optional[Path] = None,
):
    if exp_dir is not None:
        exp_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(exp_dir)

    if log_every is None:
        log_every = max(1, num_iterations // 100)

    # The policy with which we sample will be an EMA version of the base model
    env.policy = copy.deepcopy(env.base_model)
    opt = torch.optim.AdamW(env.base_model.parameters(), lr=lr)

    all_rewards = torch.tensor([])

    for it in range(1, num_iterations + 1):
        sample = env.batch_sample(samples_per_iter, sample_batch_size)
        batch = Batch.from_sample(sample)
        batch = filter_out_invalids([batch])

        if postprocess_latents is not None:
            batch.latents = postprocess_latents(batch)

        r_raw = env.reward_scale * batch.rewards
        all_rewards = torch.cat([all_rewards, r_raw.cpu()], dim=0)

        # Normalize within group
        r_norm = r_raw - r_raw.mean()
        # Compute optimality probability with global std
        r = 0.5 + 0.5 * (r_norm / (all_rewards.std() + 1e-8)).clamp(-1, 1)

        # Assign 0 probability to invalid samples (not from paper but intuitive to me)
        # r[~sample.valids] = 0.0

        metrics = {
            "r_mean": sample.rewards[sample.valids].mean(),
            "r_std": sample.rewards[sample.valids].std(),
            "r_min": sample.rewards[sample.valids].min(),
            "r_max": sample.rewards[sample.valids].max(),
            "valid": sample.valids.float().mean(),
        }
        logging.info(
            f"(iter={it:05d}) {', '.join([f'{k}: {v:.2f}' for k, v in metrics.items()])}"
        )

        dataset = DDDataset(
            [batch.latents.to(env.device)], [batch.kwargs], [r.to(env.device)]
        )
        loader = DataLoader(
            dataset,
            ft_batch_size,
            shuffle=True,
            collate_fn=dataset.collate,
            num_workers=0,
            pin_memory=False,
        )

        env.base_model.train()

        for _ in range(10):
            for x1, kwargs, opt_prob in loader:
                x1 = x1.to(env.device)
                x0 = x1.randn_like()
                t = torch.rand(len(x1), device=env.device)

                a = env.scheduler.alpha(x1, t)
                b = env.scheduler.beta(x1, t)
                xt = a * x1 + b * x0

                new_pred = env.base_model(xt, t, **kwargs)
                with torch.no_grad():
                    old_pred = env.policy(xt, t, **kwargs)

                pos_pred = (1 - beta) * old_pred + beta * new_pred
                neg_pred = (1 + beta) * old_pred - beta * new_pred

                loss = torch.mean(
                    opt_prob * env.base_model.train_loss(x1, xt=xt, t=t, pred=pos_pred)
                    + (1 - opt_prob)
                    * env.base_model.train_loss(x1, xt=xt, t=t, pred=neg_pred)
                )
                loss.backward()
                opt.step()
                opt.zero_grad()

        # ema step using eta
        with torch.no_grad():
            eta = eta_schedule(it)
            for p_old, p_new in zip(
                env.policy.parameters(), env.base_model.parameters()
            ):
                p_old.data.mul_(eta).add_(p_new.data, alpha=1 - eta)

        if exp_dir is not None:
            torch.save(env.base_model.state_dict(), exp_dir / "last.pt")

    env.policy = env.base_model
    env.base_model.eval()


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
    parser.add_argument("--sample_batch_size", type=int, default=64)
    parser.add_argument("--ft_batch_size", type=int, default=64)
    parser.add_argument("--use_last_ckpt", action="store_true")
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()
    main(args)
