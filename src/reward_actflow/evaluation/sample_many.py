import torch
import argparse
import yaml
from pathlib import Path
from diffusiongym import DummyReward, construct_env
from tqdm import trange

from adm.setups import setups as problem_setups
from adm.utils import Batch, setup_logger


@torch.no_grad()
def main(args):
    logger = setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir: Path = args.dir

    folder = exp_dir / args.samples_dir
    folder.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "args.yaml", "r") as f:
        exp_args = yaml.safe_load(f)

    logger.info(f"Loading problem setup: {exp_args['problem_setup']}")

    problem_setup = problem_setups[exp_args["problem_setup"]](exp_args, device=device)
    reward = DummyReward()
    env = construct_env(problem_setup.base_model, reward, exp_args["num_steps"], 0)

    if args.ckpt is not None:
        logger.info(f"Loading model checkpoint from {args.ckpt}")
        ckpt_path = exp_dir / "checkpoints" / args.ckpt
        state_dict = torch.load(ckpt_path, map_location=device)
        env.base_model.load_state_dict(state_dict)

    batches = []
    for i in trange(0, args.n_samples, args.batch_size):
        current_n = min(args.batch_size, args.n_samples - i)
        samples = env.sample(current_n, pbar=False)
        batch = Batch.from_sample(samples)
        batch.valids = problem_setup.validity(batch.samples, batch.kwargs)
        batches.append(batch.cpu())

    batch = Batch.concat(batches)
    problem_setup.save_samples(batch.samples, batch.kwargs, folder)
    torch.save(batch.valids, folder / "valids.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--samples_dir", type=Path, default=Path("eval_samples"))
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
