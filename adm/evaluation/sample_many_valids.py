import torch
from torch.utils.data import default_collate
import argparse
import yaml
from pathlib import Path
from diffusiongym import DummyReward, construct_env
from diffusiongym.utils import index_dict

from adm.setups import setups as problem_setups
from adm.utils import Batch, setup_logger


@torch.no_grad()
def main(args):
    logger = setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir: Path = args.dir

    folder = exp_dir / args.samples_dir
    folder.mkdir(parents=True, exist_ok=False)

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

    eval_kwargs = problem_setup.eval_sampling_kwargs(args.n_samples)
    consecutive_all_invalid = 0

    batches = []

    # We want `n_samples` valid samples
    is_valid_sampled = torch.zeros((args.n_samples,), dtype=torch.bool, device=device)
    while is_valid_sampled.int().sum() < args.n_samples:
        # We want to make sure to only sample the missing ones
        indices = []
        kwargs_list = []
        current_batch_size = 0
        for i, is_sampled in enumerate(is_valid_sampled):
            if not is_sampled:
                current_batch_size += 1
                indices.append(i)
                kwargs_list.append(index_dict(eval_kwargs, i))

            if current_batch_size >= args.batch_size:
                break

        use_random_kwargs = consecutive_all_invalid >= 3

        sampling_kwargs = {}
        if not use_random_kwargs:
            sampling_kwargs = default_collate(kwargs_list)

        sample = env.sample(current_batch_size, pbar=False, **sampling_kwargs)
        batch = Batch.from_sample(sample)
        batch.valids = problem_setup.validity(batch.samples, batch.kwargs)

        for i, index in enumerate(indices):
            if not batch.valids[i]:
                continue

            batches.append(batch[i].cpu())
            is_valid_sampled[index] = True

        n_valids = batch.valids.int().sum().item()
        if n_valids == 0:
            consecutive_all_invalid += 1
        else:
            consecutive_all_invalid = 0

        # Calculate progress
        n_valids_new = is_valid_sampled.int().sum().item()
        n_valids_prev = n_valids_new - n_valids
        prev_pct = (n_valids_prev * 100) // args.n_samples
        current_pct = (n_valids_new * 100) // args.n_samples

        # Log if we crossed a 5% boundary
        if (current_pct // 5) > (prev_pct // 5):
            display_pct = (current_pct // 5) * 5
            logger.info(f"Progress: {display_pct}% ({n_valids_new}/{args.n_samples} valid samples)")

    batch = Batch.concat(batches)
    problem_setup.save_samples(batch.samples, batch.kwargs, folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--samples_dir", type=Path, default=Path("eval_samples_valid"))
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
