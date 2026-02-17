import torch
import argparse
import yaml
from pathlib import Path
from diffusiongym import DummyReward, construct_env
import shutil

from adm.setups import setups as problem_setups
from adm.utils import index_dict, Batch, setup_logger


@torch.no_grad()
def main(args):
    logger = setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir: Path = args.dir

    folder = exp_dir / args.samples_dir
    samples_dir = folder / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "args.yaml", "r") as f:
        exp_args = yaml.safe_load(f)

    logger.info(f"Loading problem setup: {exp_args['problem_setup']}")

    problem_setup = problem_setups[exp_args["problem_setup"]](exp_args, device=device)
    reward = DummyReward()
    env = construct_env(problem_setup.base_model, reward, exp_args["num_steps"], 0)

    if args.ckpt is not None:
        logger.info(f"Loading model checkpoint from {args.ckpt}")

        ckpt_path = exp_dir / args.ckpt
        state_dict = torch.load(ckpt_path, map_location=device)
        env.base_model.load_state_dict(state_dict)

    all_valids = []

    for i in range(0, args.n_samples, args.batch_size):
        current_bs = min(args.batch_size, args.n_samples - i)
        sample = env.sample(current_bs, pbar=False)
        sample.valids = problem_setup.validity(sample.sample, sample.kwargs)
        batch = Batch.from_sample(sample)
        batch.kwargs = dict_to_cpu(batch.kwargs)

        for j in range(current_bs):
            index = i + j
            file_path = samples_dir / f"{index:06d}"

            if not batch.valids[j]:
                continue

            path = problem_setup.save_sample(
                batch.samples[j],
                index_dict(batch.kwargs, j),
                file_path,
            )
            if path is None or not path.is_file():
                logger.warning(f"Sample {index} was marked as valid but failed to save properly.")

        all_valids.append(batch.valids)

        # Log if we crossed a 5% boundary
        prev_pct = (100 * i) // args.n_samples
        current_pct = (100 * (i + current_bs)) // args.n_samples
        if (current_pct // 5) > (prev_pct // 5):
            display_pct = (current_pct // 5) * 5
            logger.info(f"Progress: {display_pct}% ({i + current_bs}/{args.n_samples} samples, {torch.cat(all_valids).float().mean().item():.2%} valid)")

    valids = torch.cat(all_valids)
    torch.save(valids, folder / "valids.pt")

    logger.info("Zipping samples...")
    shutil.make_archive(str(samples_dir), "zip", samples_dir)

    logger.info("Deleting unzipped samples...")
    shutil.rmtree(samples_dir)


def dict_to_cpu(d: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: v.cpu() for k, v in d.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--samples_dir", type=Path, default=Path("final_samples"))
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(args)
