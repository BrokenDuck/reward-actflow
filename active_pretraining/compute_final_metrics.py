import torch
from torch.utils.data import default_collate
import argparse
import yaml
from pathlib import Path
from flowgym import Reward, D, construct_env
from tqdm import tqdm
import shutil

from .setups import setups as problem_setups
from .problem_setup import SampleFile
from .utils import index_dict, Batch


@torch.no_grad()
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir: Path = args.dir

    with open(exp_dir / "args.yaml", "r") as f:
        exp_args = yaml.safe_load(f)

    folder = exp_dir / args.samples_dir
    samples_dir = folder / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    problem_setup = problem_setups[exp_args["problem_setup"]](exp_args, device=device)
    reward = DummyReward()
    env = construct_env(problem_setup.base_model, reward, exp_args["num_steps"], 0)

    if args.use_last_ckpt:
        ckpt_path = exp_dir / "base_model.pt"
        state_dict = torch.load(ckpt_path, map_location=device)
        env.base_model.load_state_dict(state_dict)

    eval_kwargs = problem_setup.eval_sampling_kwargs(args.n_samples)
    consecutive_all_invalid = 0

    batches: list[Batch] = []
    sample_files: list[SampleFile] = []

    # We want `n_samples` valid samples
    pbar = tqdm(total=args.n_samples)
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
        if use_random_kwargs:
            print("Sampling with random kwargs due to too many consecutive invalid batches.")
            out = env.sample(current_batch_size, pbar=False)
        else:
            sampling_kwargs = default_collate(kwargs_list)
            out = env.sample(current_batch_size, pbar=False, **sampling_kwargs)

        samples = out[0]
        latents = out[1][-1]
        kwargs = out[-1]
        valids = problem_setup.validity(samples, kwargs)
        batch = Batch(samples, latents, valids, kwargs)

        n_valids = batch.valids.int().sum().item()

        for i, index in enumerate(indices):
            if not batch.valids[i]:
                continue

            file_path = samples_dir / f"{index:06d}"

            batches.append(batch[i])

            # Save only valid samples
            problem_setup.save_sample(
                batch.samples[i],
                index_dict(batch.kwargs, i),
                file_path,
            )
            # Mark those kwargs as sampled
            is_valid_sampled[index] = True

        if n_valids == 0:
            consecutive_all_invalid += 1
        else:
            consecutive_all_invalid = 0

        pbar.update(n_valids)

    pbar.close()

    global_metrics = problem_setup.compute_metrics(batches)
    with open(folder / "global_metrics.yaml", "w") as f:
        yaml.dump(global_metrics, f)

    sample_files = [SampleFile(is_valid=True, file=fn) for fn in samples_dir.glob("*")]
    sample_metrics = problem_setup.compute_sample_metrics(sample_files)

    with open(folder / "sample_metrics.yaml", "w") as f:
        yaml.dump(sample_metrics, f)

    # Zip and delete samples
    shutil.make_archive(str(samples_dir), "zip", samples_dir)
    shutil.rmtree(samples_dir)


class DummyReward(Reward[D]):
    def __call__(self, x: D, **kwargs):
        return torch.zeros(len(x), device=x.device), torch.ones(len(x), device=x.device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--n_samples", type=int, required=True)
    parser.add_argument("--samples_dir", type=Path, default=Path("final_samples"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--use_last_ckpt", action="store_true")
    args = parser.parse_args()
    main(args)
