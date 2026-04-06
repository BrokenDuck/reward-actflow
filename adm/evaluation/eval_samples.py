"""Compute sample metrics separately from the training run, for all eval zip files in an experiment
directory. This saves on GPU-time during the run."""

import torch
import argparse
import yaml
from tqdm import trange
from pathlib import Path
from diffusiongym.utils import index_dict
import polars as pl

from adm.setups import setups as problem_setups
from adm.utils import setup_logger


def main(args):
    logger = setup_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir: Path = args.dir
    eval_dir = exp_dir / args.sample_dir

    if not eval_dir.is_dir():
        raise ValueError(f"Evaluation directory does not exist: {eval_dir}")

    with open(exp_dir / "args.yaml", "r") as f:
        exp_args = yaml.safe_load(f)

    logger.info(f"Loading problem setup: {exp_args['problem_setup']}...")

    problem_setup = problem_setups[exp_args["problem_setup"]](exp_args, device=device)

    # Load samples
    samples, kwargs = problem_setup.load_samples(eval_dir)

    valids_file = eval_dir / "valids.pt"
    valids = torch.load(valids_file) if valids_file.is_file() else torch.ones(len(samples), dtype=torch.bool)

    if args.do_global_metrics:
        logger.info(f"Computing global metrics...")
        global_metrics = problem_setup.compute_metrics(samples, kwargs)
        with open(eval_dir / "global_metrics.yaml", "w") as f:
            yaml.dump(global_metrics, f)

    if args.do_sample_metrics:
        logger.info(f"Computing sample metrics...")
        sample_metrics_file = eval_dir / "sample_metrics.csv"

        # Compute and save sample metrics
        batch_size = 1000
        for i in trange(0, len(samples), batch_size):
            start, end = i, min(i + batch_size, len(samples))
            current_samples = samples[start:end]
            current_kwargs = index_dict(kwargs, start, end)
            batch_metrics = problem_setup.compute_sample_metrics(current_samples, current_kwargs)

            data_rows = []

            for j, metrics in enumerate(batch_metrics):
                sample_id = start + j
                data_rows.append({ "sample_id": sample_id, **metrics })

            df_batch = pl.DataFrame(data_rows)

            if i == 0:
                df_batch.write_csv(sample_metrics_file)
            else:
                with open(sample_metrics_file, "ab") as f:
                    df_batch.write_csv(f, include_header=False)

        df = pl.read_csv(sample_metrics_file)

        # Add rows for all invalid samples that are null for all metrics
        n_samples = len(valids)
        full_range = pl.DataFrame({"sample_id": pl.arange(0, n_samples, eager=True)})
        missing = full_range.join(df.select("sample_id"), on="sample_id", how="anti")
        missing = missing.with_columns([pl.Series("is_valid", valids[missing["sample_id"].to_numpy()].numpy())])

        for col in df.columns:
            if col not in missing.columns:
                missing = missing.with_columns(pl.lit(None).cast(df.schema[col]).alias(col))

        missing = missing.select(df.columns)
        df = pl.concat([df, missing])

        df = df.sort("sample_id")
        df.write_csv(sample_metrics_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("sample_dir", type=Path)
    parser.add_argument("--do_global_metrics", action="store_true", default=False)
    parser.add_argument("--do_sample_metrics", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
