"""Compute sample metrics separately from the training run, for all eval zip files in an experiment
directory. This saves on GPU-time during the run."""

import torch
import argparse
import yaml
import zipfile
from tqdm import trange
import shutil
from pathlib import Path
import polars as pl

from adm.setups import setups as problem_setups
from adm.setups.problem_setup import SampleFile
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

    sample_zip = eval_dir / "samples.zip"
    sample_metrics_file = eval_dir / "sample_metrics.csv"
    global_metrics_file = eval_dir / "global_metrics.yaml"
    sample_dir = eval_dir / "samples_extracted"
    valids_file = eval_dir / "valids.pt"

    # Check if it is a valid evaluation directory
    if not sample_zip.is_file():
        raise ValueError(f"Invalid evaluation directory: {eval_dir}")

    # Make sure we have not already computed the metrics
    if sample_metrics_file.is_file():
        raise ValueError(f"Metrics file already exists: {sample_metrics_file}")

    logger.info(f"Extracting samples from {sample_zip} to compute metrics...")

    # Extract samples
    with zipfile.ZipFile(sample_zip, "r") as zip_ref:
        zip_ref.extractall(sample_dir)

    valids = None
    if valids_file.is_file():
        valids = torch.load(valids_file)

    # Load valids and sample files
    sample_files: list[SampleFile] = []
    for fn in sample_dir.glob("*"):
        is_valid = bool(valids[int(fn.stem)].item()) if valids is not None else True
        sample_files.append(SampleFile(is_valid=is_valid, file=fn))

    logger.info(f"Computing global metrics for {sample_dir}...")

    if args.n_global_samples is None:
        args.n_global_samples = len(sample_files)

    global_metrics = problem_setup.compute_metrics(sample_files[:args.n_global_samples])

    with open(global_metrics_file, "w") as f:
        yaml.dump(global_metrics, f)

    logger.info(f"Computing sample metrics for {sample_dir}...")

    # Compute and save sample metrics
    batch_size = 1000
    for i in trange(0, len(sample_files), batch_size):
        batch_files = sample_files[i:i+batch_size]
        batch_metrics = problem_setup.compute_sample_metrics(batch_files)
        data_rows = []

        for sf in batch_files:
            sample_id = sf.file.stem
            metrics = batch_metrics.get(sample_id, dict())

            data_rows.append({
                "sample_id": int(sample_id),
                "is_valid": sf.is_valid,
                **metrics,
            })

        df_batch = pl.DataFrame(data_rows)

        if i == 0:
            df_batch.write_csv(sample_metrics_file)
        else:
            with open(sample_metrics_file, "ab") as f:
                df_batch.write_csv(f, include_header=False)

    df = pl.read_csv(sample_metrics_file)

    # Add rows for all invalid samples that are null for all metrics
    if valids is not None:
        n_samples = len(valids)
        full_range = pl.DataFrame({ "sample_id": pl.arange(0, n_samples, eager=True) })
        missing = full_range.join(df.select("sample_id"), on="sample_id", how="anti")
        missing = missing.with_columns([pl.lit(False).alias("is_valid")])

        for col in df.columns:
            if col not in missing.columns:
                missing = missing.with_columns(pl.lit(None).cast(df.schema[col]).alias(col))

        missing = missing.select(df.columns)
        df = pl.concat([df, missing])

    df = df.sort("sample_id")
    df.write_csv(sample_metrics_file)

    shutil.rmtree(sample_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path, help="Experiment directory.")
    parser.add_argument("sample_dir", type=Path, help="Path to samples within `dir`.")
    parser.add_argument("--n_global_samples", type=int, default=None, help="Number of samples to use for global stats.")
    args = parser.parse_args()
    main(args)
