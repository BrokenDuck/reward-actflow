"""Compute sample metrics separately from the training run, for all eval zip files in an experiment
directory. This saves on GPU-time during the run."""

import torch
from tqdm import tqdm
import argparse
import yaml
import zipfile
import shutil
from pathlib import Path

from .setups import setups as problem_setups
from .problem_setup import SampleFile


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir: Path = args.dir

    with open(exp_dir / "args.yaml", "r") as f:
        exp_args = yaml.safe_load(f)

    problem_setup = problem_setups[exp_args["problem_setup"]](exp_args, device=device)

    for eval_dir in tqdm(sorted(exp_dir.glob("eval/*"), reverse=True)):
        if not eval_dir.is_dir():
            continue

        sample_zip = eval_dir / "samples.zip"
        valids_file = eval_dir / "valids.pt"
        metrics_file = eval_dir / "sample_metrics.yaml"
        sample_dir = eval_dir / "samples_extracted"

        # Check if it is a valid evaluation directory
        if not sample_zip.is_file() or not valids_file.is_file():
            continue

        # Make sure we have not already computed the metrics
        if metrics_file.is_file():
            continue

        # Extract samples
        with zipfile.ZipFile(sample_zip, "r") as zip_ref:
            zip_ref.extractall(sample_dir)

        # Load valids and sample files
        valids: torch.Tensor = torch.load(valids_file)
        sample_files: list[SampleFile] = []
        for fn in sample_dir.glob("*"):
            is_valid = bool(valids[int(fn.stem)].item())
            sample_files.append(SampleFile(is_valid=is_valid, file=fn))

        # Compute and save sample metrics
        sample_metrics = problem_setup.compute_sample_metrics(sample_files)
        with open(metrics_file, "w") as f:
            yaml.dump(sample_metrics, f)

        shutil.rmtree(sample_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path, help="Experiment directory.")
    args = parser.parse_args()
    main(args)
