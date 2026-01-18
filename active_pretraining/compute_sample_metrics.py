"""Compute sample metrics separately from the training run, for all eval zip files in an experiment
directory. This saves on GPU-time during the run."""

import torch
from tqdm import tqdm
import argparse
import yaml
from glob import glob
import zipfile
import shutil
import os

from .setups import setups as problem_setups
from .problem_setup import SampleFile


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = args.dir
    with open(os.path.join(exp_dir, "args.yaml"), "r") as f:
        exp_args = yaml.safe_load(f)

    problem_setup = problem_setups[exp_args["problem_setup"]](exp_args, device=device)

    eval_dirs = list(sorted(glob(os.path.join(exp_dir, "eval", "*"))))
    for eval_dir in tqdm(eval_dirs):
        if not os.path.isdir(eval_dir):
            continue

        sample_zip = os.path.join(eval_dir, "samples.zip")
        valids_file = os.path.join(eval_dir, "valids.pt")

        # Make sure this is a valid evaluation folder
        if not os.path.exists(sample_zip) or not os.path.exists(valids_file):
            continue

        metrics_file = os.path.join(eval_dir, "sample_metrics.yaml")

        # Make sure we have not already computed the metrics
        if os.path.exists(metrics_file):
            continue

        # Extract samples
        sample_dir = os.path.join(eval_dir, "samples_extracted")
        with zipfile.ZipFile(sample_zip, "r") as zip_ref:
            zip_ref.extractall(sample_dir)

        # Load validity tensor
        valids: torch.Tensor = torch.load(valids_file)

        sample_files: list[SampleFile] = []
        for fn in glob(os.path.join(sample_dir, "*")):
            no_ext = os.path.splitext(os.path.basename(fn))[0]
            is_valid = bool(valids[int(no_ext)].item())
            sample_files.append(SampleFile(is_valid=is_valid, file=fn))

        sample_metrics = problem_setup.compute_sample_metrics(sample_files)
        with open(metrics_file, "w") as f:
            yaml.dump(sample_metrics, f)

        shutil.rmtree(sample_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Experiment directory.")
    args = parser.parse_args()
    main(args)
