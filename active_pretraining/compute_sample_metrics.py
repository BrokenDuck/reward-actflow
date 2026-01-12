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


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = args.dir
    with open(os.path.join(exp_dir, "args.yaml"), "r") as f:
        exp_args = yaml.safe_load(f)

    problem_setup = problem_setups[exp_args["problem_setup"]](exp_args, device=device)

    zip_files = list(sorted(glob(os.path.join(exp_dir, "eval", "*.zip"))))
    for zip_file in tqdm(zip_files):
        iteration = os.path.basename(zip_file).replace(".zip", "")
        metrics_file = os.path.join(exp_dir, "eval", f"{iteration}_metrics.yaml")
        if os.path.exists(metrics_file):
            continue

        output_dir = zip_file.replace(".zip", "_extracted")
        with zipfile.ZipFile(zip_file, "r") as zip_ref:
            zip_ref.extractall(output_dir)

        sample_metrics = problem_setup.compute_sample_metrics(output_dir)
        with open(metrics_file, "w") as f:
            yaml.dump(sample_metrics, f)

        # clean up extracted directory
        shutil.rmtree(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=str, help="Experiment directory.")
    args = parser.parse_args()
    main(args)
