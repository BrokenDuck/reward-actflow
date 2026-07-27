import argparse
import subprocess
import sys
import yaml
from pathlib import Path

from adm.utils import setup_logger


def is_sgpo(run_dir: Path) -> bool:
    with open(run_dir / "args.yaml") as f:
        return "algorithm" in yaml.safe_load(f)


def run(cmd: list[str], logger):
    logger.info("$ " + " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def eval_sgpo(run_dir: Path, args, logger):
    for eval_subdir in sorted((run_dir / "eval").iterdir()):
        if not (eval_subdir / "samples.pt").exists():
            continue
        if (eval_subdir / "metrics.yaml").exists():
            logger.info(f"  Skipping {eval_subdir.name}: already evaluated")
            continue
        logger.info(f"  Computing metrics for {eval_subdir.name}...")
        eval_cmd = [sys.executable, "-m", "adm.evaluation.eval_samples",
                    str(run_dir), f"eval/{eval_subdir.name}", "--do_global_metrics",
                    "--top_p", str(args.top_p)]
        if args.reward:
            eval_cmd += ["--reward", args.reward]
        run(eval_cmd, logger)


def eval_adm(run_dir: Path, n_samples: int, batch_size: int, out_name: str, args, logger):
    out_dir = run_dir / "eval" / out_name
    if (out_dir / "metrics.yaml").exists():
        logger.info(f"Skipping {run_dir.name}: already evaluated")
        return
    sample_cmd = [
        sys.executable, "-m", "adm.evaluation.sample_many",
        str(run_dir),
        "--n_samples", str(n_samples),
        "--batch_size", str(batch_size),
        "--ckpt", "base_model.pt",
        "--samples_dir", f"eval/{out_name}",
    ]
    if args.dps:
        sample_cmd += ["--dps", "--dps_weight", str(args.dps_weight)]
        if args.reward:
            sample_cmd += ["--reward", args.reward]
    run(sample_cmd, logger)
    eval_cmd = [sys.executable, "-m", "adm.evaluation.eval_samples",
                str(run_dir), f"eval/{out_name}", "--do_global_metrics",
                "--top_p", str(args.top_p)]
    if args.reward:
        eval_cmd += ["--reward", args.reward]
    run(eval_cmd, logger)


def main(args):
    logger = setup_logger()
    sweep_dir = Path(args.sweep_dir)

    run_dirs = sorted(p for p in sweep_dir.iterdir() if p.is_dir() and (p / "args.yaml").exists())
    logger.info(f"Found {len(run_dirs)} run(s) in {sweep_dir}")

    for run_dir in run_dirs:
        if is_sgpo(run_dir):
            logger.info(f"Evaluating SGPO run {run_dir.name}...")
            eval_sgpo(run_dir, args, logger)
        elif (run_dir / "base_model.pt").exists():
            logger.info(f"Evaluating ADM run {run_dir.name}...")
            eval_adm(run_dir, args.n_samples, args.batch_size, args.out_name, args, logger)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sweep_dir")
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out_name", type=str, default="final")
    parser.add_argument("--top_p", type=float, default=0.05)
    parser.add_argument("--dps", action="store_true", default=False)
    parser.add_argument("--dps_weight", type=float, default=100.0)
    parser.add_argument("--reward", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        args.n_samples = 4
        args.batch_size = 4
        args.out_name = "debug"

    main(args)
