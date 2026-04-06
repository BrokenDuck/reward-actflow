#!/bin/bash
#SBATCH --job-name="Eval samples"
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00

source ~/.bashrc
export PATH="$HOME/.pixi/bin:$PATH"

cd ~/adm

module load eth_proxy
module load stack/2024-06 gcc/12.2.0 xtb

# Required positional arguments
ARG1=$1
ARG2=$2

# Shift first two args so remaining ones are optional flags
shift 2

# Run evaluation with optional flags
pixi run python -m adm.evaluation.eval_samples "$ARG1" "$ARG2" "$@"
