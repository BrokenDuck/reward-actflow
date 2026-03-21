#!/usr/bin/env bash

#SBATCH --job-name=baseline_no_valid
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_2080:1
#SBATCH --output=output/baseline_no_valid.out
#SBATCH --error=output/baseline_no_valid.err


OUTDIR=/cluster/scratch/$USER/baseline_no_valid
mkdir -p "$OUTDIR"

.pixi/envs/default/bin/python -m adm.task_agnostic toy gp \
    --dir "$OUTDIR" \
    --validity_mode grid3 \
    --no_uncertainty \
    --no_verifier \
    --eval_samples 24000 \
    --eval_batch_size 24000 \
    --sample_batch_size 512 \
    --samples_per_iter 512
