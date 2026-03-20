#!/usr/bin/env bash

#SBATCH --job-name=toy_sweep
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-15
#SBATCH --gpus=rtx_2080:1
#SBATCH --output=output/toy_sweep_%A_%a.out
#SBATCH --error=output/toy_sweep_%A_%a.err


DPS_WEIGHTS=(10.0 20.0 30.0 40.0)
GP_LENGTHSCALES=(0.05 0.1 0.2 0.4)

N_LS=${#GP_LENGTHSCALES[@]}

DPS=${DPS_WEIGHTS[$((SLURM_ARRAY_TASK_ID / N_LS))]}
LS=${GP_LENGTHSCALES[$((SLURM_ARRAY_TASK_ID % N_LS))]}

OUTDIR=/cluster/scratch/$USER/toy_sweep/dps_${DPS}_ls_${LS}
mkdir -p "$OUTDIR"

.pixi/envs/default/bin/python -m adm.task_agnostic toy gp \
    --dir "$OUTDIR" \
    --validity_mode grid \
    --dps_weight "$DPS" \
    --gp_lengthscale "$LS" \
    --eval_samples 24000 \
    --eval_batch_size 24000 \
    --sample_batch_size 512 \
    --samples_per_iter 512
