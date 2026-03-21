#!/usr/bin/env bash

#SBATCH --job-name=toy_sweep
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-35
#SBATCH --gpus=rtx_2080:1
#SBATCH --output=output/toy_sweep_%A_%a.out
#SBATCH --error=output/toy_sweep_%A_%a.err


DPS_WEIGHTS=(7.0 8.0 8.5 9.0 11.0 13.0)
GP_LENGTHSCALES=(0.1)
NUM_ITERS=(2000)
FT_STEPS=(20 50 70 100 130 150)

N_DPS=${#DPS_WEIGHTS[@]}
N_FT=${#FT_STEPS[@]}

DPS=${DPS_WEIGHTS[$((SLURM_ARRAY_TASK_ID / N_FT))]}
FT_STEP=${FT_STEPS[$((SLURM_ARRAY_TASK_ID % N_FT))]}
LS=${GP_LENGTHSCALES[0]}

OUTDIR=/cluster/scratch/$USER/toy_sweep/dps_${DPS}_ls_${LS}_num_iters_${NUM_ITERS}_ft_steps_${FT_STEP}
mkdir -p "$OUTDIR"

.pixi/envs/default/bin/python -m adm.task_agnostic toy gp \
    --dir "$OUTDIR" \
    --validity_mode grid3 \
    --dps_weight "$DPS" \
    --gp_lengthscale "$LS" \
    --num_iters "$NUM_ITERS" \
    --eval_samples 3000 \
    --eval_batch_size 3000 \
    --sample_batch_size 512 \
    --samples_per_iter 64 \
    --ft_batch_size 256 \
    --ft_steps "$FT_STEP" \
    --eval_every 50
