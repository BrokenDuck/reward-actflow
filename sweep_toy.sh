#!/usr/bin/env bash

#SBATCH --job-name=toy_sweep
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0
#SBATCH --gpus=rtx_2080:1
#SBATCH --output=output/toy_sweep_%A_%a.out
#SBATCH --error=output/toy_sweep_%A_%a.err


DPS_WEIGHTS=(13.0)
GP_LENGTHSCALES=(0.08)
NUM_ITERS=(501) # 501 is the default
FT_STEPS=(250)
# Uniform grid (step 0.5) over a moderate range; tune bounds if too weak/strong
# PARTICLE_COEFFS=(1.55 1.6 1.65 1.7 1.75 1.8 1.85 1.9 1.95 2.0)
INITIAL_MODEL_INVALID=true

# DPS_WEIGHTS=(20.0)
# GP_LENGTHSCALES=(0.1)
# NUM_ITERS=(50)
# FT_STEPS=(150)


N_DPS=${#DPS_WEIGHTS[@]}
N_LS=${#GP_LENGTHSCALES[@]}
N_FT=${#FT_STEPS[@]}

DPS=${DPS_WEIGHTS[$((SLURM_ARRAY_TASK_ID / (N_LS * N_FT)))]}
LS=${GP_LENGTHSCALES[$(( (SLURM_ARRAY_TASK_ID / N_FT) % N_LS ))]}
FT_STEP=${FT_STEPS[$((SLURM_ARRAY_TASK_ID % N_FT))]}

OUTDIR=/cluster/scratch/$USER/toy_sweep/dps_${DPS}_ls_${LS}_num_iters_${NUM_ITERS}_ft_steps_${FT_STEP}_initinvalid_${INITIAL_MODEL_INVALID}
mkdir -p "$OUTDIR"

.pixi/envs/default/bin/python -m adm.task_agnostic toy gp \
    --dir "$OUTDIR" \
    --validity_mode grid3 \
    --dps_weight "$DPS" \
    --gp_lengthscale "$LS" \
    --num_iters "$NUM_ITERS" \
    --seed 42 \
    --eval_samples 50000 \
    --eval_samples_curves 3000 \
    --eval_batch_size 50000 \
    --sample_batch_size 512 \
    --samples_per_iter 64 \
    --ft_batch_size 256 \
    --ft_steps "$FT_STEP" \
    --eval_every 50 \
    --guidance_method uncertainty_tilting \
    --initial_model_invalid "$INITIAL_MODEL_INVALID"
