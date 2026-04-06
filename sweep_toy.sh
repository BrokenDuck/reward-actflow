#!/usr/bin/env bash

#SBATCH --job-name=toy_sweep
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-2
#SBATCH --gpus=rtx_2080:1
#SBATCH --output=output/toy_sweep_%A_%a.out
#SBATCH --error=output/toy_sweep_%A_%a.err

# ---------------------------------------------------------------------------
# Standard method (no NSM) — 3 seeds
# ---------------------------------------------------------------------------
DPS=13.0
LS=0.08
NUM_ITERS=501
FT_STEPS=250
INITIAL_MODEL_INVALID=true

SEED=$((42 + SLURM_ARRAY_TASK_ID))
[ "$SLURM_ARRAY_TASK_ID" -eq 0 ] && SAVE_FRAMES="--save_frames" || SAVE_FRAMES=""

OUTDIR=/cluster/scratch/$USER/toy_sweep/dps_${DPS}_ls_${LS}_ft${FT_STEPS}_initinvalid_${INITIAL_MODEL_INVALID}_baseline/seed_${SEED}
mkdir -p "$OUTDIR"

.pixi/envs/default/bin/python -m adm.task_agnostic toy gp \
    --dir "$OUTDIR" \
    --validity_mode grid3 \
    --dps_weight "$DPS" \
    --gp_lengthscale "$LS" \
    --num_iters "$NUM_ITERS" \
    --seed "$SEED" \
    --eval_samples 50000 \
    --eval_samples_curves 3000 \
    --eval_batch_size 50000 \
    --sample_batch_size 512 \
    --samples_per_iter 64 \
    --ft_batch_size 256 \
    --ft_steps "$FT_STEPS" \
    --eval_every 50 \
    --guidance_method uncertainty_tilting \
    --initial_model_invalid "$INITIAL_MODEL_INVALID" \
    $SAVE_FRAMES
