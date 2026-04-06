#!/usr/bin/env bash

#SBATCH --job-name=toy_posneg
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-19
#SBATCH --gpus=rtx_2080:1
#SBATCH --output=output/toy_posneg_%A_%a.out
#SBATCH --error=output/toy_posneg_%A_%a.err

# ---------------------------------------------------------------------------
# Sweep: combined ngs=0.005, 20 seeds
# 1 config x 20 seeds = 20 jobs (array 0-19)
#
#   Combined, neg_grad_scale = 0.005
#
#   Seeds: 42..61
# ---------------------------------------------------------------------------
DPS=13.0
LS=0.08
NUM_ITERS=501
FT_STEPS=250
INITIAL_MODEL_INVALID=true

SEEDS=(42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

EXTRA_FLAGS="--combined_finetuning --neg_grad_scale 0.005"
TAG="combined_ngs0.005"

OUTDIR=/cluster/scratch/$USER/toy_sweep_posneg/dps_${DPS}_ls_${LS}_ft${FT_STEPS}_initinvalid_${INITIAL_MODEL_INVALID}_${TAG}/seed_${SEED}
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
    --save_frames \
    $EXTRA_FLAGS
