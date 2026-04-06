#!/usr/bin/env bash

#SBATCH --job-name=toy_posneg
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-34
#SBATCH --gpus=rtx_2080:1
#SBATCH --output=output/toy_posneg_%A_%a.out
#SBATCH --error=output/toy_posneg_%A_%a.err

# ---------------------------------------------------------------------------
# Sweep: combined pos+neg finetuning (SISS No-IS, gradient norm scaling)
# 7 configs × 5 seeds = 35 jobs (array 0-34)
#
#   Config 0:    Baseline — positive-only
#   Config 1-6:  Combined, neg_grad_scale = 0.0, 0.005, 0.01, 0.015, 0.02, 0.03
#
#   Seeds: 42, 43, 44, 45, 46
# ---------------------------------------------------------------------------
DPS=13.0
LS=0.08
NUM_ITERS=501
FT_STEPS=250
INITIAL_MODEL_INVALID=true

SCALE_VALUES=(0.0 0.005 0.006 0.007 0.008 0.009 0.01)
SEEDS=(42 43 44 45 46)
N_SEEDS=${#SEEDS[@]}

CONFIG_IDX=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
SEED=${SEEDS[$SEED_IDX]}

if [ "$CONFIG_IDX" -eq 0 ]; then
    EXTRA_FLAGS=""
    TAG="baseline"
else
    S_IDX=$(( CONFIG_IDX - 1 ))
    SCALE=${SCALE_VALUES[$S_IDX]}
    EXTRA_FLAGS="--combined_finetuning --neg_grad_scale $SCALE"
    TAG="combined_ngs${SCALE}"
fi

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
