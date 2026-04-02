#!/usr/bin/env bash

#SBATCH --job-name=baseline_no_filter
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-19
#SBATCH --gpus=rtx_2080:1
#SBATCH --output=output/baseline_no_filter_%A_%a.out
#SBATCH --error=output/baseline_no_filter_%A_%a.err


NUM_ITERS=501
FT_STEPS=250
INITIAL_MODEL_INVALID=true
N_SEEDS=20

SEED_IDX=$((SLURM_ARRAY_TASK_ID % N_SEEDS))
SEED=$((42 + SEED_IDX))
[ "$SEED_IDX" -eq 0 ] && SAVE_FRAMES="--save_frames" || SAVE_FRAMES=""

OUTDIR=/cluster/scratch/$USER/baseline_no_filter_initinvalid_${INITIAL_MODEL_INVALID}/seed_${SEED}
mkdir -p "$OUTDIR"

.pixi/envs/default/bin/python -m adm.task_agnostic toy gp \
    --dir "$OUTDIR" \
    --validity_mode grid3 \
    --guidance_method none \
    --no_verifier \
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
    --initial_model_invalid "$INITIAL_MODEL_INVALID" \
    $SAVE_FRAMES
