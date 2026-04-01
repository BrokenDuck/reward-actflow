#!/usr/bin/env bash

#SBATCH --job-name=baseline_with_filter
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --gpus=rtx_2080:1
#SBATCH --output=output/baseline_with_filter.out
#SBATCH --error=output/baseline_with_filter.err


NUM_ITERS=501
FT_STEPS=250
INITIAL_MODEL_INVALID=true

OUTDIR=/cluster/scratch/$USER/baseline_with_filter_initinvalid_${INITIAL_MODEL_INVALID}
mkdir -p "$OUTDIR"

.pixi/envs/default/bin/python -m adm.task_agnostic toy gp \
    --dir "$OUTDIR" \
    --validity_mode grid3 \
    --guidance_method none \
    --num_iters "$NUM_ITERS" \
    --seed 42 \
    --eval_samples 50000 \
    --eval_samples_curves 3000 \
    --eval_batch_size 50000 \
    --sample_batch_size 512 \
    --samples_per_iter 64 \
    --ft_batch_size 256 \
    --ft_steps "$FT_STEPS" \
    --eval_every 50 \
    --initial_model_invalid "$INITIAL_MODEL_INVALID"
