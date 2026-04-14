#!/bin/bash
#SBATCH --job-name="QM9 Warmup"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --mem-bind=prefer
#SBATCH --gres-flags=enforce-binding
#SBATCH --gpu-bind=closest
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=24:00:00

export PATH="$HOME/.pixi/bin:$PATH"

cd ~/adm

module load eth_proxy

WARMUP_CACHE_DIR=$1
N_REG_SAMPLES="${2:-10000}"
FT_MIN_DATASET_SIZE="${3:-4096}"

SAMPLES_PER_ITER=64
WARMUP_ITERS=$(( FT_MIN_DATASET_SIZE * 2 / SAMPLES_PER_ITER + 10 ))

mkdir -p "$WARMUP_CACHE_DIR"

REG_ARG=""
if [ "$N_REG_SAMPLES" -gt 0 ]; then
    REG_ARG="--reg_data --n_reg_samples $N_REG_SAMPLES"
fi

pixi run python -m adm.task_agnostic \
    qm9 \
    ensemble \
    --dir "$WARMUP_CACHE_DIR/run" \
    --eval_samples 1000 \
    --eval_valid_samples 500 \
    --eval_batch_size 512 \
    --eval_every 999999 \
    --num_iters $WARMUP_ITERS \
    --samples_per_iter $SAMPLES_PER_ITER \
    --sample_batch_size 64 \
    --ft_min_dataset_size $FT_MIN_DATASET_SIZE \
    --ft_batch_size 64 \
    --ft_accumulate_steps 1 \
    --ft_steps 1 \
    --ft_lr 1e-4 \
    --no_uncertainty \
    --no_wandb \
    --warmup_cache_dir "$WARMUP_CACHE_DIR" \
    $REG_ARG
