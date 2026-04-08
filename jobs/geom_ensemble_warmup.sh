#!/bin/bash
#SBATCH --job-name="GEOM Warmup"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32GB
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
FT_MIN_DATASET_SIZE="${3:-2048}"

SAMPLES_PER_ITER=64
# ~3x ratio accounts for ~33% validity rate during warmup, +10 buffer
WARMUP_ITERS=$(( FT_MIN_DATASET_SIZE * 3 / SAMPLES_PER_ITER + 10 ))

mkdir -p "$WARMUP_CACHE_DIR"

REG_ARG=""
if [ "$N_REG_SAMPLES" -gt 0 ]; then
    REG_ARG="--reg_data --n_reg_samples $N_REG_SAMPLES"
fi

pixi run python -m adm.task_agnostic \
    geom_drugs \
    ensemble \
    --dir "$WARMUP_CACHE_DIR/run" \
    --eval_samples 0 \
    --num_iters $WARMUP_ITERS \
    --samples_per_iter $SAMPLES_PER_ITER \
    --sample_batch_size 16 \
    --ft_min_dataset_size $FT_MIN_DATASET_SIZE \
    --ft_batch_size 16 \
    --ft_accumulate_steps 4 \
    --ft_steps 1 \
    --ft_lr 1e-4 \
    --no_uncertainty \
    --warmup_cache_dir "$WARMUP_CACHE_DIR" \
    $REG_ARG
