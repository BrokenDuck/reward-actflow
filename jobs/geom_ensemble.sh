#!/bin/bash
#SBATCH --job-name="GEOM Ensemble"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=32GB
#SBATCH --mem-bind=prefer
#SBATCH --gres-flags=enforce-binding
#SBATCH --gpu-bind=closest
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=120:00:00

export PATH="$HOME/.pixi/bin:$PATH"

cd ~/adm

module load eth_proxy

TIMESTEP=$1
DPS_WEIGHT=$2
FT_STEPS=$3
FT_LR=1e-4
BASE_DIR="${4:-$SCRATCH/adm/geom_ensemble_long}"
WARMUP_CACHE_DIR="${5:-}"
ALPHA_REG="${6:-0}"
SEED="${7:-}"
FT_MIN_DATASET_SIZE="${8:-2048}"
NEG_SCALE="${9:-0}"
N_SAMPLES_DIVERSITY=500
COMPUTE_VENDI=1

NUM_FT_ITERS=20000
SAMPLES_PER_ITER=64
# ~3x ratio accounts for ~33% validity rate during warmup
WARMUP_ITERS=$(( FT_MIN_DATASET_SIZE * 3 / SAMPLES_PER_ITER + 10 ))
NUM_ITERS=$(( NUM_FT_ITERS + WARMUP_ITERS ))

FOLDER="${BASE_DIR}/ade_t${TIMESTEP}_dps${DPS_WEIGHT}_ftsteps${FT_STEPS}_ftmin${FT_MIN_DATASET_SIZE}_neg${NEG_SCALE}"
if [ -n "$SEED" ]; then
    FOLDER="${FOLDER}_seed${SEED}"
fi
mkdir -p "$FOLDER"

sbatch --dependency=afterany:${SLURM_JOB_ID} \
    --output=$FOLDER/slurm_sample_valids_%j.out \
    jobs/sample_many_valids.sh \
    --folder $FOLDER \
    --ckpt last.pt \
    --samples-dir eval_samples_valid \
    --n-samples 10_000

sbatch --dependency=afterany:${SLURM_JOB_ID} \
    --output=$FOLDER/slurm_sample_%j.out \
    jobs/sample_many.sh \
    --folder $FOLDER \
    --ckpt last.pt \
    --samples-dir eval_samples \
    --n-samples 50_000

WARMUP_ARG=""
if [ -n "$WARMUP_CACHE_DIR" ]; then
    WARMUP_ARG="--warmup_cache_dir $WARMUP_CACHE_DIR"
fi

REG_ARG=""
if [ "$(echo "$ALPHA_REG > 0" | bc -l)" -eq 1 ]; then
    REG_ARG="--reg_data --alpha_reg $ALPHA_REG"
fi

pixi run python -m adm.task_agnostic \
    geom_drugs \
    ensemble \
    --dir $FOLDER \
    --eval_samples 1000 \
    --eval_batch_size 16 \
    --eval_every 50 \
    --dps_weight $DPS_WEIGHT \
    --feat_timestep $TIMESTEP \
    --num_iters $NUM_ITERS \
    --samples_per_iter $SAMPLES_PER_ITER \
    --sample_batch_size 16 \
    --ft_lr $FT_LR \
    --ft_min_dataset_size $FT_MIN_DATASET_SIZE \
    --ft_batch_size 16 \
    --ft_accumulate_steps 4 \
    --ft_steps $FT_STEPS \
    --eval_valid_samples $N_SAMPLES_DIVERSITY \
    --neg_grad_scale $NEG_SCALE \
    ${COMPUTE_VENDI:+--compute_vendi} \
    $WARMUP_ARG \
    $REG_ARG \
    ${SEED:+--seed $SEED}
