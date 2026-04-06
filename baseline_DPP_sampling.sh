#!/usr/bin/env bash

#SBATCH --job-name=baseline_dpp
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-47
#SBATCH --gpus=rtx_2080:1
#SBATCH --output=output/baseline_dpp_%A_%a.out
#SBATCH --error=output/baseline_dpp_%A_%a.err


NUM_ITERS=501
FT_STEPS=250
INITIAL_MODEL_INVALID=true
SAMPLES_PER_ITER=64
DPP_SPACE=data  # "features" (learned representation) or "data" (raw x,y)

# DPP parameters: grid search over pool sizes and kernel configs
# Kernel configs are (kernel, lengthscale) pairs; lengthscale is ignored for linear
DPP_POOL_SIZES=(128 256 512 1024)
DPP_LENGTHSCALES=(3.0 4.0 5.0 6.0 8.0 10.0)
N_POOL=${#DPP_POOL_SIZES[@]}
N_LS=${#DPP_LENGTHSCALES[@]}

LS_IDX=$((SLURM_ARRAY_TASK_ID % N_LS))
POOL_IDX=$((SLURM_ARRAY_TASK_ID / N_LS))

DPP_POOL_SIZE=${DPP_POOL_SIZES[$POOL_IDX]}
DPP_KERNEL=rbf
DPP_RBF_LENGTHSCALE=${DPP_LENGTHSCALES[$LS_IDX]}
SEED=42
SAVE_FRAMES="--save_frames"

if [ "$DPP_KERNEL" = "rbf" ]; then
    KERN_TAG=rbf_ls${DPP_RBF_LENGTHSCALE}
else
    KERN_TAG=linear
fi

OUTDIR=/cluster/scratch/$USER/baseline_dpp_pool${DPP_POOL_SIZE}_k${SAMPLES_PER_ITER}_${DPP_SPACE}_${KERN_TAG}_initinvalid_${INITIAL_MODEL_INVALID}/seed_${SEED}
mkdir -p "$OUTDIR"

.pixi/envs/default/bin/python -m adm.task_agnostic toy gp \
    --dir "$OUTDIR" \
    --validity_mode grid3 \
    --guidance_method dpp_sampling \
    --dpp_pool_size "$DPP_POOL_SIZE" \
    --dpp_space "$DPP_SPACE" \
    --dpp_kernel "$DPP_KERNEL" \
    --dpp_rbf_lengthscale "$DPP_RBF_LENGTHSCALE" \
    --num_iters "$NUM_ITERS" \
    --seed "$SEED" \
    --eval_samples 50000 \
    --eval_samples_curves 3000 \
    --eval_batch_size 50000 \
    --sample_batch_size 512 \
    --samples_per_iter "$SAMPLES_PER_ITER" \
    --ft_batch_size 256 \
    --ft_steps "$FT_STEPS" \
    --eval_every 50 \
    --initial_model_invalid "$INITIAL_MODEL_INVALID" \
    $SAVE_FRAMES
