#!/bin/bash
#SBATCH --job-name="QM9 Long"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
#SBATCH --mem-bind=prefer
#SBATCH --gres-flags=enforce-binding
#SBATCH --gpu-bind=closest
#SBATCH --gpus=a100:1
#SBATCH --time=120:00:00

export PATH="$HOME/.pixi/bin:$PATH"

cd ~/adm

module load eth_proxy

GP_KERNEL=$1
TIMESTEP=$2
DPS_WEIGHT=$3

# Decide GP kernel arguments
if [[ "$GP_KERNEL" == "linear" ]]; then
    GP_KERNEL_ARG="--gp_kernel linear"
else
    # assume GP_KERNEL is a numeric lengthscale
    GP_KERNEL_ARG="--gp_kernel rbf --gp_lengthscale $GP_KERNEL"
fi

FOLDER="$SCRATCH/adm/qm9_long/apt_${TIMESTEP}_${DPS_WEIGHT}_${GP_KERNEL}"

sbatch --dependency=afterany:${SLURM_JOB_ID} jobs/sample_many_valids.sh \
    --folder $FOLDER \
    --ckpt base_model.pt \
    --samples-dir eval_samples_valid \
    --n-samples 10_000 \

sbatch --dependency=afterany:${SLURM_JOB_ID} jobs/sample_many.sh \
    --folder $FOLDER \
    --ckpt base_model.pt \
    --samples-dir eval_samples \
    --n-samples 50_000 \

pixi run python -m adm.task_agnostic \
    qm9 \
    gp \
    --dir $FOLDER \
    --eval_samples 1000 \
    --eval_batch_size 256 \
    --eval_every 10 \
    $GP_KERNEL_ARG \
    --dps_weight $DPS_WEIGHT \
    --feat_timestep $TIMESTEP \
    --num_iters 1000 \
    --samples_per_iter 64 \
    --sample_batch_size 64 \
    --ft_lr 1e-4 \
    --ft_min_dataset_size 4096 \
    --ft_batch_size 64 \
    --ft_accumulate_steps 1 \
    --ft_steps 1000 \
    --num_iters 10_000
