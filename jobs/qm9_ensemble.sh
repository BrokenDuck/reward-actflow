#!/bin/bash
#SBATCH --job-name="QM9 Ensemble"
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8GB
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

FOLDER="$SCRATCH/adm/qm9_ensemble_long/apt_${TIMESTEP}_${DPS_WEIGHT}"

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
    ensemble \
    --dir $FOLDER \
    --eval_samples 1000 \
    --eval_batch_size 512 \
    --eval_every 10 \
    --dps_weight $DPS_WEIGHT \
    --feat_timestep $TIMESTEP \
    --num_iters 1000 \
    --samples_per_iter 64 \
    --sample_batch_size 64 \
    --ft_lr 1e-4 \
    --ft_min_dataset_size 4096 \
    --ft_batch_size 64 \
    --ft_accumulate_steps 1 \
    --ft_steps 500 \
    --num_iters 10_000
