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
BASE_DIR="${3:-$SCRATCH/adm/geom_ensemble_long}"

FOLDER="${BASE_DIR}/ade_${TIMESTEP}_${DPS_WEIGHT}"
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

pixi run python -m adm.task_agnostic \
    geom_drugs \
    ensemble \
    --dir $FOLDER \
    --eval_samples 1000 \
    --eval_batch_size 16 \
    --eval_every 10 \
    --dps_weight $DPS_WEIGHT \
    --feat_timestep $TIMESTEP \
    --num_iters 10_000 \
    --samples_per_iter 64 \
    --sample_batch_size 16 \
    --ft_lr 1e-4 \
    --ft_min_dataset_size 2048 \
    --ft_batch_size 16 \
    --ft_accumulate_steps 4 \
    --ft_steps 2000
