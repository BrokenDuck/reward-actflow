#!/bin/bash
# Launch QM9 baseline: no uncertainty (no DPS guidance, still filters to valid data)
# Usage: bash jobs/run_qm9_baseline_no_uncertainty.sh

BASE_DIR="$SCRATCH/adm/qm9_ensemble_v1"
TIMESTEP=0.8
FT_STEPS_LIST=(500)
FT_MIN_DS=4096
SEEDS=(0 1 2 3 4)
ALPHA_REG=0
N_REG_SAMPLES=0

WARMUP_CACHE_DIR="$BASE_DIR/warmup_cache_ftmin${FT_MIN_DS}"
mkdir -p "$WARMUP_CACHE_DIR"
WARMUP_JOB=$(sbatch --parsable --output=$WARMUP_CACHE_DIR/slurm_warmup_%j.out \
    jobs/qm9_ensemble_warmup.sh "$WARMUP_CACHE_DIR" "$N_REG_SAMPLES" "$FT_MIN_DS")
echo "Submitted warmup job: $WARMUP_JOB -> $WARMUP_CACHE_DIR"

for FT_STEPS in "${FT_STEPS_LIST[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        FOLDER="$BASE_DIR/baseline_no_uncertainty_t${TIMESTEP}_ftsteps${FT_STEPS}_ftmin${FT_MIN_DS}_seed${SEED}"
        mkdir -p "$FOLDER"
        echo "Submitting no_uncertainty: ft_steps=$FT_STEPS seed=$SEED -> $FOLDER"
        sbatch --dependency=afterok:$WARMUP_JOB \
            --output=$FOLDER/slurm_train_%j.out \
            jobs/qm9_ensemble_no_uncertainty.sh "$TIMESTEP" "$FT_STEPS" "$BASE_DIR" "$WARMUP_CACHE_DIR" "$ALPHA_REG" "$SEED" "$FT_MIN_DS"
    done
done
