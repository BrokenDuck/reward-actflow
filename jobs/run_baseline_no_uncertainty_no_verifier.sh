#!/bin/bash
# Launch baseline: no uncertainty + no verifier (no DPS guidance, fine-tunes on all data)
# Usage: bash jobs/run_baseline_no_uncertainty_no_verifier.sh

BASE_DIR="$SCRATCH/adm/geom_ensemble_v3"
TIMESTEP=0.8
DPS_WEIGHT=7
FT_STEPS=2000
FT_MIN_DS=8192
SEEDS=(0)
ALPHA_REG=0
N_REG_SAMPLES=0

WARMUP_CACHE_DIR="$BASE_DIR/warmup_cache_ftmin${FT_MIN_DS}"
mkdir -p "$WARMUP_CACHE_DIR"
WARMUP_JOB=$(sbatch --parsable --output=$WARMUP_CACHE_DIR/slurm_warmup_%j.out \
    jobs/geom_ensemble_warmup.sh "$WARMUP_CACHE_DIR" "$N_REG_SAMPLES" "$FT_MIN_DS")
echo "Submitted warmup job: $WARMUP_JOB -> $WARMUP_CACHE_DIR"

for SEED in "${SEEDS[@]}"; do
    FOLDER="$BASE_DIR/baseline_no_uncertainty_no_verifier_t${TIMESTEP}_dps${DPS_WEIGHT}_ftsteps${FT_STEPS}_ftmin${FT_MIN_DS}_seed${SEED}"
    mkdir -p "$FOLDER"
    echo "Submitting no_uncertainty_no_verifier: seed=$SEED -> $FOLDER"
    sbatch --dependency=afterok:$WARMUP_JOB \
        --output=$FOLDER/slurm_train_%j.out \
        jobs/geom_ensemble_no_uncertainty_no_verifier.sh "$TIMESTEP" "$DPS_WEIGHT" "$FT_STEPS" "$BASE_DIR" "$WARMUP_CACHE_DIR" "$ALPHA_REG" "$SEED" "$FT_MIN_DS"
done
