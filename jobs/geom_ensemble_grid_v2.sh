#!/bin/bash
# Grid search v3: sweep timestep, dps_weight, ft_steps for GEOM-Drugs ensemble (lr=1e-4 fixed)
# Runs a shared warmup job first, then submits all fine-tuning jobs after it completes.

BASE_DIR="$SCRATCH/adm/geom_ensemble_v3"
WARMUP_CACHE_DIR="$BASE_DIR/warmup_cache"
TIMESTEP=0.8
DPS_WEIGHT=7
FT_STEPS=2000
SEEDS=(0 1 2 3 4)
ALPHA_REG=0.8
N_REG_SAMPLES=2048

# Submit warmup job (also generates and caches reg data)
mkdir -p "$WARMUP_CACHE_DIR"
WARMUP_JOB=$(sbatch --parsable --output=$WARMUP_CACHE_DIR/slurm_warmup_%j.out \
    jobs/geom_ensemble_warmup.sh "$WARMUP_CACHE_DIR" "$N_REG_SAMPLES")
echo "Submitted warmup job: $WARMUP_JOB -> $WARMUP_CACHE_DIR"

# Submit fine-tuning jobs (one per seed) after warmup completes
for SEED in "${SEEDS[@]}"; do
    FOLDER="$BASE_DIR/ade_t${TIMESTEP}_dps${DPS_WEIGHT}_ftsteps${FT_STEPS}_seed${SEED}"
    mkdir -p "$FOLDER"
    echo "Submitting: seed=$SEED -> $FOLDER"
    sbatch --dependency=afterok:$WARMUP_JOB \
        --output=$FOLDER/slurm_train_%j.out \
        jobs/geom_ensemble.sh "$TIMESTEP" "$DPS_WEIGHT" "$FT_STEPS" "$BASE_DIR" "$WARMUP_CACHE_DIR" "$ALPHA_REG" "$SEED"
done
