#!/bin/bash
# Grid search: sweep DPS_WEIGHT x FT_STEPS x NEG_SCALE for QM9 ensemble
# Usage: bash jobs/qm9_ensemble_grid.sh

BASE_DIR="$SCRATCH/adm/qm9_ensemble_v1"
TIMESTEP=0.9
DPS_WEIGHT_LIST=(10)
FT_STEPS_LIST=(500)
FT_MIN_DS=4096
SEEDS=(0)
ALPHA_REG=0
N_REG_SAMPLES=0
NEG_SCALE_LIST=(0 0.001 0.005 0.01)

WARMUP_CACHE_DIR="$BASE_DIR/warmup_cache_ftmin${FT_MIN_DS}"
mkdir -p "$WARMUP_CACHE_DIR"
WARMUP_JOB=$(sbatch --parsable --output=$WARMUP_CACHE_DIR/slurm_warmup_%j.out \
    jobs/qm9_ensemble_warmup.sh "$WARMUP_CACHE_DIR" "$N_REG_SAMPLES" "$FT_MIN_DS")
echo "Submitted warmup job: $WARMUP_JOB -> $WARMUP_CACHE_DIR"

for DPS_WEIGHT in "${DPS_WEIGHT_LIST[@]}"; do
    for FT_STEPS in "${FT_STEPS_LIST[@]}"; do
        for NEG_SCALE in "${NEG_SCALE_LIST[@]}"; do
            for SEED in "${SEEDS[@]}"; do
                FOLDER="$BASE_DIR/ade_t${TIMESTEP}_dps${DPS_WEIGHT}_ftsteps${FT_STEPS}_ftmin${FT_MIN_DS}_neg${NEG_SCALE}_seed${SEED}"
                mkdir -p "$FOLDER"
                echo "Submitting: dps=$DPS_WEIGHT ft_steps=$FT_STEPS neg=$NEG_SCALE seed=$SEED -> $FOLDER"
                sbatch --dependency=afterok:$WARMUP_JOB \
                    --output=$FOLDER/slurm_train_%j.out \
                    jobs/qm9_ensemble.sh "$TIMESTEP" "$DPS_WEIGHT" "$FT_STEPS" "$BASE_DIR" "$WARMUP_CACHE_DIR" "$ALPHA_REG" "$SEED" "$FT_MIN_DS" "$NEG_SCALE"
            done
        done
    done
done
