#!/bin/bash
# Grid search v2: refine low dps_weight region for GEOM-Drugs ensemble

BASE_DIR="$SCRATCH/adm/geom_ensemble_v2"
TIMESTEPS=(0.75 0.8 0.85)
DPS_WEIGHTS=(1 3 5 7 10 15)

for TIMESTEP in "${TIMESTEPS[@]}"; do
    for DPS_WEIGHT in "${DPS_WEIGHTS[@]}"; do
        FOLDER="$BASE_DIR/ade_${TIMESTEP}_${DPS_WEIGHT}"
        mkdir -p "$FOLDER"
        echo "Submitting: feat_timestep=$TIMESTEP dps_weight=$DPS_WEIGHT -> $FOLDER"
        sbatch --output=$FOLDER/slurm_train_%j.out \
            jobs/geom_ensemble.sh "$TIMESTEP" "$DPS_WEIGHT" "$BASE_DIR"
    done
done
