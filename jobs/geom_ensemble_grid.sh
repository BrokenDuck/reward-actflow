#!/bin/bash
# Grid search over feat_timestep and dps_weight for GEOM-Drugs ensemble

TIMESTEPS=(0.8 0.85 0.9 0.95)
DPS_WEIGHTS=(10 30 60 100)

for TIMESTEP in "${TIMESTEPS[@]}"; do
    for DPS_WEIGHT in "${DPS_WEIGHTS[@]}"; do
        FOLDER="$SCRATCH/adm/geom_ensemble_long/ade_${TIMESTEP}_${DPS_WEIGHT}"
        mkdir -p "$FOLDER"
        echo "Submitting: feat_timestep=$TIMESTEP dps_weight=$DPS_WEIGHT -> $FOLDER"
        sbatch --output=$FOLDER/slurm_train_%j.out \
            jobs/geom_ensemble.sh "$TIMESTEP" "$DPS_WEIGHT"
    done
done
