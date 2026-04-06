#!/bin/bash

FOLDER="$SCRATCH/adm/geom_ensemble_long/no_uncertainty"
mkdir -p "$FOLDER"

sbatch \
    --output=$FOLDER/slurm_sample_valids_pretrained_%j.out \
    jobs/sample_many_valids.sh \
    --folder $FOLDER \
    --samples-dir eval_samples_valid_pretrained \
    --n-samples 10_000

sbatch \
    --output=$FOLDER/slurm_sample_pretrained_%j.out \
    jobs/sample_many.sh \
    --folder $FOLDER \
    --samples-dir eval_samples_pretrained \
    --n-samples 100_000
