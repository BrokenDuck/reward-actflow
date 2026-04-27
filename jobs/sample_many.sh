#!/bin/bash
#SBATCH --job-name="Sample many"
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --gpus=rtx_4090:1
#SBATCH --time=24:00:00

source ~/.bashrc
export PATH="$HOME/.pixi/bin:$PATH"

cd ~/adm

module load eth_proxy

# Defaults
BATCH_SIZE=256
CKPT=""

# Argument parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        --folder)
            FOLDER="$2"
            shift 2
            ;;
        --ckpt)
            CKPT="$2"
            shift 2
            ;;
        --samples-dir)
            SAMPLES_DIR="$2"
            shift 2
            ;;
        --n-samples)
            N_SAMPLES="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Required args check
if [ -z "$FOLDER" ] || [ -z "$SAMPLES_DIR" ] || [ -z "$N_SAMPLES" ]; then
    echo "Usage:"
    echo "  $0 --folder <folder> --samples-dir <dir> --n-samples <n> [--ckpt <ckpt>] [--batch-size <bs>]"
    exit 1
fi

# Base command
CMD=(
    pixi run python -m adm.evaluation.sample_many
    "$FOLDER"
    --samples_dir "$SAMPLES_DIR"
    --n_samples "$N_SAMPLES"
    --batch_size "$BATCH_SIZE"
)

# Optional ckpt
if [ -n "$CKPT" ]; then
    CMD+=(--ckpt "$CKPT")
fi

echo "Running command: ${CMD[*]}"

# Run
"${CMD[@]}"

sbatch --dependency=afterok:${SLURM_JOB_ID} \
    --output=$FOLDER/slurm_eval_${SAMPLES_DIR}_%j.out \
    jobs/eval_samples.sh \
    "$FOLDER" \
    "$SAMPLES_DIR" \
    --do_sample_metrics
