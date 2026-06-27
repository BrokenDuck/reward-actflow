#!/bin/bash
# ===========================================================================
#  evaluate_diversity_seeds_dgx.slurm — Evaluate diversity for seeds
#                  (50, 51, 52) across uncertainty_only and
#                  continued_pretraining_filtered on a SINGLE dgx-b200 node
#                  (8 GPUs). Each (seed, mode) task is pinned to its own GPU
#                  and auto-discovers all iter checkpoints in its run dir.
#
#                  continued_pretraining (unfiltered) is excluded: its
#                  validity collapsed. seed 52 has no
#                  continued_pretraining_filtered dir, so the job list skips
#                  it -> 5 tasks total, 5 of 8 GPUs used concurrently.
#
#  Submit with:    sbatch scripts/evaluate_diversity_seeds_dgx.slurm
# ===========================================================================

#SBATCH --job-name=eval-diversity-seeds
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512GB
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/slurm-%j.%x.log

# --- edit these for your environment ---
REPO_DIR=/path/to/actflow_pep   # parent of this scripts/ dir
CONDA_BASE=/path/to/miniconda3  # conda install (has bin/activate)
CONDA_ENV=actflow_pep             # env from `conda env create -f src/environment.yml`
# ----------------------------------------
SCRIPT_LOC=$REPO_DIR
LOG_LOC=$SCRIPT_LOC/logs
PYTHON_EXECUTABLE=$CONDA_ENV/bin/python
DATE=$(date +%m_%d)
NUM_GPUS=8

source $CONDA_BASE/bin/activate $CONDA_ENV
cd $SCRIPT_LOC
mkdir -p $LOG_LOC

# ---- Experiment config ----
ALPHA=0.01
SEEDS=(50 51 52)
# Parallel arrays: MODE_TAGS[i] checkpoint dirs end with MODE_SUFFIXES[i].
# all current dirs are named ${seed}_${mode}_alpha${ALPHA} (no suffix).
MODE_TAGS=(uncertainty_only continued_pretraining_filtered)
MODE_SUFFIXES=("" "")

PRETRAINED_CKPT=$SCRIPT_LOC/src/pretrained/peptune-pretrained.ckpt
CKPT_BASE=$SCRIPT_LOC/checkpoints

# ---- Build (seed, mode) job list, skipping missing run dirs ----
JOB_SEEDS=()
JOB_MODE_TAGS=()
JOB_MODE_SUFFIXES=()
for seed in "${SEEDS[@]}"; do
    for j in "${!MODE_TAGS[@]}"; do
        candidate="$CKPT_BASE/${seed}_${MODE_TAGS[$j]}_alpha${ALPHA}${MODE_SUFFIXES[$j]}"
        if [[ -d "$candidate" ]]; then
            JOB_SEEDS+=("$seed")
            JOB_MODE_TAGS+=("${MODE_TAGS[$j]}")
            JOB_MODE_SUFFIXES+=("${MODE_SUFFIXES[$j]}")
        fi
    done
done

TOTAL_JOBS=${#JOB_SEEDS[@]}
if [[ $TOTAL_JOBS -eq 0 ]]; then
    echo "ERROR: no (seed, mode) run dirs found under $CKPT_BASE"
    exit 1
fi

echo "=================================================================="
echo "  Evaluate Diversity on single dgx-b200 node  —  $(date)"
echo "  Total jobs: $TOTAL_JOBS   GPUs: $NUM_GPUS   Alpha: $ALPHA"
echo "=================================================================="

# ---- GPU scheduler: one job per GPU, max NUM_GPUS concurrent ----
declare -A SLOT_PID    # gpu -> pid of running job
declare -A SLOT_TAG    # gpu -> tag of running job
OVERALL_STATUS=0
FREE_GPU=-1

# launch job index $2 on GPU $1 (in the background)
launch_job() {
    local g=$1 i=$2
    local seed="${JOB_SEEDS[$i]}"
    local mode_tag="${JOB_MODE_TAGS[$i]}"
    local mode_suffix="${JOB_MODE_SUFFIXES[$i]}"
    local run_dir="$CKPT_BASE/${seed}_${mode_tag}_alpha${ALPHA}${mode_suffix}"
    local tag="seed${seed}_${mode_tag}_alpha${ALPHA}${mode_suffix}"
    local logfile="${LOG_LOC}/${DATE}_evaluate_diversity_${tag}.log"

    # auto-discover iter checkpoints in this run dir
    local variant_names=() variant_ckpts=()
    local ckpt iter
    shopt -s nullglob
    for ckpt in "$run_dir"/model_tfr_iter*_epoch*.ckpt; do
        iter=$(basename "$ckpt" | sed -E 's/.*iter([0-9]+)_epoch.*/\1/')
        variant_names+=("seed${seed}_${mode_tag}_alpha${ALPHA}_iter${iter}")
        variant_ckpts+=("$ckpt")
    done
    shopt -u nullglob

    if [[ ${#variant_names[@]} -eq 0 ]]; then
        echo "  [SKIP] job $i: no iter ckpts in $run_dir"
        return 1
    fi

    echo "------------------------------------------------------------------"
    echo "  Launch job $i / $((TOTAL_JOBS - 1))  on GPU $g  —  $(date)"
    echo "    Seed: $seed   Mode: $mode_tag   Variants: ${#variant_names[@]}"
    echo "    Run dir: $run_dir"
    echo "    Log: $logfile"

    CUDA_VISIBLE_DEVICES=$g $PYTHON_EXECUTABLE $SCRIPT_LOC/evaluate_diversity.py \
        --device cuda:0 \
        --n_valid 1000 \
        --seq_length 100 \
        --total_num_steps 128 \
        --noise_removal \
        --batch_size 500 \
        --pclm_batch_size 16 \
        --max_attempts 20 \
        --sigmas 0.5 \
        --pclm_cluster_thresholds 0.1 \
        --seeds 42 \
        --pretrained_ckpt "$PRETRAINED_CKPT" \
        --variant_names "${variant_names[@]}" \
        --variant_ckpts "${variant_ckpts[@]}" \
        --output_dir "$SCRIPT_LOC/results/diversity_eval" \
        --tag "${tag}_all_iters" \
        > "$logfile" 2>&1 &
    SLOT_PID[$g]=$!
    SLOT_TAG[$g]=$tag
}

# wait on the job in GPU slot $1, record its status, free the slot
reap_slot() {
    local g=$1
    local pid="${SLOT_PID[$g]}"
    local tag="${SLOT_TAG[$g]}"
    wait "$pid"
    local st=$?
    if [[ $st -eq 0 ]]; then
        echo "[DONE]   $tag  (GPU $g)"
    else
        echo "[FAILED] $tag  (GPU $g, exit=$st)"
        OVERALL_STATUS=1
    fi
    unset 'SLOT_PID[$g]'
    unset 'SLOT_TAG[$g]'
}

# block until a GPU is free, reaping any finished job; sets FREE_GPU
acquire_slot() {
    while true; do
        for g in $(seq 0 $((NUM_GPUS - 1))); do
            if [[ -z "${SLOT_PID[$g]:-}" ]]; then FREE_GPU=$g; return; fi
        done
        for g in $(seq 0 $((NUM_GPUS - 1))); do
            local pid="${SLOT_PID[$g]:-}"
            if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
                reap_slot "$g"; FREE_GPU=$g; return
            fi
        done
        sleep 10
    done
}

# ---- Dispatch all jobs ----
for i in $(seq 0 $((TOTAL_JOBS - 1))); do
    acquire_slot
    launch_job "$FREE_GPU" "$i"
done

# ---- Drain remaining jobs ----
for g in $(seq 0 $((NUM_GPUS - 1))); do
    if [[ -n "${SLOT_PID[$g]:-}" ]]; then reap_slot "$g"; fi
done

echo "=================================================================="
echo "  All eval jobs finished  —  $(date)  (overall exit=$OVERALL_STATUS)"
echo "  Results: $SCRIPT_LOC/results/diversity_eval/run_*_all_iters/"
echo "=================================================================="
exit $OVERALL_STATUS
