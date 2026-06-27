#!/bin/bash
# ===========================================================================
#  run_all.slurm — Train 3 exploration modes across multiple random
#                  seeds on a single dgx-b200 node (8 GPUs). Each
#                  (mode, seed) job is pinned to its own GPU; with
#                  3 seeds x 3 modes = 9 jobs and 8 GPUs, 8 run at once
#                  and the 9th starts as soon as a GPU frees.
#
#  Submit with:    sbatch scripts/run_all.slurm
#
#  Modes:
#    1. uncertainty_only              — GP uncertainty tilting
#    2. continued_pretraining         — Denoising loss on all self-generated samples
#    3. continued_pretraining_filtered — Denoising loss on valid-only self-generated samples
# ===========================================================================

#SBATCH --job-name=run-all-modes
#SBATCH --partition=dgx-b200
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=512GB
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/slurm-%j.%x.log

# --- edit these for your environment ---
REPO_DIR=/path/to/actflow_pep   # parent of this scripts/ dir
CONDA_BASE=/path/to/miniconda3  # conda install (has bin/activate)
CONDA_ENV=actflow_pep             # env from `conda env create -f src/environment.yml`
# ----------------------------------------
SCRIPT_LOC=$REPO_DIR
LOG_LOC=$REPO_DIR/logs
CKPT_LOC=$REPO_DIR/checkpoints
PYTHON_EXECUTABLE=$CONDA_ENV/bin/python
DATE=$(date +%m_%d)
SPECIAL_PREFIX='run-all-modes'
NUM_GPUS=8

source $CONDA_BASE/bin/activate $CONDA_ENV
cd $SCRIPT_LOC
mkdir -p $LOG_LOC

# Experiment config
ALPHA=0.005
SEEDS=(50 51 52)  # 3 seeds x 3 modes = 9 jobs
GP_KERNEL=rbf
INITIAL_POOL_SIZE=1000
POOL_REFRESH_FRACTION=1

COMMON_ARGS=(
    --noise_removal
    --grad_clip
    --wdce_num_replicates 16
    --seq_length 100
    --total_num_steps 128
    --num_epochs 100
    --save_every_n_epochs 100
    --learning_rate 1e-4
    --batch_size 100
    --num_accum_steps 4
    --truncate_steps 50
    --buffer_size 500
    --training_mini_batch_size 20
    --num_children 50
    --num_iter 10
    --exploration 0.1
    --resample_every_n_step 10
    --reset_every_n_step 1
    --num_al_iterations 100
    --num_exploration_samples 100
    --gp_kernel "$GP_KERNEL"
    --gp_lengthscale 0  
    --gp_conditional_reward  
    --vendi_eval_samples 200
    --vendi_n_valid 100
    --vendi_kernel cosine
    --vendi_sigma 0.25
    --skip_eval
    --save_every_n_iters 2
    --prot_name "tfr"
    --scalarization "sum"
    --initial_pool_size "$INITIAL_POOL_SIZE"
    --pool_refresh_fraction "$POOL_REFRESH_FRACTION"
)

# ---- Build full job list: (mode, mode_tag, seed, run_name, extra_args) ----
# run_name → checkpoint subfolder: checkpoints/{seed}_{mode_tag}_alpha{ALPHA}
JOB_MODES=()
JOB_TAGS=()
JOB_SEEDS=()
JOB_RUN_NAMES=()
JOB_EXTRA=()  # space-separated extra args per job (empty string if none)

for seed in "${SEEDS[@]}"; do
    JOB_MODES+=("uncertainty_only");      JOB_TAGS+=("uncertainty_only_seed${seed}");               JOB_SEEDS+=("$seed"); JOB_RUN_NAMES+=("${seed}_uncertainty_only_alpha${ALPHA}");               JOB_EXTRA+=("--filter_invalid --gp_filter_invalid")
    JOB_MODES+=("continued_pretraining"); JOB_TAGS+=("continued_pretraining_seed${seed}");          JOB_SEEDS+=("$seed"); JOB_RUN_NAMES+=("${seed}_continued_pretraining_alpha${ALPHA}");          JOB_EXTRA+=("")
    JOB_MODES+=("continued_pretraining"); JOB_TAGS+=("continued_pretraining_filtered_seed${seed}"); JOB_SEEDS+=("$seed"); JOB_RUN_NAMES+=("${seed}_continued_pretraining_filtered_alpha${ALPHA}"); JOB_EXTRA+=("--filter_invalid --gp_filter_invalid")
done

TOTAL_JOBS=${#JOB_MODES[@]}

echo "=================================================================="
echo "  run_all on single dgx-b200 node  —  $(date)"
echo "  Total jobs: $TOTAL_JOBS   GPUs: $NUM_GPUS   Alpha: $ALPHA   Kernel: $GP_KERNEL"
echo "=================================================================="

# ---- GPU scheduler: one job per GPU, max NUM_GPUS concurrent ----
declare -A SLOT_PID    # gpu -> pid of running job
declare -A SLOT_TAG    # gpu -> tag of running job
OVERALL_STATUS=0
FREE_GPU=-1

# launch job index $2 on GPU $1 (in the background)
launch_job() {
    local g=$1 i=$2
    local mode="${JOB_MODES[$i]}"
    local tag="${JOB_TAGS[$i]}"
    local seed="${JOB_SEEDS[$i]}"
    local run_name="${JOB_RUN_NAMES[$i]}"
    local extra="${JOB_EXTRA[$i]}"
    local logfile="${LOG_LOC}/${DATE}_${SPECIAL_PREFIX}_${tag}.log"

    echo "------------------------------------------------------------------"
    echo "  Launch job $i / $((TOTAL_JOBS - 1))  on GPU $g  —  $(date)"
    echo "    Mode: $mode   Tag: $tag   Seed: $seed   Extra: $extra"
    echo "    Run name: $run_name"
    echo "    Log: $logfile"

    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES=$g $PYTHON_EXECUTABLE $SCRIPT_LOC/finetune_with_uncertainty.py \
        --device "cuda:0" \
        --mode "$mode" \
        --seed "$seed" \
        --alpha "$ALPHA" \
        --run_name "$run_name" \
        "${COMMON_ARGS[@]}" \
        $extra \
        > "$logfile" 2>&1 &
    SLOT_PID[$g]=$!
    SLOT_TAG[$g]=$tag
}

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

for i in $(seq 0 $((TOTAL_JOBS - 1))); do
    acquire_slot
    launch_job "$FREE_GPU" "$i"
done

for g in $(seq 0 $((NUM_GPUS - 1))); do
    if [[ -n "${SLOT_PID[$g]:-}" ]]; then reap_slot "$g"; fi
done

echo "=================================================================="
echo "  All jobs finished  —  $(date)  (overall exit=$OVERALL_STATUS)"
echo "=================================================================="
exit $OVERALL_STATUS
