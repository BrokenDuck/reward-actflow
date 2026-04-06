#!/usr/bin/env bash

#SBATCH --job-name=toy_nsm
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --array=0-63
#SBATCH --gpus=rtx_2080:1
#SBATCH --output=output/toy_nsm_%A_%a.out
#SBATCH --error=output/toy_nsm_%A_%a.err

# ---------------------------------------------------------------------------
# Shared hyperparameters
# ---------------------------------------------------------------------------
LS=0.08
NUM_ITERS=501
INITIAL_MODEL_INVALID=true

# ---------------------------------------------------------------------------
# Sweep layout  (4 pos-ft configs) × (1 baseline + 5 NSM weights × 3 seeds)
#   = 4 × 16 = 64 jobs  (array 0-63)
#
# Positive ft configs (more aggressive DPS / FT_STEPS):
#   0: DPS=13  FT_STEPS=250   (current default)
#   1: DPS=20  FT_STEPS=250   (stronger guidance)
#   2: DPS=13  FT_STEPS=500   (more ft steps)
#   3: DPS=20  FT_STEPS=500   (both higher)
#
# NSM: fixed steps=5, lr=1e-6, sweep weight 8→128.
# ---------------------------------------------------------------------------
DPS_VALUES=(13.0 17.0 13.0 17.0)
FT_VALUES=(250  250  300  300)
N_POS_CONFIGS=${#DPS_VALUES[@]}

NSM_WEIGHT_VALUES=(8.0 10.0 14.0 20.0 30.0)
N_NSM_WEIGHTS=${#NSM_WEIGHT_VALUES[@]}
NSM_STEPS=5
NSM_LR=1e-6

N_SEEDS=3
JOBS_PER_POS=$(( 1 + N_NSM_WEIGHTS * N_SEEDS ))   # 16

# ---------------------------------------------------------------------------
# Decode array task id
# ---------------------------------------------------------------------------
POS_IDX=$(( SLURM_ARRAY_TASK_ID / JOBS_PER_POS ))
LOCAL_IDX=$(( SLURM_ARRAY_TASK_ID % JOBS_PER_POS ))

DPS=${DPS_VALUES[$POS_IDX]}
FT_STEPS=${FT_VALUES[$POS_IDX]}

if [ "$LOCAL_IDX" -eq 0 ]; then
    # ---- Baseline for this pos-ft config ------------------------------------
    SEED=42
    NSM_FLAGS=""
    TAG="baseline"
    SAVE_FRAMES="--save_frames"
else
    # ---- NSM run -------------------------------------------------------------
    NSM_IDX=$(( (LOCAL_IDX - 1) / N_SEEDS ))
    SEED_IDX=$(( (LOCAL_IDX - 1) % N_SEEDS ))
    SEED=$((42 + SEED_IDX))

    NSM_W=${NSM_WEIGHT_VALUES[$NSM_IDX]}
    NSM_FLAGS="--nsm_enabled --nsm_weight $NSM_W --nsm_steps $NSM_STEPS --nsm_lr $NSM_LR"
    TAG="nsm_w${NSM_W}_s${NSM_STEPS}_lr${NSM_LR}"

    [ "$SEED_IDX" -eq 0 ] && SAVE_FRAMES="--save_frames" || SAVE_FRAMES=""
fi

OUTDIR=/cluster/scratch/$USER/toy_sweep_negative/dps_${DPS}_ls_${LS}_ft${FT_STEPS}_initinvalid_${INITIAL_MODEL_INVALID}_${TAG}/seed_${SEED}
mkdir -p "$OUTDIR"

.pixi/envs/default/bin/python -m adm.task_agnostic toy gp \
    --dir "$OUTDIR" \
    --validity_mode grid3 \
    --dps_weight "$DPS" \
    --gp_lengthscale "$LS" \
    --num_iters "$NUM_ITERS" \
    --seed "$SEED" \
    --eval_samples 50000 \
    --eval_samples_curves 3000 \
    --eval_batch_size 50000 \
    --sample_batch_size 512 \
    --samples_per_iter 64 \
    --ft_batch_size 256 \
    --ft_steps "$FT_STEPS" \
    --eval_every 50 \
    --guidance_method uncertainty_tilting \
    --initial_model_invalid "$INITIAL_MODEL_INVALID" \
    $NSM_FLAGS \
    $SAVE_FRAMES
