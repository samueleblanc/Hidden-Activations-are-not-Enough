#!/bin/bash
# ==============================================================
# experiment_config.sh — Shared Configuration
#
# Single source of truth for experiment settings, resource
# profiles, accounts, and helper functions.
# Sourced by both calibration.sh and run_experiment.sh.
# ==============================================================

# --- Default configuration ---
ACCOUNT="def-assem"
GPU_ACCOUNT=""              # Account for GPU jobs (defaults to ACCOUNT if empty)
CPU_ACCOUNT=""              # Account for CPU jobs (defaults to ACCOUNT if empty)
TOTAL_CHUNKS=8
BATCH_SIZE=1800
NUM_SAMPLES_PER_CLASS=100
SAMPLES_PER_ATTACK=500
TEST_SIZE=-1
ENV_NAME="env"
MODULES="StdEnv/2023 python/3.11.5 scipy-stack/2025a"
SLURM_OUT_DIR="slurm_out"
SLURM_ERR_DIR="slurm_err"
# --- Incremental save settings ---
SAVE_INTERVAL=200           # Incremental save every N new matrices
SAVE_CHECK_SECONDS=60       # How often background process checks
SAVE_GRACE_SECONDS=180      # Seconds before wall time to trigger emergency save

# --- Resource profiles (normal mode) ---
# Step A
A_GPU="--gpus=h100:1"
A_CPUS=2
A_TIME="06:00:00"
A_MEM="15G"
# Step B
B_GPU="--gpus=h100:1"
B_CPUS=12
B_TIME="00:20:00"
B_MEM="280G"
# Step C
C_GPU="--gres=gpu:1"
C_CPUS=16
C_TIME="12:00:00"
C_MEM="124G"
# Step D (Adv Matrices)
D_GPU="--gpus=h100:1"
D_CPUS=12
D_TIME="12:00:00"
D_MEM="280G"
# Step E (Representation Comparison - GPU)
E_GPU="--gpus=h100:1"
E_CPUS=8
E_TIME="08:00:00"
E_MEM="64G"
# Step G (Theorem 4.5 Validation - GPU)
G_GPU="--gpus=h100:1"
G_CPUS=4
G_TIME="03:00:00"
G_MEM="64G"
# Step F (LaTeX Tables - CPU-only, lightweight)
F_CPUS=2
F_TIME="00:15:00"
F_MEM="4G"
# Audit
AUDIT_CPUS=4
AUDIT_TIME="02:00:00"
AUDIT_MEM="100G"
# Calibration
CALIB_GPU="--gpus=h100:1"
CALIB_CPUS=4
CALIB_TIME="01:00:00"
CALIB_MEM="32G"

# ==============================================================
# Experiments to process
# ==============================================================
EXPERIMENTS=("alexnet_cifar10")

# --- Resolve per-type accounts (default to ACCOUNT) ---
GPU_ACCOUNT="${GPU_ACCOUNT:-$ACCOUNT}"
CPU_ACCOUNT="${CPU_ACCOUNT:-$ACCOUNT}"

# --- Load environment (needed for pre-flight Python calls) ---
module load $MODULES 2>/dev/null || true
if [ -d "$ENV_NAME" ]; then
    source $ENV_NAME/bin/activate
fi

# ==============================================================
# Helper functions
# ==============================================================
submit_job() {
    local script="$1"
    local deps="$2"
    local sbatch_cmd="sbatch --parsable"
    if [ -n "$deps" ]; then
        sbatch_cmd="sbatch --parsable --dependency=afterok:${deps}"
    fi
    local job_id
    job_id=$($sbatch_cmd "$script")
    echo "$job_id"
}

submit_job_afterany() {
    local script="$1"
    local deps="$2"
    local sbatch_cmd="sbatch --parsable"
    if [ -n "$deps" ]; then
        sbatch_cmd="sbatch --parsable --dependency=afterany:${deps}"
    fi
    local job_id
    job_id=$($sbatch_cmd "$script")
    echo "$job_id"
}

enforce_min_time() {
    # Ensures SLURM time is at least a minimum floor (default 30 min)
    # Usage: enforce_min_time "HH:MM:SS" ["HH:MM:SS_floor"]
    local time_str="$1"
    local min_time="${2:-00:30:00}"
    local h m s
    IFS=: read -r h m s <<< "$time_str"
    local total=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
    IFS=: read -r h m s <<< "$min_time"
    local min_seconds=$(( 10#$h * 3600 + 10#$m * 60 + 10#$s ))
    if [ "$total" -lt "$min_seconds" ]; then
        echo "$min_time"
    else
        echo "$time_str"
    fi
}

# Determine dataset dirs to copy based on experiment
get_dataset_copy_commands() {
    local dataset="$1"
    case "$dataset" in
        cifar10)
            echo 'mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/ 2>/dev/null || true'
            ;;
        cifar100)
            echo 'mkdir -p $SLURM_TMPDIR/data/cifar-100-python/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/ 2>/dev/null || true'
            ;;
        mnist)
            echo 'mkdir -p $SLURM_TMPDIR/data/MNIST/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/MNIST/* $SLURM_TMPDIR/data/MNIST/ 2>/dev/null || true'
            ;;
        fashion)
            echo 'mkdir -p $SLURM_TMPDIR/data/FashionMNIST/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/FashionMNIST/* $SLURM_TMPDIR/data/FashionMNIST/ 2>/dev/null || true'
            ;;
        imagenet)
            echo '# ImageNet: reading directly from /datashare/imagenet/ILSVRC2012/ (NFS)'
            ;;
    esac
}

# ==============================================================
# Generate and submit pipeline for each experiment
# ==============================================================

# Get dataset for each experiment
get_experiment_dataset() {
    python3 -c "
from constants.constants import DEFAULT_EXPERIMENTS
print(DEFAULT_EXPERIMENTS.get('$1', {}).get('dataset', 'cifar10'))
"
}

get_experiment_epochs() {
    python3 -c "
from constants.constants import DEFAULT_EXPERIMENTS
print(DEFAULT_EXPERIMENTS.get('$1', {}).get('epochs', 0))
"
}

get_experiment_num_classes() {
    python3 -c "
from constants.constants import DEFAULT_EXPERIMENTS
d = DEFAULT_EXPERIMENTS.get('$1', {}).get('dataset', 'cifar10')
print({'cifar10':10,'cifar100':100,'mnist':10,'fashion':10,'imagenet':1000}.get(d, 10))
"
}

read_checkpoint_status() {
    # $1 = checkpoint file path
    # Returns: complete, partial, or missing
    if [ -f "$1" ]; then
        python3 -c "import json; print(json.load(open('$1')).get('status','partial'))"
    else
        echo "missing"
    fi
}
