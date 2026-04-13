#!/bin/bash
# ==============================================================
# hyperparameters.sh -- HP Tuning Pipeline Entry Point
#
# Single command entry point for the hyperparameter tuning
# pipeline. Downloads pretrained weights and datasets, generates
# planner + executor Slurm scripts, and submits them.
#
# Usage:
#   bash slurm/hyperparameters.sh [--dry-run]
#
# Prerequisites:
#   - Python venv at $VENV with torch, torchvision, optuna
#   - Internet access (login node) for weight/data downloads
#   - SLURM cluster with H100 MIG support
# ==============================================================

set -euo pipefail

# ==================== USER CONFIGURATION ====================
GPU_ACCOUNT="def-assem"               # Slurm account for GPU jobs
CPU_ACCOUNT="def-assem"               # Slurm account for CPU jobs
MAX_SENTINEL_CYCLES=5                 # Max rounds
INITIAL_TIME="6:00:00"                # Time limit for first round
INITIAL_MEM="15G"                     # System memory for first round
INITIAL_MIG="H100-1g.10gb"           # Starting MIG tier
TRIALS_PER_CONFIG=50                  # Optuna trials per config
MAX_JOBS=800                          # Max Slurm jobs per cycle
MODULES="StdEnv/2023 python/3.11.5 scipy-stack/2025a"
VENV="env"
# =============================================================

# ==================== DERIVED VARIABLES ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BASE_DIR="$PROJECT_DIR/experiments/hp_tuning"
REGISTRY="$BASE_DIR/job_registry.json"
MANIFEST="$BASE_DIR/manifest.json"
DRY_RUN=""
# =============================================================

# ==================== PARSE CLI ARGS ====================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Usage: bash slurm/hyperparameters.sh [--dry-run]"
            exit 1
            ;;
    esac
done

# ==================== STEP 1: CONFIGURATION SUMMARY ====================
echo "=============================================================="
echo "  HP Tuning Pipeline"
echo "=============================================================="
echo ""
echo "  GPU account:        $GPU_ACCOUNT"
echo "  CPU account:        $CPU_ACCOUNT"
echo "  Max sentinel cycles: $MAX_SENTINEL_CYCLES"
echo "  Initial time:       $INITIAL_TIME"
echo "  Initial memory:     $INITIAL_MEM"
echo "  Initial MIG tier:   $INITIAL_MIG"
echo "  Trials per config:  $TRIALS_PER_CONFIG"
echo "  Max jobs per cycle: $MAX_JOBS"
echo "  Modules:            $MODULES"
echo "  Venv:               $VENV"
echo "  Project dir:        $PROJECT_DIR"
echo "  Base dir:           $BASE_DIR"
echo "  Registry:           $REGISTRY"
echo "  Manifest:           $MANIFEST"
if [[ -n "$DRY_RUN" ]]; then
    echo "  Mode:               DRY RUN (no jobs will be submitted)"
fi
echo ""
echo "=============================================================="
echo ""

# ==================== STEP 2: DOWNLOAD PRETRAINED WEIGHTS ====================
echo "--- Downloading pretrained weights (if not cached) ---"

# Activate venv
# shellcheck disable=SC1091
source "$PROJECT_DIR/$VENV/bin/activate"

python -c "
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,
)
model_fns = [
    resnet18, resnet34, resnet50, resnet101, resnet152,
    vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn,
]
for fn in model_fns:
    name = fn.__name__
    print(f'  Ensuring weights for {name}...')
    fn(weights='DEFAULT')
print('  All 9 pretrained weight files ready.')
"

echo ""

# ==================== STEP 3: DOWNLOAD TINY IMAGENET ====================
TINY_DIR="$PROJECT_DIR/data/tiny-imagenet-200"
if [[ ! -d "$TINY_DIR" ]]; then
    echo "--- Downloading Tiny ImageNet ---"
    mkdir -p "$PROJECT_DIR/data"
    wget -q http://cs231n.stanford.edu/tiny-imagenet-200.zip -O /tmp/tiny-imagenet-200.zip
    unzip -q /tmp/tiny-imagenet-200.zip -d "$PROJECT_DIR/data/"
    rm /tmp/tiny-imagenet-200.zip
    echo "  Restructuring validation directory..."
    python "$SCRIPT_DIR/restructure_tiny_imagenet_val.py" --dir "$TINY_DIR"
    echo "  Tiny ImageNet ready."
    echo ""
else
    echo "--- Tiny ImageNet already present at $TINY_DIR ---"
    echo ""
fi

# ==================== STEP 4: CREATE DIRECTORIES ====================
echo "--- Creating directories ---"
mkdir -p "$BASE_DIR"
mkdir -p "$PROJECT_DIR/slurm_out"
mkdir -p "$PROJECT_DIR/slurm_err"
echo "  $BASE_DIR"
echo "  $PROJECT_DIR/slurm_out"
echo "  $PROJECT_DIR/slurm_err"
echo ""

# ==================== STEP 5: GENERATE SLURM SCRIPTS ====================
echo "--- Generating Slurm scripts ---"

PLANNER_SCRIPT="$BASE_DIR/planner.sh"
EXECUTOR_SCRIPT="$BASE_DIR/executor.sh"

cat > "$PLANNER_SCRIPT" << PLANNER_EOF
#!/bin/bash
#SBATCH --account=$CPU_ACCOUNT
#SBATCH --job-name=hp_planner
#SBATCH --output=$PROJECT_DIR/slurm_out/hp_planner_%j.out
#SBATCH --error=$PROJECT_DIR/slurm_err/hp_planner_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00

module load $MODULES
source "$PROJECT_DIR/$VENV/bin/activate"
cd "$PROJECT_DIR"

python -m slurm.hp_planner \\
    --registry "$REGISTRY" \\
    --max-jobs $MAX_JOBS \\
    --trials-per-config $TRIALS_PER_CONFIG \\
    --initial-mig "$INITIAL_MIG" \\
    --initial-mem "$INITIAL_MEM" \\
    --initial-time "$INITIAL_TIME" \\
    --max-cycles $MAX_SENTINEL_CYCLES \\
    --manifest "$MANIFEST" \\
    --base-dir "$BASE_DIR"
PLANNER_EOF

cat > "$EXECUTOR_SCRIPT" << EXECUTOR_EOF
#!/bin/bash
#SBATCH --account=$CPU_ACCOUNT
#SBATCH --job-name=hp_executor
#SBATCH --output=$PROJECT_DIR/slurm_out/hp_executor_%j.out
#SBATCH --error=$PROJECT_DIR/slurm_err/hp_executor_%j.err
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00

module load $MODULES
source "$PROJECT_DIR/$VENV/bin/activate"
cd "$PROJECT_DIR"

python -m slurm.hp_executor \\
    --manifest "$MANIFEST" \\
    --registry "$REGISTRY" \\
    --gpu-account "$GPU_ACCOUNT" \\
    --cpu-account "$CPU_ACCOUNT" \\
    --modules "$MODULES" \\
    --venv "$VENV" \\
    --max-cycles $MAX_SENTINEL_CYCLES \\
    --max-jobs $MAX_JOBS \\
    --trials-per-config $TRIALS_PER_CONFIG \\
    --initial-mig "$INITIAL_MIG" \\
    --initial-mem "$INITIAL_MEM" \\
    --initial-time "$INITIAL_TIME" \\
    --base-dir "$BASE_DIR" \\
    $DRY_RUN
EXECUTOR_EOF

chmod +x "$PLANNER_SCRIPT"
chmod +x "$EXECUTOR_SCRIPT"

echo "  $PLANNER_SCRIPT"
echo "  $EXECUTOR_SCRIPT"
echo ""

# ==================== STEP 6: SUBMIT JOBS ====================
if [[ -n "$DRY_RUN" ]]; then
    echo "--- DRY RUN: would submit ---"
    echo "  sbatch $PLANNER_SCRIPT"
    echo "  sbatch --dependency=afterok:<PLANNER_JOB_ID> $EXECUTOR_SCRIPT"
    echo ""
else
    echo "--- Submitting jobs ---"
    PLANNER_JOB=$(sbatch --parsable "$PLANNER_SCRIPT")
    echo "  Planner submitted:  $PLANNER_JOB"

    EXECUTOR_JOB=$(sbatch --parsable --dependency=afterok:"$PLANNER_JOB" "$EXECUTOR_SCRIPT")
    echo "  Executor submitted: $EXECUTOR_JOB (depends on planner $PLANNER_JOB)"
    echo ""
fi

# ==================== STEP 7: MONITORING COMMANDS ====================
echo "=============================================================="
echo "  Monitoring Commands"
echo "=============================================================="
echo ""
echo "  # Watch job queue"
echo "  squeue -u \$USER"
echo ""
echo "  # Check planner output"
echo "  ls -t $PROJECT_DIR/slurm_out/hp_planner_*.out | head -1 | xargs cat"
echo ""
echo "  # Check executor output"
echo "  ls -t $PROJECT_DIR/slurm_out/hp_executor_*.out | head -1 | xargs cat"
echo ""
echo "  # View registry status"
echo "  python -c \"import json; r=json.load(open('$REGISTRY')); print(f'Total jobs: {len(r[\\\"jobs\\\"])}'); print(f'Statuses: {dict((s, sum(1 for j in r[\\\"jobs\\\"] if j[\\\"status\\\"]==s)) for s in set(j[\\\"status\\\"] for j in r[\\\"jobs\\\"]))}')\" 2>/dev/null || echo '  (registry not yet created)'"
echo ""
echo "  # View manifest"
echo "  cat $MANIFEST 2>/dev/null || echo '  (manifest not yet created)'"
echo ""
echo "=============================================================="
