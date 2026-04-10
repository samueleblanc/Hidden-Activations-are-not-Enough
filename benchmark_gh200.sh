#!/bin/bash
# ==============================================================
# benchmark_gh200.sh — SLURM launcher for GH200 benchmark
#
# Runs benchmark_gh200.py on the gh-aria partition to determine
# optimal batch_size and memory budget for KM computation.
#
# Usage:
#   sbatch benchmark_gh200.sh [experiment_name]
#   sbatch benchmark_gh200.sh vgg_cifar10
# ==============================================================

#SBATCH --partition=gh-aria
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --mem=480G
#SBATCH --output=slurm_out/benchmark_gh200_%j.out
#SBATCH --error=slurm_err/benchmark_gh200_%j.err

set -euo pipefail

EXPERIMENT="${1:-vgg_cifar10}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source config for env setup
source "$SCRIPT_DIR/experiment_config.sh"

mkdir -p slurm_out slurm_err

# Activate environment
$ENV_SETUP

# SLURM_TMPDIR fallback
SLURM_TMPDIR="${SLURM_TMPDIR:-/tmp/slurm-$SLURM_JOB_ID}"
mkdir -p "$SLURM_TMPDIR"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Copy datasets to local SSD
eval "$(get_dataset_copy_commands 2>/dev/null || true)"

echo "=============================================================="
echo "  GH200 Benchmark: $EXPERIMENT"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $(hostname)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "=============================================================="

python benchmark_gh200.py \
    --experiment_name "$EXPERIMENT" \
    --temp_dir "$SLURM_TMPDIR" \
    --target_utilization 0.90 \
    --timing_samples 20 \
    --max_concurrent 5 \
    --force

echo "Benchmark complete."
