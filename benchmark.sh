#!/bin/bash
#SBATCH --partition=gh-aria
#SBATCH --chdir=/net/nfs-iq/home-gh/armenta/Hidden-Activations-are-not-Enough
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --time=16:00:00
#SBATCH --mem=480G
#SBATCH --output=slurm_out/BENCH_1k_%A.out
#SBATCH --error=slurm_err/BENCH_1k_%A.err
#SBATCH --job-name=bench_1k

# ==============================================================
# GH200 Matrix Benchmark — 1k matrices × 4 experiments
#
# Submits a single job that benchmarks knowledge matrix computation
# for all 4 main experiments, measuring time and memory.
#
# Usage:  sbatch benchmark.sh
# ==============================================================

set -euo pipefail

echo "======================================================"
echo "  GH200 Matrix Benchmark Job"
echo "  Job ID:    $SLURM_JOB_ID"
echo "  Node:      $(hostname)"
echo "  Date:      $(date -Iseconds)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "  TMPDIR:    $SLURM_TMPDIR"
echo "======================================================"

# --- Environment ---
source /net/nfs-iq/home-gh/armenta/Hidden-Activations-are-not-Enough/gh_env/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Create output dirs ---
mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err

# --- Copy datasets to fast local SSD ---
echo ""
echo "Copying datasets to SLURM_TMPDIR..."

mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/
cp -r $SLURM_SUBMIT_DIR/data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/ 2>/dev/null || true

mkdir -p $SLURM_TMPDIR/data/cifar-100-python/
cp -r $SLURM_SUBMIT_DIR/data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/ 2>/dev/null || true

echo "Dataset copy complete."

# --- GPU monitoring (background, every 30s) ---
mkdir -p $SLURM_SUBMIT_DIR/gpu-monitor/
GPU_LOGFILE="$SLURM_SUBMIT_DIR/gpu-monitor/benchmark_1k_${SLURM_JOB_ID}.log"
monitor_gpu() {
  echo "Timestamp, GPU_Util(%), Mem_Used(MiB), Mem_Total(MiB), Temp(C)" > "$GPU_LOGFILE"
  while true; do
    ts=$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
      --format=csv,noheader,nounits \
      | awk -v t="$ts" '{print t", "$1", "$2", "$3", "$4}' >> "$GPU_LOGFILE"
    sleep 30
  done
}
monitor_gpu &
MONITOR_PID=$!

# --- System memory baseline ---
echo ""
echo "System memory baseline:"
free -h
echo ""

# --- Run benchmark ---
echo "Starting benchmark..."
echo ""

python test_calibration.py \
    --experiments alexnet_cifar10 resnet_cifar10 resnet_cifar100 vgg_cifar100 \
    --num_matrices 1000 \
    --max_batch_size 8192 \
    --num_samples_per_class 500 \
    --samples_per_attack 500 \
    --total_chunks 1 \
    --temp_dir $SLURM_TMPDIR \
    --output experiments/benchmark_1k.json

PYTHON_EXIT=$?

# --- Cleanup ---
kill $MONITOR_PID 2>/dev/null || true

echo ""
echo "======================================================"
echo "  Benchmark finished with exit code: $PYTHON_EXIT"
echo "  Date: $(date -Iseconds)"
echo "======================================================"

# --- Final system state ---
echo ""
echo "Final system memory:"
free -h
echo ""
echo "Final GPU state:"
nvidia-smi
echo ""

exit $PYTHON_EXIT
