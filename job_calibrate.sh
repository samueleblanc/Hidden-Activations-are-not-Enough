#!/bin/bash
#SBATCH --account=def-jcbus
#SBATCH --time=00:20:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_out/A1_calibrate_%A_%a.out
#SBATCH --error=slurm_err/A1_calibrate_%A_%a.err

set -euo pipefail

# Step A1 — per-architecture KM batch-size calibration.
# 3 array tasks (one per arch). Dispatches bin/calibrate.py.
# All later Phase 1 workers (S1/S2/S3) read the resulting calibration.json.

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

ARCHS=("resnet152" "densenet121" "googlenet")
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}

OUT_PATH="experiments/calibration/${ARCH}_imagenet/calibration.json"
mkdir -p "$(dirname "$OUT_PATH")"

# Use `python -m` so the repo root is on sys.path; otherwise
# `from utils.utils import …` inside bin/calibrate.py fails with
# ModuleNotFoundError (sys.path[0] would be bin/, not the repo root).
python -m bin.calibrate --arch "$ARCH" --out_path "$OUT_PATH"
