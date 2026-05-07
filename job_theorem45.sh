#!/bin/bash
#SBATCH --array=0-17
#SBATCH --time=16:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=slurm_out/C_thm45_%A_%a.out
#SBATCH --error=slurm_err/C_thm45_%A_%a.err

set -euo pipefail
# Without `set -e`, a non-zero exit from python is masked by the trailing
# `echo "Task X completed"` below — SLURM then mistakenly marks the array
# task COMPLETED and the per-attack file silently goes missing (observed
# 2026-05-06 when 12 tasks landed on a broken-GPU node and silently failed).

# Pillar 3 alignment (TMLR resubmission): Step C now estimates γ for the same
# three architectures used by Pillar 3 KM-feature-viz (commit 9e5e9f4) and
# Steps A/B. ResNet152's KM is ~2× the size of ResNet18's; bumped to
# 16h/96G/h100:1 to match the larger memory profile of the new model set.

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

# Map array task ID to (experiment, attack) pair
# 18 tasks = 3 experiments × 6 attacks
EXPERIMENTS=("resnet152_imagenet" "densenet121_imagenet" "googlenet_imagenet")
ATTACKS=("FGSM" "PGD" "CW" "DeepFool" "APGD" "Square")

EXP_IDX=$(( SLURM_ARRAY_TASK_ID / 6 ))
ATK_IDX=$(( SLURM_ARRAY_TASK_ID % 6 ))
EXPERIMENT=${EXPERIMENTS[$EXP_IDX]}
ATTACK=${ATTACKS[$ATK_IDX]}

echo "Step C: Theorem 4.5 — $EXPERIMENT / $ATTACK (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# May 2026 scale-up: N bumped 200 -> 500 to tighten gamma CI on the
# headline cell. DeepFool/resnet152 stays at N=200 (already capped at
# 100 steps via _RESNET152_OVERRIDES; N=500 would exceed Nibi's 24h
# gpubackfill limit). 17 of 18 (exp, attack) cells run N=500.
NUM_SAMPLES=500
if [ "$EXPERIMENT" = "resnet152_imagenet" ] && [ "$ATTACK" = "DeepFool" ]; then
    NUM_SAMPLES=200
    echo "  Using N=$NUM_SAMPLES (resnet152/DeepFool capped per time budget)"
fi

python validate_theorem45.py \
    --experiment $EXPERIMENT \
    --attacks $ATTACK \
    --num_samples $NUM_SAMPLES \
    --matrix_batch_size 1024 \
    --no-aggregate
# matrix_batch_size: 1800 caused Step A OOM with TWO models, but Step C
# only holds one model — 1024 fits comfortably under 80 GB on H100 and is
# ~4× faster than 256. The previous 256 ran resnet152 in 6h per attack;
# DeepFool (200 steps × 200 samples) timed out at 8h. 1024 brings DeepFool
# under 4h on resnet152. May 2026 scale-up uses 16h budget for N=500
# across the 17 non-capped cells.

echo "Task $SLURM_ARRAY_TASK_ID ($EXPERIMENT / $ATTACK) completed"
