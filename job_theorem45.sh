#!/bin/bash
#SBATCH --array=0-17
#SBATCH --time=08:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=slurm_out/C_thm45_%A_%a.out
#SBATCH --error=slurm_err/C_thm45_%A_%a.err

# Pillar 3 alignment (TMLR resubmission): Step C now estimates γ for the same
# three architectures used by Pillar 3 KM-feature-viz (commit 9e5e9f4) and
# Steps A/B. ResNet152's KM is ~2× the size of ResNet18's; bumped to
# 8h/96G/h100:1 to match the larger memory profile of the new model set.

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

python validate_theorem45.py \
    --experiment $EXPERIMENT \
    --attacks $ATTACK \
    --num_samples 200 \
    --matrix_batch_size 256 \
    --no-aggregate
# matrix_batch_size lowered from 1800 → 256: resnet152's KM tensor is ~2×
# ResNet18's, and the Pillar 3 D1 jobs already run KMs at batch_size=512
# for a single model. Step C holds one model + KM forward; 256 is a safe
# margin under H100's 80 GB.

echo "Task $SLURM_ARRAY_TASK_ID ($EXPERIMENT / $ATTACK) completed"
