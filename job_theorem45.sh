#!/bin/bash
#SBATCH --array=0-17
#SBATCH --time=04:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=slurm_out/C_thm45_%A_%a.out
#SBATCH --error=slurm_err/C_thm45_%A_%a.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

# Map array task ID to (experiment, attack) pair
# 18 tasks = 3 experiments × 6 attacks
EXPERIMENTS=("alexnet_imagenet" "resnet_imagenet" "vgg_imagenet")
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
    --matrix_batch_size 1800

echo "Task $SLURM_ARRAY_TASK_ID ($EXPERIMENT / $ATTACK) completed"
