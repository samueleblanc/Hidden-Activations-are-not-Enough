#!/bin/bash
#SBATCH --array=0-2
#SBATCH --time=02:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=slurm_out/C_thm45_%A_%a.out
#SBATCH --error=slurm_err/C_thm45_%A_%a.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

# Map array task ID to experiment
EXPERIMENTS=("alexnet_imagenet" "resnet_imagenet" "vgg_imagenet")
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "Step C: Theorem 4.5 validation for $EXPERIMENT (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

python validate_theorem45.py \
    --experiment $EXPERIMENT \
    --num_samples 200 \
    --matrix_batch_size 1800

echo "Task $SLURM_ARRAY_TASK_ID ($EXPERIMENT) completed"
