#!/bin/bash
#SBATCH --array=0-2
#SBATCH --time=04:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_out/debug_vgg_%A_%a.out
#SBATCH --error=slurm_err/debug_vgg_%A_%a.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

ATTACKS=("APGD" "DeepFool" "Square")
ATTACK=${ATTACKS[$SLURM_ARRAY_TASK_ID]}

echo "VGG gamma=0 diagnostic: $ATTACK (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python debug_vgg_gamma_zero.py --attack $ATTACK --num_samples 200

echo "Task $SLURM_ARRAY_TASK_ID ($ATTACK) completed"
