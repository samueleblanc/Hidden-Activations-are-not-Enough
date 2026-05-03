#!/bin/bash
#SBATCH --array=0-1
#SBATCH --time=00:45:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_out/iso_debug_alex_vgg_%A_%a.out
#SBATCH --error=slurm_err/iso_debug_alex_vgg_%A_%a.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

# Task 0: alexnet_imagenet
# Task 1: vgg_imagenet
# (resnet_imagenet was already --debug-measured in job 12456822.)
EXPERIMENTS=("alexnet_imagenet" "vgg_imagenet")
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "Isomorphism --debug for $EXPERIMENT (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# Reduced permutations + samples — only need relative-error measurements,
# not full statistics. The full 5 perms x 50 samples JSONs are preserved
# in git history on the `refactor` branch (commit 9e7d47d).
python isomorphism_experiment.py \
    --experiment $EXPERIMENT \
    --num_permutations 2 \
    --num_matrix_samples 20 \
    --debug
