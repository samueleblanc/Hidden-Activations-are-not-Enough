#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=slurm_out/D1_kms_%A_%a.out
#SBATCH --error=slurm_err/D1_kms_%A_%a.err

set -euo pipefail

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
cd "$SLURM_SUBMIT_DIR"

MODELS=("alexnet" "resnet18" "vgg11")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Step D1: compute_kms for $MODEL (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m km_feature_viz.compute_kms \
    --manifest results/km-feature-viz/manifest.json \
    --batch-size 512 \
    --limit-models "$MODEL" \
    --state-suffix "$MODEL"

echo "Task $SLURM_ARRAY_TASK_ID ($MODEL) completed"
