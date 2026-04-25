#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_out/D3_deepdream_%A_%a.out
#SBATCH --error=slurm_err/D3_deepdream_%A_%a.err

set -euo pipefail

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
cd "$SLURM_SUBMIT_DIR"

MODELS=("alexnet" "resnet18" "vgg11")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Step D3: compute_deepdream for $MODEL (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

python -m km_feature_viz.compute_deepdream \
    --models "$MODEL" \
    --state-suffix "$MODEL"

echo "Task $SLURM_ARRAY_TASK_ID ($MODEL) completed"
