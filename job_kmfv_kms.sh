#!/bin/bash
# Step D1: KM compute for the 3-arch Pillar 3 launch (resnet152, densenet121,
# googlenet — all 224×224, KM column dim 150 529). ResNet152 dominates the
# memory budget: per-layer Jacobian intermediates inside
# KnowledgeMatrixComputer.forward() scale with model depth, so 192G is sized
# for resnet152's chunked compute (densenet121 / googlenet are smaller and
# comfortably fit). The per-image full M is (1000, 150529) fp32 ≈ 600 MB;
# per-batch peak inside the chunked computation is multiple GB depending on
# --batch-size. Time held at 08:00:00 to cover the deepest arch.
#SBATCH --time=08:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --output=slurm_out/D1_kms_%A_%a.out
#SBATCH --error=slurm_err/D1_kms_%A_%a.err

set -euo pipefail

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
cd "$SLURM_SUBMIT_DIR"

# Pillar 3 launch: 3 archs in parallel via --array=0-2 (orchestrator handles
# the array spec). Keep this array in sync with D_MODELS in run_pipeline.sh.
MODELS=("resnet152" "densenet121" "googlenet")
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
