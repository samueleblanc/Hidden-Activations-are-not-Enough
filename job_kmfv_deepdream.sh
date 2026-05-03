#!/bin/bash
# Step D3: DeepDream activation-maximization for 5 neurons × 3 layers per arch.
# Per arch: an inline 03a neuron-selection pass (class-conditional Grad-CAM for
# resnet152/densenet121, catalogued from Distill Circuits for googlenet) writes
# state/03a_neuron_selection_<arch>.json, then 200 optimization steps × 15
# (layer, neuron) pairs = 3000 forward+backward passes. Mem 32G covers the
# deepest arch (resnet152) plus the 60-image+grad neuron-selection pass.
# Time 01:00:00 fine on H100 for selection + 15 deepdream optimizations.
#SBATCH --time=01:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=slurm_out/D3_deepdream_%A_%a.out
#SBATCH --error=slurm_err/D3_deepdream_%A_%a.err

set -euo pipefail

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
cd "$SLURM_SUBMIT_DIR"

# Pillar 3 launch: 3 archs in parallel via --array=0-2 (orchestrator handles
# the array spec). Keep this array in sync with D_MODELS in run_pipeline.sh.
MODELS=("resnet152" "densenet121" "googlenet")
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

echo "Step D3: compute_deepdream for $MODEL (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

python -m km_feature_viz.compute_deepdream \
    --manifest results/km-feature-viz/manifest.json \
    --models "$MODEL" \
    --state-suffix "$MODEL"

echo "Task $SLURM_ARRAY_TASK_ID ($MODEL) completed"
