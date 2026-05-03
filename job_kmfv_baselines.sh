#!/bin/bash
# Step D2: baselines (gradcam, ig, smoothgrad, feature_maps, pgd) for all archs
# in one process via the manifest. Pillar 3 launch: 3 archs × 20 imgs × 5
# methods = 300 individual baseline computations on a single H100. Time bumped
# 01:00:00 → 02:00:00 to be safe across the 3-arch sweep. Mem bumped 32G → 48G
# to accommodate Captum IG + SmoothGrad intermediates on the deepest arch
# (resnet152) plus per-method state churn.
#SBATCH --time=02:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=slurm_out/D2_baselines_%j.out
#SBATCH --error=slurm_err/D2_baselines_%j.err

set -euo pipefail

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
cd "$SLURM_SUBMIT_DIR"

echo "Step D2: compute_baselines (gradcam, ig, smoothgrad, feature_maps, pgd)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m km_feature_viz.compute_baselines \
    --manifest results/km-feature-viz/manifest.json \
    --methods gradcam ig smoothgrad feature_maps pgd

echo "D2 completed"
