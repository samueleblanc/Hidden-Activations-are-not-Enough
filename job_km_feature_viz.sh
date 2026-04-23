#!/bin/bash
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm_out/km_feature_viz_%j.out
#SBATCH --error=slurm_err/km_feature_viz_%j.err

set -euo pipefail

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

cd "$SLURM_SUBMIT_DIR"

MANIFEST=results/km-feature-viz/manifest.json

mkdir -p results/km-feature-viz
mkdir -p slurm_out slurm_err

# Step 0: regenerate manifest (cheap; idempotent).
# IMAGENET_ROOT defaults to nibi's standard location (matches validate_theorem45.py
# and teleportation_experiment.py). Override via env var if needed.
python -m km_feature_viz.manifest_cli \
    --imagenet-root "${IMAGENET_ROOT:-/datashare/imagenet/ILSVRC2012}" \
    --output "$MANIFEST"

# Step 01: KMs (heaviest)
python -m km_feature_viz.compute_kms \
    --manifest "$MANIFEST" \
    --batch-size 512

# Step 02: cheap baselines (Grad-CAM, IG, SmoothGrad, feature_maps, PGD)
python -m km_feature_viz.compute_baselines \
    --manifest "$MANIFEST" \
    --methods gradcam ig smoothgrad feature_maps pgd

# Step 03: DeepDream
python -m km_feature_viz.compute_deepdream

# Steps 05/06 are patch-blocked. Skip if patch not available; the stub
# exits with code 2 — capture and continue.
python -m km_feature_viz.counterfactual_lp || \
    echo "counterfactual_lp skipped (patch not available)"
python -m km_feature_viz.jacobian_sensitivity || \
    echo "jacobian_sensitivity skipped (patch not available)"

echo "Done."
