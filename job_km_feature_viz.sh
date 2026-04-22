#!/bin/bash
#SBATCH --account=def-jcbus
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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

# Step 0: regenerate manifest (cheap; idempotent)
python -m km_feature_viz.manifest_cli \
    --imagenet-val "${IMAGENET_VAL_DIR:-$SLURM_TMPDIR/imagenet/val}" \
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
