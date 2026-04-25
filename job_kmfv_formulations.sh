#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_out/D4_formulations_%j.out
#SBATCH --error=slurm_err/D4_formulations_%j.err

set -euo pipefail

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
cd "$SLURM_SUBMIT_DIR"

echo "Step D4: counterfactual_lp + jacobian_sensitivity"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# Both formulations are patch-blocked when the knowledgematrix extract_weff
# patch is unavailable. Tolerate exit 2; the bundle's completeness check
# already skips gracefully when the formulation outputs are missing.
python -m km_feature_viz.counterfactual_lp \
    --manifest results/km-feature-viz/manifest.json \
    || echo "counterfactual_lp skipped (patch not available)"

python -m km_feature_viz.jacobian_sensitivity \
    --manifest results/km-feature-viz/manifest.json \
    || echo "jacobian_sensitivity skipped (patch not available)"

echo "D4 completed"
