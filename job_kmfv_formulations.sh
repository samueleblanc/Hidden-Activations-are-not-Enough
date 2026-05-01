#!/bin/bash
# Step D4: counterfactual_lp + jacobian_sensitivity formulations. Loops all
# Pillar 3 archs (resnet152, densenet121, googlenet) internally via the
# manifest. Per-arch W_eff is (1000, 150529) fp32 ≈ 600 MB; the LP and
# Jacobian-sensitivity steps hold W_eff in memory plus solver/gradient
# intermediates. Mem bumped 32G → 48G to absorb the 3-arch fp32 W_eff plus
# solver intermediates.
#
# Time bumped 1h → 4h after job 13073434 timed out at 1h with zero LPs
# completed. Closed-form LP (no scipy.linprog) makes the LP itself ~ms,
# but W_eff extraction on resnet152 was likely the bottleneck. 4h gives
# headroom for: 27 W_eff extractions × ~2-5min each + the LP/post work.
# If we again see a timeout with the per-stage timing prints in stdout,
# that pinpoints the slow path for the next fix.
#SBATCH --time=04:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=slurm_out/D4_formulations_%j.out
#SBATCH --error=slurm_err/D4_formulations_%j.err

set -euo pipefail

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
cd "$SLURM_SUBMIT_DIR"

echo "Step D4: counterfactual_lp + jacobian_sensitivity"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# Both formulations may be patch-blocked when the knowledgematrix extract_weff
# patch is unavailable, OR may legitimately fail per-sample. We track exit
# codes per script and only fail the SLURM job if BOTH return non-zero —
# this surfaces full-formulation failure in seff while tolerating one
# subscript failing on its own. Each script returns:
#   0  if at least one sample completed (or all were already done)
#   1  if zero samples completed this run (hard failure)
#   2  if the knowledgematrix extract_weff patch is missing
rc_cf=0
rc_jc=0
python -m km_feature_viz.counterfactual_lp \
    --manifest results/km-feature-viz/manifest.json \
    || rc_cf=$?

python -m km_feature_viz.jacobian_sensitivity \
    --manifest results/km-feature-viz/manifest.json \
    || rc_jc=$?

[ "$rc_cf" -ne 0 ] && echo "WARNING: counterfactual_lp exited $rc_cf"
[ "$rc_jc" -ne 0 ] && echo "WARNING: jacobian_sensitivity exited $rc_jc"

if [ "$rc_cf" -ne 0 ] && [ "$rc_jc" -ne 0 ]; then
    echo "ERROR: both formulations failed (rc_cf=$rc_cf rc_jc=$rc_jc)"
    exit 1
fi

echo "D4 completed (rc_cf=$rc_cf rc_jc=$rc_jc)"
