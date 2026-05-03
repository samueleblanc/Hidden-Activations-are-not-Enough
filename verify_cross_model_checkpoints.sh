#!/bin/bash
# Pre-flight: verify all 7 cross-model checkpoints load + KM-completeness pass.
# Run BEFORE submitting job_cross_model.sh to surface any state-dict porting
# issues without spending cluster GPU-hours on a doomed run.
#
# Usage:
#   bash verify_cross_model_checkpoints.sh
#
# Each check loads the KM wrapper, ports alternate weights via positional
# state-dict remap, and verifies M(x).sum(1) ≈ f(x) on one random sample.
# Exits 0 if all 7 pass; non-zero with a summary if any fail.

set -uo pipefail

PYTHON=${PYTHON:-env/bin/python}
PASSED=()
FAILED=()

declare -A CKPTS=(
    ["resnet152:tv_v1"]=1
    ["resnet152:tv_v2"]=1
    ["resnet152:timm_a1"]=1
    ["resnet152:timm_a2"]=1
    ["resnet152:timm_a3"]=1
    ["densenet121:tv_v1"]=1
    ["densenet121:timm_ra"]=1
)

for spec in "${!CKPTS[@]}"; do
    ARCH=${spec%:*}
    ALIAS=${spec#*:}
    echo ""
    echo "=== Verifying $ARCH / $ALIAS ==="
    if $PYTHON cross_model_experiment.py --arch "$ARCH" --verify "$ALIAS"; then
        PASSED+=("$spec")
    else
        FAILED+=("$spec")
    fi
done

echo ""
echo "========================================"
echo "  SUMMARY: ${#PASSED[@]} passed, ${#FAILED[@]} failed"
echo "========================================"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED checkpoints:"
    for spec in "${FAILED[@]}"; do echo "  $spec"; done
    echo ""
    echo "Investigate before submitting job_cross_model.sh — a failed checkpoint"
    echo "indicates the timm/torchvision state-dict layout differs from the"
    echo "knowledgematrix wrapper. Likely fix: extend the positional remap in"
    echo "build_km_model_with_alt_weights() or report the upstream wrapper bug."
    exit 1
fi
echo "All checkpoints pass — safe to sbatch job_cross_model.sh"
