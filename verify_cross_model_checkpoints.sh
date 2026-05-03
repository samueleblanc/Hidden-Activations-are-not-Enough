#!/bin/bash
# Pre-flight: verify all 7 cross-model checkpoints load + KM-completeness pass.
# Used by run_pipeline.sh on the login node (where there's internet access
# for timm/torchvision weight downloads) before dispatching Step E pairs.
# Also runnable standalone for debugging.
#
# Usage:
#   bash verify_cross_model_checkpoints.sh
#
# Each check loads the KM wrapper, ports alternate weights via positional
# state-dict remap, and verifies M(x).sum(1) ≈ f(x) on one random sample.
# Writes results/cross_model/verify_results.json so the orchestrator can
# filter the Step E array down to runnable pairs (pair runnable iff both
# checkpoints in pair pass).
#
# Exits 0 always — partial failures are recorded in the JSON for the
# orchestrator to act on. Stdout still summarizes for the user.

set -uo pipefail

PYTHON=${PYTHON:-env/bin/python}
RESULTS_DIR=results/cross_model
mkdir -p "$RESULTS_DIR"
RESULTS_TMP="$RESULTS_DIR/verify_results.json.tmp"
RESULTS_FILE="$RESULTS_DIR/verify_results.json"

PASSED=()
FAILED=()

# Order matters for human-readable output (resnet152 first, densenet121 second)
SPECS=(
    "resnet152:tv_v1"
    "resnet152:tv_v2"
    "resnet152:timm_a1"
    "resnet152:timm_a2"
    "resnet152:timm_a3"
    "densenet121:tv_v1"
    "densenet121:timm_ra"
)

# Build the JSON incrementally
echo "{" > "$RESULTS_TMP"
echo "  \"timestamp\": \"$(date -Iseconds)\"," >> "$RESULTS_TMP"
echo "  \"checkpoints\": {" >> "$RESULTS_TMP"

NUM=${#SPECS[@]}
for ((idx=0; idx<NUM; idx++)); do
    spec=${SPECS[$idx]}
    ARCH=${spec%:*}
    ALIAS=${spec#*:}
    echo ""
    echo "=== Verifying $ARCH / $ALIAS ==="
    if $PYTHON cross_model_experiment.py --arch "$ARCH" --verify "$ALIAS"; then
        PASSED+=("$spec")
        STATUS="passed"
    else
        FAILED+=("$spec")
        STATUS="failed"
    fi
    SEP=$([ $idx -lt $((NUM-1)) ] && echo "," || echo "")
    echo "    \"$spec\": \"$STATUS\"$SEP" >> "$RESULTS_TMP"
done

echo "  }," >> "$RESULTS_TMP"
echo "  \"passed_count\": ${#PASSED[@]}," >> "$RESULTS_TMP"
echo "  \"failed_count\": ${#FAILED[@]}" >> "$RESULTS_TMP"
echo "}" >> "$RESULTS_TMP"
mv "$RESULTS_TMP" "$RESULTS_FILE"

echo ""
echo "========================================"
echo "  SUMMARY: ${#PASSED[@]} passed, ${#FAILED[@]} failed"
echo "  Results: $RESULTS_FILE"
echo "========================================"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "FAILED checkpoints:"
    for spec in "${FAILED[@]}"; do echo "  $spec"; done
    echo ""
    echo "Investigate the timm/torchvision state-dict layout vs knowledgematrix"
    echo "wrapper. Likely fix: extend the positional remap in"
    echo "cross_model_experiment.py:build_km_model_with_alt_weights()."
    echo "Pairs containing a failed checkpoint will be SKIPPED by run_pipeline.sh;"
    echo "passing pairs will still run."
fi
exit 0  # never block the orchestrator on partial failure
