#!/bin/bash
# ==============================================================
# run_pipeline.sh -- Scan & Submit for pretrained experiment pipeline
#
# Phase 1: Scan for existing results, write pipeline_state.json
# Phase 2: Submit only the SLURM jobs that are still needed
#
# Steps:
#   A: Isomorphism    (job_isomorphism.sh,  array 0-2)
#   B: Teleportation  (job_teleportation.sh, array 0-2)
#   C: Theorem 4.5    (job_theorem45.sh,    array 0-2)
#
# Usage:
#   bash run_pipeline.sh              # Scan and submit
#   bash run_pipeline.sh --dry-run    # Scan only, do not submit
# ==============================================================
set -euo pipefail

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

TIMESTAMP=$(date -Iseconds)

# ---- Task definitions ----

# Step A: Isomorphism
A_NAMES=("alexnet_imagenet" "resnet_imagenet" "vgg_imagenet")
A_FILES=(
    "experiments/alexnet_imagenet/isomorphism/isomorphism_results.json"
    "experiments/resnet_imagenet/isomorphism/isomorphism_results.json"
    "experiments/vgg_imagenet/isomorphism/isomorphism_results.json"
)

# Step B: Teleportation
B_NAMES=("resnet18" "vgg11_bn" "resnet50")
B_FILES=(
    "results/teleportation/resnet18_imagenet_teleportation.json"
    "results/teleportation/vgg11_bn_imagenet_teleportation.json"
    "results/teleportation/resnet50_imagenet_teleportation.json"
)

# Step C: Theorem 4.5
C_NAMES=("alexnet_imagenet" "resnet_imagenet" "vgg_imagenet")
C_RESULTS=(
    "experiments/alexnet_imagenet/theorem45/theorem45_results.json"
    "experiments/resnet_imagenet/theorem45/theorem45_results.json"
    "experiments/vgg_imagenet/theorem45/theorem45_results.json"
)
C_CHECKPOINTS=(
    "experiments/alexnet_imagenet/theorem45/theorem45_checkpoint.json"
    "experiments/resnet_imagenet/theorem45/theorem45_checkpoint.json"
    "experiments/vgg_imagenet/theorem45/theorem45_checkpoint.json"
)

# ==============================================================
# Phase 1: Scan
# ==============================================================
echo "========================================"
echo "  PIPELINE SCAN"
echo "  $(date)"
echo "========================================"
echo ""

A_NEEDED=()
B_NEEDED=()
C_NEEDED=()

declare -a A_STATUS B_STATUS C_STATUS

echo "Step A: Isomorphism"
for i in 0 1 2; do
    if [ -f "${A_FILES[$i]}" ]; then
        A_STATUS[$i]="done"
        echo "  [$i] ${A_NAMES[$i]}: DONE"
    else
        A_STATUS[$i]="pending"
        A_NEEDED+=("$i")
        echo "  [$i] ${A_NAMES[$i]}: PENDING"
    fi
done
echo ""

echo "Step B: Teleportation"
for i in 0 1 2; do
    if [ -f "${B_FILES[$i]}" ]; then
        B_STATUS[$i]="done"
        echo "  [$i] ${B_NAMES[$i]}: DONE"
    else
        B_STATUS[$i]="pending"
        B_NEEDED+=("$i")
        echo "  [$i] ${B_NAMES[$i]}: PENDING"
    fi
done
echo ""

echo "Step C: Theorem 4.5"
for i in 0 1 2; do
    if [ -f "${C_RESULTS[$i]}" ]; then
        C_STATUS[$i]="done"
        echo "  [$i] ${C_NAMES[$i]}: DONE"
    elif [ -f "${C_CHECKPOINTS[$i]}" ]; then
        C_STATUS[$i]="in_progress"
        C_NEEDED+=("$i")
        echo "  [$i] ${C_NAMES[$i]}: IN PROGRESS (checkpoint found, will resume)"
    else
        C_STATUS[$i]="pending"
        C_NEEDED+=("$i")
        echo "  [$i] ${C_NAMES[$i]}: PENDING"
    fi
done
echo ""

# ==============================================================
# Write pipeline_state.json
# ==============================================================
join_array() { local IFS=','; echo "$*"; }

A_ARRAY_STR=""
B_ARRAY_STR=""
C_ARRAY_STR=""
[ ${#A_NEEDED[@]} -gt 0 ] && A_ARRAY_STR=$(join_array "${A_NEEDED[@]}")
[ ${#B_NEEDED[@]} -gt 0 ] && B_ARRAY_STR=$(join_array "${B_NEEDED[@]}")
[ ${#C_NEEDED[@]} -gt 0 ] && C_ARRAY_STR=$(join_array "${C_NEEDED[@]}")

STATE_FILE="pipeline_state.json"
TMP_FILE="${STATE_FILE}.tmp"

printf '%s\n' '{
  "timestamp": "'"$TIMESTAMP"'",
  "step_A_isomorphism": {
    "0_'"${A_NAMES[0]}"'": "'"${A_STATUS[0]}"'",
    "1_'"${A_NAMES[1]}"'": "'"${A_STATUS[1]}"'",
    "2_'"${A_NAMES[2]}"'": "'"${A_STATUS[2]}"'"
  },
  "step_B_teleportation": {
    "0_'"${B_NAMES[0]}"'": "'"${B_STATUS[0]}"'",
    "1_'"${B_NAMES[1]}"'": "'"${B_STATUS[1]}"'",
    "2_'"${B_NAMES[2]}"'": "'"${B_STATUS[2]}"'"
  },
  "step_C_theorem45": {
    "0_'"${C_NAMES[0]}"'": "'"${C_STATUS[0]}"'",
    "1_'"${C_NAMES[1]}"'": "'"${C_STATUS[1]}"'",
    "2_'"${C_NAMES[2]}"'": "'"${C_STATUS[2]}"'"
  },
  "submitted": {
    "step_A": "'"${A_ARRAY_STR:-none}"'",
    "step_B": "'"${B_ARRAY_STR:-none}"'",
    "step_C": "'"${C_ARRAY_STR:-none}"'"
  }
}' > "$TMP_FILE"
mv "$TMP_FILE" "$STATE_FILE"

echo "Pipeline state written to $STATE_FILE"
echo ""

# ==============================================================
# Phase 2: Submit
# ==============================================================
TOTAL_NEEDED=$(( ${#A_NEEDED[@]} + ${#B_NEEDED[@]} + ${#C_NEEDED[@]} ))
TOTAL_DONE=$(( 9 - TOTAL_NEEDED ))

echo "========================================"
echo "  SUMMARY: $TOTAL_DONE/9 done, $TOTAL_NEEDED to submit"
echo "========================================"
echo ""

if [ "$TOTAL_NEEDED" -eq 0 ]; then
    echo "All 9 tasks are complete. Nothing to submit."
    exit 0
fi

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would submit:"
    [ ${#A_NEEDED[@]} -gt 0 ] && echo "  sbatch --array=$A_ARRAY_STR job_isomorphism.sh"
    [ ${#B_NEEDED[@]} -gt 0 ] && echo "  sbatch --array=$B_ARRAY_STR job_teleportation.sh"
    [ ${#C_NEEDED[@]} -gt 0 ] && echo "  sbatch --array=$C_ARRAY_STR job_theorem45.sh"
    exit 0
fi

echo "Submitting SLURM jobs..."
echo ""

if [ ${#A_NEEDED[@]} -gt 0 ]; then
    echo "  Step A (Isomorphism):  --array=$A_ARRAY_STR"
    sbatch --array="$A_ARRAY_STR" job_isomorphism.sh
fi

if [ ${#B_NEEDED[@]} -gt 0 ]; then
    echo "  Step B (Teleportation): --array=$B_ARRAY_STR"
    sbatch --array="$B_ARRAY_STR" job_teleportation.sh
fi

if [ ${#C_NEEDED[@]} -gt 0 ]; then
    echo "  Step C (Theorem 4.5):  --array=$C_ARRAY_STR"
    sbatch --array="$C_ARRAY_STR" job_theorem45.sh
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Re-run this script after jobs complete to check remaining work."
