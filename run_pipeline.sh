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

C_ATTACKS=("FGSM" "PGD" "CW" "DeepFool" "APGD" "Square")
C_ATTACK_TASKS=()  # per-attack SLURM array indices (0-17)
C_AGG_NEEDED=()    # experiments needing aggregation

echo "Step C: Theorem 4.5"
for i in 0 1 2; do
    if [ -f "${C_RESULTS[$i]}" ]; then
        C_STATUS[$i]="done"
        echo "  [$i] ${C_NAMES[$i]}: DONE"
    else
        # Check per-attack files
        PA_DIR="experiments/${C_NAMES[$i]}/theorem45/per_attack"
        CKPT_FILE="${C_CHECKPOINTS[$i]}"
        PA_COUNT=0
        [ -d "$PA_DIR" ] && PA_COUNT=$(ls "$PA_DIR"/*.json 2>/dev/null | wc -l)
        CKPT_COUNT=0
        if [ -f "$CKPT_FILE" ]; then
            CKPT_COUNT=$(python3 -c "
import json
with open('$CKPT_FILE') as f: d=json.load(f)
print(len(d.get('per_attack',{})))
" 2>/dev/null || echo 0)
        fi
        TOTAL_DONE_ATTACKS=$((PA_COUNT > CKPT_COUNT ? PA_COUNT : CKPT_COUNT))

        if [ "$TOTAL_DONE_ATTACKS" -ge 6 ]; then
            C_STATUS[$i]="needs_aggregation"
            C_AGG_NEEDED+=("$i")
            echo "  [$i] ${C_NAMES[$i]}: ALL ATTACKS DONE (${TOTAL_DONE_ATTACKS}/6), needs aggregation"
        else
            C_STATUS[$i]="in_progress"
            echo "  [$i] ${C_NAMES[$i]}: ${TOTAL_DONE_ATTACKS}/6 attacks done"
            # Add missing per-attack SLURM tasks
            for a in 0 1 2 3 4 5; do
                ATK=${C_ATTACKS[$a]}
                PA_FILE="$PA_DIR/${ATK}.json"
                TASK_ID=$(( i * 6 + a ))
                if [ ! -f "$PA_FILE" ]; then
                    # Also check checkpoint for this attack
                    IN_CKPT=false
                    if [ -f "$CKPT_FILE" ]; then
                        IN_CKPT=$(python3 -c "
import json
with open('$CKPT_FILE') as f: d=json.load(f)
print('true' if '$ATK' in d.get('per_attack',{}) else 'false')
" 2>/dev/null || echo false)
                    fi
                    if [ "$IN_CKPT" = "false" ]; then
                        C_ATTACK_TASKS+=("$TASK_ID")
                    fi
                fi
            done
        fi
    fi
done
echo ""

# ==============================================================
# Write pipeline_state.json
# ==============================================================
join_array() { local IFS=','; echo "$*"; }

A_ARRAY_STR=""
B_ARRAY_STR=""
C_ATTACK_ARRAY_STR=""
C_AGG_ARRAY_STR=""
[ ${#A_NEEDED[@]} -gt 0 ] && A_ARRAY_STR=$(join_array "${A_NEEDED[@]}")
[ ${#B_NEEDED[@]} -gt 0 ] && B_ARRAY_STR=$(join_array "${B_NEEDED[@]}")
[ ${#C_ATTACK_TASKS[@]} -gt 0 ] && C_ATTACK_ARRAY_STR=$(join_array "${C_ATTACK_TASKS[@]}")
[ ${#C_AGG_NEEDED[@]} -gt 0 ] && C_AGG_ARRAY_STR=$(join_array "${C_AGG_NEEDED[@]}")

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
    "step_C_attacks": "'"${C_ATTACK_ARRAY_STR:-none}"'",
    "step_C_aggregate": "'"${C_AGG_ARRAY_STR:-none}"'"
  }
}' > "$TMP_FILE"
mv "$TMP_FILE" "$STATE_FILE"

echo "Pipeline state written to $STATE_FILE"
echo ""

# ==============================================================
# Phase 2: Submit
# ==============================================================
C_TOTAL_JOBS=$(( ${#C_ATTACK_TASKS[@]} + ${#C_AGG_NEEDED[@]} ))
TOTAL_NEEDED=$(( ${#A_NEEDED[@]} + ${#B_NEEDED[@]} + C_TOTAL_JOBS ))
# Count fully done: A(3) + B(3) + C(3 experiments)
C_DONE=0
for i in 0 1 2; do [ "${C_STATUS[$i]}" = "done" ] && C_DONE=$((C_DONE + 1)); done
TOTAL_DONE=$(( 3 - ${#A_NEEDED[@]} + 3 - ${#B_NEEDED[@]} + C_DONE ))

echo "========================================"
echo "  SUMMARY: $TOTAL_DONE/9 done, $TOTAL_NEEDED jobs to submit"
if [ ${#C_ATTACK_TASKS[@]} -gt 0 ]; then
    echo "  Step C: ${#C_ATTACK_TASKS[@]} per-attack jobs"
fi
if [ ${#C_AGG_NEEDED[@]} -gt 0 ]; then
    echo "  Step C: ${#C_AGG_NEEDED[@]} aggregation jobs"
fi
echo "========================================"
echo ""

if [ "$TOTAL_NEEDED" -eq 0 ]; then
    echo "All 9 tasks are complete. Nothing to submit."
    exit 0
fi

ACCOUNT="def-bruestle_gpu"

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would submit (account=$ACCOUNT):"
    [ ${#A_NEEDED[@]} -gt 0 ] && echo "  sbatch --account=$ACCOUNT --array=$A_ARRAY_STR job_isomorphism.sh"
    [ ${#B_NEEDED[@]} -gt 0 ] && echo "  sbatch --account=$ACCOUNT --array=$B_ARRAY_STR job_teleportation.sh"
    [ ${#C_ATTACK_TASKS[@]} -gt 0 ] && echo "  sbatch --account=$ACCOUNT --array=$C_ATTACK_ARRAY_STR job_theorem45.sh"
    [ ${#C_AGG_NEEDED[@]} -gt 0 ] && echo "  sbatch --account=$ACCOUNT --array=$C_AGG_ARRAY_STR --dependency=afterany:\$C_JOB_ID job_theorem45_agg.sh"
    exit 0
fi

echo "Submitting SLURM jobs (account=$ACCOUNT)..."
echo ""

if [ ${#A_NEEDED[@]} -gt 0 ]; then
    echo "  Step A (Isomorphism):  --array=$A_ARRAY_STR"
    sbatch --account="$ACCOUNT" --array="$A_ARRAY_STR" job_isomorphism.sh
fi

if [ ${#B_NEEDED[@]} -gt 0 ]; then
    echo "  Step B (Teleportation): --array=$B_ARRAY_STR"
    sbatch --account="$ACCOUNT" --array="$B_ARRAY_STR" job_teleportation.sh
fi

C_JOB_ID=""
if [ ${#C_ATTACK_TASKS[@]} -gt 0 ]; then
    echo "  Step C (Theorem 4.5 per-attack):  --array=$C_ATTACK_ARRAY_STR"
    C_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" --array="$C_ATTACK_ARRAY_STR" job_theorem45.sh)
    echo "    Job ID: $C_JOB_ID"
fi

if [ ${#C_AGG_NEEDED[@]} -gt 0 ]; then
    if [ -n "$C_JOB_ID" ]; then
        echo "  Step C (Theorem 4.5 aggregation):  --array=$C_AGG_ARRAY_STR (after $C_JOB_ID)"
        sbatch --account="$ACCOUNT" --array="$C_AGG_ARRAY_STR" --dependency=afterany:"$C_JOB_ID" job_theorem45_agg.sh
    else
        echo "  Step C (Theorem 4.5 aggregation):  --array=$C_AGG_ARRAY_STR"
        sbatch --account="$ACCOUNT" --array="$C_AGG_ARRAY_STR" job_theorem45_agg.sh
    fi
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Re-run this script after jobs complete to check remaining work."
