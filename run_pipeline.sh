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

# ---- Cluster account ----
# Edit this on the cluster to the real account (e.g., "def-bruestle_gpu").
# Locally it stays as the placeholder so the value isn't checked in.
ACCOUNT="def-xxxx"

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

TIMESTAMP=$(date -Iseconds)

# ==============================================================
# Phase 0: Inline manifest regen for km-feature-viz (D-line)
# ==============================================================
# The manifest is a deterministic listing of (model, class, image) triples
# from the ImageNet val set; cheap (milliseconds) and CPU-only. Running it
# inline lets Phase 1 use the manifest to count expected work counts.
mkdir -p results/km-feature-viz
echo "Phase 0: regenerating km-feature-viz manifest..."
python -m km_feature_viz.manifest_cli \
    --imagenet-root "${IMAGENET_ROOT:-/datashare/imagenet/ILSVRC2012}" \
    --output results/km-feature-viz/manifest.json
echo ""

# ---- Task definitions ----

# Step A: Isomorphism
A_NAMES=("alexnet_imagenet" "resnet_imagenet" "vgg_imagenet")
A_FILES=(
    "experiments/alexnet_imagenet/isomorphism/isomorphism_results.json"
    "experiments/resnet_imagenet/isomorphism/isomorphism_results.json"
    "experiments/vgg_imagenet/isomorphism/isomorphism_results.json"
)

# Step B: Teleportation (non-BN VGG — vgg11_bn COB drifts, see teleportation_experiment.py)
B_NAMES=("resnet18" "vgg11" "resnet50")
B_FILES=(
    "results/teleportation/resnet18_imagenet_teleportation.json"
    "results/teleportation/vgg11_imagenet_teleportation.json"
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

# Step D: km-feature-viz (5 sub-steps)
D_MODELS=("alexnet" "resnet18" "vgg11")

# D1: compute_kms — per-model state files, 500 entries each
D1_STATE_FILES=(
    "results/km-feature-viz/state/01_compute_kms_alexnet.json"
    "results/km-feature-viz/state/01_compute_kms_resnet18.json"
    "results/km-feature-viz/state/01_compute_kms_vgg11.json"
)
D1_EXPECTED=500

# D2: compute_baselines — 5 per-method state files, 1500 entries each
D2_METHODS=("gradcam" "ig" "smoothgrad" "feature_maps" "pgd")
D2_EXPECTED=1500

# D3: compute_deepdream — per-model state files, 15 entries each
D3_STATE_FILES=(
    "results/km-feature-viz/state/03_deepdream_alexnet.json"
    "results/km-feature-viz/state/03_deepdream_resnet18.json"
    "results/km-feature-viz/state/03_deepdream_vgg11.json"
)
D3_EXPECTED=15

# D4: formulations — existence-only (patch-blocked may produce empty file)
D4_STATE_FILES=(
    "results/km-feature-viz/state/05_counterfactual_lp.json"
    "results/km-feature-viz/state/06_jacobian_sensitivity.json"
)

# D5: bundle — single tarball at repo root
D5_BUNDLE_PATH="km-feature-viz.tar"

# ==============================================================
# Phase 1: Scan
# ==============================================================
echo "========================================"
echo "  PIPELINE SCAN"
echo "  $(date)"
echo "========================================"
echo ""

count_state_entries() {
    # $1 = path to state json file. Echoes the entry count, or 0 if missing.
    local path="$1"
    if [ ! -f "$path" ]; then
        echo 0
        return
    fi
    python3 -c "
import json
with open('$path') as f: d = json.load(f)
print(len(d))
" 2>/dev/null || echo 0
}

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

# ---- D1 scan: compute_kms (per-model) ----
declare -a D1_STATUS
D1_NEEDED=()  # array task IDs (model indices) needing submission

echo "Step D1: km-feature-viz compute_kms"
for i in 0 1 2; do
    COUNT=$(count_state_entries "${D1_STATE_FILES[$i]}")
    if [ "$COUNT" -ge "$D1_EXPECTED" ]; then
        D1_STATUS[$i]="done"
        echo "  [$i] ${D_MODELS[$i]}: DONE (${COUNT}/${D1_EXPECTED})"
    else
        D1_STATUS[$i]="pending"
        D1_NEEDED+=("$i")
        echo "  [$i] ${D_MODELS[$i]}: ${COUNT}/${D1_EXPECTED}"
    fi
done
echo ""

# ---- D2 scan: compute_baselines ----
D2_STATUS="done"
D2_NEEDED=false
echo "Step D2: km-feature-viz compute_baselines"
for method in "${D2_METHODS[@]}"; do
    F="results/km-feature-viz/state/02_${method}.json"
    COUNT=$(count_state_entries "$F")
    if [ "$COUNT" -lt "$D2_EXPECTED" ]; then
        D2_STATUS="pending"
        D2_NEEDED=true
        echo "  ${method}: ${COUNT}/${D2_EXPECTED}"
    else
        echo "  ${method}: DONE (${COUNT}/${D2_EXPECTED})"
    fi
done
echo ""

# ---- D3 scan: compute_deepdream (per-model) ----
declare -a D3_STATUS
D3_NEEDED=()
echo "Step D3: km-feature-viz compute_deepdream"
for i in 0 1 2; do
    COUNT=$(count_state_entries "${D3_STATE_FILES[$i]}")
    if [ "$COUNT" -ge "$D3_EXPECTED" ]; then
        D3_STATUS[$i]="done"
        echo "  [$i] ${D_MODELS[$i]}: DONE (${COUNT}/${D3_EXPECTED})"
    else
        D3_STATUS[$i]="pending"
        D3_NEEDED+=("$i")
        echo "  [$i] ${D_MODELS[$i]}: ${COUNT}/${D3_EXPECTED}"
    fi
done
echo ""

# ---- D4 scan: formulations (existence only — patch-blocked tolerant) ----
D4_STATUS="done"
D4_NEEDED=false
echo "Step D4: km-feature-viz formulations"
for state_file in "${D4_STATE_FILES[@]}"; do
    if [ -f "$state_file" ]; then
        echo "  $(basename "$state_file"): EXISTS"
    else
        D4_STATUS="pending"
        D4_NEEDED=true
        echo "  $(basename "$state_file"): MISSING"
    fi
done
echo ""

# ---- D5 scan: bundle ----
echo "Step D5: km-feature-viz bundle"
if [ -f "$D5_BUNDLE_PATH" ]; then
    D5_STATUS="done"
    D5_NEEDED=false
    echo "  $D5_BUNDLE_PATH: EXISTS"
else
    D5_STATUS="pending"
    D5_NEEDED=true
    echo "  $D5_BUNDLE_PATH: MISSING"
fi
echo ""

# ==============================================================
# Write pipeline_state.json
# ==============================================================
join_array() { local IFS=','; echo "$*"; }

# Union of experiments needing final aggregation:
#   - those with pre-existing per_attack files but no final results
#     (C_AGG_NEEDED from scan above), AND
#   - those for which per-attack tasks are being queued in this run
#     (derived from C_ATTACK_TASKS via task_id / 6, deduplicated).
# Per-attack jobs use --no-aggregate, so the aggregation job is the ONLY
# writer of theorem45_results.json; always submit it when any per-attack
# job is queued to avoid the race condition where each concurrent task
# would otherwise write its own 1-attack results file.
# Kept portable for bash 3.x (no associative arrays) — flags[0..2] act as a set.
C_AGG_FLAGS=(0 0 0)
if [ ${#C_AGG_NEEDED[@]} -gt 0 ]; then
    for i in "${C_AGG_NEEDED[@]}"; do C_AGG_FLAGS[$i]=1; done
fi
if [ ${#C_ATTACK_TASKS[@]} -gt 0 ]; then
    for tid in "${C_ATTACK_TASKS[@]}"; do
        exp_idx=$(( tid / 6 ))
        C_AGG_FLAGS[$exp_idx]=1
    done
fi
C_EXPS_NEEDING_FINAL_AGG=()
for i in 0 1 2; do
    if [ "${C_AGG_FLAGS[$i]}" = "1" ]; then
        C_EXPS_NEEDING_FINAL_AGG+=("$i")
    fi
done

# Build D-line array strings + boolean-as-string substitutions for state JSON
D1_ARRAY_STR=""
D3_ARRAY_STR=""
[ ${#D1_NEEDED[@]} -gt 0 ] && D1_ARRAY_STR=$(join_array "${D1_NEEDED[@]}")
[ ${#D3_NEEDED[@]} -gt 0 ] && D3_ARRAY_STR=$(join_array "${D3_NEEDED[@]}")
D2_SUB=$([ "$D2_NEEDED" = true ] && echo "yes" || echo "none")
D4_SUB=$([ "$D4_NEEDED" = true ] && echo "yes" || echo "none")
D5_SUB=$([ "$D5_NEEDED" = true ] && echo "yes" || echo "none")

A_ARRAY_STR=""
B_ARRAY_STR=""
C_ATTACK_ARRAY_STR=""
C_AGG_ARRAY_STR=""
[ ${#A_NEEDED[@]} -gt 0 ] && A_ARRAY_STR=$(join_array "${A_NEEDED[@]}")
[ ${#B_NEEDED[@]} -gt 0 ] && B_ARRAY_STR=$(join_array "${B_NEEDED[@]}")
[ ${#C_ATTACK_TASKS[@]} -gt 0 ] && C_ATTACK_ARRAY_STR=$(join_array "${C_ATTACK_TASKS[@]}")
[ ${#C_EXPS_NEEDING_FINAL_AGG[@]} -gt 0 ] && C_AGG_ARRAY_STR=$(join_array "${C_EXPS_NEEDING_FINAL_AGG[@]}")

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
  "step_D1_kms": {
    "0_'"${D_MODELS[0]}"'": "'"${D1_STATUS[0]}"'",
    "1_'"${D_MODELS[1]}"'": "'"${D1_STATUS[1]}"'",
    "2_'"${D_MODELS[2]}"'": "'"${D1_STATUS[2]}"'"
  },
  "step_D2_baselines": "'"$D2_STATUS"'",
  "step_D3_deepdream": {
    "0_'"${D_MODELS[0]}"'": "'"${D3_STATUS[0]}"'",
    "1_'"${D_MODELS[1]}"'": "'"${D3_STATUS[1]}"'",
    "2_'"${D_MODELS[2]}"'": "'"${D3_STATUS[2]}"'"
  },
  "step_D4_formulations": "'"$D4_STATUS"'",
  "step_D5_bundle": "'"$D5_STATUS"'",
  "submitted": {
    "step_A": "'"${A_ARRAY_STR:-none}"'",
    "step_B": "'"${B_ARRAY_STR:-none}"'",
    "step_C_attacks": "'"${C_ATTACK_ARRAY_STR:-none}"'",
    "step_C_aggregate": "'"${C_AGG_ARRAY_STR:-none}"'",
    "step_D1": "'"${D1_ARRAY_STR:-none}"'",
    "step_D2": "'"$D2_SUB"'",
    "step_D3": "'"${D3_ARRAY_STR:-none}"'",
    "step_D4": "'"$D4_SUB"'",
    "step_D5": "'"$D5_SUB"'"
  }
}' > "$TMP_FILE"
mv "$TMP_FILE" "$STATE_FILE"

echo "Pipeline state written to $STATE_FILE"
echo ""

# ==============================================================
# Phase 2: Submit
# ==============================================================
C_TOTAL_JOBS=$(( ${#C_ATTACK_TASKS[@]} + ${#C_EXPS_NEEDING_FINAL_AGG[@]} ))
D2_JOBS=$([ "$D2_NEEDED" = true ] && echo 1 || echo 0)
D4_JOBS=$([ "$D4_NEEDED" = true ] && echo 1 || echo 0)
D5_JOBS=$([ "$D5_NEEDED" = true ] && echo 1 || echo 0)
TOTAL_NEEDED=$(( ${#A_NEEDED[@]} + ${#B_NEEDED[@]} + C_TOTAL_JOBS \
                + ${#D1_NEEDED[@]} + D2_JOBS + ${#D3_NEEDED[@]} + D4_JOBS + D5_JOBS ))

# Count fully done: A(3) + B(3) + C(3) + D1(3) + D2(1) + D3(3) + D4(1) + D5(1) = 15
# But the user-facing tally counts whole workstreams done out of 14 named slots.
C_DONE=0
for i in 0 1 2; do [ "${C_STATUS[$i]}" = "done" ] && C_DONE=$((C_DONE + 1)); done
D1_DONE=0
for i in 0 1 2; do [ "${D1_STATUS[$i]}" = "done" ] && D1_DONE=$((D1_DONE + 1)); done
D3_DONE=0
for i in 0 1 2; do [ "${D3_STATUS[$i]}" = "done" ] && D3_DONE=$((D3_DONE + 1)); done
D2_DONE=$([ "$D2_STATUS" = "done" ] && echo 1 || echo 0)
D4_DONE=$([ "$D4_STATUS" = "done" ] && echo 1 || echo 0)
D5_DONE=$([ "$D5_STATUS" = "done" ] && echo 1 || echo 0)
TOTAL_DONE=$(( 3 - ${#A_NEEDED[@]} + 3 - ${#B_NEEDED[@]} + C_DONE \
              + D1_DONE + D2_DONE + D3_DONE + D4_DONE + D5_DONE ))

echo "========================================"
echo "  SUMMARY: $TOTAL_DONE/14 done, $TOTAL_NEEDED jobs to submit"
if [ ${#C_ATTACK_TASKS[@]} -gt 0 ]; then
    echo "  Step C: ${#C_ATTACK_TASKS[@]} per-attack jobs"
fi
if [ ${#C_EXPS_NEEDING_FINAL_AGG[@]} -gt 0 ]; then
    echo "  Step C: ${#C_EXPS_NEEDING_FINAL_AGG[@]} aggregation jobs"
fi
[ ${#D1_NEEDED[@]} -gt 0 ] && echo "  Step D1: ${#D1_NEEDED[@]} per-model jobs"
[ "$D2_NEEDED" = true ]    && echo "  Step D2: 1 baselines job"
[ ${#D3_NEEDED[@]} -gt 0 ] && echo "  Step D3: ${#D3_NEEDED[@]} per-model jobs"
[ "$D4_NEEDED" = true ]    && echo "  Step D4: 1 formulations job"
[ "$D5_NEEDED" = true ]    && echo "  Step D5: 1 bundle job"
echo "========================================"
echo ""

if [ "$TOTAL_NEEDED" -eq 0 ]; then
    echo "All 14 tasks are complete. Nothing to submit."
    exit 0
fi

build_d5_dep_dryrun() {
    # Build a placeholder dep string for the dry-run banner.
    local deps=()
    [ ${#D1_NEEDED[@]} -gt 0 ] && deps+=("\$D1_JOB_ID")
    [ "$D2_NEEDED" = true ]    && deps+=("\$D2_JOB_ID")
    [ ${#D3_NEEDED[@]} -gt 0 ] && deps+=("\$D3_JOB_ID")
    [ "$D4_NEEDED" = true ]    && deps+=("\$D4_JOB_ID")
    if [ ${#deps[@]} -eq 0 ]; then
        echo ""
    else
        local IFS=':'; echo "--dependency=afterany:${deps[*]}"
    fi
}

if [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] Would submit (account=$ACCOUNT):"
    [ ${#A_NEEDED[@]} -gt 0 ] && echo "  sbatch --account=$ACCOUNT --array=$A_ARRAY_STR job_isomorphism.sh"
    [ ${#B_NEEDED[@]} -gt 0 ] && echo "  sbatch --account=$ACCOUNT --array=$B_ARRAY_STR job_teleportation.sh"
    [ ${#C_ATTACK_TASKS[@]} -gt 0 ] && echo "  sbatch --account=$ACCOUNT --array=$C_ATTACK_ARRAY_STR job_theorem45.sh"
    if [ ${#C_EXPS_NEEDING_FINAL_AGG[@]} -gt 0 ]; then
        if [ ${#C_ATTACK_TASKS[@]} -gt 0 ]; then
            echo "  sbatch --account=$ACCOUNT --array=$C_AGG_ARRAY_STR --dependency=afterany:\$C_JOB_ID job_theorem45_agg.sh"
        else
            echo "  sbatch --account=$ACCOUNT --array=$C_AGG_ARRAY_STR job_theorem45_agg.sh"
        fi
    fi
    [ ${#D1_NEEDED[@]} -gt 0 ] && echo "  sbatch --account=$ACCOUNT --array=$D1_ARRAY_STR job_kmfv_kms.sh"
    [ "$D2_NEEDED" = true ]    && echo "  sbatch --account=$ACCOUNT job_kmfv_baselines.sh"
    [ ${#D3_NEEDED[@]} -gt 0 ] && echo "  sbatch --account=$ACCOUNT --array=$D3_ARRAY_STR job_kmfv_deepdream.sh"
    if [ "$D4_NEEDED" = true ]; then
        if [ ${#D1_NEEDED[@]} -gt 0 ]; then
            echo "  sbatch --account=$ACCOUNT --dependency=afterok:\$D1_JOB_ID job_kmfv_formulations.sh"
        else
            echo "  sbatch --account=$ACCOUNT job_kmfv_formulations.sh"
        fi
    fi
    if [ "$D5_NEEDED" = true ]; then
        D5_DEP=$(build_d5_dep_dryrun)
        if [ -n "$D5_DEP" ]; then
            echo "  sbatch --account=$ACCOUNT $D5_DEP job_kmfv_bundle.sh"
        else
            echo "  sbatch --account=$ACCOUNT job_kmfv_bundle.sh"
        fi
    fi
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

if [ ${#C_EXPS_NEEDING_FINAL_AGG[@]} -gt 0 ]; then
    if [ -n "$C_JOB_ID" ]; then
        echo "  Step C (Theorem 4.5 aggregation):  --array=$C_AGG_ARRAY_STR (after $C_JOB_ID)"
        sbatch --account="$ACCOUNT" --array="$C_AGG_ARRAY_STR" --dependency=afterany:"$C_JOB_ID" job_theorem45_agg.sh
    else
        echo "  Step C (Theorem 4.5 aggregation):  --array=$C_AGG_ARRAY_STR"
        sbatch --account="$ACCOUNT" --array="$C_AGG_ARRAY_STR" job_theorem45_agg.sh
    fi
fi

# ---- D-line submissions ----
D1_JOB_ID=""
D2_JOB_ID=""
D3_JOB_ID=""
D4_JOB_ID=""

if [ ${#D1_NEEDED[@]} -gt 0 ]; then
    echo "  Step D1 (km-feature-viz kms): --array=$D1_ARRAY_STR"
    D1_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" --array="$D1_ARRAY_STR" job_kmfv_kms.sh)
    echo "    Job ID: $D1_JOB_ID"
fi

if [ "$D2_NEEDED" = true ]; then
    echo "  Step D2 (km-feature-viz baselines): single"
    D2_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" job_kmfv_baselines.sh)
    echo "    Job ID: $D2_JOB_ID"
fi

if [ ${#D3_NEEDED[@]} -gt 0 ]; then
    echo "  Step D3 (km-feature-viz deepdream): --array=$D3_ARRAY_STR"
    D3_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" --array="$D3_ARRAY_STR" job_kmfv_deepdream.sh)
    echo "    Job ID: $D3_JOB_ID"
fi

if [ "$D4_NEEDED" = true ]; then
    if [ -n "$D1_JOB_ID" ]; then
        echo "  Step D4 (km-feature-viz formulations): single (after $D1_JOB_ID)"
        D4_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" --dependency=afterok:"$D1_JOB_ID" job_kmfv_formulations.sh)
    else
        echo "  Step D4 (km-feature-viz formulations): single"
        D4_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" job_kmfv_formulations.sh)
    fi
    echo "    Job ID: $D4_JOB_ID"
fi

if [ "$D5_NEEDED" = true ]; then
    # Bundle depends (afterany) on whichever D1–D4 jobs were queued in this run.
    D5_DEPS=()
    [ -n "$D1_JOB_ID" ] && D5_DEPS+=("$D1_JOB_ID")
    [ -n "$D2_JOB_ID" ] && D5_DEPS+=("$D2_JOB_ID")
    [ -n "$D3_JOB_ID" ] && D5_DEPS+=("$D3_JOB_ID")
    [ -n "$D4_JOB_ID" ] && D5_DEPS+=("$D4_JOB_ID")
    if [ ${#D5_DEPS[@]} -gt 0 ]; then
        DEP_STR="afterany:$(IFS=:; echo "${D5_DEPS[*]}")"
        echo "  Step D5 (km-feature-viz bundle): single (--dependency=$DEP_STR)"
        sbatch --account="$ACCOUNT" --dependency="$DEP_STR" job_kmfv_bundle.sh
    else
        echo "  Step D5 (km-feature-viz bundle): single"
        sbatch --account="$ACCOUNT" job_kmfv_bundle.sh
    fi
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Re-run this script after jobs complete to check remaining work."
