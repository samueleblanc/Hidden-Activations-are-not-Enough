#!/bin/bash
# ==============================================================
# run_pipeline.sh -- Scan & Submit for pretrained experiment pipeline
#
# Phase 1: Scan for existing results, write pipeline_state.json
# Phase 2: Submit only the SLURM jobs that are still needed
#
# Steps:
#   B: Teleportation       (job_teleportation.sh, array 0-2)
#   C: Theorem 4.5         (job_theorem45.sh,    array 0-17 + agg)
#   D: km-feature-viz      (5 sub-steps, D1-D5; D4 deps on D1, D5 deps on D1-D4)
#   E: Cross-model         (job_cross_model.sh, array 0-10; pre-flight verify
#                           on login node first)
#
# Note: Step A (random neuron permutation isomorphism) was dropped from the
# new TMLR direction; Pillar 1 evidence comes entirely from teleportation
# (Step B). The old script lives at legacy/isomorphism_experiment.py.
#
# All independent steps (B, C-attacks, D1, D2, D3, E) launch in parallel.
# Dependencies: C-aggregate after C-attacks; D4 after D1; D5 after D1-D4.
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
#
# Login-node Python doesn't carry torch — auto-load the cluster modules
# (Compute Canada `+computecanada` wheels like typing_extensions only resolve
# when `scipy-stack` is loaded) and activate the project venv. Idempotent
# (no-op if torch is already importable). The SLURM jobs do the same dance
# inside each job; this block handles the orchestrator's preflight.
# Always prefer env/ when present — defends against system pythons that
# happen to have torch but are too old to parse the knowledgematrix typing
# (e.g. anaconda 3.8 on a dev laptop).
if type module >/dev/null 2>&1; then
    module load StdEnv/2023 python/3.11.5 scipy-stack/2025a 2>/dev/null || true
fi
if [ -f env/bin/activate ]; then
    # shellcheck disable=SC1091
    source env/bin/activate
fi
if ! python -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: torch not importable after module load + venv activate." >&2
    echo "  On the cluster, run interactively first:" >&2
    echo "    module load StdEnv/2023 python/3.11.5 scipy-stack/2025a" >&2
    echo "    source env/bin/activate" >&2
    echo "    pip install -r requirements-slurm.txt" >&2
    echo "  Then retry: bash run_pipeline.sh" >&2
    exit 1
fi
mkdir -p results/km-feature-viz
if [ "$DRY_RUN" = true ]; then
    echo "Phase 0: skipping manifest regen (--dry-run)"
else
    echo "Phase 0: regenerating km-feature-viz manifest..."
    python -m km_feature_viz.manifest_cli \
        --imagenet-root "${IMAGENET_ROOT:-/datashare/imagenet/ILSVRC2012}" \
        --output results/km-feature-viz/manifest.json
fi
echo ""

# ---- Task definitions ----

# Step B: Teleportation — Pillar-3 archs (matches Step C and D-line, matches
# job_teleportation.sh's ARCHITECTURES). resnet18/vgg11/resnet50 results from
# the pre-migration pipeline are intentionally unreferenced; the paper uses
# only the Pillar-3 trio across all three pillars for consistency.
# Library extensions (parallel-branch patch + googlenetcob.py) verified safe
# to rel diff < 1e-6 in commit 369c9dc.
B_NAMES=("resnet152" "densenet121" "googlenet")
B_FILES=(
    "results/teleportation/resnet152_imagenet_teleportation.json"
    "results/teleportation/densenet121_imagenet_teleportation.json"
    "results/teleportation/googlenet_imagenet_teleportation.json"
)

# Step C: Theorem 4.5 — Pillar 3 archs (matches Step B / D-line; commit 369c9dc)
C_NAMES=("resnet152_imagenet" "densenet121_imagenet" "googlenet_imagenet")
C_RESULTS=(
    "experiments/resnet152_imagenet/theorem45/theorem45_results.json"
    "experiments/densenet121_imagenet/theorem45/theorem45_results.json"
    "experiments/googlenet_imagenet/theorem45/theorem45_results.json"
)
C_CHECKPOINTS=(
    "experiments/resnet152_imagenet/theorem45/theorem45_checkpoint.json"
    "experiments/densenet121_imagenet/theorem45/theorem45_checkpoint.json"
    "experiments/googlenet_imagenet/theorem45/theorem45_checkpoint.json"
)

# Step D: km-feature-viz (5 sub-steps) — Pillar 3 launch with 3 archs.
# D_MODELS is the source of truth for the arch list; the bundle gate in
# job_kmfv_bundle.sh and the MODELS arrays in job_kmfv_kms.sh /
# job_kmfv_deepdream.sh must be kept in sync.
D_MODELS=("resnet152" "densenet121" "googlenet")

# D1: compute_kms — per-model state files, 20 entries each
# (3 classes × 7+7+6 = 20 ImageNet-val images per arch).
D1_STATE_FILES=(
    "results/km-feature-viz/state/01_compute_kms_resnet152.json"
    "results/km-feature-viz/state/01_compute_kms_densenet121.json"
    "results/km-feature-viz/state/01_compute_kms_googlenet.json"
)
D1_EXPECTED=20

# D2: compute_baselines — 5 per-method state files, 60 entries each
# (#archs × #images = 3 × 20 = 60 per method).
D2_METHODS=("gradcam" "ig" "smoothgrad" "feature_maps" "pgd")
D2_EXPECTED=60

# D3: compute_deepdream — per-model state files, 15 entries each
# (5 selected channels × 3 layers per arch).
D3_STATE_FILES=(
    "results/km-feature-viz/state/03_deepdream_resnet152.json"
    "results/km-feature-viz/state/03_deepdream_densenet121.json"
    "results/km-feature-viz/state/03_deepdream_googlenet.json"
)
D3_EXPECTED=15

# D4: formulations — existence-only (patch-blocked may produce empty file)
D4_STATE_FILES=(
    "results/km-feature-viz/state/05_counterfactual.json"
    "results/km-feature-viz/state/06_jacobian.json"
)

# D5: bundle — single tarball at repo root
D5_BUNDLE_PATH="km-feature-viz.tar"

# Step E: Cross-model representation comparison (Pillar 3 May-2026 reframing).
# All checkpoints are differently-trained (different recipes, not same-recipe
# seed variants); see km-notes.md 2026-05-02 + paper-plan.md §1.1.
# Pair indices match job_cross_model.sh's ALL_PAIRS array exactly.
# 11 pairs = 10 resnet152 (k=5) + 1 densenet121 (k=2). GoogLeNet excluded.
E_PAIR_SPECS=(
    "resnet152:tv_v1:tv_v2"
    "resnet152:tv_v1:timm_a1"
    "resnet152:tv_v1:timm_a2"
    "resnet152:tv_v1:timm_a3"
    "resnet152:tv_v2:timm_a1"
    "resnet152:tv_v2:timm_a2"
    "resnet152:tv_v2:timm_a3"
    "resnet152:timm_a1:timm_a2"
    "resnet152:timm_a1:timm_a3"
    "resnet152:timm_a2:timm_a3"
    "densenet121:tv_v1:timm_ra"
)
E_FILES=()
for spec in "${E_PAIR_SPECS[@]}"; do
    IFS=':' read -r _arch _i _j <<< "$spec"
    E_FILES+=("results/cross_model/${_arch}/per_pair/${_i}__${_j}.json")
done
E_VERIFY_RESULTS="results/cross_model/verify_results.json"

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

B_NEEDED=()
C_NEEDED=()

declare -a B_STATUS C_STATUS

echo "Step B: Teleportation"
for i in "${!B_NAMES[@]}"; do
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

# ---- E scan: cross-model pairs ----
declare -a E_STATUS
E_NEEDED=()  # candidate task IDs (pre-verify)

echo "Step E: cross-model pairs"
for i in "${!E_PAIR_SPECS[@]}"; do
    if [ -f "${E_FILES[$i]}" ]; then
        E_STATUS[$i]="done"
        echo "  [$i] ${E_PAIR_SPECS[$i]}: DONE"
    else
        E_STATUS[$i]="pending"
        E_NEEDED+=("$i")
        echo "  [$i] ${E_PAIR_SPECS[$i]}: PENDING"
    fi
done
echo ""

# ---- E pre-flight: verify alternate-weight checkpoints on the login node ----
# Only run verify if there are pairs that need to launch — saves ~5 min on
# repeat invocations once everything is done. The verify script itself caches
# downloaded weights in ~/.cache so subsequent runs are fast.
declare -a E_BLOCKED  # pair indices skipped due to a failed checkpoint
E_BLOCKED=()
if [ ${#E_NEEDED[@]} -gt 0 ]; then
    if [ "$DRY_RUN" = true ]; then
        echo "Step E pre-flight: skipping checkpoint verify (--dry-run)"
        echo ""
    else
        echo "Step E pre-flight: verifying cross-model checkpoints on login node…"
        bash verify_cross_model_checkpoints.sh
        echo ""
    fi
    if [ "$DRY_RUN" != true ] && [ ! -f "$E_VERIFY_RESULTS" ]; then
        echo "WARNING: verify_results.json not produced. Skipping all Step E pairs." >&2
        E_BLOCKED=("${E_NEEDED[@]}")
        E_NEEDED=()
    elif [ "$DRY_RUN" = true ] && [ ! -f "$E_VERIFY_RESULTS" ]; then
        : # dry-run with no prior verify → assume all runnable for the report
    else
        # Filter E_NEEDED: a pair is runnable iff BOTH its checkpoints passed.
        E_RUNNABLE=()
        for i in "${E_NEEDED[@]}"; do
            spec=${E_PAIR_SPECS[$i]}
            IFS=':' read -r _arch _i _j <<< "$spec"
            STATUS_I=$(python3 -c "
import json
with open('$E_VERIFY_RESULTS') as f: d=json.load(f)
print(d['checkpoints'].get('${_arch}:${_i}','missing'))
" 2>/dev/null || echo missing)
            STATUS_J=$(python3 -c "
import json
with open('$E_VERIFY_RESULTS') as f: d=json.load(f)
print(d['checkpoints'].get('${_arch}:${_j}','missing'))
" 2>/dev/null || echo missing)
            if [ "$STATUS_I" = "passed" ] && [ "$STATUS_J" = "passed" ]; then
                E_RUNNABLE+=("$i")
            else
                E_STATUS[$i]="blocked_by_verify"
                E_BLOCKED+=("$i")
                echo "  [$i] ${spec}: BLOCKED (${_arch}:${_i}=${STATUS_I}, ${_arch}:${_j}=${STATUS_J})"
            fi
        done
        E_NEEDED=("${E_RUNNABLE[@]}")
    fi
    echo ""
fi

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

B_ARRAY_STR=""
C_ATTACK_ARRAY_STR=""
C_AGG_ARRAY_STR=""
E_ARRAY_STR=""
[ ${#B_NEEDED[@]} -gt 0 ] && B_ARRAY_STR=$(join_array "${B_NEEDED[@]}")
[ ${#C_ATTACK_TASKS[@]} -gt 0 ] && C_ATTACK_ARRAY_STR=$(join_array "${C_ATTACK_TASKS[@]}")
[ ${#C_EXPS_NEEDING_FINAL_AGG[@]} -gt 0 ] && C_AGG_ARRAY_STR=$(join_array "${C_EXPS_NEEDING_FINAL_AGG[@]}")
[ ${#E_NEEDED[@]} -gt 0 ] && E_ARRAY_STR=$(join_array "${E_NEEDED[@]}")

STATE_FILE="pipeline_state.json"
TMP_FILE="${STATE_FILE}.tmp"

printf '%s\n' '{
  "timestamp": "'"$TIMESTAMP"'",
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
  "step_E_cross_model": {
    "pending": '${#E_NEEDED[@]}',
    "blocked": '${#E_BLOCKED[@]}',
    "done": '$(( ${#E_PAIR_SPECS[@]} - ${#E_NEEDED[@]} - ${#E_BLOCKED[@]} ))'
  },
  "submitted": {
    "step_B": "'"${B_ARRAY_STR:-none}"'",
    "step_C_attacks": "'"${C_ATTACK_ARRAY_STR:-none}"'",
    "step_C_aggregate": "'"${C_AGG_ARRAY_STR:-none}"'",
    "step_D1": "'"${D1_ARRAY_STR:-none}"'",
    "step_D2": "'"$D2_SUB"'",
    "step_D3": "'"${D3_ARRAY_STR:-none}"'",
    "step_D4": "'"$D4_SUB"'",
    "step_D5": "'"$D5_SUB"'",
    "step_E": "'"${E_ARRAY_STR:-none}"'"
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
TOTAL_NEEDED=$(( ${#B_NEEDED[@]} + C_TOTAL_JOBS \
                + ${#D1_NEEDED[@]} + D2_JOBS + ${#D3_NEEDED[@]} + D4_JOBS + D5_JOBS \
                + ${#E_NEEDED[@]} ))

# Count fully-done workstreams: B(3) + C(3) + D1(1) + D2(1) + D3(1) + D4(1) + D5(1) + E(11 pairs) = 22.
# B/C are scored per-experiment; D1/D3 are scored 1 iff ALL models done; E counts per-pair.
C_DONE=0
for i in 0 1 2; do [ "${C_STATUS[$i]}" = "done" ] && C_DONE=$((C_DONE + 1)); done
D1_DONE=$([ ${#D1_NEEDED[@]} -eq 0 ] && echo 1 || echo 0)
D3_DONE=$([ ${#D3_NEEDED[@]} -eq 0 ] && echo 1 || echo 0)
D2_DONE=$([ "$D2_STATUS" = "done" ] && echo 1 || echo 0)
D4_DONE=$([ "$D4_STATUS" = "done" ] && echo 1 || echo 0)
D5_DONE=$([ "$D5_STATUS" = "done" ] && echo 1 || echo 0)
E_DONE=$(( ${#E_PAIR_SPECS[@]} - ${#E_NEEDED[@]} - ${#E_BLOCKED[@]} ))
TOTAL_DONE=$(( 3 - ${#B_NEEDED[@]} + C_DONE \
              + D1_DONE + D2_DONE + D3_DONE + D4_DONE + D5_DONE \
              + E_DONE ))
TOTAL_TASKS=$(( 3 + 3 + 1 + 1 + 1 + 1 + 1 + ${#E_PAIR_SPECS[@]} ))

echo "========================================"
echo "  SUMMARY: $TOTAL_DONE/$TOTAL_TASKS done, $TOTAL_NEEDED jobs to submit"
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
[ ${#E_NEEDED[@]} -gt 0 ]  && echo "  Step E: ${#E_NEEDED[@]} cross-model pair jobs"
[ ${#E_BLOCKED[@]} -gt 0 ] && echo "  Step E: ${#E_BLOCKED[@]} pairs BLOCKED by verify failure (see results/cross_model/verify_results.json)"
echo "========================================"
echo ""

if [ "$TOTAL_NEEDED" -eq 0 ]; then
    echo "All $TOTAL_TASKS tasks are complete. Nothing to submit."
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
    [ ${#E_NEEDED[@]} -gt 0 ] && echo "  sbatch --account=$ACCOUNT --array=$E_ARRAY_STR job_cross_model.sh"
    exit 0
fi

echo "Submitting SLURM jobs (account=$ACCOUNT)..."
echo ""

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

# Step E (cross-model) — independent of B/C/D. Verify already ran on the
# login node; E_NEEDED is filtered to runnable pairs only.
if [ ${#E_NEEDED[@]} -gt 0 ]; then
    echo "  Step E (cross-model): --array=$E_ARRAY_STR"
    sbatch --account="$ACCOUNT" --array="$E_ARRAY_STR" job_cross_model.sh
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Re-run this script after jobs complete to check remaining work."
