#!/bin/bash
# ==============================================================
# run_pipeline.sh -- Scan & Submit for pretrained experiment pipeline
#
# Phase 1: Scan for existing results, write pipeline_state.json
# Phase 2: Submit only the SLURM jobs that are still needed
#
# Steps:
#   B: Teleportation       (Study 1b, job_teleportation.sh, array 0-2)
#   C: Theorem 4.5         (Study 2,  job_theorem45.sh,    array 0-17 + agg)
#   E: Cross-model         (positioned vs. Study 3, job_cross_model.sh, array 0-10;
#                           pre-flight verify on login node first)
#
# Note: Step A (random neuron permutation isomorphism, Study 1a) was dropped
# from the orchestrator. Direct invocation still works via
# isomorphism_experiment.py at the repo root.
#
# All independent steps (B, C-attacks, E) launch in parallel.
# Dependencies: C-aggregate after C-attacks.
#
# Usage:
#   bash run_pipeline.sh              # Scan and submit
#   bash run_pipeline.sh --dry-run    # Scan only, do not submit
# ==============================================================
set -euo pipefail

# ---- Cluster account ----
# Defaults to def-amorales (Nibi) per reference_nibi_sbatch.md. Override
# by exporting ACCOUNT=<acct> before invoking the orchestrator, e.g.:
#   ACCOUNT=def-bruestle_gpu bash run_pipeline.sh
ACCOUNT="${ACCOUNT:-def-amorales}"

DRY_RUN=false
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done

if [ "$DRY_RUN" != true ] && command -v squeue >/dev/null 2>&1; then
    RUNNING_JOBS=$(squeue -u "$USER" -h -o "%j" 2>/dev/null || true)
    KNOWN_JOB_NAMES="job_calibrate.sh|job_phase1_s1.sh|job_phase1_s2.sh|job_phase1_s3.sh|job_phase1_reduce.sh|job_adv_scaleup.sh|job_teleportation.sh|job_theorem45.sh|job_theorem45_agg.sh|job_cross_model.sh|job_tar_artifacts.sh"
    if echo "$RUNNING_JOBS" | grep -qE "^($KNOWN_JOB_NAMES)$"; then
        echo "ERROR: existing pipeline jobs running. Cancel them first or wait:" >&2
        echo "$RUNNING_JOBS" | grep -E "^($KNOWN_JOB_NAMES)$" >&2
        echo "" >&2
        echo "  Cancel: scancel -u \$USER --name=\"<job_name>\"" >&2
        exit 1
    fi
fi

TIMESTAMP=$(date -Iseconds)

# ==============================================================
# Phase 0: Login-node preflights (BEFORE any sbatch)
# ==============================================================
# All preflights run before any sbatch so a broken env fails fast at the
# login node rather than after a SLURM allocation. Idempotent throughout —
# re-runs are fast no-ops on the cached path. Run on --dry-run too so the
# dry-run output reflects a real verified state.

# --- Phase 0a: module + venv -----------------------------------
# Activate the project venv. The SLURM jobs do the same dance inside each
# job; this block handles the orchestrator's preflight. Always prefer env/
# when present — defends against system pythons that happen to have torch
# but are too old to parse the knowledgematrix typing (e.g. anaconda 3.8).
if type module >/dev/null 2>&1; then
    module load StdEnv/2023 python/3.11.5 scipy-stack/2025a 2>/dev/null || true
fi
HAS_ENV=false
if [ -f env/bin/activate ]; then
    # shellcheck disable=SC1091
    source env/bin/activate
    HAS_ENV=true
fi

# --- Phase 0b: neuralteleportation patches ---------------------
# Apply the local PyTorch-2.x compatibility patches + GoogLeNetCOB drop-in
# to the venv-installed neuralteleportation package. Idempotent: cmp -s
# short-circuits on already-applied patches. Doing this once on the login
# node patches the shared-FS venv, eliminating the per-array-task race
# from job_phase1_s1.sh's safety-net invocation.
# Skipped on local dev without env/ (developer is presumably running a
# different python and can apply patches manually).
if [ "$HAS_ENV" = true ] && [ -f patches/apply_neuralteleportation_patches.sh ]; then
    echo "Preflight: applying neuralteleportation patches..."
    if ! bash patches/apply_neuralteleportation_patches.sh ./env >/dev/null 2>&1; then
        echo "ERROR: neuralteleportation patches failed. Run interactively to see details:" >&2
        echo "  bash patches/apply_neuralteleportation_patches.sh ./env" >&2
        exit 1
    fi
fi

# --- Phase 0c: import smoke-test -------------------------------
# Import every entry point the orchestrator will submit. A NameError or
# missing module surfaces here, not after a SLURM allocation.
if ! python -c "
import sys
try:
    from validate_theorem45 import validate_theorem45
    from cross_model_experiment import main as _xm
    from teleportation_experiment import run_experiment as _tp
    from bin.calibrate import main as _cal
    from cka_similarity.workers import s1_within_arch_invariance as _s1
    from cka_similarity.workers import s2_cross_architecture as _s2
    from cka_similarity.workers import s3_distance_amplification as _s3
except Exception as e:
    print(f'preflight import failure: {type(e).__name__}: {e}', file=sys.stderr)
    sys.exit(1)
" >/dev/null 2>&1; then
    echo "ERROR: preflight imports failed. Run interactively to see details:" >&2
    echo "  module load StdEnv/2023 python/3.11.5 scipy-stack/2025a" >&2
    echo "  source env/bin/activate" >&2
    echo "  python -c 'from validate_theorem45 import validate_theorem45'" >&2
    exit 1
fi

# --- Phase 0d: pretrained weight cache (torchvision + timm) ----
# Compute nodes have no internet — a missing pretrained weight would only
# surface after a SLURM allocation. Cache every weight variant any
# downstream code path will request:
#
#   - Phase-1 architectures (resnet152, densenet121, googlenet) at the
#     DEFAULT torchvision weights — needed by Steps B/C/D, the calibration
#     job, every Phase-1 worker.
#   - Cross-model alternate recipes (Step E only) — torchvision V2 +
#     5 timm RSB variants. Without this, Step E's pre-flight verify
#     pauses for ~250MB downloads scattered across 7 separate sub-steps,
#     which reads to the operator like the verify is hung.
#
# Idempotent: torchvision/timm detect cached files and skip downloads on
# re-runs (~milliseconds cached; ~30s–2min for first-time of the lot).
# Skipped on local dev without env/.
if [ "$HAS_ENV" = true ]; then
    echo "Preflight: caching pretrained weights (torchvision DEFAULT + cross-model recipes)..."
    if ! python -c "
import torchvision.models as tvm

# Phase-1 archs (every downstream sub-study uses these)
tvm.resnet152(weights='DEFAULT')      # = IMAGENET1K_V1, also covers cross-model tv_v1
tvm.densenet121(weights='DEFAULT')    # = IMAGENET1K_V1, also covers cross-model tv_v1
tvm.googlenet(weights='DEFAULT')

# Cross-model (Step E) alternate recipes — verify_cross_model_checkpoints.sh
# would otherwise download these ad-hoc on the login node.
tvm.resnet152(weights='IMAGENET1K_V2')   # Step E tv_v2

import timm
for tag in (
    'resnet152.a1_in1k',                  # Step E timm_a1 (RSB A1)
    'resnet152.a2_in1k',                  # Step E timm_a2 (RSB A2)
    'resnet152.a3_in1k',                  # Step E timm_a3 (RSB A3 / 160px)
    'densenet121.ra_in1k',                # Step E timm_ra
):
    timm.create_model(tag, pretrained=True)
" 2>&1; then
        echo "ERROR: pretrained weight caching failed. The login node needs internet" >&2
        echo "  access on first run (timm fetches from HuggingFace, torchvision from" >&2
        echo "  download.pytorch.org). Re-run on a login node with internet:" >&2
        echo "    bash run_pipeline.sh" >&2
        exit 1
    fi
fi
echo ""

# ==============================================================
# Phase -1: Calibration (must precede all other Phase 1 steps)
# ==============================================================
# Step A1 produces experiments/calibration/{arch}_imagenet/calibration.json.
# All Phase 1 workers (S1/S2/S3) read it at startup; bin/sentinel.sh
# steps the active tier down on CUDA-OOM. Submit only the array tasks
# whose calibration JSON is missing — runs are checkpointable via
# atomic_json_dump, so re-running is safe.
ARCHS_PHASE1=("resnet152" "densenet121" "googlenet")
CALIB_NEEDED=()
for i in "${!ARCHS_PHASE1[@]}"; do
    arch=${ARCHS_PHASE1[$i]}
    if [ ! -f "experiments/calibration/${arch}_imagenet/calibration.json" ]; then
        CALIB_NEEDED+=("$i")
    fi
done

CALIB_JOB_ID=""
if [ ${#CALIB_NEEDED[@]} -gt 0 ]; then
    CALIB_ARRAY=$(IFS=,; echo "${CALIB_NEEDED[*]}")
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] sbatch --account=$ACCOUNT --array=$CALIB_ARRAY job_calibrate.sh"
    else
        echo "Phase 1 Step A1 (Calibration): --array=$CALIB_ARRAY"
        CALIB_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" --array="$CALIB_ARRAY" job_calibrate.sh)
        echo "  Job ID: $CALIB_JOB_ID"
    fi
fi

# ---- Task definitions ----

# Step B: Teleportation — 3 main archs (matches Step C, matches
# job_teleportation.sh's ARCHITECTURES).
# Library extensions (parallel-branch patch + googlenetcob.py) verified safe
# to rel diff < 1e-6 in commit 369c9dc.
B_NAMES=("resnet152" "densenet121" "googlenet")
B_FILES=(
    "results/teleportation/resnet152_imagenet_teleportation.json"
    "results/teleportation/densenet121_imagenet_teleportation.json"
    "results/teleportation/googlenet_imagenet_teleportation.json"
)

# Step C: Theorem 4.5 — 3 main archs (matches Step B; commit 369c9dc)
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

# Step E: Cross-model representation comparison (positioned vs. Study 3, May 2026 reframing).
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
        [ -d "$PA_DIR" ] && PA_COUNT=$(find "$PA_DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
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
B_ARRAY_STR=""
C_ATTACK_ARRAY_STR=""
C_AGG_ARRAY_STR=""
E_ARRAY_STR=""
[ ${#B_NEEDED[@]} -gt 0 ] && B_ARRAY_STR=$(join_array "${B_NEEDED[@]}")
[ ${#C_ATTACK_TASKS[@]} -gt 0 ] && C_ATTACK_ARRAY_STR=$(join_array "${C_ATTACK_TASKS[@]}")
[ ${#C_EXPS_NEEDING_FINAL_AGG[@]} -gt 0 ] && C_AGG_ARRAY_STR=$(join_array "${C_EXPS_NEEDING_FINAL_AGG[@]}")
[ ${#E_NEEDED[@]} -gt 0 ] && E_ARRAY_STR=$(join_array "${E_NEEDED[@]}")

STATE_FILE="pipeline_state.json"

if [ "$DRY_RUN" != true ]; then
    TMP_FILE=$(mktemp -p "$(dirname "$STATE_FILE")" "${STATE_FILE}.XXXXXX.tmp")
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
  "step_E_cross_model": {
    "pending": '${#E_NEEDED[@]}',
    "blocked": '${#E_BLOCKED[@]}',
    "done": '$(( ${#E_PAIR_SPECS[@]} - ${#E_NEEDED[@]} - ${#E_BLOCKED[@]} ))'
  },
  "submitted": {
    "step_B": "'"${B_ARRAY_STR:-none}"'",
    "step_C_attacks": "'"${C_ATTACK_ARRAY_STR:-none}"'",
    "step_C_aggregate": "'"${C_AGG_ARRAY_STR:-none}"'",
    "step_E": "'"${E_ARRAY_STR:-none}"'"
  }
}' > "$TMP_FILE"
    mv "$TMP_FILE" "$STATE_FILE"
    echo "Pipeline state written to $STATE_FILE"
else
    echo "[DRY RUN] Would write pipeline state to $STATE_FILE"
fi
echo ""

# ==============================================================
# Phase 2: Submit
# ==============================================================
C_TOTAL_JOBS=$(( ${#C_ATTACK_TASKS[@]} + ${#C_EXPS_NEEDING_FINAL_AGG[@]} ))
TOTAL_NEEDED=$(( ${#B_NEEDED[@]} + C_TOTAL_JOBS + ${#E_NEEDED[@]} ))

# Count fully-done workstreams: B(3) + C(3) + E(11 pairs) = 17.
# B/C are scored per-experiment; E counts per-pair.
C_DONE=0
for i in 0 1 2; do [ "${C_STATUS[$i]}" = "done" ] && C_DONE=$((C_DONE + 1)); done
E_DONE=$(( ${#E_PAIR_SPECS[@]} - ${#E_NEEDED[@]} - ${#E_BLOCKED[@]} ))
TOTAL_DONE=$(( 3 - ${#B_NEEDED[@]} + C_DONE + E_DONE ))
TOTAL_TASKS=$(( 3 + 3 + ${#E_PAIR_SPECS[@]} ))

echo "========================================"
echo "  SUMMARY: $TOTAL_DONE/$TOTAL_TASKS done, $TOTAL_NEEDED jobs to submit"
if [ ${#C_ATTACK_TASKS[@]} -gt 0 ]; then
    echo "  Step C: ${#C_ATTACK_TASKS[@]} per-attack jobs"
fi
if [ ${#C_EXPS_NEEDING_FINAL_AGG[@]} -gt 0 ]; then
    echo "  Step C: ${#C_EXPS_NEEDING_FINAL_AGG[@]} aggregation jobs"
fi
[ ${#E_NEEDED[@]} -gt 0 ]  && echo "  Step E: ${#E_NEEDED[@]} cross-model pair jobs"
[ ${#E_BLOCKED[@]} -gt 0 ] && echo "  Step E: ${#E_BLOCKED[@]} pairs BLOCKED by verify failure (see results/cross_model/verify_results.json)"
echo "========================================"
echo ""

if [ "$TOTAL_NEEDED" -eq 0 ]; then
    echo "All $TOTAL_TASKS legacy B/C/E tasks are complete (Phase 1 still evaluated below)."
fi

# Initialize C_JOB_ID up front so the (optional) real-submit branch can
# refer to it whether or not Step C per-attacks are queued. Distinct from
# Phase 1's P1C_JOB_ID below.
C_JOB_ID=""

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
    [ ${#E_NEEDED[@]} -gt 0 ] && echo "  sbatch --account=$ACCOUNT --array=$E_ARRAY_STR job_cross_model.sh"
else
    echo "Submitting SLURM jobs (account=$ACCOUNT)..."
    echo ""

    if [ ${#B_NEEDED[@]} -gt 0 ]; then
        echo "  Step B (Teleportation): --array=$B_ARRAY_STR"
        sbatch --account="$ACCOUNT" --array="$B_ARRAY_STR" job_teleportation.sh
    fi

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

    # Step E (cross-model) — independent of B/C. Verify already ran on the
    # login node; E_NEEDED is filtered to runnable pairs only.
    if [ ${#E_NEEDED[@]} -gt 0 ]; then
        echo "  Step E (cross-model): --array=$E_ARRAY_STR"
        sbatch --account="$ACCOUNT" --array="$E_ARRAY_STR" job_cross_model.sh
    fi
fi

# ==============================================================
# Phase 1 (CKA / similarity-measure expansion) submissions
# Dependencies: A1 -> {A2, B1, B2, B3}; B3 also waits on A2;
# Phase-1 C waits on B1+B2+B3; Step D waits on Phase-1 C.
# Variable naming: P1C_JOB_ID (NOT C_JOB_ID) to avoid collision with
# the existing Theorem 4.5 C_JOB_ID above.
# ==============================================================
DEP_FLAG_A1=""
[ -n "$CALIB_JOB_ID" ] && DEP_FLAG_A1="--dependency=afterany:$CALIB_JOB_ID"

# Step A2 — Adversarial scale-up (depends on A1)
# Enumerate all 3 archs × 6 attacks and submit only missing combinations.
# Index mapping must match job_adv_scaleup.sh: ARCH_IDX=(TASK_ID / 6), ATTACK_IDX=(TASK_ID % 6).
A2_JOB_ID=""
A2_ARCHS=("resnet152" "densenet121" "googlenet")
A2_ATTACKS=("fgsm" "pgd" "cw" "deepfool" "apgd" "square")
A2_NEEDED=()
for ai in "${!A2_ARCHS[@]}"; do
    for atki in "${!A2_ATTACKS[@]}"; do
        ARCH=${A2_ARCHS[$ai]}
        ATK=${A2_ATTACKS[$atki]}
        PFILE="experiments/${ARCH}_imagenet/adversarial_pairs_N5000/${ATK}/pairs.pth"
        if [ ! -f "$PFILE" ]; then
            TASK_ID=$(( ai * 6 + atki ))
            A2_NEEDED+=("$TASK_ID")
        fi
    done
done

if [ ${#A2_NEEDED[@]} -gt 0 ]; then
    A2_ARRAY_STR=$(join_array "${A2_NEEDED[@]}")
    if [ "$DRY_RUN" != true ]; then
        A2_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" --array="$A2_ARRAY_STR" $DEP_FLAG_A1 job_adv_scaleup.sh)
        echo "Phase 1 Step A2 (adv scaleup): $A2_JOB_ID --array=$A2_ARRAY_STR (${#A2_NEEDED[@]} of 18)"
    else
        echo "[DRY RUN] sbatch --array=$A2_ARRAY_STR job_adv_scaleup.sh (${#A2_NEEDED[@]} of 18)"
    fi
fi

# Step B1 — S1 measure panel (depends on A1, soft)
B1_JOB_ID=""
if [ ! -f "results/phase1/s1/.complete" ]; then
    if [ "$DRY_RUN" != true ]; then
        B1_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" $DEP_FLAG_A1 job_phase1_s1.sh)
        echo "Phase 1 Step B1 (S1): $B1_JOB_ID"
    else
        echo "[DRY RUN] sbatch --array=0-63 job_phase1_s1.sh"
    fi
fi

# Step B2 — S2 cross-arch (depends on A1)
B2_JOB_ID=""
if [ ! -f "results/phase1/s2/.complete" ]; then
    if [ "$DRY_RUN" != true ]; then
        B2_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" $DEP_FLAG_A1 job_phase1_s2.sh)
        echo "Phase 1 Step B2 (S2): $B2_JOB_ID"
    else
        echo "[DRY RUN] sbatch --array=0-63 job_phase1_s2.sh"
    fi
fi

# Step B3 — S3 measure panel (depends on A1 + A2)
B3_JOB_ID=""
B3_DEPS=""
[ -n "$CALIB_JOB_ID" ] && B3_DEPS="${B3_DEPS}${CALIB_JOB_ID}:"
[ -n "$A2_JOB_ID" ] && B3_DEPS="${B3_DEPS}${A2_JOB_ID}:"
B3_DEPS="${B3_DEPS%:}"
B3_DEP_FLAG=""
[ -n "$B3_DEPS" ] && B3_DEP_FLAG="--dependency=afterany:$B3_DEPS"
if [ ! -f "results/phase1/s3/.complete" ]; then
    if [ "$DRY_RUN" != true ]; then
        B3_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" $B3_DEP_FLAG job_phase1_s3.sh)
        echo "Phase 1 Step B3 (S3): $B3_JOB_ID"
    else
        echo "[DRY RUN] sbatch --array=0-63 job_phase1_s3.sh"
    fi
fi

# Phase-1 Step C — Reduce (depends on B1+B2+B3)
P1C_DEPS=""
[ -n "$B1_JOB_ID" ] && P1C_DEPS="${P1C_DEPS}${B1_JOB_ID}:"
[ -n "$B2_JOB_ID" ] && P1C_DEPS="${P1C_DEPS}${B2_JOB_ID}:"
[ -n "$B3_JOB_ID" ] && P1C_DEPS="${P1C_DEPS}${B3_JOB_ID}:"
P1C_DEPS="${P1C_DEPS%:}"
P1C_DEP_FLAG=""
[ -n "$P1C_DEPS" ] && P1C_DEP_FLAG="--dependency=afterany:$P1C_DEPS"
P1C_JOB_ID=""
P1C_QUEUED=false
if [ ! -f "results/phase1/aggregated/sanity_report.json" ]; then
    P1C_QUEUED=true
    if [ "$DRY_RUN" != true ]; then
        P1C_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" $P1C_DEP_FLAG job_phase1_reduce.sh)
        echo "Phase-1 Step C (reduce): $P1C_JOB_ID"
    else
        echo "[DRY RUN] sbatch job_phase1_reduce.sh"
    fi
fi

# Step D — Tar (depends on Phase-1 Step C). Triggered iff Phase-1 C
# was queued (in real mode we have a job id; in dry-run we use the flag).
D_JOB_ID=""
if [ -n "$P1C_JOB_ID" ] && [ "$DRY_RUN" != true ]; then
    D_JOB_ID=$(sbatch --parsable --account="$ACCOUNT" --dependency=afterany:"$P1C_JOB_ID" job_tar_artifacts.sh)
    echo "Step D (tar): $D_JOB_ID"
elif [ "$P1C_QUEUED" = "true" ] && [ "$DRY_RUN" = true ]; then
    echo "[DRY RUN] sbatch job_tar_artifacts.sh"
fi

# ==============================================================
# Opt-in sentinel hook (USE_SENTINEL=true bash run_pipeline.sh)
# ==============================================================
# For each queued Phase 1 job, schedule a tiny afterany sentinel that
# bin/sentinel.sh handles (system-OOM, timeout, CUDA-OOM tier-step,
# generic retry). No-op when USE_SENTINEL is unset/false or in dry-run
# (no real job ids exist to attach to).
if [ "${USE_SENTINEL:-false}" = "true" ] && [ "$DRY_RUN" != true ]; then
    for SENT_PAIR in \
        "$B1_JOB_ID:job_phase1_s1.sh" \
        "$B2_JOB_ID:job_phase1_s2.sh" \
        "$B3_JOB_ID:job_phase1_s3.sh" \
        "$A2_JOB_ID:job_adv_scaleup.sh" \
        "$P1C_JOB_ID:job_phase1_reduce.sh"; do
        IFS=':' read -r SENT_JID SENT_SCRIPT <<< "$SENT_PAIR"
        [ -z "$SENT_JID" ] && continue
        # Pass the calibration directory (not a single-arch file) so the
        # sentinel can step down ALL arch calibrations on CUDA-OOM. Phase-1
        # workers iterate all 3 archs per chunk, so the failing arch is
        # unknown; stepping all calibrations is the safe conservative choice.
        SENT_CALIB_DIR="experiments/calibration"
        sbatch --parsable --account="$ACCOUNT" --time=00:05:00 --mem=4G \
            --dependency=afterany:"$SENT_JID" \
            --wrap="bash bin/sentinel.sh --account=$ACCOUNT $SENT_JID $SENT_SCRIPT $SENT_CALIB_DIR" \
            > /dev/null
    done
fi

echo ""
echo "Monitor with: squeue -u \$USER"
echo "Re-run this script after jobs complete to check remaining work."
