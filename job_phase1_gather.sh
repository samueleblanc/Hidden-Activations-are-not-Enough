#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --time=02:00:00
#SBATCH --mem=480G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:1
#SBATCH --output=slurm_out/C_gather_%A.out
#SBATCH --error=slurm_err/C_gather_%A.err

set -euo pipefail

# Phase-1 Step C (reduce) — GATHER stage.
#
# Runs AFTER job_phase1_reduce_array.sh (--dependency=afterany; the gather is
# robust to a few failed array tasks — any combo missing from the cache is
# recomputed in-process with a warning). Assembles the per-combo cache into
# s1/s2/s3_results.json, runs the Cui/Murphy controls, the sanity gate (with
# the controls-ran guard), and emits the paper LaTeX tables — exactly the same
# outputs the old monolithic job_phase1_reduce.sh produced, just reading
# pre-finalized combos instead of recomputing them serially.
#
# GPU + ImageNet val are requested for the controls (the only GPU consumer in
# the reduce). The per-combo measures already ran on CPU in the array; the
# gather's aggregate passes are cheap cache loads. The walltime that the
# monolithic job blew (the serial OT finalize) is gone, so 2h is ample.
# Pass --skip_controls (via CONTROLS_SKIP=1) for an offline GPU-less gather.
#
# mem 480G: Rorqual H100 nodes report RealMemory=510000M, so 512G never
# schedules; 480G clears it and is ample (the 419GB soft-matching broadcast
# was fixed in 35013aa, and that path now runs in the array anyway).

mkdir -p "$SLURM_SUBMIT_DIR/slurm_out" "$SLURM_SUBMIT_DIR/slurm_err"
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

ARCHS="${ARCHS:-resnet152 densenet121 googlenet}"
ATTACKS="${ATTACKS:-fgsm pgd cw deepfool apgd square}"
NUM_TELEPORTS="${NUM_TELEPORTS:-50}"
NUM_CHUNKS="${NUM_CHUNKS:-64}"
COMBO_DIR="${COMBO_DIR:-results/phase1/combos}"

# --- Tar-staging mode (matches job_phase1_reduce.sh) -----------------------
# The gather only recomputes combos that are MISSING from the cache, so it
# still needs the chunk dirs available as a fallback. Stage them like the
# array job and the original reduce job did.
if [ -n "${STATE_TAR:-}" ]; then
    echo "Staging state tar -> $SLURM_TMPDIR/state (s1/s2/s3 incl. panels)"
    mkdir -p "$SLURM_TMPDIR/state"
    tar -xf "$STATE_TAR" -C "$SLURM_TMPDIR/state" \
        --exclude='results/phase1/s3_N5000_archived*' \
        --exclude='results/phase1/artifact*' \
        results/phase1
    S_BASE="$SLURM_TMPDIR/state/results/phase1"
else
    S_BASE="results/phase1"
fi

# --- ImageNet val for the Cui/Murphy controls ------------------------------
# IMAGENET_TAR: stage from the migration tar (contains ILSVRC2012/{val,devkit}).
# Unset => copy from IMAGENET_ROOT (Nibi /datashare default). Skipped entirely
# when CONTROLS_SKIP=1 (no val data needed for a controls-free gather).
SKIP_FLAG=""
if [ "${CONTROLS_SKIP:-0}" = "1" ]; then
    SKIP_FLAG="--skip_controls"
    DATA_DIR="results/phase1"   # unused with --skip_controls; harmless default
else
    mkdir -p "$SLURM_TMPDIR/data"
    if [ -n "${IMAGENET_TAR:-}" ]; then
        echo "Staging ImageNet from $IMAGENET_TAR"
        tar -xf "$IMAGENET_TAR" -C "$SLURM_TMPDIR/data"
    else
        IMAGENET_ROOT="${IMAGENET_ROOT:-/datashare/imagenet/ILSVRC2012}"
        mkdir -p "$SLURM_TMPDIR/data/ILSVRC2012"
        cp -r "$IMAGENET_ROOT/val" "$SLURM_TMPDIR/data/ILSVRC2012/"
    fi
    DATA_DIR="$SLURM_TMPDIR/data/ILSVRC2012"
fi

echo "=== Gather from combo cache $COMBO_DIR ==="
python -m cka_similarity.reduce \
    --combo_dir "$COMBO_DIR" \
    --s1_dir "$S_BASE/s1" \
    --s2_dir "$S_BASE/s2" \
    --s3_dir "$S_BASE/s3" \
    --out_dir results/phase1/aggregated \
    --paper_tables_dir docs/Final-twist/paper/tables \
    --archs $ARCHS \
    --attacks $ATTACKS \
    --num_teleports "$NUM_TELEPORTS" \
    --num_chunks "$NUM_CHUNKS" \
    --data_dir "$DATA_DIR" \
    $SKIP_FLAG
