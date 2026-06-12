#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --time=15:00:00
#SBATCH --mem=480G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:1
#SBATCH --output=slurm_out/C_reduce_%A.out
#SBATCH --error=slurm_err/C_reduce_%A.err

set -euo pipefail

# Phase-1 Step C — Reduce + sanity + LaTeX tables.
# Single task; depends on B′, C′, F finishing.
# mem 512G -> 480G (2026-06-12): Rorqual H100 nodes report RealMemory=510000M,
# so 512G can never schedule there; 480G clears it and is ample for the
# 35013aa soft-matching path (the historic 419GB broadcast is gone).

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# --- Tar-staging mode (quota-constrained clusters, e.g. Rorqual) ----------
# STATE_TAR: migration tar holding results/phase1 (chunk JSONs + .pt panels).
# When set, the panels are staged to node-local disk instead of living on the
# shared FS; reduce reads s1/s2/s3 from the staged copy. Outputs (aggregated
# JSONs, stage checkpoints, paper tables) still land in the repo. Unset =>
# Nibi behavior (read results/phase1 in place).
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
# Unset => copy from IMAGENET_ROOT (Nibi /datashare default).
mkdir -p "$SLURM_TMPDIR/data"
if [ -n "${IMAGENET_TAR:-}" ]; then
    echo "Staging ImageNet from $IMAGENET_TAR"
    tar -xf "$IMAGENET_TAR" -C "$SLURM_TMPDIR/data"
else
    IMAGENET_ROOT="${IMAGENET_ROOT:-/datashare/imagenet/ILSVRC2012}"
    mkdir -p $SLURM_TMPDIR/data/ILSVRC2012
    cp -r "$IMAGENET_ROOT/val" $SLURM_TMPDIR/data/ILSVRC2012/
fi

python -m cka_similarity.reduce \
    --s1_dir "$S_BASE/s1" \
    --s2_dir "$S_BASE/s2" \
    --s3_dir "$S_BASE/s3" \
    --out_dir results/phase1/aggregated \
    --paper_tables_dir docs/Final-twist/paper/tables \
    --data_dir $SLURM_TMPDIR/data/ILSVRC2012
