#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --time=03:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --array=0-170
#SBATCH --output=slurm_out/C_reduce_array_%A_%a.out
#SBATCH --error=slurm_err/C_reduce_array_%A_%a.err

set -euo pipefail

# Phase-1 Step C (reduce) — PER-COMBO array stage.
#
# Replaces the monolithic serial aggregate (job_phase1_reduce.sh), which timed
# out at the 3h wall: aggregate_s1 ran the expensive optimal-transport measures
# (entropic Gromov-Wasserstein + Sinkhorn soft-matching) for every (arch ×
# teleport) combo serially at finalize. Here each array task finalizes ONE
# combo (mapped from $SLURM_ARRAY_TASK_ID via cka_similarity.reduce.aggregate.
# enumerate_combos) and caches it to a SHARED combo_dir. The gather
# (job_phase1_gather.sh) then assembles the cached combos with no recompute.
#
# CPU-only: no measure here needs a GPU (controls — the only GPU consumer —
# run in the gather). Each combo loads ~64 chunk files (~3 GB) and builds the
# fp64 OT cost matrices (entropic Gromov-Wasserstein on a 5000-point subsample
# + Sinkhorn soft-matching on a p × p cost); 64G is ample. Those OT measures
# dominate the runtime: one S1 combo takes ~1h12 on 16 cores WITH the BLAS
# thread caps set below, so 3h is a safe cap. (An uncapped 4-core run thrashed
# — 5h50m of CPU at the 2h wall, never finished — which is why the caps and the
# core bump matter; see the thread-cap note below.)
#
# IDEMPOTENT: a combo already present in combo_dir is skipped (no recompute),
# so a partial array can be resubmitted to fill only the missing combos.
#
# --array bound: the default above (0-170) matches the production config
# (3 archs × 50 teleports + 3 pairs + 3 archs × 6 attacks = 171 combos).
# run_pipeline.sh derives N from the enumerator and OVERRIDES this with an
# explicit `sbatch --array=0-$((N-1))%CONC`, so the in-script default is only
# a hand-submit convenience — keep it in sync with the production combo count.

mkdir -p "$SLURM_SUBMIT_DIR/slurm_out" "$SLURM_SUBMIT_DIR/slurm_err"
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# BLAS thread caps — pin OpenBLAS/MKL/OMP to the SLURM allocation. The OT
# measures call heavily into BLAS; uncapped, each defaults to the NODE's
# physical core count and oversubscribes the cgroup (a 4-core run thrashed:
# 5h50m CPU at the 2h wall, unfinished). Capping to $SLURM_CPUS_PER_TASK lets
# the 16 cores cooperate instead of fighting (combo finishes in ~1h12).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# Combo config (must match the gather's). Override via env if the run differs.
ARCHS="${ARCHS:-resnet152 densenet121 googlenet}"
ATTACKS="${ATTACKS:-fgsm pgd cw deepfool apgd square}"
NUM_TELEPORTS="${NUM_TELEPORTS:-50}"
NUM_CHUNKS="${NUM_CHUNKS:-64}"

# Shared combo cache — MUST live on the shared FS (persists past this task and
# is read by the gather). Defaults into the repo's results tree.
COMBO_DIR="${COMBO_DIR:-results/phase1/combos}"
mkdir -p "$COMBO_DIR"

# --- Tar-staging mode (quota-constrained clusters, e.g. Rorqual) ----------
# Mirror job_phase1_reduce.sh: when STATE_TAR is set, stage the chunk dirs to
# node-local disk and read s1/s2/s3 from there. Outputs (the combo cache) still
# land on the shared FS via COMBO_DIR.
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

echo "=== Reduce combo task $SLURM_ARRAY_TASK_ID -> $COMBO_DIR ==="
python -m cka_similarity.reduce.combo \
    --index "$SLURM_ARRAY_TASK_ID" \
    --combo_dir "$COMBO_DIR" \
    --archs $ARCHS \
    --attacks $ATTACKS \
    --num_teleports "$NUM_TELEPORTS" \
    --num_chunks "$NUM_CHUNKS" \
    --s1_dir "$S_BASE/s1" \
    --s2_dir "$S_BASE/s2" \
    --s3_dir "$S_BASE/s3"
