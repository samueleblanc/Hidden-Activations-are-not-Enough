#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_out/E_verify_%j.out
#SBATCH --error=slurm_err/E_verify_%j.err

set -euo pipefail

# Step E pre-flight: verify all cross-model checkpoints load + remap correctly.
# CPU-only — wrapper construction (~30s-1min per checkpoint × 7 checkpoints)
# plus a single 1×3×224×224 forward to compare KM-wrapper logits against the
# source model's logits. No GPU, no internet (weights pre-cached on the login
# node by run_pipeline.sh:Phase 0d).
#
# Writes results/cross_model/verify_results.json. Each Step E pair task
# (job_cross_model.sh) reads this file at startup and self-skips if either
# of its pair members has status "failed".
#
# Always exits 0 — partial verify failures are recorded in the JSON, never
# block the Step E array via SLURM dependency. The orchestrator submits Step
# E with --dependency=afterany on this job for ordering, not gating.

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# Patches needed for the KM wrappers to load alternate (tv_v2 / timm RSB)
# state-dicts. Idempotent — cmp -s short-circuits on already-applied patches.
bash patches/apply_neuralteleportation_patches.sh ./env

bash verify_cross_model_checkpoints.sh
