#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --array=0-1
#SBATCH --time=08:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --output=slurm_out/D2_kmdrift_%A_%a.out
#SBATCH --error=slurm_err/D2_kmdrift_%A_%a.err

set -euo pipefail

# D2 — KM drift under neural teleportation (Gate-B requirement).
# 2 array tasks: 0=resnet152, 1=densenet121 (googlenet is not wired for
# KM alt-weight loading; see teleportation_km_drift.py docstring).
#
# Cost: (T+1)*N_km KM extractions. resnet152 @ ~104 s/KM => T=5, N_km=50
# needs ~8.7 h — i.e. ONE resubmit after the 8h wall (the script's
# per-teleport resume picks up where the wall hit; base KMs are recomputed
# per slot, ~1.5 h overhead). densenet121 fits a single slot. Submit a
# back-to-back resume slot up front:
#   JID=$(sbatch --parsable job_km_drift.sh)
#   sbatch --array=0 --dependency=afterany:$JID job_km_drift.sh
# The resumed task no-ops in minutes if the first slot finished.
#
# Wall stays at 8h per repo policy — never raise it; the resume exists
# precisely so we don't have to.

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# neuralteleportation patches (idempotent) — COB models + GoogLeNetCOB drop-in.
bash patches/apply_neuralteleportation_patches.sh ./env || {
    echo "ERROR: failed to apply neuralteleportation patches"; exit 1;
}

ARCHS=("resnet152" "densenet121")
ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}

echo "D2: KM drift under teleportation for $ARCH (task $SLURM_ARRAY_TASK_ID)"

# --data tensor = the Step-B sample set: load_dataset('imagenet','test',
# 1000, seed=42) — the same loader/split/seed teleportation_experiment.py
# uses, over the SAME val tree (ImageFolder ordering is filename-determined,
# so the tensor is reproducible iff the val tree matches Nibi's bit-for-bit).
# Generated once here if missing; per-PID tmp + atomic rename makes the
# concurrent generation by both array tasks safe.
DATA_PTH="results/teleportation/imagenet_test_samples_N1000_seed42.pth"
mkdir -p results/teleportation results/teleportation_km_drift
if [ ! -f "$DATA_PTH" ]; then
    # IMAGENET_TAR (quota-constrained clusters): stage ILSVRC2012/{val,devkit}
    # to node-local disk just for this one-time tensor generation. Unset =>
    # read IMAGENET_ROOT in place (Nibi /datashare default).
    if [ -n "${IMAGENET_TAR:-}" ]; then
        echo "Staging ImageNet from $IMAGENET_TAR"
        mkdir -p "$SLURM_TMPDIR/data"
        tar -xf "$IMAGENET_TAR" -C "$SLURM_TMPDIR/data"
        IMAGENET_ROOT="$SLURM_TMPDIR/data/ILSVRC2012"
    else
        IMAGENET_ROOT="${IMAGENET_ROOT:-/datashare/imagenet/ILSVRC2012}"
    fi
    echo "Generating Step-B sample tensor at $DATA_PTH ..."
    python - "$IMAGENET_ROOT" "$DATA_PTH" <<'PYEOF'
import os, sys, torch
from teleportation_experiment import load_dataset
root, out = sys.argv[1], sys.argv[2]
x = load_dataset('imagenet', 'test', 1000, data_dir=root, seed=42)
tmp = f"{out}.{os.getpid()}.tmp"
torch.save(x, tmp)
os.replace(tmp, out)
print('saved', out, tuple(x.shape))
PYEOF
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Per-arch KM batch cap. teleportation_km_drift.py defaults --km_batch=1800, but
# that EXCEEDS resnet152's calibrated 85%-tier batch (1088, peak 70.8/79 GB) and
# OOMs in KnowledgeMatrixComputer.forward (job 14500625_0, 2026-06-20). densenet121's
# calibrated batch is 4288, so 1800 fits there with margin. Source of truth:
# experiments/calibration/<arch>_imagenet/calibration.json (active_tier km_batch_size).
if [ "$ARCH" = "resnet152" ]; then KM_BATCH=1088; else KM_BATCH=1800; fi
echo "KM batch for $ARCH: $KM_BATCH"

python teleportation_km_drift.py \
    --arch "$ARCH" \
    --num_teleportations 5 \
    --num_km_samples 50 \
    --num_gate_samples 500 \
    --data "$DATA_PTH" \
    --out results/teleportation_km_drift \
    --device cuda \
    --seed 0 \
    --km_batch "$KM_BATCH"

echo "Task $SLURM_ARRAY_TASK_ID ($ARCH) completed"
