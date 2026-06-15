#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --array=0-17
#SBATCH --time=06:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=slurm_out/D5_dmpart_%A_%a.out
#SBATCH --error=slurm_err/D5_dmpart_%A_%a.err

set -euo pipefail
# NOTE: set -euo pipefail is mandatory (repo policy). Without it, the trailing
# echo would mask a python crash and SLURM would mark the task COMPLETED.

# D5 — ΔM participation (entrywise + spectral concentration) for the trio.
# 18 array tasks = 3 architectures × 6 attacks.
#   ARCH   = ARCHS[task / 6]
#   ATTACK = ATTACKS[task % 6]
#
# Reads the stored A2 adversarial endpoints (NO attack regeneration, NO
# ImageNet val copy): TWO KM forwards per pair, then one 1000×1000 Gram
# eigendecomposition (NEVER a full ΔM SVD). The A2 pairs.pth live at
#   experiments/{arch}_imagenet/adversarial_pairs_N5000/{attack}/pairs.pth
# and total ~108G across all (arch, attack). This therefore runs on Nibi
# where they already live, OR stages a per-(arch, attack) pairs tar to
# node-local disk via PAIRS_TAR (see below) on quota-constrained clusters.
#
# Cost: 2 KM extractions/pair. resnet152 @ ~104 s/KM is the worst case, so
# N_PAIRS (default 100) keeps a cell at ~3.5h on H100 — under the 6h wall
# with margin. Per-sample atomic checkpointing means a wall hit resumes from
# the last completed sample (do NOT raise the wall; the checkpoint exists so
# we don't have to — repo policy caps KM/GPU walls at 8h).

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err

# Map array task ID to (arch, attack). Lowercase ATTACK matches the A2 pairs
# directory naming; ATTACK_UC is the uppercase value dm_participation.py's
# --attack choices expect (FGSM/PGD/CW/DeepFool/APGD/Square).
ARCHS=("resnet152" "densenet121" "googlenet")
ATTACKS=("fgsm" "pgd" "cw" "deepfool" "apgd" "square")
ATTACKS_UC=("FGSM" "PGD" "CW" "DeepFool" "APGD" "Square")

ARCH_IDX=$((SLURM_ARRAY_TASK_ID / 6))
ATTACK_IDX=$((SLURM_ARRAY_TASK_ID % 6))
ARCH=${ARCHS[$ARCH_IDX]}
ATTACK=${ATTACKS[$ATTACK_IDX]}
ATTACK_UC=${ATTACKS_UC[$ATTACK_IDX]}

echo "D5: ΔM participation for $ARCH / $ATTACK_UC (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Pairs location. Default: read in place (Nibi, where A2 already wrote them).
# PAIRS_TAR (quota-constrained clusters): a tar of the per-(arch, attack) A2
# pairs dir, staged to $SLURM_TMPDIR and read from there. The tar must unpack
# to experiments/{arch}_imagenet/adversarial_pairs_N5000/{attack}/pairs.pth
# (same relative layout as the repo) so the path below resolves either way.
# Honors the same PAIRS_TAR / IMAGENET_TAR staging contract as
# job_mask_hamming.sh (this script does NOT touch the ImageNet val set).
REL="experiments/${ARCH}_imagenet/adversarial_pairs_N5000/${ATTACK}/pairs.pth"
if [ -n "${PAIRS_TAR:-}" ]; then
    echo "Staging pairs from $PAIRS_TAR"
    mkdir -p "$SLURM_TMPDIR/stage"
    tar -xf "$PAIRS_TAR" -C "$SLURM_TMPDIR/stage"
    PAIRS="$SLURM_TMPDIR/stage/$REL"
else
    PAIRS="$REL"
fi

if [ ! -f "$PAIRS" ]; then
    echo "ERROR: pairs file not found: $PAIRS"
    echo "  (A2 must have produced it; on quota-constrained clusters set PAIRS_TAR.)"
    exit 1
fi

# N_PAIRS bounds the per-task work under the 6h wall (override via env).
N_PAIRS="${N_PAIRS:-100}"

python dm_participation.py \
    --arch "$ARCH" \
    --attack "$ATTACK_UC" \
    --pairs "$PAIRS" \
    --temp_dir "$SLURM_TMPDIR" \
    --n_pairs "$N_PAIRS" \
    --matrix_batch_size 4096 \
    --device cuda

echo "Task $SLURM_ARRAY_TASK_ID ($ARCH / $ATTACK_UC) completed"
