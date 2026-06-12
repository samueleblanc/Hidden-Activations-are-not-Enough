#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --array=0-2
#SBATCH --time=18:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=192G
#SBATCH --output=slurm_out/B_teleport_%A_%a.out
#SBATCH --error=slurm_err/B_teleport_%A_%a.err

set -euo pipefail

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

# Map array task ID to architecture (Pillar-3 alignment).
#
# Pillar 1 (this experiment, "Step B") demonstrates that penultimate-layer
# activations are NOT invariant under neural teleportation, while the
# Knowledge Matrix is. We use the same three architectures as Pillar 3
# (resnet152, densenet121, googlenet) so the feature-viz comparisons can
# be cross-referenced against the canonical-representation argument.
#
# All three are now teleportable thanks to two new patches in patches/:
#   - neuralteleportation_parallel_branch.patch: fixes a cob-bleed bug in
#     parallel branches (required for GoogLeNet/Inception modules).
#   - googlenetcob.py: new GoogLeNetCOB module (the upstream library does
#     not ship one).
# Apply both via:  bash patches/apply_neuralteleportation_patches.sh
#
# Sizing rationale: full ImageNet val pass with
# 100 teleportations on resnet152 and densenet121 needs ~3-4h on H100 with
# 192G CPU RAM (CPU memory holds the per-teleportation feature matrices).
# May 2026 scale-up: N bumped 500 -> 1000 samples/split, time 8h -> 18h
# linearly, to tighten between-teleport SD on the headline cell.
# Also adds linear CKA column per the May 2026 Pillar 1B reframing.
ARCHITECTURES=("resnet152" "densenet121" "googlenet")
ARCH=${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}

echo "Step B: Teleportation experiment for $ARCH / imagenet (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# Ensure GoogLeNetCOB and the parallel-branch patch are installed. The
# apply script is idempotent (no-op if already applied).
bash patches/apply_neuralteleportation_patches.sh ./env || {
    echo "ERROR: failed to apply neuralteleportation patches"; exit 1;
}

python teleportation_experiment.py \
    --architecture $ARCH \
    --dataset imagenet \
    --pretrained \
    --num_teleportations 100 \
    --num_samples 1000 \
    --data_dir "${IMAGENET_ROOT:-/datashare/imagenet/ILSVRC2012}"

echo "Task $SLURM_ARRAY_TASK_ID ($ARCH) completed"
