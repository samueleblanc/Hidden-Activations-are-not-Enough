#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --array=0-17
#SBATCH --time=08:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=slurm_out/A2_advscale_%A_%a.out
#SBATCH --error=slurm_err/A2_advscale_%A_%a.err

set -euo pipefail

# Step A2 — adversarial pair scale-up.
# 18 array tasks = 6 attacks × 3 architectures.

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err

# Map array task ID to (arch, attack)
ARCHS=("resnet152" "densenet121" "googlenet")
ATTACKS=("fgsm" "pgd" "cw" "deepfool" "apgd" "square")

ARCH_IDX=$((SLURM_ARRAY_TASK_ID / 6))
ATTACK_IDX=$((SLURM_ARRAY_TASK_ID % 6))
ARCH=${ARCHS[$ARCH_IDX]}
ATTACK=${ATTACKS[$ATTACK_IDX]}

echo "Step A2: scale up $ATTACK on $ARCH"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# Copy ImageNet val to local scratch for fast access
mkdir -p $SLURM_TMPDIR/data/ILSVRC2012
cp -r /datashare/imagenet/ILSVRC2012/val $SLURM_TMPDIR/data/ILSVRC2012/

python generate_adversarial_pairs_scaleup.py \
    --arch "$ARCH" --attack "$ATTACK" --target_n 5000 \
    --temp_dir "$SLURM_TMPDIR" \
    --batch_size 32 --checkpoint_every 100
