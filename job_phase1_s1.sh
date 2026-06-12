#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --array=0-63
#SBATCH --time=01:30:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=slurm_out/B1_s1_%A_%a.out
#SBATCH --error=slurm_err/B1_s1_%A_%a.err

set -euo pipefail

# Step B1 — S1 within-arch invariance measure panel.

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate
bash patches/apply_neuralteleportation_patches.sh ./env || exit 1

# Copy ImageNet val to local scratch. IMAGENET_ROOT overrides the Nibi
# default for other clusters (e.g. Rorqual); sbatch propagates the
# submitting shell's environment to the job.
IMAGENET_ROOT="${IMAGENET_ROOT:-/datashare/imagenet/ILSVRC2012}"
mkdir -p $SLURM_TMPDIR/data/ILSVRC2012
cp -r "$IMAGENET_ROOT/val" $SLURM_TMPDIR/data/ILSVRC2012/

python -m cka_similarity.workers.s1_within_arch_invariance \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --num_chunks 64 \
    --num_samples 25000 \
    --num_teleports 50 \
    --archs resnet152 densenet121 googlenet \
    --out_dir results/phase1/s1 \
    --data_dir $SLURM_TMPDIR/data/ILSVRC2012
