#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --time=15:00:00
#SBATCH --mem=512G
#SBATCH --cpus-per-task=16
#SBATCH --gpus=h100:1
#SBATCH --output=slurm_out/C_reduce_%A.out
#SBATCH --error=slurm_err/C_reduce_%A.err

set -euo pipefail

# Phase-1 Step C — Reduce + sanity + LaTeX tables.
# Single task; depends on B′, C′, F finishing.

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

IMAGENET_ROOT="${IMAGENET_ROOT:-/datashare/imagenet/ILSVRC2012}"
mkdir -p $SLURM_TMPDIR/data/ILSVRC2012
cp -r "$IMAGENET_ROOT/val" $SLURM_TMPDIR/data/ILSVRC2012/

python -m cka_similarity.reduce \
    --s1_dir results/phase1/s1 \
    --s2_dir results/phase1/s2 \
    --s3_dir results/phase1/s3 \
    --out_dir results/phase1/aggregated \
    --paper_tables_dir docs/Final-twist/paper/tables \
    --data_dir $SLURM_TMPDIR/data/ILSVRC2012
