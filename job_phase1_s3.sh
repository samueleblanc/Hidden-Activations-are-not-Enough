#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --array=0-63
#SBATCH --time=08:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=slurm_out/B3_s3_%A_%a.out
#SBATCH --error=slurm_err/B3_s3_%A_%a.err

set -euo pipefail

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# total_pairs reduced 5000 -> 1000 (2026-06-04): at N=5000 each chunk task hit
# the 8h wall inside resnet152's KM extraction, so densenet121/googlenet were
# starved (0/64). N=1000 (~16 pairs/chunk/attack) lets every chunk cover all
# three archs within 8h. Amplification-by-attack means + bootstrap CI stay
# stable at N=1000. NOTE: changing N changes chunk slicing — old N=5000
# results/phase1/s3 chunks MUST be cleared before resubmitting (incompatible
# append-resume would keep stale rows).
python -m cka_similarity.workers.s3_distance_amplification \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --num_chunks 64 \
    --total_pairs 1000 \
    --archs resnet152 densenet121 googlenet \
    --attacks fgsm pgd cw deepfool apgd square \
    --out_dir results/phase1/s3 \
    --pairs_root experiments
