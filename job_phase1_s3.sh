#!/bin/bash
#SBATCH --account=def-jcbus
#SBATCH --array=0-63
#SBATCH --time=02:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=slurm_out/B3_s3_%A_%a.out
#SBATCH --error=slurm_err/B3_s3_%A_%a.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

python -m cka_similarity.workers.s3_distance_amplification \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --num_chunks 64 \
    --total_pairs 5000 \
    --archs resnet152 densenet121 googlenet \
    --attacks fgsm pgd cw deepfool apgd square \
    --out_dir results/phase1/s3 \
    --pairs_root experiments
