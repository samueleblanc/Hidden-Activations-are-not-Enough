#!/bin/bash
#SBATCH --account=def-jcbus
#SBATCH --array=0-63
#SBATCH --time=00:45:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=slurm_out/B2_s2_%A_%a.out
#SBATCH --error=slurm_err/B2_s2_%A_%a.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

mkdir -p $SLURM_TMPDIR/data/ILSVRC2012
cp -r /datashare/imagenet/ILSVRC2012/val $SLURM_TMPDIR/data/ILSVRC2012/

python -m cka_similarity.workers.s2_cross_architecture \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --num_chunks 64 \
    --num_samples 25000 \
    --archs resnet152 densenet121 googlenet \
    --out_dir results/phase1/s2 \
    --data_dir $SLURM_TMPDIR/data/ILSVRC2012
