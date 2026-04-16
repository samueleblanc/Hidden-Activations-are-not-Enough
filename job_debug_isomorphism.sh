#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_out/debug_iso_%j.out
#SBATCH --error=slurm_err/debug_iso_%j.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

python debug_isomorphism.py --experiment resnet_imagenet --num_samples 5
