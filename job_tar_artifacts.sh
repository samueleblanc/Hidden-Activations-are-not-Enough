#!/bin/bash
#SBATCH --account=def-jcbus
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --output=slurm_out/D_tar_%A.out
#SBATCH --error=slurm_err/D_tar_%A.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a 2>/dev/null || true
source env/bin/activate 2>/dev/null || true

bash bin/tar_artifacts.sh
