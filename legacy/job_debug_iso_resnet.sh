#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=slurm_out/iso_debug_%j.out
#SBATCH --error=slurm_err/iso_debug_%j.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

echo "ResNet isomorphism with --debug (relative-error diagnostic)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# Reduced permutations + samples — we only need relative-error measurements,
# not full statistics. Writes over experiments/resnet_imagenet/isomorphism/
# isomorphism_results.json, but the existing file is already committed so
# we can recover the "full" run if needed.
python isomorphism_experiment.py \
    --experiment resnet_imagenet \
    --num_permutations 2 \
    --num_matrix_samples 20 \
    --debug
