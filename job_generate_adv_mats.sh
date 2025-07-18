#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=0:20:00 #hour:minutes:seconds
#SBATCH --gres=gpu:1
#SBATCH --mem=20G #memory requested
#SBATCH --output=slurm_out/adv_mats_%j.out
#SBATCH --error=slurm_err/adv_mats_%j.err

EXPERIMENT="alexnet_cifar10"

module load StdEnv/2023 python/3.10.13 scipy-stack/2025a #load the required module
source env_nibi/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Copying weights..."
cp experiments/$EXPERIMENT/weights/* $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Weights copied to temp directory..."

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/
echo "Copying adversarial examples..."
cp -r experiments/$EXPERIMENT/adversarial_examples/* $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/
echo "Adv examples ready."

mkdir -p experiments/$EXPERIMENT/adversarial_matrices/

python generate_adversarial_matrices.py --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR --batch_size 256

echo "Done!"
