#!/bin/bash

#SBATCH --account=def-bruestle #account to charge the calculation
#SBATCH --time=00:20:00 #hour:minutes:seconds
#SBATCH --gres=gpu:1
#SBATCH --mem=10G #memory requested
#SBATCH --output=slurm_out/adv_examples_gpu_%j.out
#SBATCH --error=slurm_err/adv_examples_gpu_%j.err

# Create output and error directories if they don't exist
mkdir -p $PWD/slurm_out
mkdir -p $PWD/slurm_err

# Set variables
EXPERIMENT="alexnet_cifar10"

# Prepare environment
echo "Using temporary directory: $SLURM_TMPDIR"
module load StdEnv/2023 python/3.10.13 scipy-stack/2025a cuda/12.2 cudnn
source env_nibi/bin/activate

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Copying weights..."
cp experiments/$EXPERIMENT/weights/* $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Weights copied to temp directory..."

mkdir -p "$SLURM_TMPDIR/data/cifar-10-batches-py/"
echo "Copying datasets..."
cp -r data/cifar-10-batches-py/* "$SLURM_TMPDIR/data/cifar-10-batches-py/" || { echo "Failed to copy dataset"; exit 1; }
echo "CIFAR10 ready"

python generate_adversarial_examples.py --experiment_name $EXPERIMENT --temp_dir=$SLURM_TMPDIR
