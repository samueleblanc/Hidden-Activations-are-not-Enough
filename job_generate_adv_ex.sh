#!/bin/bash

#SBATCH --account=def-bruestle #account to charge the calculation
#SBATCH --time=2:00:00 #hour:minutes:seconds
#SBATCH --gres=gpu:1
#SBATCH --mem=16G #memory requested
#SBATCH --output=adv_examples_gpu_%j.out
#SBATCH --error=adv_examples_gpu_%j.err

# Set variables
EXPERIMENT="alexnet_cifar10"
ADV_FILE="$PWD/experiments/$EXPERIMENT/adversarial_examples.zip"

# Prepare environment
TEMP_DIR=$SLURM_TMPDIR
echo "Using temporary directory: $TEMP_DIR"
module load StdEnv/2023 python/3.10.13 scipy-stack/2025a cuda/12.2 cudnn
source env_nibi/bin/activate

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Copying weights..."
cp experiments/$EXPERIMENT/weights/* $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Weights copied to temp directory..."

mkdir -p "$TEMP_DIR/data/cifar-10-batches-py/"
echo "Copying datasets..."
cp -r data/cifar-10-batches-py/* "$TEMP_DIR/data/cifar-10-batches-py/" || { echo "Failed to copy dataset"; exit 1; }
echo "CIFAR10 ready"

python generate_adversarial_examples.py --experiment_name $EXPERIMENT --temp_dir=$SLURM_TMPDIR

# Zip the adversarial examples
cd $TEMP_DIR/experiments/$EXPERIMENT
zip -r adversarial_examples.zip adversarial_examples

# Copy the zip file to the login node
cp adversarial_examples.zip $PWD/experiments/$EXPERIMENT/