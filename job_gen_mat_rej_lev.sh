#!/bin/bash

#SBATCH --account=def-bruestle #account to charge the calculation
#SBATCH --time=0:20:00 #hour:minutes:seconds
#SBATCH --gres=gpu:1
#SBATCH --mem=16G #memory requested
#SBATCH --output=slurm_out/mats_rej_lev_gpu_%j.out
#SBATCH --error=slurm_err/mats_rej_lev_gpu_%j.err

EXPERIMENT="alexnet_cifar10"

# Create output and error directories if they don't exist
mkdir -p $PWD/slurm_out
mkdir -p $PWD/slurm_err

module load StdEnv/2023 python/3.10.13 scipy-stack/2025a #load the required module
source env_nibi/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Copying weights..."
cp experiments/$EXPERIMENT/weights/* $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Weights copied to temp directory..."

EXPERIMENT_DATA_TRAIN = "$PWD/experiments/$EXPERIMENT/rejection_levels/exp_dataset_train.pth"
EXPERIMENT_DATA_LABELS = "$PWD/experiments/$EXPERIMENT/rejection_levels/exp_dataset_labels.pth"
mkdir -p "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/"

mkdir -p "$SLURM_TMPDIR/data/cifar-10-batches-py/"
echo "Copying datasets..."
cp -r data/cifar-10-batches-py/* "$SLURM_TMPDIR/data/cifar-10-batches-py/" || { echo "Failed to copy dataset"; exit 1; }
echo "CIFAR10 ready"

if [ -f "$EXPERIMENT_DATA_TRAIN" ]; then
    echo "Found existing experiment data train file: $EXPERIMENT_DATA_TRAIN"
    cp "$EXPERIMENT_DATA_TRAIN" "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/exp_dataset_train.pth" || { echo "Failed to copy file"; exit 1; }
fi

if [ -f "$EXPERIMENT_DATA_LABELS" ]; then
    echo "Found existing experiment data labels file: $EXPERIMENT_DATA_LABELS"
    cp "$EXPERIMENT_DATA_LABELS" "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/exp_dataset_labels.pth" || {echo "Failed to copy file."; exit 1;}
fi

python compute_matrices_for_rejection_level.py --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR --batch_size 512

echo "Matrices for rejection level computed!"