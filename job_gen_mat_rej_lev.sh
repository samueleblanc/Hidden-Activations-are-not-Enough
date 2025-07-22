#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=00:30:00 #hour:minutes:seconds
#SBATCH --gres=gpu:1
#SBATCH --mem=18G #memory requested
#SBATCH --output=slurm_out/mats_rej_lev_gpu_%j.out
#SBATCH --error=slurm_err/mats_rej_lev_gpu_%j.err

EXPERIMENT="alexnet_cifar10"
ZIP_FILE="$PWD/experiments/$EXPERIMENT/rejection_levels/matrices.zip"

# Create output and error directories if they don't exist
mkdir -p $PWD/slurm_out
mkdir -p $PWD/slurm_err

module load StdEnv/2023 python/3.10.13 scipy-stack/2025a #load the required module
source env_nibi/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Copying weights..."
cp experiments/$EXPERIMENT/weights/* $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Weights copied to temp directory..."

EXPERIMENT_DATA_TRAIN="$PWD/experiments/$EXPERIMENT/rejection_levels/exp_dataset_train.pth"
EXPERIMENT_DATA_LABELS="$PWD/experiments/$EXPERIMENT/rejection_levels/exp_dataset_labels.pth"
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
    cp "$EXPERIMENT_DATA_LABELS" "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/exp_dataset_labels.pth" || { echo "Failed to copy file"; exit 1; }
fi

# Check for existing zip file and unzip if it exists
if [ -f "$ZIP_FILE" ]; then
    echo "Found existing zip file: $ZIP_FILE"
    cp "$ZIP_FILE" "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/"
    cd "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/"
    unzip -o matrices.zip
    echo "Unzipped existing matrices"
fi

python compute_matrices_for_rejection_level.py --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR --batch_size 512 &
PYTHON_PID=$!

time_limit=$(scontrol show job $SLURM_JOB_ID | grep TimeLimit | awk '{print $2}' | cut -d= -f2)
IFS=':' read -r hours minutes seconds <<< "$time_limit"
total_seconds=$((hours*3600 + minutes*60 + seconds))
sleep_time=$((total_seconds - 900))  # Reserve 15 minutes for cleanup
if [ $sleep_time -lt 0 ]; then
    sleep_time=0
fi
echo "Sleeping for $sleep_time seconds"
sleep $sleep_time

# Kill Python script if still running
if ps -p $PYTHON_PID > /dev/null; then
    echo "Killing Python process $PYTHON_PID"
    kill $PYTHON_PID
    wait $PYTHON_PID 2>/dev/null
fi

# Zip the matrices
echo "Zipping matrices..."
cd $SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels
zip -r matrices.zip matrices

# Copy the zip file to $PWD
echo "Copying zip file to $PWD/experiments/$EXPERIMENT/rejection_levels/..."
mkdir -p $PWD/experiments/$EXPERIMENT/rejection_levels
cp matrices.zip $PWD/experiments/$EXPERIMENT/rejection_levels/

echo "Job completed"
