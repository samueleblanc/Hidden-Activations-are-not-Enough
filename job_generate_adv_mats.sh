#!/bin/bash

#SBATCH --account=def-bruestle #account to charge the calculation
#SBATCH --time=04:00:00 #hour:minutes:seconds
#SBATCH --gres=gpu:1
#SBATCH --mem=10G #memory requested
#SBATCH --output=slurm_out/adv_mats_%j.out
#SBATCH --error=slurm_err/adv_mats_%j.err

EXPERIMENT="alexnet_cifar10"
ZIP_TIME=20 # in minutes
hours=4
minutes=0
seconds=0


module load StdEnv/2023 python/3.10.13 scipy-stack/2025a #load the required module
source env_nibi/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

HOME_DIR="/home/armenta/scratch/KnowledgeMatrices/Hidden-Activations-are-not-Enough"

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Copying weights..."
cp experiments/$EXPERIMENT/weights/* $SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
echo "Weights copied to temp directory..."

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/
echo "Copying adversarial examples..."
cp -r experiments/$EXPERIMENT/adversarial_examples/* $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/
echo "Adv examples ready."

mkdir -p experiments/$EXPERIMENT/adversarial_matrices/

ZIP_FILE="experiments/$EXPERIMENT/adversarial_matrices/matrices.zip"
if [ -f "$ZIP_FILE" ]; then
    echo "Found existing experiment data labels file: $ZIP_FILE"
    cp "$ZIP_FILE" "$SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_matrices/matrices.zip" || { echo "Failed to copy file"; exit 1; }
    echo "Unzipping matrices.zip to temporary directory..."
    unzip "$SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_matrices/matrices.zip" -d "$SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_matrices/" || { echo "Failed to unzip file"; exit 1; }
    echo "Unzip completed."
fi

# Run Python script in background and capture its PID
python generate_adversarial_matrices.py --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR --batch_size 512 &
PYTHON_PID=$!

# Calculate sleep time (total time - 15 minutes)
total_seconds=$((hours*3600 + minutes*60 + seconds))
sleep_time=$((total_seconds - 60*ZIP_TIME))  # 900=15*60 seconds = 15 minutes

# Ensure sleep_time is not negative
if [ $sleep_time -lt 0 ]; then
    sleep_time=0
fi

echo "Sleeping for $sleep_time seconds"
sleep $sleep_time

# Kill Python process if still running
if kill -0 $PYTHON_PID 2>/dev/null; then
    echo "Killing Python process $PYTHON_PID"
    kill $PYTHON_PID
    wait $PYTHON_PID  # Wait for process to terminate
fi
# Zip the matrices
echo "Zipping matrices..."
cd $SLURM_TMPDIR/experiments/$EXPERIMENT
zip -r adversarial_matrices.zip adversarial_matrices
cd -
# Copy the zip file to the permanent directory
#echo "Zip file: $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_matrices.zip"
echo "Copying zip file $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_matrices.zip to $PWD/experiments/$EXPERIMENT/adversarial_matrices/"
mkdir -p $PWD/experiments/$EXPERIMENT/adversarial_matrices/
cp $SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_matrices.zip $PWD/experiments/$EXPERIMENT/adversarial_matrices/ || { echo "Failed to copy zip file"; exit 1; }
#cp adversarial_matrices.zip $PWD/experiments/$EXPERIMENT/adversarial_matrices/

echo "Done!"
