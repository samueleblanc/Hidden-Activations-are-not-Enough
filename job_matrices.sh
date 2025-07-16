#!/bin/bash
#SBATCH --account=def-bruestle
#SBATCH --job-name=alexnet_cifar10_matrices
#SBATCH --array=0-3
#SBATCH --time=2:30:00  # Increased to accommodate potential longer runs
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=17G  # Increased to prevent segmentation faults
#SBATCH --output=alexnet_cifar10_%A_%a.out
#SBATCH --error=alexnet_cifar10_%A_%a.err

# Set variables
EXPERIMENT="alexnet_cifar10"
TASK_ID=$SLURM_ARRAY_TASK_ID
ZIP_FILE="$PWD/experiments/$EXPERIMENT/matrices_task_$TASK_ID.zip"

# Prepare environment
TEMP_DIR=$SLURM_TMPDIR
echo "Using temporary directory: $TEMP_DIR"
module load StdEnv/2023 python/3.10.13 scipy-stack/2025a cuda/12.2 cudnn
source env_nibi/bin/activate

# Copy data and weights to temporary directory
echo "Copying datasets..."
mkdir -p "$TEMP_DIR/data/cifar-10-batches-py/"
cp -r data/cifar-10-batches-py/* "$TEMP_DIR/data/cifar-10-batches-py/" || { echo "Failed to copy dataset"; exit 1; }
echo "Copying weights for task $TASK_ID..."
mkdir -p "$TEMP_DIR/experiments/$EXPERIMENT/weights/"
cp experiments/$EXPERIMENT/weights/* "$TEMP_DIR/experiments/$EXPERIMENT/weights/"

# Check for existing zip file and unzip if present
if [ -f "$ZIP_FILE" ]; then
    echo "Found existing zip file: $ZIP_FILE"
    cp "$ZIP_FILE" "$TEMP_DIR/experiments/$EXPERIMENT/" || { echo "Failed to copy zip file"; exit 1; }
    echo "Unzipping existing matrices for task $TASK_ID..."
    unzip -o "$TEMP_DIR/experiments/$EXPERIMENT/matrices_task_$TASK_ID.zip" -d "$TEMP_DIR/experiments/$EXPERIMENT/" || { echo "Unzipping failed"; exit 1; }
    echo "Existing matrices unzipped to $TEMP_DIR/experiments/$EXPERIMENT/matrices"
fi

# Run Python script in the foreground
echo "Generating matrices for task $TASK_ID..."
python generate_matrices.py \
    --temp_dir "$TEMP_DIR" \
    --experiment "$EXPERIMENT" \
    --chunk_id $TASK_ID \
    --total_chunks 4 \
    --batch_size 16

# Zip the matrices directory
echo "Zipping matrices for task $TASK_ID..."
cd "$TEMP_DIR/experiments/$EXPERIMENT"
zip -r "matrices_task_$TASK_ID.zip" matrices || { echo "Zipping failed"; exit 1; }

# Copy the zip file to $PWD
echo "Copying zip file to $PWD..."
cp "matrices_task_$TASK_ID.zip" "$ZIP_FILE" || { echo "Failed to copy zip file"; exit 1; }
echo "Task $TASK_ID completed successfully"