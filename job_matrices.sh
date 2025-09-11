#!/bin/bash
#SBATCH --account=def-ko1
#SBATCH --job-name=alexnet_imagenet_matrices
#SBATCH --array=0-3
#SBATCH --time=00:20:00  # Increased to accommodate potential longer runs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=10G  # Increased to prevent segmentation faults
#SBATCH --output=alexnet_imagenet_%A_%a.out
#SBATCH --error=alexnet_imagenet_%A_%a.err

#export CUDA_VISIBLE_DEVICES=0,1
# Set variables
EXPERIMENT="alexnet_imagenet"
TASK_ID=$SLURM_ARRAY_TASK_ID
ZIP_FILE="$PWD/experiments/$EXPERIMENT/matrices_task_$TASK_ID.zip"

# Prepare environment
TEMP_DIR=$SLURM_TMPDIR
echo "Using temporary directory: $TEMP_DIR"
module load StdEnv/2023 python/3.11.5 scipy-stack/2025a cuda/12.2 cudnn
source env_nibi_311/bin/activate

# Copy data and weights to temporary directory
echo "Copying datasets..."
#mkdir -p "$TEMP_DIR/data/cifar-10-batches-py/"
#cp -r data/cifar-10-batches-py/* "$TEMP_DIR/data/cifar-10-batches-py/" || { echo "Failed to copy dataset"; exit 1; }
echo "Copying weights for task $TASK_ID..."
mkdir -p "$TEMP_DIR/experiments/$EXPERIMENT/weights/"
cp experiments/$EXPERIMENT/weights/* "$TEMP_DIR/experiments/$EXPERIMENT/weights/"

echo "copy imagenet.."
mkdir -p "$TEMP_DIR/data/ILSVRC2012/"
cp -r /datashare/imagenet/ILSVRC2012/* $TEMP_DIR/data/ILSVRC2012/
echo "imagenet ready!!!"

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
    --batch_size 8000

# Zip the matrices directory
echo "Zipping matrices for task $TASK_ID..."
cd "$TEMP_DIR/experiments/$EXPERIMENT"
zip -r "matrices_task_$TASK_ID.zip" matrices || { echo "Zipping failed"; exit 1; }

# Copy the zip file to $PWD
echo "Copying zip file to $PWD..."
cp "matrices_task_$TASK_ID.zip" "$ZIP_FILE" || { echo "Failed to copy zip file"; exit 1; }
echo "Task $TASK_ID completed successfully"