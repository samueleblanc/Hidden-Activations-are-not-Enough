#!/bin/bash
#SBATCH --account=<ACCOUNT_NAME>
#SBATCH --job-name=alexnet_cifar10_matrices
#SBATCH --array=0-3                # Adjust based on your total_chunks
#SBATCH --time=00:30:00            # Set total job time (e.g., 1 hour)
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=10G                   # Adjust as needed
#SBATCH --output=alexnet_cifar10_%A_%a.out  # %A is job ID, %a is array index
#SBATCH --error=alexnet_cifar10_%A_%a.err

# Set variables
EXPERIMENT="alexnet_cifar10"
TOTAL_CHUNKS=4  # Match your total number of chunks
TIME_LIMIT_MINUTES=30  # Match --time in minutes (e.g., 01:00:00 = 60 minutes)
SLEEP_DURATION=$((TIME_LIMIT_MINUTES - 10))  # Sleep until 10 minutes remain

# Ensure temporary directory is set
TEMP_DIR=$SLURM_TMPDIR
echo "Using temporary directory: $TEMP_DIR"

# Prepare python run
module load StdEnv/2020 python/3.10.2 scipy-stack/2023a cuda cudnn
source env/bin/activate
echo "Copying datasets..."
cp -r data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/
echo "data ready..."
mkdir -p $SLURM_TMPDIR/experiments/alexnet_cifar10/weights/
mkdir -p $SLURM_TMPDIR/experiments/alexnet_cifar10/matrices/
echo "Copying weights for task $SLURM_ARRAY_TASK_ID..."
cp experiments/alexnet_cifar10/weights/* $SLURM_TMPDIR/experiments/alexnet_cifar10/weights/
echo "Weights copied to temp directory..."

# Check if zip file exists in destination and copy/unzip if it does
EXISTING_ZIP="$PWD/experiments/$EXPERIMENT/matrices.zip"
if [ -f "$EXISTING_ZIP" ]; then
    echo "Found existing zip file: $EXISTING_ZIP"
    cp -f "$EXISTING_ZIP" "$TEMP_DIR/experiments/$EXPERIMENT/matrices.zip"
    echo "Unzipping existing matrices..."
    unzip -o "$TEMP_DIR/experiments/$EXPERIMENT/matrices.zip" -d "$TEMP_DIR/experiments/$EXPERIMENT"
    echo "Existing matrices unzipped to $TEMP_DIR/experiments/$EXPERIMENT/matrices"
fi

# Run Python computation in the background
python generate_matrices.py \
    --temp_dir "$TEMP_DIR" \
    --experiment "$EXPERIMENT" \
    --chunk_id $SLURM_ARRAY_TASK_ID \
    --total_chunks $TOTAL_CHUNKS \
    &  # Background execution
PYTHON_PID=$!

# Calculate sleep time in seconds
SLEEP_SECONDS=$((SLEEP_DURATION * 60))
echo "Sleeping for $SLEEP_SECONDS seconds to wait until 10 minutes remain"

# Sleep until 10 minutes before the job ends
sleep $SLEEP_SECONDS

# Forcefully kill the Python process if still running
if ps -p "$PYTHON_PID" > /dev/null; then
    echo "Killing Python process $PYTHON_PID for task $SLURM_ARRAY_TASK_ID"
    kill -SIGTERM "$PYTHON_PID"
    # Wait briefly to ensure process terminates
    sleep 10
    # Check if process is still alive and force kill if necessary
    if ps -p "$PYTHON_PID" > /dev/null; then
        echo "Process $PYTHON_PID did not terminate, sending SIGKILL"
        kill -SIGKILL "$PYTHON_PID"
    fi
else
    echo "Python process $PYTHON_PID already completed"
fi

# When 10 minutes remain, zip and copy with a lock to avoid conflicts
LOCK_FILE="$TEMP_DIR/zip.lock"
MATRICES_DIR="$TEMP_DIR/experiments/$EXPERIMENT/matrices"
ZIP_FILE="$TEMP_DIR/experiments/$EXPERIMENT/matrices.zip"
DESTINATION="$PWD/experiments/$EXPERIMENT/matrices.zip"

if ( set -C; > "$LOCK_FILE" ) 2>/dev/null; then
    # Lock acquired, zip and copy
    echo "Zipping matrices from $MATRICES_DIR to $ZIP_FILE"
    zip -r "$ZIP_FILE" "matrices" -C "$MATRICES_DIR/.."
    mkdir -p "$(dirname "$DESTINATION")"
    cp -f "$ZIP_FILE" "$DESTINATION"
    echo "Copied $ZIP_FILE to $DESTINATION"
    rm "$LOCK_FILE"
else
    echo "Another task is zipping, skipping"
fi

echo "Job completed"