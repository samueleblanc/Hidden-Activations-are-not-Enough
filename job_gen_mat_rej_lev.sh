#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=00:20:00 #hour:minutes:seconds
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G #memory requested
#SBATCH --output=slurm_out/mats_rej_lev_%A.out
#SBATCH --error=slurm_err/mats_rej_lev_%A.err

hours=0
minutes=20
seconds=0
zip_time=10

EXPERIMENT="alexnet_cifar10"
ZIP_FILE="$PWD/experiments/$EXPERIMENT/rejection_levels/matrices.zip"
HOME_DIR="links/scratch/armenta"

# Create output and error directories if they don't exist
mkdir -p $PWD/slurm_out
mkdir -p $PWD/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env_rorqual/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

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
    cd -
fi

GPU_LOGFILE="gpu_monitor.$EXPERIMENT.rej_lev.log"
INTERVAL=30  # seconds between GPU checks

monitor_gpu() {
  echo "Timestamp, GPU Utilization (%), GPU Memory Used (MiB), GPU Memory Total (MiB)" > "$GPU_LOGFILE"
  while true; do
    timestamp=$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
               --format=csv,noheader,nounits \
    | awk -v ts="$timestamp" '{print ts", "$1", "$2", "$3}' >> "$GPU_LOGFILE"
    sleep $INTERVAL
  done
}

python compute_matrices_for_rejection_level.py --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR --batch_size 18816 &
PYTHON_PID=$!

#time_limit=$(scontrol show job $SLURM_JOB_ID | grep TimeLimit | awk '{print $2}' | cut -d= -f2)
#IFS=':' read -r hours minutes seconds <<< "$time_limit"
total_seconds=$((hours*3600 + minutes*60 + seconds))
sleep_time=$((total_seconds - zip_time*60))
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

MATRICES_DIR="$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices"
ZIP_OUTPUT_DIR="$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels"
if [ -d "$MATRICES_DIR" ]; then
    echo "Zipping matrices from $MATRICES_DIR ..."
    cd "$ZIP_OUTPUT_DIR" || { echo "Failed to cd to $ZIP_OUTPUT_DIR"; }
    zip -r matrices.zip matrices || echo "Zip failed"
    cd - >/dev/null || true
else
    echo "No matrices directory found at $MATRICES_DIR, skipping zipping"
fi

# Copy the zip file back to HOME_DIR (permanent storage)
TEMP_ZIP="$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices.zip"
DEST_DIR="$HOME_DIR/experiments/$EXPERIMENT/rejection_levels/"

if [ -f "$TEMP_ZIP" ]; then
    echo "Copying zip file $TEMP_ZIP to $DEST_DIR"
    mkdir -p "$DEST_DIR"
    cp "$TEMP_ZIP" "$DEST_DIR" || { echo "Failed to copy zip file"; exit 1; }
else
    echo "No zip file to copy from temp dir ($TEMP_ZIP)"
fi

echo "Job completed"