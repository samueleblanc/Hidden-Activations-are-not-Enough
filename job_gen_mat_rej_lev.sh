#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=00:20:00 #hour:minutes:seconds
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G #memory requested
#SBATCH --output=slurm_out/D_rej_lev_%A.out
#SBATCH --error=slurm_err/D_rej_lev_%A.err

hours=0
minutes=20
seconds=0
zip_time=10

EXPERIMENT="alexnet_cifar10"
HOME_DIR="links/scratch/armenta"

# Create output and error directories if they don't exist
mkdir -p $PWD/slurm_out
mkdir -p $PWD/slurm_err

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env_rorqual/bin/activate

# Prepare temp directories and weights
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

# If matrices.zip exists on permanent storage, copy and unzip into tmp
ZIP_FILE="$PWD/experiments/$EXPERIMENT/rejection_levels/matrices.zip"
if [ -f "$ZIP_FILE" ]; then
    echo "Found existing zip file: $ZIP_FILE"
    cp "$ZIP_FILE" "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/"
    cd "$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/"
    unzip -o matrices.zip
    echo "Unzipped existing matrices"
    cd -
fi

mkdir gpu-monitor
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

# start monitor in background
monitor_gpu &
MONITOR_PID=$!
echo "GPU monitor started in background (PID $MONITOR_PID)"

# Launch 4 workers (chunk_id 0..3), binding each to one GPU
TOTAL_CHUNKS=4
BATCH_SIZE=18816

PIDS=()
for CHUNK_ID in $(seq 0 $((TOTAL_CHUNKS-1))); do
    # Bind process to a single GPU index using CUDA_VISIBLE_DEVICES
    export CUDA_VISIBLE_DEVICES="$CHUNK_ID"
    echo "Starting worker for chunk $CHUNK_ID on CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
    python compute_matrices_for_rejection_level.py \
        --experiment_name $EXPERIMENT \
        --temp_dir $SLURM_TMPDIR \
        --batch_size $BATCH_SIZE \
        --chunk_id $CHUNK_ID \
        --total_chunks $TOTAL_CHUNKS \
        &
    PIDS+=($!)
    sleep 1  # small stagger
done

# compute sleep_time until we should start zipping (like before)
total_seconds=$((hours*3600 + minutes*60 + seconds))
sleep_time=$((total_seconds - zip_time*60))
if [ $sleep_time -lt 0 ]; then
    sleep_time=0
fi
echo "Sleeping for $sleep_time seconds"
sleep $sleep_time

# After the sleep window, check if workers are still alive and kill them (to ensure we start zipping)
for pid in "${PIDS[@]}"; do
    if ps -p $pid > /dev/null; then
        echo "Killing Python worker $pid"
        kill $pid
        wait $pid 2>/dev/null || true
    fi
done

# Stop GPU monitor
if ps -p $MONITOR_PID > /dev/null; then
    kill $MONITOR_PID
    wait $MONITOR_PID 2>/dev/null || true
fi

# Zip matrices directory
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
