#!/bin/bash
# ==============================================================
# run_experiment.sh — Slurm Pipeline Orchestrator
#
# Runs entire experiment pipelines with a single command.
# Supports checkpointing (audit), skip-audit, test mode, and
# multiple experiments.
#
# Pipeline: A(Train) -> B(Matrices),C(AdvExamples),D(RejLevel)
#           -> E(MatStats),F(AdvMatrices) -> G(GridSearch)
#
# Usage:
#   bash run_experiment.sh alexnet_cifar10
#   bash run_experiment.sh --test alexnet_cifar10
#   bash run_experiment.sh --skip-audit alexnet_cifar10 resnet_cifar10
#   bash run_experiment.sh --dry-run --test alexnet_cifar10
#
# Run from the project root directory on a login node.
# ==============================================================

set -euo pipefail

# --- Default configuration ---
ACCOUNT="def-assem"
TOTAL_CHUNKS=8
BATCH_SIZE=1800
NUM_SAMPLES_PER_CLASS=100
SAMPLES_PER_ATTACK=500
NUM_SAMPLES_REJECTION_LEVEL=10000
TEST_SIZE=-1
ENV_NAME="env"
DRY_RUN=false
SKIP_AUDIT=false
TEST_MODE=false
NO_CALIBRATE=false
MODULES="StdEnv/2023 python/3.11.5 scipy-stack/2025a"

# --- Resource profiles (normal mode) ---
# Step A
A_GPU="--gpus=H100-1g.10gb :1"
A_CPUS=2
A_TIME="06:00:00"
A_MEM="15G"
# Step B
B_GPU="--gpus=h100:1"
B_CPUS=12
B_TIME="00:20:00"
B_MEM="280G"
# Step C
C_GPU="--gres=gpu:1"
C_CPUS=16
C_TIME="04:00:00"
C_MEM="124G"
# Step D
D_GPU="--gpus=h100:1"
D_CPUS=16
D_TIME="09:00:00"
D_MEM="180G"
# Step E
E_CPUS=2
E_TIME="01:00:00"
E_MEM="8G"
# Step F
F_GPU="--gpus=h100:1"
F_CPUS=12
F_TIME="12:00:00"
F_MEM="280G"
# Step G
G_CPUS=64
G_TIME="48:00:00"
G_MEM_PER_CPU="42G"
# Audit
AUDIT_CPUS=4
AUDIT_TIME="02:00:00"
AUDIT_MEM="100G"
# Calibration
CALIB_GPU="--gpus=h100:1"
CALIB_CPUS=4
CALIB_TIME="00:30:00"
CALIB_MEM="32G"

# ==============================================================
# Argument parsing
# ==============================================================
EXPERIMENTS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --account)
            ACCOUNT="$2"; shift 2 ;;
        --total-chunks)
            TOTAL_CHUNKS="$2"; shift 2 ;;
        --batch-size)
            BATCH_SIZE="$2"; shift 2 ;;
        --samples-per-class)
            NUM_SAMPLES_PER_CLASS="$2"; shift 2 ;;
        --samples-per-attack)
            SAMPLES_PER_ATTACK="$2"; shift 2 ;;
        --samples-rejection-level)
            NUM_SAMPLES_REJECTION_LEVEL="$2"; shift 2 ;;
        --test-size)
            TEST_SIZE="$2"; shift 2 ;;
        --env)
            ENV_NAME="$2"; shift 2 ;;
        --dry-run)
            DRY_RUN=true; shift ;;
        --skip-audit)
            SKIP_AUDIT=true; shift ;;
        --test)
            TEST_MODE=true; shift ;;
        --no-calibrate)
            NO_CALIBRATE=true; shift ;;
        --help|-h)
            echo "Usage: bash run_experiment.sh [OPTIONS] experiment_name [experiment_name ...]"
            echo ""
            echo "Options:"
            echo "  --account ACCOUNT         Slurm account (default: def-assem)"
            echo "  --total-chunks N          Number of chunks for parallel jobs (default: 8)"
            echo "  --batch-size N            Matrix computation batch size (default: 1800)"
            echo "  --samples-per-class N     Samples per class for matrices (default: 100)"
            echo "  --samples-per-attack N    Adversarial samples per attack (default: 500)"
            echo "  --samples-rejection-level N  Rejection level samples (default: 10000)"
            echo "  --test-size N             Test set size for adv examples (default: -1 = all)"
            echo "  --env ENV_NAME            Virtual environment name (default: env_rorqual)"
            echo "  --dry-run                 Generate scripts but don't submit"
            echo "  --skip-audit              Skip audit, submit full pipeline directly"
            echo "  --test                    Test mode: small samples, short time limits"
            echo "  --no-calibrate            Skip GPU calibration, use hardcoded resource defaults"
            echo "  --help                    Show this help message"
            exit 0 ;;
        -*)
            echo "ERROR: Unknown option: $1"
            exit 1 ;;
        *)
            EXPERIMENTS+=("$1"); shift ;;
    esac
done

if [ ${#EXPERIMENTS[@]} -eq 0 ]; then
    echo "ERROR: No experiment names provided."
    echo "Usage: bash run_experiment.sh [OPTIONS] experiment_name [experiment_name ...]"
    echo "Run 'bash run_experiment.sh --help' for options."
    exit 1
fi

# ==============================================================
# Test mode overrides
# ==============================================================
if [ "$TEST_MODE" = "true" ]; then
    echo "[TEST MODE] Using small sample sizes and short time limits."
    TOTAL_CHUNKS=2
    BATCH_SIZE=100
    NUM_SAMPLES_PER_CLASS=10
    SAMPLES_PER_ATTACK=10
    NUM_SAMPLES_REJECTION_LEVEL=100
    TEST_SIZE=100
    # Shorter time limits
    A_TIME="00:10:00"
    A_MEM="8G"
    B_GPU="--gpus=h100:1"
    B_CPUS=4
    B_TIME="00:15:00"
    B_MEM="32G"
    C_GPU="--gres=gpu:1"
    C_CPUS=4
    C_TIME="00:30:00"
    C_MEM="32G"
    D_GPU="--gpus=h100:1"
    D_CPUS=4
    D_TIME="00:30:00"
    D_MEM="32G"
    E_CPUS=2
    E_TIME="00:15:00"
    E_MEM="4G"
    F_GPU="--gpus=h100:1"
    F_CPUS=4
    F_TIME="00:30:00"
    F_MEM="32G"
    G_CPUS=4
    G_TIME="01:00:00"
    G_MEM_PER_CPU="8G"
    AUDIT_CPUS=2
    AUDIT_TIME="00:30:00"
    AUDIT_MEM="16G"
fi

# ==============================================================
# Pre-flight checks
# ==============================================================
echo "=============================================================="
echo "  Pipeline Orchestrator — Pre-flight Checks"
echo "=============================================================="

# Check we're in the project root
if [ ! -f "constants/constants.py" ]; then
    echo "ERROR: Must run from the project root directory."
    echo "       Expected to find constants/constants.py"
    exit 1
fi

# Validate experiment names
echo ""
echo "Validating experiments..."
for EXP in "${EXPERIMENTS[@]}"; do
    python3 -c "
from constants.constants import DEFAULT_EXPERIMENTS
if '$EXP' not in DEFAULT_EXPERIMENTS:
    import sys
    print('ERROR: \"$EXP\" not found in DEFAULT_EXPERIMENTS (constants/constants.py)')
    print('Available experiments:', ', '.join(sorted(DEFAULT_EXPERIMENTS.keys())))
    sys.exit(1)
print('  OK: $EXP')
" || exit 1
done

# Check datasets
echo ""
echo "Checking datasets..."
python3 << 'DATASET_CHECK_EOF'
import sys
import os
sys.path.insert(0, '.')
from constants.constants import DEFAULT_EXPERIMENTS

experiments = os.environ.get("EXPERIMENT_LIST", "").split()
datasets_needed = set()
for exp in experiments:
    if exp in DEFAULT_EXPERIMENTS:
        ds = DEFAULT_EXPERIMENTS[exp].get('dataset', 'mnist')
        datasets_needed.add(ds)

dataset_dirs = {
    'cifar10': 'data/cifar-10-batches-py',
    'cifar100': 'data/cifar-100-python',
    'mnist': 'data/MNIST',
    'fashion': 'data/FashionMNIST',
    'imagenet': 'data/ILSVRC2012',
}

missing = []
for ds in datasets_needed:
    dir_path = dataset_dirs.get(ds)
    if dir_path and os.path.isdir(dir_path):
        print(f'  OK: {ds} ({dir_path})')
    elif ds == 'imagenet':
        # ImageNet is at /datashare/imagenet/ILSVRC2012/ on the cluster
        if os.path.isdir('/datashare/imagenet/ILSVRC2012'):
            print(f'  OK: {ds} (/datashare/imagenet/ILSVRC2012)')
        else:
            print(f'  DOWNLOAD NEEDED: {ds}')
            missing.append(ds)
    else:
        print(f'  DOWNLOAD NEEDED: {ds}')
        missing.append(ds)

if missing:
    print('\nDownloading missing datasets...')
    for ds in missing:
        if ds == 'imagenet':
            print('  ERROR: ImageNet must be available at /datashare/imagenet/ILSVRC2012/')
            sys.exit(1)
        elif ds == 'cifar10':
            from torchvision.datasets import CIFAR10
            CIFAR10(root='./data', train=True, download=True)
            CIFAR10(root='./data', train=False, download=True)
            print(f'  Downloaded: {ds}')
        elif ds == 'cifar100':
            from torchvision.datasets import CIFAR100
            CIFAR100(root='./data', train=True, download=True)
            CIFAR100(root='./data', train=False, download=True)
            print(f'  Downloaded: {ds}')
        elif ds == 'mnist':
            import torchvision
            torchvision.datasets.MNIST(root='./data', train=True, download=True)
            print(f'  Downloaded: {ds}')
        elif ds == 'fashion':
            import torchvision
            torchvision.datasets.FashionMNIST(root='./data', train=True, download=True)
            print(f'  Downloaded: {ds}')

print('Datasets ready.')
DATASET_CHECK_EOF
export EXPERIMENT_LIST="${EXPERIMENTS[*]}"

# Check pretrained weights
echo ""
echo "Checking pretrained weights..."
python3 << 'WEIGHTS_CHECK_EOF'
import sys
import os
import torch
sys.path.insert(0, '.')
from constants.constants import DEFAULT_EXPERIMENTS

experiments = os.environ.get("EXPERIMENT_LIST", "").split()
arch_map = {-3: 'alexnet', -2: 'resnet', -1: 'vgg'}
weight_paths = {
    'alexnet': 'experiments/alexnet_imagenet/weights/pretrained-weights.pth',
    'resnet': 'experiments/resnet_imagenet/weights/pretrained-weights.pth',
    'vgg': 'experiments/vgg_imagenet/weights/pretrained-weights.pth',
}

needed_archs = set()
for exp in experiments:
    if exp in DEFAULT_EXPERIMENTS:
        idx = DEFAULT_EXPERIMENTS[exp].get('architecture_index', 0)
        if idx in arch_map:
            needed_archs.add(arch_map[idx])

for arch in needed_archs:
    path = weight_paths[arch]
    if os.path.exists(path):
        print(f'  OK: {arch} pretrained weights ({path})')
    else:
        print(f'  DOWNLOAD NEEDED: {arch} pretrained weights')
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if arch == 'alexnet':
            from torchvision.models import alexnet, AlexNet_Weights
            model = alexnet(weights=AlexNet_Weights.DEFAULT)
            torch.save(model.state_dict(), path)
        elif arch == 'resnet':
            from torchvision.models import resnet18, ResNet18_Weights
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
            torch.save(model.state_dict(), path)
        elif arch == 'vgg':
            from torchvision.models import vgg11, VGG11_Weights
            model = vgg11(weights=VGG11_Weights.DEFAULT)
            torch.save(model.state_dict(), path)
        print(f'  Downloaded: {arch} pretrained weights -> {path}')

print('Pretrained weights ready.')
WEIGHTS_CHECK_EOF

echo ""
echo "Pre-flight checks complete."
echo ""

# ==============================================================
# Helper functions
# ==============================================================
submit_job() {
    local script="$1"
    local deps="$2"
    local sbatch_cmd="sbatch --parsable"
    if [ -n "$deps" ]; then
        sbatch_cmd="sbatch --parsable --dependency=afterok:${deps}"
    fi
    local job_id
    job_id=$($sbatch_cmd "$script")
    echo "$job_id"
}

# Determine dataset dirs to copy based on experiment
get_dataset_copy_commands() {
    local dataset="$1"
    case "$dataset" in
        cifar10)
            echo 'mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/ 2>/dev/null || true'
            ;;
        cifar100)
            echo 'mkdir -p $SLURM_TMPDIR/data/cifar-100-python/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/ 2>/dev/null || true'
            ;;
        mnist)
            echo 'mkdir -p $SLURM_TMPDIR/data/MNIST/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/MNIST/* $SLURM_TMPDIR/data/MNIST/ 2>/dev/null || true'
            ;;
        fashion)
            echo 'mkdir -p $SLURM_TMPDIR/data/FashionMNIST/'
            echo 'cp -r $SLURM_SUBMIT_DIR/data/FashionMNIST/* $SLURM_TMPDIR/data/FashionMNIST/ 2>/dev/null || true'
            ;;
        imagenet)
            echo 'mkdir -p $SLURM_TMPDIR/data/ILSVRC2012/'
            echo 'cp -r /datashare/imagenet/ILSVRC2012/* $SLURM_TMPDIR/data/ILSVRC2012/ 2>/dev/null || true'
            ;;
    esac
}

# ==============================================================
# Generate and submit pipeline for each experiment
# ==============================================================

# Get dataset for each experiment
get_experiment_dataset() {
    python3 -c "
from constants.constants import DEFAULT_EXPERIMENTS
print(DEFAULT_EXPERIMENTS.get('$1', {}).get('dataset', 'cifar10'))
"
}

submit_full_pipeline() {
    # Submits the full A->G pipeline for a single experiment.
    # Used by both --skip-audit and the dispatcher.
    local EXP="$1"
    local DEP_PREFIX="$2"  # optional dependency (e.g., audit job)
    local JOB_DIR="experiments/$EXP/orchestrator_jobs"
    local DATASET
    DATASET=$(get_experiment_dataset "$EXP")
    local COPY_DATA
    COPY_DATA=$(get_dataset_copy_commands "$DATASET")

    # Track job IDs
    local JOB_A="" JOB_B_IDS="" JOB_C="" JOB_D_IDS="" JOB_E="" JOB_F_IDS="" JOB_G=""

    # ==========================================================
    # Step A: Training
    # ==========================================================
    cat > "$JOB_DIR/step_A.sh" << STEPA_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH $A_GPU
#SBATCH --cpus-per-task=$A_CPUS
#SBATCH --time=$A_TIME
#SBATCH --mem=$A_MEM
#SBATCH --output=slurm_out/PIPE_A_${EXP}_%A.out
#SBATCH --error=slurm_err/PIPE_A_${EXP}_%A.err

mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate

$COPY_DATA

python training.py --experiment_name $EXP --temp_dir \$SLURM_TMPDIR
echo "Step A (training) complete for $EXP."
STEPA_EOF

    JOB_A=$(submit_job "$JOB_DIR/step_A.sh" "$DEP_PREFIX")
    echo "  [A] Training:            $JOB_A"

    # ==========================================================
    # Step B: Generate matrices (per chunk)
    # ==========================================================
    for CHUNK in $(seq 0 $((TOTAL_CHUNKS - 1))); do
        cat > "$JOB_DIR/step_B_chunk_${CHUNK}.sh" << STEPB_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH $B_GPU
#SBATCH --cpus-per-task=$B_CPUS
#SBATCH --time=$B_TIME
#SBATCH --mem=$B_MEM
#SBATCH --output=slurm_out/PIPE_B_${EXP}_c${CHUNK}_%A.out
#SBATCH --error=slurm_err/PIPE_B_${EXP}_c${CHUNK}_%A.err

mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate

EXPERIMENT="$EXP"
TASK_ID=$CHUNK
TEMP_DIR=\$SLURM_TMPDIR

$COPY_DATA

mkdir -p \$TEMP_DIR/experiments/\$EXPERIMENT/weights/
cp \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/weights/* \$TEMP_DIR/experiments/\$EXPERIMENT/weights/

ZIP_FILE=\$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/matrices_task_\$TASK_ID.zip
if [ -f "\$ZIP_FILE" ]; then
    cp "\$ZIP_FILE" "\$TEMP_DIR/experiments/\$EXPERIMENT/"
    unzip -o "\$TEMP_DIR/experiments/\$EXPERIMENT/matrices_task_\$TASK_ID.zip" -d "\$TEMP_DIR/experiments/\$EXPERIMENT/"
fi

BATCH_SIZE=$BATCH_SIZE
CALIB_FILE="\$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/calibration.json"
if [ -f "\$CALIB_FILE" ]; then
    BATCH_SIZE=\$(python3 -c "import json; print(json.load(open('\$CALIB_FILE'))['batch_size'])")
    echo "Using calibrated batch_size=\$BATCH_SIZE"
fi

# GPU monitoring
mkdir -p \$SLURM_SUBMIT_DIR/gpu-monitor/
GPU_LOGFILE="\$SLURM_SUBMIT_DIR/gpu-monitor/\$EXPERIMENT.B.\$TASK_ID.log"
monitor_gpu() {
  echo "Timestamp, GPU Util (%), Mem Used (MiB), Mem Total (MiB)" > "\$GPU_LOGFILE"
  while true; do
    ts=\$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -v t="\$ts" '{print t", "\$1", "\$2", "\$3}' >> "\$GPU_LOGFILE"
    sleep 30
  done
}
monitor_gpu &
MONITOR_PID=\$!

python generate_matrices.py --temp_dir \$TEMP_DIR --experiment \$EXPERIMENT --chunk_id \$TASK_ID --total_chunks $TOTAL_CHUNKS --batch_size \$BATCH_SIZE --num_samples_per_class $NUM_SAMPLES_PER_CLASS

kill \$MONITOR_PID 2>/dev/null || true

cd \$TEMP_DIR/experiments/\$EXPERIMENT
zip -r matrices_task_\$TASK_ID.zip matrices || { echo "Zipping failed"; exit 1; }

cd \$SLURM_SUBMIT_DIR
python -m utils.data_integrity --verify-zip \$TEMP_DIR/experiments/\$EXPERIMENT/matrices_task_\$TASK_ID.zip || { echo "Zip verification failed"; exit 1; }

cp \$TEMP_DIR/experiments/\$EXPERIMENT/matrices_task_\$TASK_ID.zip \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/ || { echo "Copy failed"; exit 1; }
echo "Step B chunk $CHUNK complete for $EXP."
STEPB_EOF

        JOB_ID=$(submit_job "$JOB_DIR/step_B_chunk_${CHUNK}.sh" "$JOB_A")
        JOB_B_IDS="${JOB_B_IDS:+$JOB_B_IDS:}$JOB_ID"
    done
    echo "  [B] Matrices (x$TOTAL_CHUNKS): ${JOB_B_IDS//:/, }"

    # ==========================================================
    # Step C: Adversarial examples
    # ==========================================================
    local C_TEST_SIZE_ARG=""
    if [ "$TEST_SIZE" != "-1" ]; then
        C_TEST_SIZE_ARG="--test_size $TEST_SIZE"
    fi

    cat > "$JOB_DIR/step_C.sh" << STEPC_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH $C_GPU
#SBATCH --cpus-per-task=$C_CPUS
#SBATCH --time=$C_TIME
#SBATCH --mem=$C_MEM
#SBATCH --output=slurm_out/PIPE_C_${EXP}_%A.out
#SBATCH --error=slurm_err/PIPE_C_${EXP}_%A.err

mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate

EXPERIMENT="$EXP"

$COPY_DATA

mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/weights/
cp \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/weights/* \$SLURM_TMPDIR/experiments/\$EXPERIMENT/weights/

python generate_adversarial_examples.py --experiment_name \$EXPERIMENT --temp_dir=\$SLURM_TMPDIR $C_TEST_SIZE_ARG

# Copy results back
mkdir -p \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/adversarial_examples/
cp -r \$SLURM_TMPDIR/experiments/\$EXPERIMENT/adversarial_examples/* \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/adversarial_examples/ 2>/dev/null || true
echo "Step C (adversarial examples) complete for $EXP."
STEPC_EOF

    JOB_C=$(submit_job "$JOB_DIR/step_C.sh" "$JOB_A")
    echo "  [C] Adv examples:        $JOB_C"

    # ==========================================================
    # Step D: Rejection level matrices (per chunk)
    # ==========================================================
    for CHUNK in $(seq 0 $((TOTAL_CHUNKS - 1))); do
        cat > "$JOB_DIR/step_D_chunk_${CHUNK}.sh" << STEPD_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH $D_GPU
#SBATCH --cpus-per-task=$D_CPUS
#SBATCH --time=$D_TIME
#SBATCH --mem=$D_MEM
#SBATCH --output=slurm_out/PIPE_D_${EXP}_c${CHUNK}_%A.out
#SBATCH --error=slurm_err/PIPE_D_${EXP}_c${CHUNK}_%A.err

mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate

EXPERIMENT="$EXP"
TASK_ID=$CHUNK
ZIP_OUTPUT_FILE="matrices_task_\$TASK_ID.zip"

$COPY_DATA

mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/weights/
cp \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/weights/* \$SLURM_TMPDIR/experiments/\$EXPERIMENT/weights/

mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/

# Copy existing rejection level data if present
for F in exp_dataset_train.pth exp_dataset_labels.pth; do
    SRC="\$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/rejection_levels/\$F"
    [ -f "\$SRC" ] && cp "\$SRC" "\$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/"
done

ZIP_FILE="\$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/rejection_levels/\$ZIP_OUTPUT_FILE"
if [ -f "\$ZIP_FILE" ]; then
    cp "\$ZIP_FILE" "\$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/"
    cd "\$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/"
    unzip -o "\$ZIP_OUTPUT_FILE"
    cd -
fi

BATCH_SIZE=$BATCH_SIZE
CALIB_FILE="\$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/calibration.json"
if [ -f "\$CALIB_FILE" ]; then
    BATCH_SIZE=\$(python3 -c "import json; print(json.load(open('\$CALIB_FILE'))['batch_size'])")
    echo "Using calibrated batch_size=\$BATCH_SIZE"
fi

# GPU monitoring
mkdir -p \$SLURM_SUBMIT_DIR/gpu-monitor/
GPU_LOGFILE="\$SLURM_SUBMIT_DIR/gpu-monitor/\$EXPERIMENT.D.\$TASK_ID.log"
monitor_gpu() {
  echo "Timestamp, GPU Util (%), Mem Used (MiB), Mem Total (MiB)" > "\$GPU_LOGFILE"
  while true; do
    ts=\$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -v t="\$ts" '{print t", "\$1", "\$2", "\$3}' >> "\$GPU_LOGFILE"
    sleep 30
  done
}
monitor_gpu &
MONITOR_PID=\$!

python compute_matrices_for_rejection_level.py \
    --experiment_name \$EXPERIMENT \
    --temp_dir \$SLURM_TMPDIR \
    --batch_size \$BATCH_SIZE \
    --chunk_id \$TASK_ID \
    --total_chunks $TOTAL_CHUNKS \
    --num_samples_rejection_level $NUM_SAMPLES_REJECTION_LEVEL

kill \$MONITOR_PID 2>/dev/null || true

MATRICES_DIR="\$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/matrices"
ZIP_OUTPUT_DIR="\$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels"

if [ -d "\$MATRICES_DIR" ]; then
    cd "\$ZIP_OUTPUT_DIR"
    zip -r "\$ZIP_OUTPUT_FILE" matrices || { echo "Zip failed"; exit 1; }

    cd \$SLURM_SUBMIT_DIR
    python -m utils.data_integrity --verify-zip "\$ZIP_OUTPUT_DIR/\$ZIP_OUTPUT_FILE" || { echo "Zip verification failed"; exit 1; }
fi

# Copy results and dataset files back
DEST_DIR="\$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/rejection_levels/"
mkdir -p "\$DEST_DIR"
[ -f "\$ZIP_OUTPUT_DIR/\$ZIP_OUTPUT_FILE" ] && cp "\$ZIP_OUTPUT_DIR/\$ZIP_OUTPUT_FILE" "\$DEST_DIR"
for F in exp_dataset_train.pth exp_dataset_labels.pth; do
    SRC="\$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/\$F"
    [ -f "\$SRC" ] && cp "\$SRC" "\$DEST_DIR"
done
echo "Step D chunk $CHUNK complete for $EXP."
STEPD_EOF

        JOB_ID=$(submit_job "$JOB_DIR/step_D_chunk_${CHUNK}.sh" "$JOB_A")
        JOB_D_IDS="${JOB_D_IDS:+$JOB_D_IDS:}$JOB_ID"
    done
    echo "  [D] Rejection levels (x$TOTAL_CHUNKS): ${JOB_D_IDS//:/, }"

    # ==========================================================
    # Step E: Matrix statistics (depends on all B jobs)
    # ==========================================================
    cat > "$JOB_DIR/step_E.sh" << STEPE_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --cpus-per-task=$E_CPUS
#SBATCH --time=$E_TIME
#SBATCH --mem-per-cpu=$E_MEM
#SBATCH --output=slurm_out/PIPE_E_${EXP}_%A.out
#SBATCH --error=slurm_err/PIPE_E_${EXP}_%A.err

mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate

EXPERIMENT="$EXP"

mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/matrices/

for i in \$(seq 0 $((TOTAL_CHUNKS - 1))); do
    cp \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/matrices_task_\$i.zip \$SLURM_TMPDIR/experiments/\$EXPERIMENT/
    unzip -o "\$SLURM_TMPDIR/experiments/\$EXPERIMENT/matrices_task_\$i.zip" -d "\$SLURM_TMPDIR/experiments/\$EXPERIMENT/"
done

echo "Matrices unzipped. Computing statistics..."
python compute_matrix_statistics.py --experiment_name \$EXPERIMENT --temp_dir \$SLURM_TMPDIR

# Copy result back
cp \$SLURM_TMPDIR/experiments/\$EXPERIMENT/matrices/matrix_statistics.json \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/matrices/ 2>/dev/null || true
echo "Step E (matrix statistics) complete for $EXP."
STEPE_EOF

    JOB_E=$(submit_job "$JOB_DIR/step_E.sh" "$JOB_B_IDS")
    echo "  [E] Matrix statistics:   $JOB_E"

    # ==========================================================
    # Step F: Adversarial matrices (per chunk, depends on C)
    # ==========================================================
    for CHUNK in $(seq 0 $((TOTAL_CHUNKS - 1))); do
        cat > "$JOB_DIR/step_F_chunk_${CHUNK}.sh" << STEPF_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH $F_GPU
#SBATCH --cpus-per-task=$F_CPUS
#SBATCH --time=$F_TIME
#SBATCH --mem=$F_MEM
#SBATCH --output=slurm_out/PIPE_F_${EXP}_c${CHUNK}_%A.out
#SBATCH --error=slurm_err/PIPE_F_${EXP}_c${CHUNK}_%A.err

mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate

EXPERIMENT="$EXP"
TASK_ID=$CHUNK

$COPY_DATA

mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/weights/
cp \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/weights/* \$SLURM_TMPDIR/experiments/\$EXPERIMENT/weights/

mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/adversarial_examples/
tar cf - -C \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/adversarial_examples . | tar xf - -C \$SLURM_TMPDIR/experiments/\$EXPERIMENT/adversarial_examples

ZIP_FILE="\$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/adv_matrices_task_\$TASK_ID.zip"
if [ -f "\$ZIP_FILE" ]; then
    cp "\$ZIP_FILE" "\$SLURM_TMPDIR/experiments/\$EXPERIMENT/"
    unzip -o "\$SLURM_TMPDIR/experiments/\$EXPERIMENT/adv_matrices_task_\$TASK_ID.zip" -d "\$SLURM_TMPDIR/experiments/\$EXPERIMENT/"
fi

BATCH_SIZE=$BATCH_SIZE
CALIB_FILE="\$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/calibration.json"
if [ -f "\$CALIB_FILE" ]; then
    BATCH_SIZE=\$(python3 -c "import json; print(json.load(open('\$CALIB_FILE'))['batch_size'])")
    echo "Using calibrated batch_size=\$BATCH_SIZE"
fi

# GPU monitoring
mkdir -p \$SLURM_SUBMIT_DIR/gpu-monitor/
GPU_LOGFILE="\$SLURM_SUBMIT_DIR/gpu-monitor/\$EXPERIMENT.F.\$TASK_ID.log"
monitor_gpu() {
  echo "Timestamp, GPU Util (%), Mem Used (MiB), Mem Total (MiB)" > "\$GPU_LOGFILE"
  while true; do
    ts=\$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -v t="\$ts" '{print t", "\$1", "\$2", "\$3}' >> "\$GPU_LOGFILE"
    sleep 30
  done
}
monitor_gpu &
MONITOR_PID=\$!

python generate_adversarial_matrices.py \
    --experiment_name \$EXPERIMENT \
    --temp_dir \$SLURM_TMPDIR \
    --chunk_id \$TASK_ID \
    --total_chunks $TOTAL_CHUNKS \
    --batch_size \$BATCH_SIZE \
    --samples_per_attack $SAMPLES_PER_ATTACK

kill \$MONITOR_PID 2>/dev/null || true

cd \$SLURM_TMPDIR/experiments/\$EXPERIMENT/
zip -r adv_matrices_task_\$TASK_ID.zip adversarial_matrices/ || { echo "Zipping failed"; exit 1; }

cd \$SLURM_SUBMIT_DIR
python -m utils.data_integrity --verify-zip \$SLURM_TMPDIR/experiments/\$EXPERIMENT/adv_matrices_task_\$TASK_ID.zip || { echo "Zip verification failed"; exit 1; }

cp \$SLURM_TMPDIR/experiments/\$EXPERIMENT/adv_matrices_task_\$TASK_ID.zip \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/ || { echo "Copy failed"; exit 1; }
echo "Step F chunk $CHUNK complete for $EXP."
STEPF_EOF

        JOB_ID=$(submit_job "$JOB_DIR/step_F_chunk_${CHUNK}.sh" "$JOB_C")
        JOB_F_IDS="${JOB_F_IDS:+$JOB_F_IDS:}$JOB_ID"
    done
    echo "  [F] Adv matrices (x$TOTAL_CHUNKS): ${JOB_F_IDS//:/, }"

    # ==========================================================
    # Step G: Grid search (depends on E + all F + all D)
    # ==========================================================
    cat > "$JOB_DIR/step_G.sh" << STEPG_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --cpus-per-task=$G_CPUS
#SBATCH --time=$G_TIME
#SBATCH --mem-per-cpu=$G_MEM_PER_CPU
#SBATCH --output=slurm_out/PIPE_G_${EXP}_%A.out
#SBATCH --error=slurm_err/PIPE_G_${EXP}_%A.err

mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

EXPERIMENT="$EXP"

$COPY_DATA

mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/weights/
cp \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/weights/* \$SLURM_TMPDIR/experiments/\$EXPERIMENT/weights/

# Unzip matrices for statistics
mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/matrices/
cp \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/matrices/matrix_statistics.json \$SLURM_TMPDIR/experiments/\$EXPERIMENT/matrices/

# Unzip adversarial matrices
mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/adversarial_matrices/
for i in \$(seq 0 $((TOTAL_CHUNKS - 1))); do
    if [ -f "\$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/adv_matrices_task_\$i.zip" ]; then
        cp \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/adv_matrices_task_\$i.zip \$SLURM_TMPDIR/experiments/\$EXPERIMENT/
        unzip -o \$SLURM_TMPDIR/experiments/\$EXPERIMENT/adv_matrices_task_\$i.zip -d \$SLURM_TMPDIR/experiments/\$EXPERIMENT/
    fi
done

# Copy adversarial examples
mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/adversarial_examples/
cp -r \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/adversarial_examples/* \$SLURM_TMPDIR/experiments/\$EXPERIMENT/adversarial_examples/ 2>/dev/null || true

# Unzip rejection level matrices
mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/matrices/
cp -r \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/rejection_levels/* \$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/ 2>/dev/null || true

for i in \$(seq 0 $((TOTAL_CHUNKS - 1))); do
    if [ -f "\$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/matrices_task_\$i.zip" ]; then
        unzip -o \$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/matrices_task_\$i.zip -d \$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/
    fi
done

echo "All data ready. Starting grid search..."
python grid_search.py --nb_workers \$SLURM_CPUS_PER_TASK --experiment_name \$EXPERIMENT --temp_dir \$SLURM_TMPDIR --rej_lev 0

# Copy results back
mkdir -p \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/grid_search/
cp -r \$SLURM_TMPDIR/experiments/\$EXPERIMENT/grid_search/* \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/grid_search/ 2>/dev/null || true
cp -r \$SLURM_TMPDIR/experiments/\$EXPERIMENT/rejection_levels/reject_at_* \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/rejection_levels/ 2>/dev/null || true
echo "Step G (grid search) complete for $EXP."
STEPG_EOF

    # Build G dependencies: E + all F + all D
    local G_DEPS="$JOB_E"
    G_DEPS="${G_DEPS:+$G_DEPS:}$JOB_F_IDS"
    G_DEPS="${G_DEPS:+$G_DEPS:}$JOB_D_IDS"

    JOB_G=$(submit_job "$JOB_DIR/step_G.sh" "$G_DEPS")
    echo "  [G] Grid search:         $JOB_G"

    # ==========================================================
    # Final audit
    # ==========================================================
    cat > "$JOB_DIR/final_audit.sh" << FINALAUDIT_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --cpus-per-task=$AUDIT_CPUS
#SBATCH --time=$AUDIT_TIME
#SBATCH --mem=$AUDIT_MEM
#SBATCH --output=slurm_out/PIPE_AUDIT_${EXP}_%A.out
#SBATCH --error=slurm_err/PIPE_AUDIT_${EXP}_%A.err

mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate

EXPERIMENT="$EXP"
TOTAL_CHUNKS=$TOTAL_CHUNKS

mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/
cp -r \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/* \$SLURM_TMPDIR/experiments/\$EXPERIMENT/

cd \$SLURM_SUBMIT_DIR

python << 'AUDIT_PY_EOF'
import sys, os, json
sys.path.insert(0, os.environ["SLURM_SUBMIT_DIR"])
from utils.data_integrity import verify_experiment, _print_report
from constants.constants import ATTACKS, DEFAULT_EXPERIMENTS

experiment = os.environ["EXPERIMENT"]
total_chunks = int(os.environ["TOTAL_CHUNKS"])
tmpdir = os.environ["SLURM_TMPDIR"]
experiment_dir = os.path.join(tmpdir, "experiments", experiment)

num_classes = 10
if experiment in DEFAULT_EXPERIMENTS:
    dataset = DEFAULT_EXPERIMENTS[experiment].get("dataset", "cifar10")
    if dataset == "cifar100": num_classes = 100
    elif dataset == "imagenet": num_classes = 1000

report = verify_experiment(
    experiment_dir=experiment_dir, experiment_name=experiment,
    num_classes=num_classes, num_samples_per_class=1000,
    total_chunks=total_chunks, num_samples_rejection_level=10000,
    attacks_list=ATTACKS, sample_ratio=1.0,
)
_print_report(report)

report_path = os.path.join(experiment_dir, "audit_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"Final audit report saved to: {report_path}")
AUDIT_PY_EOF

cp \$SLURM_TMPDIR/experiments/\$EXPERIMENT/audit_report.json \
   \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/audit_report.json 2>/dev/null || true
echo "Final audit complete for $EXP."
FINALAUDIT_EOF

    local FINAL_AUDIT_JOB
    FINAL_AUDIT_JOB=$(submit_job "$JOB_DIR/final_audit.sh" "$JOB_G")
    echo "  [*] Final audit:         $FINAL_AUDIT_JOB"
    echo ""
}

# ==============================================================
# Main: process each experiment
# ==============================================================
echo "=============================================================="
echo "  Pipeline Orchestrator — Submission"
echo "=============================================================="

mkdir -p slurm_out slurm_err

for EXP in "${EXPERIMENTS[@]}"; do
    JOB_DIR="experiments/$EXP/orchestrator_jobs"
    mkdir -p "$JOB_DIR"

    echo ""
    echo "--- $EXP ---"

    # --- Generate calibration job if needed ---
    CALIB_DEP=""
    if [ "$NO_CALIBRATE" = "false" ]; then
        DATASET_FOR_CALIB=$(get_experiment_dataset "$EXP")
        CALIB_COPY_DATA=$(get_dataset_copy_commands "$DATASET_FOR_CALIB")

        cat > "$JOB_DIR/calibrate.sh" << CALIB_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH $CALIB_GPU
#SBATCH --cpus-per-task=$CALIB_CPUS
#SBATCH --time=$CALIB_TIME
#SBATCH --mem=$CALIB_MEM
#SBATCH --output=slurm_out/PIPE_CALIB_${EXP}_%A.out
#SBATCH --error=slurm_err/PIPE_CALIB_${EXP}_%A.err

mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate

$CALIB_COPY_DATA

# GPU monitoring during calibration
mkdir -p \$SLURM_SUBMIT_DIR/gpu-monitor/
GPU_LOGFILE="\$SLURM_SUBMIT_DIR/gpu-monitor/$EXP.calibration.log"
monitor_gpu() {
  echo "Timestamp, GPU Util (%), Mem Used (MiB), Mem Total (MiB)" > "\$GPU_LOGFILE"
  while true; do
    ts=\$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -v t="\$ts" '{print t", "\$1", "\$2", "\$3}' >> "\$GPU_LOGFILE"
    sleep 30
  done
}
monitor_gpu &
MONITOR_PID=\$!

python calibrate.py \\
    --experiment_name $EXP \\
    --temp_dir \$SLURM_TMPDIR \\
    --target_utilization 0.93 \\
    --timing_samples 50 \\
    --total_chunks $TOTAL_CHUNKS \\
    --num_samples_per_class $NUM_SAMPLES_PER_CLASS \\
    --samples_per_attack $SAMPLES_PER_ATTACK \\
    --num_samples_rejection_level $NUM_SAMPLES_REJECTION_LEVEL

kill \$MONITOR_PID 2>/dev/null || true
echo "Calibration complete for $EXP."
CALIB_EOF

        if [ "$DRY_RUN" = "false" ]; then
            # Check if calibration.json already exists on login node
            if [ -f "experiments/$EXP/calibration.json" ]; then
                echo "  [0] Calibration:         SKIPPED (calibration.json exists)"
            else
                CALIB_JOB=$(submit_job "$JOB_DIR/calibrate.sh" "")
                CALIB_DEP="$CALIB_JOB"
                echo "  [0] Calibration:         $CALIB_JOB"
            fi
        else
            echo "  [DRY RUN] Calibration script generated: $JOB_DIR/calibrate.sh"
        fi
    fi

    # --- Load calibrated resource profiles if available ---
    CALIB_FILE="experiments/$EXP/calibration.json"
    if [ -f "$CALIB_FILE" ] && [ "$NO_CALIBRATE" = "false" ]; then
        echo "  Loading calibrated resources from $CALIB_FILE"
        A_TIME=$(python3 -c "import json; print(json.load(open('$CALIB_FILE'))['slurm_resources']['A']['time'])")
        A_MEM=$(python3 -c "import json; print(json.load(open('$CALIB_FILE'))['slurm_resources']['A']['mem'])")
        B_TIME=$(python3 -c "import json; print(json.load(open('$CALIB_FILE'))['slurm_resources']['B']['time'])")
        B_MEM=$(python3 -c "import json; print(json.load(open('$CALIB_FILE'))['slurm_resources']['B']['mem'])")
        D_TIME=$(python3 -c "import json; print(json.load(open('$CALIB_FILE'))['slurm_resources']['D']['time'])")
        D_MEM=$(python3 -c "import json; print(json.load(open('$CALIB_FILE'))['slurm_resources']['D']['mem'])")
        F_TIME=$(python3 -c "import json; print(json.load(open('$CALIB_FILE'))['slurm_resources']['F']['time'])")
        F_MEM=$(python3 -c "import json; print(json.load(open('$CALIB_FILE'))['slurm_resources']['F']['mem'])")
        BATCH_SIZE=$(python3 -c "import json; print(json.load(open('$CALIB_FILE'))['batch_size'])")
        echo "    A: time=$A_TIME mem=$A_MEM"
        echo "    B: time=$B_TIME mem=$B_MEM  batch_size=$BATCH_SIZE"
        echo "    D: time=$D_TIME mem=$D_MEM"
        echo "    F: time=$F_TIME mem=$F_MEM"
    fi

    if [ "$SKIP_AUDIT" = "true" ]; then
        # --skip-audit: submit full pipeline directly from login node
        if [ "$DRY_RUN" = "true" ]; then
            echo "  [DRY RUN] Would submit full pipeline for $EXP"
            # Still generate scripts for inspection
            submit_full_pipeline "$EXP" "" 2>/dev/null || true
            echo "  Scripts generated in: $JOB_DIR/"
        else
            submit_full_pipeline "$EXP" "$CALIB_DEP"
        fi
    else
        # Normal mode: audit -> dispatcher -> pipeline

        # --- Generate audit script ---
        cat > "$JOB_DIR/audit.sh" << AUDIT_EOF
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --cpus-per-task=$AUDIT_CPUS
#SBATCH --time=$AUDIT_TIME
#SBATCH --mem=$AUDIT_MEM
#SBATCH --output=slurm_out/PIPE_PREAUDIT_${EXP}_%A.out
#SBATCH --error=slurm_err/PIPE_PREAUDIT_${EXP}_%A.err

mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate

EXPERIMENT="$EXP"
TOTAL_CHUNKS=$TOTAL_CHUNKS

echo "Copying experiment data to temporary directory..."
mkdir -p \$SLURM_TMPDIR/experiments/\$EXPERIMENT/
cp -r \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/* \$SLURM_TMPDIR/experiments/\$EXPERIMENT/ 2>/dev/null || true
echo "Copy complete."

cd \$SLURM_SUBMIT_DIR

# --- Run audit ---
python << 'AUDIT_PY_EOF'
import sys, os, json
sys.path.insert(0, os.environ["SLURM_SUBMIT_DIR"])
from utils.data_integrity import verify_experiment, _print_report
from constants.constants import ATTACKS, DEFAULT_EXPERIMENTS

experiment = os.environ["EXPERIMENT"]
total_chunks = int(os.environ["TOTAL_CHUNKS"])
tmpdir = os.environ["SLURM_TMPDIR"]
experiment_dir = os.path.join(tmpdir, "experiments", experiment)

num_classes = 10
if experiment in DEFAULT_EXPERIMENTS:
    dataset = DEFAULT_EXPERIMENTS[experiment].get("dataset", "cifar10")
    if dataset == "cifar100": num_classes = 100
    elif dataset == "imagenet": num_classes = 1000

report = verify_experiment(
    experiment_dir=experiment_dir, experiment_name=experiment,
    num_classes=num_classes, num_samples_per_class=1000,
    total_chunks=total_chunks, num_samples_rejection_level=10000,
    attacks_list=ATTACKS, sample_ratio=1.0,
)
_print_report(report)

report_path = os.path.join(experiment_dir, "audit_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"JSON report saved to: {report_path}")
AUDIT_PY_EOF

echo "Audit complete. Generating recovery plan..."

# --- Generate recovery plan ---
python << 'RECOVERY_PLAN_PY_EOF'
import sys, os, json
from datetime import datetime
sys.path.insert(0, os.environ["SLURM_SUBMIT_DIR"])

experiment = os.environ["EXPERIMENT"]
total_chunks = int(os.environ["TOTAL_CHUNKS"])
tmpdir = os.environ["SLURM_TMPDIR"]
submit_dir = os.environ["SLURM_SUBMIT_DIR"]

experiment_dir = os.path.join(tmpdir, "experiments", experiment)
report_path = os.path.join(experiment_dir, "audit_report.json")

with open(report_path, "r") as f:
    report = json.load(f)

now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
summary = report["summary"]

# --- Determine per-step failures ---
weights_status = report["steps"]["weights"]["status"]
recover_a = weights_status != "OK"
reason_a = [f"weights: {weights_status}"] if recover_a else []

recover_b = False; recover_b_chunks = []; reason_b = []
for i, entry in enumerate(report["steps"]["matrices_zips"]):
    if entry["status"] != "OK":
        recover_b = True
        recover_b_chunks.append(str(i))
        reason_b.append(f"matrices_task_{i}.zip: {entry['status']}")

recover_c = False; reason_c = []
for entry in report["steps"]["adversarial_examples"]:
    if entry["status"] != "OK":
        recover_c = True
        fname = os.path.basename(os.path.dirname(entry["path"]))
        reason_c.append(f"adversarial_examples/{fname}: {entry['status']}")

recover_d = False; recover_d_chunks = []; reason_d = []
for i, entry in enumerate(report["steps"]["rejection_level_zips"]):
    if entry["status"] != "OK":
        recover_d = True
        recover_d_chunks.append(str(i))
        reason_d.append(f"rejection_levels/matrices_task_{i}.zip: {entry['status']}")

mat_stats = report["steps"]["matrix_statistics"]
recover_e = mat_stats["status"] != "OK"
reason_e = [f"matrix_statistics.json: {mat_stats['status']}"] if recover_e else []

recover_f = False; recover_f_chunks = []; reason_f = []
for i, entry in enumerate(report["steps"]["adv_matrices_zips"]):
    if entry["status"] != "OK":
        recover_f = True
        recover_f_chunks.append(str(i))
        reason_f.append(f"adv_matrices_task_{i}.zip: {entry['status']}")

grid_status = report["steps"]["grid_search"]["status"]
recover_g = grid_status != "OK"
reason_g = [f"grid_search: {grid_status}"] if recover_g else []

for entry in report["steps"].get("rejection_level_jsons", []):
    if entry["status"] != "OK":
        recover_g = True
        reason_g.append(f"rejection_level_jsons: {entry['status']}")

# --- Propagate dependencies ---
if recover_a:
    if not recover_b:
        recover_b = True; recover_b_chunks = [str(i) for i in range(total_chunks)]
        reason_b.append("Propagated: depends on Step A")
    if not recover_c:
        recover_c = True; reason_c.append("Propagated: depends on Step A")
    if not recover_d:
        recover_d = True; recover_d_chunks = [str(i) for i in range(total_chunks)]
        reason_d.append("Propagated: depends on Step A")
if recover_b and not recover_e:
    recover_e = True; reason_e.append("Propagated: depends on Step B")
if recover_c and not recover_f:
    recover_f = True; recover_f_chunks = [str(i) for i in range(total_chunks)]
    reason_f.append("Propagated: depends on Step C")
if (recover_e or recover_f or recover_d) and not recover_g:
    recover_g = True; deps = []
    if recover_e: deps.append("E")
    if recover_f: deps.append("F")
    if recover_d: deps.append("D")
    reason_g.append(f"Propagated: depends on {'+'.join(deps)}")

recovery_needed = any([recover_a, recover_b, recover_c, recover_d, recover_e, recover_f, recover_g])

# --- Write recovery_plan.sh ---
plan_path = os.path.join(submit_dir, "experiments", experiment, "recovery_plan.sh")
os.makedirs(os.path.dirname(plan_path), exist_ok=True)

with open(plan_path, "w") as f:
    f.write("#!/bin/bash\n")
    f.write(f"# Recovery plan for: {experiment} (generated {now})\n")
    f.write(f'RECOVERY_EXPERIMENT="{experiment}"\n')
    f.write(f'RECOVERY_TOTAL_CHUNKS={total_chunks}\n')
    f.write(f'RECOVERY_GENERATED_AT="{now}"\n\n')

    def write_step(f, name, label, recover, reasons, chunks=None):
        f.write(f"RECOVER_STEP_{name}={'true' if recover else 'false'}\n")
        if chunks is not None:
            f.write(f'RECOVER_STEP_{name}_CHUNKS="{" ".join(chunks) if recover else ""}"\n')
        for r in reasons:
            f.write(f"#   {r}\n")
        f.write("\n")

    write_step(f, "A", "Training", recover_a, reason_a)
    write_step(f, "B", "Generate matrices", recover_b, reason_b, recover_b_chunks)
    write_step(f, "C", "Adversarial examples", recover_c, reason_c)
    write_step(f, "D", "Rejection level matrices", recover_d, reason_d, recover_d_chunks)
    write_step(f, "E", "Matrix statistics", recover_e, reason_e)
    write_step(f, "F", "Adversarial matrices", recover_f, reason_f, recover_f_chunks)
    write_step(f, "G", "Grid search", recover_g, reason_g)
    f.write(f"RECOVERY_NEEDED={'true' if recovery_needed else 'false'}\n")

print(f"Recovery plan saved to: {plan_path}")
if recovery_needed:
    print("RECOVERY NEEDED.")
else:
    print("No recovery needed - all artifacts OK.")
RECOVERY_PLAN_PY_EOF

cp \$SLURM_TMPDIR/experiments/\$EXPERIMENT/audit_report.json \
   \$SLURM_SUBMIT_DIR/experiments/\$EXPERIMENT/audit_report.json 2>/dev/null || true
echo "Audit job complete for $EXP."
AUDIT_EOF

        # --- Generate dispatcher script ---
        cat > "$JOB_DIR/dispatch.sh" << 'DISPATCH_HEADER'
#!/bin/bash
#SBATCH --account=__ACCOUNT__
#SBATCH --cpus-per-task=1
#SBATCH --time=00:15:00
#SBATCH --mem=1G
#SBATCH --output=slurm_out/PIPE_DISPATCH___EXP___%A.out
#SBATCH --error=slurm_err/PIPE_DISPATCH___EXP___%A.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err

EXPERIMENT="__EXP__"
ACCOUNT="__ACCOUNT__"
TOTAL_CHUNKS=__TOTAL_CHUNKS__
BATCH_SIZE=__BATCH_SIZE__
NUM_SAMPLES_PER_CLASS=__NUM_SAMPLES_PER_CLASS__
SAMPLES_PER_ATTACK=__SAMPLES_PER_ATTACK__
NUM_SAMPLES_REJECTION_LEVEL=__NUM_SAMPLES_REJECTION_LEVEL__
TEST_SIZE="__TEST_SIZE__"
ENV_NAME="__ENV_NAME__"
MODULES="__MODULES__"

# Resource profiles
A_GPU="__A_GPU__"
A_CPUS=__A_CPUS__
A_TIME="__A_TIME__"
A_MEM="__A_MEM__"
B_GPU="__B_GPU__"
B_CPUS=__B_CPUS__
B_TIME="__B_TIME__"
B_MEM="__B_MEM__"
C_GPU="__C_GPU__"
C_CPUS=__C_CPUS__
C_TIME="__C_TIME__"
C_MEM="__C_MEM__"
D_GPU="__D_GPU__"
D_CPUS=__D_CPUS__
D_TIME="__D_TIME__"
D_MEM="__D_MEM__"
E_CPUS=__E_CPUS__
E_TIME="__E_TIME__"
E_MEM="__E_MEM__"
F_GPU="__F_GPU__"
F_CPUS=__F_CPUS__
F_TIME="__F_TIME__"
F_MEM="__F_MEM__"
G_CPUS=__G_CPUS__
G_TIME="__G_TIME__"
G_MEM_PER_CPU="__G_MEM_PER_CPU__"
AUDIT_CPUS=__AUDIT_CPUS__
AUDIT_TIME="__AUDIT_TIME__"
AUDIT_MEM="__AUDIT_MEM__"
DISPATCH_HEADER
        # Append the dispatcher body
        cat >> "$JOB_DIR/dispatch.sh" << 'DISPATCH_BODY'

# --- Source recovery plan ---
PLAN_FILE="$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/recovery_plan.sh"
if [ ! -f "$PLAN_FILE" ]; then
    echo "ERROR: Recovery plan not found: $PLAN_FILE"
    exit 1
fi
source "$PLAN_FILE"

if [ "$RECOVERY_NEEDED" = "false" ]; then
    echo "No recovery needed for $EXPERIMENT. Pipeline complete."
    exit 0
fi

echo "=============================================================="
echo "  Dispatcher: submitting pipeline for $EXPERIMENT"
echo "=============================================================="

JOB_DIR="$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/orchestrator_jobs/recovery"
mkdir -p "$JOB_DIR"

submit_job() {
    local script="$1"
    local deps="$2"
    local sbatch_cmd="sbatch --parsable"
    if [ -n "$deps" ]; then
        sbatch_cmd="sbatch --parsable --dependency=afterok:${deps}"
    fi
    local job_id
    job_id=$($sbatch_cmd "$script")
    echo "$job_id"
}

# Get dataset copy commands
get_dataset_copy() {
    python3 -c "
import sys; sys.path.insert(0, '$SLURM_SUBMIT_DIR')
from constants.constants import DEFAULT_EXPERIMENTS
ds = DEFAULT_EXPERIMENTS.get('$EXPERIMENT', {}).get('dataset', 'cifar10')
cmds = {
    'cifar10': 'mkdir -p \$SLURM_TMPDIR/data/cifar-10-batches-py/\ncp -r \$SLURM_SUBMIT_DIR/data/cifar-10-batches-py/* \$SLURM_TMPDIR/data/cifar-10-batches-py/ 2>/dev/null || true',
    'cifar100': 'mkdir -p \$SLURM_TMPDIR/data/cifar-100-python/\ncp -r \$SLURM_SUBMIT_DIR/data/cifar-100-python/* \$SLURM_TMPDIR/data/cifar-100-python/ 2>/dev/null || true',
    'mnist': 'mkdir -p \$SLURM_TMPDIR/data/MNIST/\ncp -r \$SLURM_SUBMIT_DIR/data/MNIST/* \$SLURM_TMPDIR/data/MNIST/ 2>/dev/null || true',
    'fashion': 'mkdir -p \$SLURM_TMPDIR/data/FashionMNIST/\ncp -r \$SLURM_SUBMIT_DIR/data/FashionMNIST/* \$SLURM_TMPDIR/data/FashionMNIST/ 2>/dev/null || true',
    'imagenet': 'mkdir -p \$SLURM_TMPDIR/data/ILSVRC2012/\ncp -r /datashare/imagenet/ILSVRC2012/* \$SLURM_TMPDIR/data/ILSVRC2012/ 2>/dev/null || true',
}
print(cmds.get(ds, ''))
"
}

COPY_DATA=$(get_dataset_copy)
C_TEST_SIZE_ARG=""
if [ "$TEST_SIZE" != "-1" ]; then
    C_TEST_SIZE_ARG="--test_size $TEST_SIZE"
fi

JOB_A="" JOB_B_IDS="" JOB_C="" JOB_D_IDS="" JOB_E="" JOB_F_IDS="" JOB_G=""
ALL_JOBS=""

# --- Step A ---
if [ "$RECOVER_STEP_A" = "true" ]; then
    cat > "$JOB_DIR/step_A.sh" << EOF_A
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH $A_GPU
#SBATCH --cpus-per-task=$A_CPUS
#SBATCH --time=$A_TIME
#SBATCH --mem=$A_MEM
#SBATCH --output=slurm_out/REC_A_${EXPERIMENT}_%A.out
#SBATCH --error=slurm_err/REC_A_${EXPERIMENT}_%A.err
mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate
$COPY_DATA
python training.py --experiment_name $EXPERIMENT --temp_dir \$SLURM_TMPDIR
echo "Step A complete."
EOF_A
    JOB_A=$(submit_job "$JOB_DIR/step_A.sh" "")
    ALL_JOBS="$JOB_A"
    echo "[A] Training: $JOB_A"
fi

# --- Step B (per chunk) ---
if [ "$RECOVER_STEP_B" = "true" ]; then
    for CHUNK in $RECOVER_STEP_B_CHUNKS; do
        cat > "$JOB_DIR/step_B_c${CHUNK}.sh" << EOF_B
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH $B_GPU
#SBATCH --cpus-per-task=$B_CPUS
#SBATCH --time=$B_TIME
#SBATCH --mem=$B_MEM
#SBATCH --output=slurm_out/REC_B_${EXPERIMENT}_c${CHUNK}_%A.out
#SBATCH --error=slurm_err/REC_B_${EXPERIMENT}_c${CHUNK}_%A.err
mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate
$COPY_DATA
mkdir -p \$SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
cp \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/weights/* \$SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
ZIP_FILE=\$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/matrices_task_${CHUNK}.zip
if [ -f "\$ZIP_FILE" ]; then
    cp "\$ZIP_FILE" "\$SLURM_TMPDIR/experiments/$EXPERIMENT/"
    unzip -o "\$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_${CHUNK}.zip" -d "\$SLURM_TMPDIR/experiments/$EXPERIMENT/"
fi
BATCH_SIZE=$BATCH_SIZE
CALIB_FILE="\$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/calibration.json"
if [ -f "\$CALIB_FILE" ]; then
    BATCH_SIZE=\$(python3 -c "import json; print(json.load(open('\$CALIB_FILE'))['batch_size'])")
    echo "Using calibrated batch_size=\$BATCH_SIZE"
fi
mkdir -p \$SLURM_SUBMIT_DIR/gpu-monitor/
GPU_LOGFILE="\$SLURM_SUBMIT_DIR/gpu-monitor/$EXPERIMENT.B.${CHUNK}.log"
monitor_gpu() {
  echo "Timestamp, GPU Util (%), Mem Used (MiB), Mem Total (MiB)" > "\$GPU_LOGFILE"
  while true; do
    ts=\$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -v t="\$ts" '{print t", "\$1", "\$2", "\$3}' >> "\$GPU_LOGFILE"
    sleep 30
  done
}
monitor_gpu &
MONITOR_PID=\$!
python generate_matrices.py --temp_dir \$SLURM_TMPDIR --experiment $EXPERIMENT --chunk_id $CHUNK --total_chunks $TOTAL_CHUNKS --batch_size \$BATCH_SIZE --num_samples_per_class $NUM_SAMPLES_PER_CLASS
kill \$MONITOR_PID 2>/dev/null || true
cd \$SLURM_TMPDIR/experiments/$EXPERIMENT
zip -r matrices_task_${CHUNK}.zip matrices || { echo "Zip failed"; exit 1; }
cd \$SLURM_SUBMIT_DIR
python -m utils.data_integrity --verify-zip \$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_${CHUNK}.zip || { echo "Verify failed"; exit 1; }
cp \$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_${CHUNK}.zip \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/
echo "Step B chunk $CHUNK complete."
EOF_B
        DEP="${JOB_A:-}"
        JOB_ID=$(submit_job "$JOB_DIR/step_B_c${CHUNK}.sh" "$DEP")
        JOB_B_IDS="${JOB_B_IDS:+$JOB_B_IDS:}$JOB_ID"
        ALL_JOBS="${ALL_JOBS:+$ALL_JOBS:}$JOB_ID"
        echo "[B] Matrices chunk $CHUNK: $JOB_ID"
    done
fi

# --- Step C ---
if [ "$RECOVER_STEP_C" = "true" ]; then
    cat > "$JOB_DIR/step_C.sh" << EOF_C
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH $C_GPU
#SBATCH --cpus-per-task=$C_CPUS
#SBATCH --time=$C_TIME
#SBATCH --mem=$C_MEM
#SBATCH --output=slurm_out/REC_C_${EXPERIMENT}_%A.out
#SBATCH --error=slurm_err/REC_C_${EXPERIMENT}_%A.err
mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate
$COPY_DATA
mkdir -p \$SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
cp \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/weights/* \$SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
python generate_adversarial_examples.py --experiment_name $EXPERIMENT --temp_dir=\$SLURM_TMPDIR $C_TEST_SIZE_ARG
mkdir -p \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adversarial_examples/
cp -r \$SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/* \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adversarial_examples/ 2>/dev/null || true
echo "Step C complete."
EOF_C
    DEP="${JOB_A:-}"
    JOB_C=$(submit_job "$JOB_DIR/step_C.sh" "$DEP")
    ALL_JOBS="${ALL_JOBS:+$ALL_JOBS:}$JOB_C"
    echo "[C] Adv examples: $JOB_C"
fi

# --- Step D (per chunk) ---
if [ "$RECOVER_STEP_D" = "true" ]; then
    for CHUNK in $RECOVER_STEP_D_CHUNKS; do
        cat > "$JOB_DIR/step_D_c${CHUNK}.sh" << EOF_D
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH $D_GPU
#SBATCH --cpus-per-task=$D_CPUS
#SBATCH --time=$D_TIME
#SBATCH --mem=$D_MEM
#SBATCH --output=slurm_out/REC_D_${EXPERIMENT}_c${CHUNK}_%A.out
#SBATCH --error=slurm_err/REC_D_${EXPERIMENT}_c${CHUNK}_%A.err
mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate
$COPY_DATA
mkdir -p \$SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
cp \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/weights/* \$SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
mkdir -p \$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/
for F in exp_dataset_train.pth exp_dataset_labels.pth; do
    SRC="\$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/rejection_levels/\$F"
    [ -f "\$SRC" ] && cp "\$SRC" "\$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/"
done
ZIP_FILE="\$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/rejection_levels/matrices_task_${CHUNK}.zip"
if [ -f "\$ZIP_FILE" ]; then
    cp "\$ZIP_FILE" "\$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/"
    cd "\$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/"
    unzip -o "matrices_task_${CHUNK}.zip"
    cd -
fi
BATCH_SIZE=$BATCH_SIZE
CALIB_FILE="\$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/calibration.json"
if [ -f "\$CALIB_FILE" ]; then
    BATCH_SIZE=\$(python3 -c "import json; print(json.load(open('\$CALIB_FILE'))['batch_size'])")
    echo "Using calibrated batch_size=\$BATCH_SIZE"
fi
mkdir -p \$SLURM_SUBMIT_DIR/gpu-monitor/
GPU_LOGFILE="\$SLURM_SUBMIT_DIR/gpu-monitor/$EXPERIMENT.D.${CHUNK}.log"
monitor_gpu() {
  echo "Timestamp, GPU Util (%), Mem Used (MiB), Mem Total (MiB)" > "\$GPU_LOGFILE"
  while true; do
    ts=\$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -v t="\$ts" '{print t", "\$1", "\$2", "\$3}' >> "\$GPU_LOGFILE"
    sleep 30
  done
}
monitor_gpu &
MONITOR_PID=\$!
python compute_matrices_for_rejection_level.py --experiment_name $EXPERIMENT --temp_dir \$SLURM_TMPDIR --batch_size \$BATCH_SIZE --chunk_id $CHUNK --total_chunks $TOTAL_CHUNKS --num_samples_rejection_level $NUM_SAMPLES_REJECTION_LEVEL
kill \$MONITOR_PID 2>/dev/null || true
MATRICES_DIR="\$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices"
if [ -d "\$MATRICES_DIR" ]; then
    cd "\$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels"
    zip -r "matrices_task_${CHUNK}.zip" matrices || { echo "Zip failed"; exit 1; }
    cd \$SLURM_SUBMIT_DIR
    python -m utils.data_integrity --verify-zip "\$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices_task_${CHUNK}.zip" || { echo "Verify failed"; exit 1; }
fi
DEST_DIR="\$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/rejection_levels/"
mkdir -p "\$DEST_DIR"
[ -f "\$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices_task_${CHUNK}.zip" ] && cp "\$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices_task_${CHUNK}.zip" "\$DEST_DIR"
for F in exp_dataset_train.pth exp_dataset_labels.pth; do
    SRC="\$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/\$F"
    [ -f "\$SRC" ] && cp "\$SRC" "\$DEST_DIR"
done
echo "Step D chunk $CHUNK complete."
EOF_D
        DEP="${JOB_A:-}"
        JOB_ID=$(submit_job "$JOB_DIR/step_D_c${CHUNK}.sh" "$DEP")
        JOB_D_IDS="${JOB_D_IDS:+$JOB_D_IDS:}$JOB_ID"
        ALL_JOBS="${ALL_JOBS:+$ALL_JOBS:}$JOB_ID"
        echo "[D] Rej level chunk $CHUNK: $JOB_ID"
    done
fi

# --- Step E (depends on all B) ---
if [ "$RECOVER_STEP_E" = "true" ]; then
    cat > "$JOB_DIR/step_E.sh" << EOF_E
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --cpus-per-task=$E_CPUS
#SBATCH --time=$E_TIME
#SBATCH --mem-per-cpu=$E_MEM
#SBATCH --output=slurm_out/REC_E_${EXPERIMENT}_%A.out
#SBATCH --error=slurm_err/REC_E_${EXPERIMENT}_%A.err
mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate
mkdir -p \$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
for i in \$(seq 0 $((TOTAL_CHUNKS - 1))); do
    cp \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/matrices_task_\$i.zip \$SLURM_TMPDIR/experiments/$EXPERIMENT/
    unzip -o "\$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_\$i.zip" -d "\$SLURM_TMPDIR/experiments/$EXPERIMENT/"
done
python compute_matrix_statistics.py --experiment_name $EXPERIMENT --temp_dir \$SLURM_TMPDIR
mkdir -p \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/matrices/
cp \$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/matrix_statistics.json \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/matrices/ 2>/dev/null || true
echo "Step E complete."
EOF_E
    DEP="${JOB_B_IDS:-}"
    JOB_E=$(submit_job "$JOB_DIR/step_E.sh" "$DEP")
    ALL_JOBS="${ALL_JOBS:+$ALL_JOBS:}$JOB_E"
    echo "[E] Matrix statistics: $JOB_E"
fi

# --- Step F (per chunk, depends on C) ---
if [ "$RECOVER_STEP_F" = "true" ]; then
    for CHUNK in $RECOVER_STEP_F_CHUNKS; do
        cat > "$JOB_DIR/step_F_c${CHUNK}.sh" << EOF_F
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH $F_GPU
#SBATCH --cpus-per-task=$F_CPUS
#SBATCH --time=$F_TIME
#SBATCH --mem=$F_MEM
#SBATCH --output=slurm_out/REC_F_${EXPERIMENT}_c${CHUNK}_%A.out
#SBATCH --error=slurm_err/REC_F_${EXPERIMENT}_c${CHUNK}_%A.err
mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate
$COPY_DATA
mkdir -p \$SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
cp \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/weights/* \$SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
mkdir -p \$SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/
tar cf - -C \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adversarial_examples . | tar xf - -C \$SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples
ZIP_FILE="\$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adv_matrices_task_${CHUNK}.zip"
if [ -f "\$ZIP_FILE" ]; then
    cp "\$ZIP_FILE" "\$SLURM_TMPDIR/experiments/$EXPERIMENT/"
    unzip -o "\$SLURM_TMPDIR/experiments/$EXPERIMENT/adv_matrices_task_${CHUNK}.zip" -d "\$SLURM_TMPDIR/experiments/$EXPERIMENT/"
fi
BATCH_SIZE=$BATCH_SIZE
CALIB_FILE="\$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/calibration.json"
if [ -f "\$CALIB_FILE" ]; then
    BATCH_SIZE=\$(python3 -c "import json; print(json.load(open('\$CALIB_FILE'))['batch_size'])")
    echo "Using calibrated batch_size=\$BATCH_SIZE"
fi
mkdir -p \$SLURM_SUBMIT_DIR/gpu-monitor/
GPU_LOGFILE="\$SLURM_SUBMIT_DIR/gpu-monitor/$EXPERIMENT.F.${CHUNK}.log"
monitor_gpu() {
  echo "Timestamp, GPU Util (%), Mem Used (MiB), Mem Total (MiB)" > "\$GPU_LOGFILE"
  while true; do
    ts=\$(date +%Y-%m-%dT%H:%M:%S)
    nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits \
      | awk -v t="\$ts" '{print t", "\$1", "\$2", "\$3}' >> "\$GPU_LOGFILE"
    sleep 30
  done
}
monitor_gpu &
MONITOR_PID=\$!
python generate_adversarial_matrices.py --experiment_name $EXPERIMENT --temp_dir \$SLURM_TMPDIR --chunk_id $CHUNK --total_chunks $TOTAL_CHUNKS --batch_size \$BATCH_SIZE --samples_per_attack $SAMPLES_PER_ATTACK
kill \$MONITOR_PID 2>/dev/null || true
cd \$SLURM_TMPDIR/experiments/$EXPERIMENT/
zip -r adv_matrices_task_${CHUNK}.zip adversarial_matrices/ || { echo "Zip failed"; exit 1; }
cd \$SLURM_SUBMIT_DIR
python -m utils.data_integrity --verify-zip \$SLURM_TMPDIR/experiments/$EXPERIMENT/adv_matrices_task_${CHUNK}.zip || { echo "Verify failed"; exit 1; }
cp \$SLURM_TMPDIR/experiments/$EXPERIMENT/adv_matrices_task_${CHUNK}.zip \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/
echo "Step F chunk $CHUNK complete."
EOF_F
        DEP="${JOB_C:-}"
        JOB_ID=$(submit_job "$JOB_DIR/step_F_c${CHUNK}.sh" "$DEP")
        JOB_F_IDS="${JOB_F_IDS:+$JOB_F_IDS:}$JOB_ID"
        ALL_JOBS="${ALL_JOBS:+$ALL_JOBS:}$JOB_ID"
        echo "[F] Adv matrices chunk $CHUNK: $JOB_ID"
    done
fi

# --- Step G (depends on E + all F + all D) ---
if [ "$RECOVER_STEP_G" = "true" ]; then
    cat > "$JOB_DIR/step_G.sh" << EOF_G
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --cpus-per-task=$G_CPUS
#SBATCH --time=$G_TIME
#SBATCH --mem-per-cpu=$G_MEM_PER_CPU
#SBATCH --output=slurm_out/REC_G_${EXPERIMENT}_%A.out
#SBATCH --error=slurm_err/REC_G_${EXPERIMENT}_%A.err
mkdir -p \$SLURM_SUBMIT_DIR/slurm_out \$SLURM_SUBMIT_DIR/slurm_err
module load $MODULES
source $ENV_NAME/bin/activate
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
$COPY_DATA
mkdir -p \$SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
cp \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/weights/* \$SLURM_TMPDIR/experiments/$EXPERIMENT/weights/
mkdir -p \$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
cp \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/matrices/matrix_statistics.json \$SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
for i in \$(seq 0 $((TOTAL_CHUNKS - 1))); do
    [ -f "\$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adv_matrices_task_\$i.zip" ] && {
        cp \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adv_matrices_task_\$i.zip \$SLURM_TMPDIR/experiments/$EXPERIMENT/
        unzip -o \$SLURM_TMPDIR/experiments/$EXPERIMENT/adv_matrices_task_\$i.zip -d \$SLURM_TMPDIR/experiments/$EXPERIMENT/
    }
done
mkdir -p \$SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/
cp -r \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/adversarial_examples/* \$SLURM_TMPDIR/experiments/$EXPERIMENT/adversarial_examples/ 2>/dev/null || true
mkdir -p \$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices/
cp -r \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/rejection_levels/* \$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/ 2>/dev/null || true
for i in \$(seq 0 $((TOTAL_CHUNKS - 1))); do
    [ -f "\$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices_task_\$i.zip" ] && \
        unzip -o \$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/matrices_task_\$i.zip -d \$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/
done
echo "All data ready. Starting grid search..."
python grid_search.py --nb_workers \$SLURM_CPUS_PER_TASK --experiment_name $EXPERIMENT --temp_dir \$SLURM_TMPDIR --rej_lev 0
mkdir -p \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/grid_search/
cp -r \$SLURM_TMPDIR/experiments/$EXPERIMENT/grid_search/* \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/grid_search/ 2>/dev/null || true
cp -r \$SLURM_TMPDIR/experiments/$EXPERIMENT/rejection_levels/reject_at_* \$SLURM_SUBMIT_DIR/experiments/$EXPERIMENT/rejection_levels/ 2>/dev/null || true
echo "Step G complete."
EOF_G
    G_DEPS=""
    [ -n "$JOB_E" ] && G_DEPS="$JOB_E"
    [ -n "$JOB_F_IDS" ] && G_DEPS="${G_DEPS:+$G_DEPS:}$JOB_F_IDS"
    [ -n "$JOB_D_IDS" ] && G_DEPS="${G_DEPS:+$G_DEPS:}$JOB_D_IDS"
    JOB_G=$(submit_job "$JOB_DIR/step_G.sh" "$G_DEPS")
    ALL_JOBS="${ALL_JOBS:+$ALL_JOBS:}$JOB_G"
    echo "[G] Grid search: $JOB_G"
fi

echo ""
echo "Dispatcher complete for $EXPERIMENT."
echo "Submitted jobs: ${ALL_JOBS//:/, }"
DISPATCH_BODY

        # Replace placeholders in dispatcher
        sed -i "s|__ACCOUNT__|$ACCOUNT|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__EXP__|$EXP|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__TOTAL_CHUNKS__|$TOTAL_CHUNKS|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__BATCH_SIZE__|$BATCH_SIZE|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__NUM_SAMPLES_PER_CLASS__|$NUM_SAMPLES_PER_CLASS|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__SAMPLES_PER_ATTACK__|$SAMPLES_PER_ATTACK|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__NUM_SAMPLES_REJECTION_LEVEL__|$NUM_SAMPLES_REJECTION_LEVEL|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__TEST_SIZE__|$TEST_SIZE|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__ENV_NAME__|$ENV_NAME|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__MODULES__|$MODULES|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__A_GPU__|$A_GPU|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__A_CPUS__|$A_CPUS|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__A_TIME__|$A_TIME|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__A_MEM__|$A_MEM|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__B_GPU__|$B_GPU|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__B_CPUS__|$B_CPUS|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__B_TIME__|$B_TIME|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__B_MEM__|$B_MEM|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__C_GPU__|$C_GPU|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__C_CPUS__|$C_CPUS|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__C_TIME__|$C_TIME|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__C_MEM__|$C_MEM|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__D_GPU__|$D_GPU|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__D_CPUS__|$D_CPUS|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__D_TIME__|$D_TIME|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__D_MEM__|$D_MEM|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__E_CPUS__|$E_CPUS|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__E_TIME__|$E_TIME|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__E_MEM__|$E_MEM|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__F_GPU__|$F_GPU|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__F_CPUS__|$F_CPUS|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__F_TIME__|$F_TIME|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__F_MEM__|$F_MEM|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__G_CPUS__|$G_CPUS|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__G_TIME__|$G_TIME|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__G_MEM_PER_CPU__|$G_MEM_PER_CPU|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__AUDIT_CPUS__|$AUDIT_CPUS|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__AUDIT_TIME__|$AUDIT_TIME|g" "$JOB_DIR/dispatch.sh"
        sed -i "s|__AUDIT_MEM__|$AUDIT_MEM|g" "$JOB_DIR/dispatch.sh"

        # Submit
        if [ "$DRY_RUN" = "true" ]; then
            echo "  [DRY RUN] Would submit audit + dispatcher for $EXP"
            echo "  Scripts generated in: $JOB_DIR/"
        else
            AUDIT_JOB=$(submit_job "$JOB_DIR/audit.sh" "$CALIB_DEP")
            DISPATCH_JOB=$(submit_job "$JOB_DIR/dispatch.sh" "$AUDIT_JOB")
            echo "  [*] Audit:               $AUDIT_JOB (depends on: ${CALIB_DEP:-none})"
            echo "  [*] Dispatcher:          $DISPATCH_JOB (depends on $AUDIT_JOB)"
        fi
    fi
done

# ==============================================================
# Summary
# ==============================================================
echo ""
echo "=============================================================="
echo "  Pipeline Orchestrator — Summary"
echo "=============================================================="
echo ""
echo "  Experiments: ${EXPERIMENTS[*]}"
echo "  Mode: $([ "$SKIP_AUDIT" = "true" ] && echo "skip-audit" || echo "audit+dispatch")"
echo "  Calibration: $([ "$NO_CALIBRATE" = "true" ] && echo "disabled" || echo "enabled")"
echo "  Test mode: $TEST_MODE"
echo "  Dry run: $DRY_RUN"
echo ""
if [ "$TEST_MODE" = "true" ]; then
    echo "  Test parameters:"
    echo "    Chunks: $TOTAL_CHUNKS"
    echo "    Batch size: $BATCH_SIZE"
    echo "    Samples/class: $NUM_SAMPLES_PER_CLASS"
    echo "    Samples/attack: $SAMPLES_PER_ATTACK"
    echo "    Rejection level samples: $NUM_SAMPLES_REJECTION_LEVEL"
    echo "    Test size (adv examples): $TEST_SIZE"
    echo ""
fi
echo "  Monitor with: squeue -u \$USER"
echo "  Job scripts:  experiments/<name>/orchestrator_jobs/"
echo "=============================================================="
