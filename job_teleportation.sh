#!/bin/bash
#SBATCH --account=def-xxx
#SBATCH --array=0-2
#SBATCH --time=02:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=slurm_out/B_teleport_%A_%a.out
#SBATCH --error=slurm_err/B_teleport_%A_%a.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

# Map array task ID to architecture
# COB models available: resnet18, vgg11_bn (no AlexNet COB in neuralteleportation)
ARCHITECTURES=("resnet18" "vgg11_bn" "resnet50")
ARCH=${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}

echo "Step B: Teleportation experiment for $ARCH / imagenet (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

python teleportation_experiment.py \
    --architecture $ARCH \
    --dataset imagenet \
    --pretrained \
    --num_teleportations 100 \
    --num_samples 500 \
    --data_dir /datashare/imagenet/ILSVRC2012

echo "Task $SLURM_ARRAY_TASK_ID ($ARCH) completed"
