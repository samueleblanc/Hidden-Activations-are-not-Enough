#!/bin/bash
#SBATCH --array=0-2
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm_out/C_thm45_agg_%A_%a.out
#SBATCH --error=slurm_err/C_thm45_agg_%A_%a.err

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

# Map array task ID to experiment
EXPERIMENTS=("resnet152_imagenet" "densenet121_imagenet" "googlenet_imagenet")
EXPERIMENT=${EXPERIMENTS[$SLURM_ARRAY_TASK_ID]}

echo "Step C aggregation: $EXPERIMENT (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

python validate_theorem45.py \
    --experiment $EXPERIMENT \
    --num_samples 200 \
    --aggregate

echo "Aggregation $SLURM_ARRAY_TASK_ID ($EXPERIMENT) completed"
