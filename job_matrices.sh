#!/bin/bash

#SBATCH --account=<ACCOUNT_NAME>
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=10G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3  # 4 tasks for 4 chunks

module load StdEnv/2020 python/3.10.2 scipy-stack/2023a cuda cudnn
source env/bin/activate

mkdir -p $SLURM_TMPDIR/experiments/alexnet_cifar10/weights/
echo "Copying weights for task $SLURM_ARRAY_TASK_ID..."
cp experiments/alexnet_cifar10/weights/* $SLURM_TMPDIR/experiments/alexnet_cifar10/weights/
echo "Weights copied to temp directory..."

python generate_matrices.py \
    --experiment_name=alexnet_cifar10 \
    --temp_dir=$SLURM_TMPDIR \
    --total_chunks=4 \
    --batch_size=16