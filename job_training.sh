#!/bin/bash

#SBATCH --account=def-ko1  #account to charge the calculation
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00 #hour:minutes:seconds
#SBATCH --mem=32G #memory requested
#SBATCH --exclusive

module load StdEnv/2023 python/3.10.13 scipy-stack/2025a
mkdir -p $SLURM_TMPDIR/data/cifar-100-python/
mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/
echo "Copying datasets..."
cp -r data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/
cp -r data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/

echo "data ready..."
source env_nibi/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python scratch-train.py --temp_dir $SLURM_TMPDIR --model vgg --dataset cifar100