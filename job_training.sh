#!/bin/bash

#SBATCH --account=def-ko1  #account to charge the calculation
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00 #hour:minutes:seconds
#SBATCH --mem=4G #memory requested
#SBATCH --output=slurm_out/resnet_cifar10_%j.out
#SBATCH --error=slurm_err/resnet_cifar10_%j.err


module load StdEnv/2023 python/3.10.13 scipy-stack/2025a

mkdir -p $SLURM_TMPDIR/data/cifar-100-python/
echo "Copying datasets..."
cp -r data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/

mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/
cp -r data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/

echo "data ready..."
source env_nibi/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
echo "Start training..."
python scratch-train.py
