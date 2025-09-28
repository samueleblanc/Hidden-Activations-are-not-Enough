#!/bin/bash

#SBATCH --account=def-assem
#SBATCH --gpus=a100_2g.10gb:4
#SBATCH --cpus-per-task=3
#SBATCH --time=00:30:00
#SBATCH --mem=31G

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
mkdir -p $SLURM_TMPDIR/data/cifar-100-python/
mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/
echo "Copying datasets..."
cp -r data/cifar-100-python/* $SLURM_TMPDIR/data/cifar-100-python/
cp -r data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/

echo "data ready..."
source env_narval/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

#export CUDA_VISIBLE_DEVICES=0,1
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_USE_CUDA_DSA=1

python scratch-train.py --temp_dir $SLURM_TMPDIR --model vgg --dataset cifar100
#python training.py --experiment_name vgg_cifar100 --temp_dir $SLURM_TMPDIR