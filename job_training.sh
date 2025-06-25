#!/bin/bash

#SBATCH --account=def-bruestle  #account to charge the calculation
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00 #hour:minutes:seconds
#SBATCH --mem=4G #memory requested

module load StdEnv/2020 python/3.9.6 scipy-stack/2023a cuda cudnn #load the required module
mkdir -p $SLURM_TMPDIR/data/cifar-10-batches-py/
echo "Copying datasets..."
cp -r data/cifar-10-batches-py/* $SLURM_TMPDIR/data/cifar-10-batches-py/
echo "data ready..."
source env/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

python scratch-train.py
