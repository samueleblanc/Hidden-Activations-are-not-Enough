#!/bin/bash

#SBATCH --account=<ACCOUNT_NAME>  #account to charge the calculation
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:30:00 #hour:minutes:seconds
#SBATCH --mem=8G #memory requested
#SBATCH --array=8

module load StdEnv/2020 scipy-stack/2023a cuda cudnn #load the required module

mkdir -p $SLURM_TMPDIR/data/MNIST/
mkdir -p $SLURM_TMPDIR/data/FashionMNIST/
mkdir -p $SLURM_TMPDIR/data/CIFAR10/
mkdir -p $SLURM_TMPDIR/data/CIFAR100/
mkdir -p $SLURM_TMPDIR/data/Imagenette/
echo "Copying datasets..."
cp -r data/MNIST/* $SLURM_TMPDIR/data/MNIST/
echo "MNIST ready"
cp -r data/FashionMNIST/* $SLURM_TMPDIR/data/FashionMNIST/
echo "Fashion ready"
cp -r data/CIFAR10/* $SLURM_TMPDIR/data/CIFAR10/
echo "CIFAR10 ready"
cp -r data/CIFAR100/* $SLURM_TMPDIR/data/CIFAR100/
echo "CIFAR100 ready"
cp -r data/Imagenette/* $SLURM_TMPDIR/data/Imagenette/
echo "Imagenette ready"

source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)

python training.py --default_index $SLURM_ARRAY_TASK_ID --temp_dir $SLURM_TMPDIR
