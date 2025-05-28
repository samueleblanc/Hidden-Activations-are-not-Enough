#!/bin/bash

#SBATCH --account=<ACCOUNT_NAME> #account to charge the calculation
#SBATCH --time=4:00:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=4 #number of CPU requested
#SBATCH --mem-per-cpu=3G #memory requested
#SBATCH --array=0

module load StdEnv/2020 python/3.9.6 scipy-stack/2023a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/
echo "Copying weights..."
cp experiments/$SLURM_ARRAY_TASK_ID/weights/* $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/
echo "Weights copied to temp directory..."

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

python generate_adversarial_examples.py --default_index $SLURM_ARRAY_TASK_ID --nb_workers $SLURM_CPUS_PER_TASK --temp_dir=$SLURM_TMPDIR
