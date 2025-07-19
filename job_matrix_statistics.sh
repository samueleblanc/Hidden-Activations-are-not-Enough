#!/bin/bash

#SBATCH --account=def-ko1 #account to charge the calculation
#SBATCH --time=00:15:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G #memory requested

module load StdEnv/2023 python/3.10.13 scipy-stack/2025a

source env_nibi/bin/activate

EXPERIMENT="alexnet_cifar10"
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
echo "Matrices ready"

mkdir -p experiments/$EXPERIMENT/matrices/
cd experiments/$EXPERIMENT/
unzip matrices_task_0.zip
unzip matrices_task_1.zip
unzip matrices_task_2.zip
unzip matrices_task_3.zip

cd ../../
cp -r experiments/$EXPERIMENT/matrices/* $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
echo "Matrices ready"
python compute_matrix_statistics.py --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR
