#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=00:15:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G #memory requested

module load StdEnv/2023 python/3.10.13 scipy-stack/2025a

source env_nibi/bin/activate

EXPERIMENT="alexnet_cifar10"
mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
cp experiments/$EXPERIMENT/matrices_task_0.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/matrices_task_1.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/matrices_task_2.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/matrices_task_3.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
@mkdir -p experiments/$EXPERIMENT/matrices/
#cd experiments/$EXPERIMENT/
#unzip matrices_task_0.zip
#unzip matrices_task_1.zip
#unzip matrices_task_2.zip
#unzip matrices_task_3.zip

unzip -o $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_0.zip
unzip -o $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_1.zip
unzip -o $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_2.zip
unzip -o $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_3.zip

#ls
#ls matrices/*/

cd ../../
cp -r experiments/$EXPERIMENT/matrices/* $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
echo "Matrices ready"
python compute_matrix_statistics.py --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR
