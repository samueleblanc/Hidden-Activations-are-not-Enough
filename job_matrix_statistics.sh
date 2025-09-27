#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=00:20:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=40G #memory requested
#SBATCH --output=slurm_out/E_mat_stats_%A_%a.out
#SBATCH --error=slurm_err/E_mat_stats_%A_%a.err

mkdir -p $PWD/slurm_out
mkdir -p $PWD/slurm_err

EXPERIMENT="alexnet_cifar10"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env_rorqual/bin/activate

mkdir -p $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
cp experiments/$EXPERIMENT/matrices_task_0.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/matrices_task_1.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/matrices_task_2.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/
cp experiments/$EXPERIMENT/matrices_task_3.zip $SLURM_TMPDIR/experiments/$EXPERIMENT/

mkdir -p experiments/$EXPERIMENT/matrices/
unzip -o $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_0.zip
unzip -o $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_1.zip
unzip -o $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_2.zip
unzip -o $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices_task_3.zip

cd -
cp -r experiments/$EXPERIMENT/matrices/* $SLURM_TMPDIR/experiments/$EXPERIMENT/matrices/
echo "Matrices ready. Computing statistics."
python compute_matrix_statistics.py --experiment_name $EXPERIMENT --temp_dir $SLURM_TMPDIR
