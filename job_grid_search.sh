#!/bin/bash

#SBATCH --account=def-assem #account to charge the calculation
#SBATCH --time=00:20:00 #hour:minutes:seconds
#SBATCH --cpus-per-task=2 #number of CPU requested
#SBATCH --mem-per-cpu=1G #memory requested
#SBATCH --array=0

module load StdEnv/2020 scipy-stack/2023a #load the required module
source ENV/bin/activate #load the virtualenv (absolute or relative path to where the script is submitted)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/
echo "Copying weights..."
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/weights/* $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/weights/

echo "Copying matrix statistics..."
mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/matrices/
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/matrices/matrix_statistics.json $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/matrices/

mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/matrices/

echo "Copying adversarial matrices..."
mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_matrices/
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/adversarial_matrices/adversarial_matrices.zip $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_matrices/
echo "Decompress..."
unzip /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/adversarial_matrices/adversarial_matrices.zip -d $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_matrices/

echo "Copying adversarial examples..."
mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_examples/
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/adversarial_examples/adversarial_examples.pth $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/adversarial_examples/

echo "Copying exp dataset train..."
mkdir -p $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/matrices/
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/exp_dataset_train.pth $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/

echo "Copying rejection level matrices..."
cp /home/armenta/scratch/MatrixStatistics/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/matrices.zip $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/matrices/

echo "Decompress..."
unzip $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/matrices/matrices.zip -d $SLURM_TMPDIR/experiments/$SLURM_ARRAY_TASK_ID/rejection_levels/

echo "All data ready!"

python compute_rejection_level.py --std 1 --d1 0.5 --default_index $SLURM_ARRAY_TASK_ID --temp_dir $SLURM_TMPDIR
#python grid_search.py --nb_workers=$SLURM_CPUS_PER_TASK --default_index $SLURM_ARRAY_TASK_ID --temp_dir $SLURM_TMPDIR
echo "Grid search finished !!!"
