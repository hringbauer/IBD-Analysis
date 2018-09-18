#!/bin/bash
#SBATCH --partition=broadwl
#SBATCH --ntasks=1
#SBATCH --job-name="selfing100reps"
#SBATCH --time=1:00:00
#SBATCH --mem=4G
#SBATCH --mail-user=hringbauer@uchicago.edu
#SBATCH --mail-type=ALL
#SBATCH --no-requeue
#SBATCH --export=NONE
#SBATCH --array=250-299
unset SLURM_EXPORT_ENV

export OMP_NUM_THREADS=1

module load python/2.7.12
python multi_run.py $SLURM_ARRAY_TASK_ID
