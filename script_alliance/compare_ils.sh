#!/bin/bash
#SBATCH --job-name=mpi-run
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

set -u

cd /home/yorig/tuner/mpils || exit 1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CPLEX_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"


python3 script/compare_local_search_batch.py --mpi-procs 2 --number-of-evaluations 20 --output-root compare/2proc_20eval
