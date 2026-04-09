#!/bin/bash
#SBATCH --job-name=mpils-24x12
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=24
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --distribution=block:block

set -euo pipefail

cd /home/yorig/tuner/mpils

INSTANCE_PATH="/home/yorig/Instances_MPILS/N1.lp"
SOLVER_TIME=150
TUNING_OBJECTIVE="gap"
WORKING_DIR="N1-24x12"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CPLEX_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# évite du sur-threading caché
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "CPLEX_NUM_THREADS=$CPLEX_NUM_THREADS"

srun \
  --cpu-bind=verbose,cores \
  --mem-bind=verbose,local \
  ./build/mpils \
    "$INSTANCE_PATH" \
    --solver-time "$SOLVER_TIME" \
    --solver-threads "${SLURM_CPUS_PER_TASK}" \
    --tuning-objective "$TUNING_OBJECTIVE" \
    --working-dir "$WORKING_DIR"
