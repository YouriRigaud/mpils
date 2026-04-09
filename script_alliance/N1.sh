#!/bin/bash
#SBATCH --job-name=mpils-numa
#SBATCH --nodes=4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --distribution=block:block

set -euo pipefail

cd /home/yorig/tuner/mpils

INSTANCE_PATH="/home/yorig/Instances_MPILS/N1.lp"

MPI_FROM_ARGS=""
SOLVER_TIME=120
TUNING_OBJECTIVE="gap"
WORKING_DIR="N1-4-120"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mpi)
      [[ $# -ge 2 ]] || { echo "Erreur: --mpi nécessite une valeur"; exit 1; }
      MPI_FROM_ARGS="$2"
      shift 2
      ;;
    --solver-time)
      [[ $# -ge 2 ]] || { echo "Erreur: --solver-time nécessite une valeur"; exit 1; }
      SOLVER_TIME="$2"
      shift 2
      ;;
    --tuning-objective)
      [[ $# -ge 2 ]] || { echo "Erreur: --tuning-objective nécessite une valeur"; exit 1; }
      TUNING_OBJECTIVE="$2"
      shift 2
      ;;
    --working-dir)
      [[ $# -ge 2 ]] || { echo "Erreur: --working-dir nécessite une valeur"; exit 1; }
      WORKING_DIR="$2"
      shift 2
      ;;
    --instance)
      [[ $# -ge 2 ]] || { echo "Erreur: --instance nécessite une valeur"; exit 1; }
      INSTANCE_PATH="$2"
      shift 2
      ;;
    *)
      echo "Argument ignoré: $1"
      shift
      ;;
  esac
done

if [[ -z "${SLURM_NTASKS:-}" ]]; then
  echo "Erreur: SLURM_NTASKS n'est pas défini."
  exit 1
fi

if [[ -n "$MPI_FROM_ARGS" && "$MPI_FROM_ARGS" -ne "$SLURM_NTASKS" ]]; then
  echo "Erreur: incohérence entre --mpi et l'allocation Slurm."
  echo "  --mpi        = $MPI_FROM_ARGS"
  echo "  SLURM_NTASKS = $SLURM_NTASKS"
  exit 1
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export CPLEX_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Évite qu'une autre lib crée des threads en plus de CPLEX
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "===== Job info ====="
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "CPLEX_NUM_THREADS=$CPLEX_NUM_THREADS"
echo "INSTANCE_PATH=$INSTANCE_PATH"
echo "SOLVER_TIME=$SOLVER_TIME"
echo "TUNING_OBJECTIVE=$TUNING_OBJECTIVE"
echo "WORKING_DIR=$WORKING_DIR"
echo "===================="

echo "===== Topology check ====="
taskset -pc $$
numactl --show || true
lscpu | egrep 'CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\)|NUMA node[0-9]'
echo "========================="

srun \
  --ntasks="${SLURM_NTASKS}" \
  --cpus-per-task="${SLURM_CPUS_PER_TASK}" \
  --cpu-bind=verbose,cores \
  --mem-bind=verbose,local \
  build/mpils \
    "$INSTANCE_PATH" \
    --solver-time "$SOLVER_TIME" \
    --solver-threads "${SLURM_CPUS_PER_TASK}" \
    --tuning-objective "$TUNING_OBJECTIVE" \
    --working-dir "$WORKING_DIR"
