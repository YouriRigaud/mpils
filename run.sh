#!/bin/bash
#SBATCH --job-name=mpi-run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=12:00:00
#SBATCH --mem=8G
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err

set -u

cd /home/yorig/tuner/mpils || exit 1

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export CPLEX_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

ORIGINAL_ARGS=("$@")
MPI_FROM_ARGS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mpi)
      [[ $# -ge 2 ]] || { echo "Erreur: --mpi nécessite une valeur"; exit 1; }
      MPI_FROM_ARGS="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

if [[ -z "$MPI_FROM_ARGS" ]]; then
  echo "Erreur: l'argument --mpi est obligatoire."
  echo "Exemple: sbatch --ntasks=2 run.sh --mpi 2 --solver-time 5 --tuning-objective gap"
  exit 1
fi

if ! [[ "$MPI_FROM_ARGS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Erreur: --mpi doit être un entier >= 1"
  exit 1
fi

if [[ -z "${SLURM_NTASKS:-}" ]]; then
  echo "Erreur: SLURM_NTASKS n'est pas défini."
  exit 1
fi

if [[ "$MPI_FROM_ARGS" -ne "$SLURM_NTASKS" ]]; then
  echo "Erreur: incohérence entre --mpi et l'allocation Slurm."
  echo "  --mpi         = $MPI_FROM_ARGS"
  echo "  SLURM_NTASKS  = $SLURM_NTASKS"
  exit 1
fi

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-1}"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "CPLEX_NUM_THREADS=$CPLEX_NUM_THREADS"
echo "Arguments transmis à run_all.sh: ${ORIGINAL_ARGS[*]}"

./script/run_all.sh "${ORIGINAL_ARGS[@]}"
