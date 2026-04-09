#!/bin/bash
#SBATCH --job-name=mpils-multi
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
#SBATCH --distribution=block:block

# ============================================================
# =============== VARIABLES MODIFIABLES EN HAUT ==============
# ============================================================

# --- Paramètres Slurm / exécution ---
MPI_PROCS=1
CPUS_PER_TASK=2          # 24 = 1 NUMA entier sur ta machine
CPLEX_THREADS=2          # CPLEX utilisera 16 threads à l'intérieur de ce NUMA
ENABLE_CPLEX_TEST=0      # 1 = activer le test CPLEX, 0 = désactiver
SBATCH_TIME="24:00:00"
SBATCH_MEM="0"
SBATCH_JOB_NAME="mpils-multi"

# --- Paramètres du tuner ---
SOLVER_TIME=15
TUNING_OBJECTIVE="gap"    # gap | upper_bound
ENABLE_TRACE=1

# --- Chemins racines ---
REPO_DIR="/home/yorig/tuner/mpils"
RESULT_ROOT="/home/yorig/tuner/result"
INSTANCE_DIR="/home/yorig/miplib/mpils-miplib-test"
CPLEX_APP="/home/yorig/CPLEX_Studio2212/cplex/bin/x86-64_linux/cplex"

# --- Liste des instances ---
INSTANCE_LIST=(
  "app1-2.mps"
  "brasil.mps"
  "comp08-2idx.mps"
  "fast0507.mps"
  "fastxgemm-n2r6s0t2.mps"
  "fastxgemm-n2r7s4t1.mps"
  "fiball.mps"
  "fjspeasy01i.mps"
  "gasprod2-2.mps"
  "glass4.mps"
  "mas74.mps"
  "mas76.mps"
  "mod011.mps"
  "mzzv11.mps"
  "neos-1456979.mps"
  "neos-2746589-doon.mps"
  "neos-4413714-turia.mps"
  "neos-4722843-widden.mps"
  "pk1.mps"
  "supportcase18.mps"
  "supportcase33.mps"
  "supportcase7.mps"
  "swath3.mps"
  "trento1.mps"
)

# ============================================================
# ==================== FIN VARIABLES MODIFIABLES =============
# ============================================================

usage() {
  cat <<EOF
Usage:
  sbatch numa-run.sh

Ce script lance le tuner sur toutes les instances de INSTANCE_LIST.

Variables à modifier directement en haut du script :
  MPI_PROCS
  CPUS_PER_TASK
  CPLEX_THREADS
  SOLVER_TIME
  TUNING_OBJECTIVE
  ENABLE_TRACE
  REPO_DIR
  RESULT_ROOT
  INSTANCE_DIR
  CPLEX_APP
  INSTANCE_LIST
EOF
}

run_cplex_test() {
  local instance_path="$1"
  local prm_file="$2"
  local save_dir="$3"

  local cplex_log="${save_dir}/cplex_test.log"
  local cplex_time_file="${save_dir}/cplex_test_time.txt"
  local cplex_status_file="${save_dir}/cplex_test_status.txt"

  CPLEX_TEST_RC=-1
  CPLEX_ELAPSED_SECONDS="NA"

  echo "[*] Test CPLEX avec la configuration: $prm_file"

  if [[ ! -x "$CPLEX_APP" ]]; then
    echo "[!] CPLEX introuvable ou non exécutable: $CPLEX_APP"
    echo "CPLEX executable not found: $CPLEX_APP" >"$cplex_status_file"
    return 1
  fi

  if [[ ! -f "$prm_file" ]]; then
    echo "[!] Fichier .prm introuvable: $prm_file"
    echo "Missing prm file: $prm_file" >"$cplex_status_file"
    return 1
  fi

  local start_ts
  local end_ts
  local elapsed

  start_ts=$(date +%s)

  "$CPLEX_APP" -c \
    "read $instance_path" \
    "read $prm_file" \
    "set threads 2" \
    "optimize" \
    "quit" \
    >"$cplex_log" 2>&1

  CPLEX_TEST_RC=$?

  end_ts=$(date +%s)
  elapsed=$((end_ts - start_ts))
  CPLEX_ELAPSED_SECONDS="$elapsed"

  {
    echo "instance=$instance_path"
    echo "prm_file=$prm_file"
    echo "return_code=$CPLEX_TEST_RC"
    echo "elapsed_seconds=$CPLEX_ELAPSED_SECONDS"
  } >"$cplex_time_file"

  if [[ $CPLEX_TEST_RC -eq 0 ]]; then
    echo "CPLEX test finished successfully" >"$cplex_status_file"
    echo "[*] Test CPLEX terminé en ${CPLEX_ELAPSED_SECONDS}s"
  else
    echo "CPLEX test failed with return code $CPLEX_TEST_RC" >"$cplex_status_file"
    echo "[!] Test CPLEX échoué avec code retour $CPLEX_TEST_RC après ${CPLEX_ELAPSED_SECONDS}s"
  fi

  return $CPLEX_TEST_RC
}

# === Validation simple ===
[[ "$MPI_PROCS" =~ ^[1-9][0-9]*$ ]] || { echo "Erreur: MPI_PROCS doit être un entier >= 1"; exit 1; }
[[ "$CPUS_PER_TASK" =~ ^[1-9][0-9]*$ ]] || { echo "Erreur: CPUS_PER_TASK doit être un entier >= 1"; exit 1; }
[[ "$CPLEX_THREADS" =~ ^[1-9][0-9]*$ ]] || { echo "Erreur: CPLEX_THREADS doit être un entier >= 1"; exit 1; }
[[ "$SOLVER_TIME" =~ ^[1-9][0-9]*$ ]] || { echo "Erreur: SOLVER_TIME doit être un entier >= 1"; exit 1; }
[[ "$ENABLE_TRACE" =~ ^[01]$ ]] || { echo "Erreur: ENABLE_TRACE doit valoir 0 ou 1"; exit 1; }
[[ "$ENABLE_CPLEX_TEST" =~ ^[01]$ ]] || { echo "Erreur: ENABLE_CPLEX_TEST doit valoir 0 ou 1"; exit 1; }

case "$TUNING_OBJECTIVE" in
  gap|upper_bound)
    ;;
  *)
    echo "Erreur: TUNING_OBJECTIVE doit valoir 'gap' ou 'upper_bound'"
    exit 1
    ;;
esac

if [[ "$CPLEX_THREADS" -gt "$CPUS_PER_TASK" ]]; then
  echo "Erreur: CPLEX_THREADS ne peut pas être > CPUS_PER_TASK"
  exit 1
fi

# Cohérence informative avec les directives SBATCH du haut
if [[ -n "${SLURM_NTASKS:-}" && "$SLURM_NTASKS" -ne "$MPI_PROCS" ]]; then
  echo "Erreur: MPI_PROCS=$MPI_PROCS mais Slurm a alloué SLURM_NTASKS=$SLURM_NTASKS"
  echo "Modifie soit la variable MPI_PROCS, soit la ligne #SBATCH --ntasks"
  exit 1
fi

if [[ -n "${SLURM_CPUS_PER_TASK:-}" && "$SLURM_CPUS_PER_TASK" -ne "$CPUS_PER_TASK" ]]; then
  echo "Erreur: CPUS_PER_TASK=$CPUS_PER_TASK mais Slurm a alloué SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
  echo "Modifie soit la variable CPUS_PER_TASK, soit la ligne #SBATCH --cpus-per-task"
  exit 1
fi

# === Chemins calculés ===
RUN_TAG="${MPI_PROCS}proc-${SOLVER_TIME}s-${TUNING_OBJECTIVE}"
BASE_DIR="${REPO_DIR}/tmp/${RUN_TAG}"
SAVE_BASE="${RESULT_ROOT}/${RUN_TAG}"
TUNER_APP="${REPO_DIR}/build/mpils"
SUMMARY_CSV="${SAVE_BASE}/summary.csv"

# === Options shell ===
set -e
set -u
set -o pipefail
if [[ "$ENABLE_TRACE" -eq 1 ]]; then
  set -x
fi

# === Vérifications ===
mkdir -p "$BASE_DIR"
mkdir -p "$SAVE_BASE"

[[ -x "$TUNER_APP" ]] || { echo "Erreur: tuner non exécutable: $TUNER_APP"; exit 1; }
[[ -d "$INSTANCE_DIR" ]] || { echo "Erreur: dossier d'instances introuvable: $INSTANCE_DIR"; exit 1; }
command -v srun >/dev/null 2>&1 || { echo "Erreur: srun introuvable"; exit 1; }

# === Environnement threads ===
export OMP_NUM_THREADS="$CPLEX_THREADS"
export CPLEX_NUM_THREADS="$CPLEX_THREADS"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# === Infos job ===
echo "===== Job info ====="
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-NA}"
echo "SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-NA}"
echo "SLURM_NTASKS=${SLURM_NTASKS:-NA}"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-NA}"
echo "MPI_PROCS=$MPI_PROCS"
echo "CPUS_PER_TASK=$CPUS_PER_TASK"
echo "CPLEX_THREADS=$CPLEX_THREADS"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "CPLEX_NUM_THREADS=$CPLEX_NUM_THREADS"
echo "RUN_TAG=$RUN_TAG"
echo "BASE_DIR=$BASE_DIR"
echo "SAVE_BASE=$SAVE_BASE"
echo "SUMMARY_CSV=$SUMMARY_CSV"
echo "===================="

# === Initialisation du CSV ===
echo "instance,tuner_rc,best_prm_found,cplex_test_rc,cplex_elapsed_seconds,save_dir" >"$SUMMARY_CSV"

count=0
success=0
failure=0
failed_instances=()

for instance in "${INSTANCE_LIST[@]}"; do
  ((++count))

  echo
  echo "------------------------------------------"
  echo "Instance #$count : '$instance'"
  echo "Date / Heure      : $(date)"
  echo "MPI_PROCS         : $MPI_PROCS"
  echo "CPUS_PER_TASK     : $CPUS_PER_TASK"
  echo "CPLEX_THREADS     : $CPLEX_THREADS"
  echo "SOLVER_TIME       : $SOLVER_TIME"
  echo "TUNING_OBJECTIVE  : $TUNING_OBJECTIVE"
  echo "BASE_DIR          : $BASE_DIR"
  echo "SAVE_BASE         : $SAVE_BASE"
  echo "------------------------------------------"

  CPLEX_TEST_RC="NA"
  CPLEX_ELAPSED_SECONDS="NA"
  BEST_PRM_FOUND=0

  INSTANCE_PATH="${INSTANCE_DIR}/${instance}"
  if [[ ! -f "$INSTANCE_PATH" ]]; then
    echo "[!] Instance introuvable : $INSTANCE_PATH"
    failed_instances+=("$instance (missing file)")
    ((++failure))
    echo "${instance},missing_file,0,NA,NA,NA" >>"$SUMMARY_CSV"
    continue
  fi

  echo "[*] Nettoyage temporaire"
  rm -rf "$BASE_DIR"
  mkdir -p "$BASE_DIR"

  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  CLEAN_INSTANCE_NAME="${instance%.mps}"
  SAVE_DIR="${SAVE_BASE}/${CLEAN_INSTANCE_NAME}_${TIMESTAMP}"
  mkdir -p "$SAVE_DIR"

  LOG_FILE="$SAVE_DIR/run.log"

  echo "[*] Lancement du tuner sur '$instance'"

  srun \
    --ntasks="$MPI_PROCS" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --distribution=block:block \
    --cpu-bind=cores \
    --mem-bind=local \
    "$TUNER_APP" \
      "$INSTANCE_PATH" \
      --solver-time "$SOLVER_TIME" \
      --solver-threads "$CPLEX_THREADS" \
      --tuning-objective "$TUNING_OBJECTIVE" \
      --working-dir "$BASE_DIR" \
      </dev/null >"$LOG_FILE" 2>&1

  TUNER_RC=$?
  echo "[*] Code retour du tuner pour '$instance' : $TUNER_RC"

  if [[ $TUNER_RC -ne 0 ]]; then
    echo "[!] Erreur détectée pour '$instance', mais on continue"
    ((++failure))
    failed_instances+=("$instance (rc=$TUNER_RC)")
  else
    ((++success))
  fi

  if [[ -d "$BASE_DIR" ]]; then
    echo "[*] Sauvegarde des résultats dans $SAVE_DIR"
    cp -r "$BASE_DIR"/. "$SAVE_DIR"/
  else
    echo "[!] Aucun dossier de travail à sauvegarder pour '$instance'"
  fi

  BEST_PRM_FILE="${SAVE_DIR}/best_configuration.prm"

  if [[ -f "$BEST_PRM_FILE" ]]; then
    BEST_PRM_FOUND=1

    if [[ "$ENABLE_CPLEX_TEST" -eq 1 ]]; then
      run_cplex_test "$INSTANCE_PATH" "$BEST_PRM_FILE" "$SAVE_DIR"
    else
      echo "[*] Test CPLEX désactivé (ENABLE_CPLEX_TEST=0)"
      CPLEX_TEST_RC="SKIPPED"
      CPLEX_ELAPSED_SECONDS="NA"
    fi

  else
    echo "[!] Aucun best_configuration.prm trouvé dans $SAVE_DIR"
  fi

  echo "${instance},${TUNER_RC},${BEST_PRM_FOUND},${CPLEX_TEST_RC},${CPLEX_ELAPSED_SECONDS},${SAVE_DIR}" >>"$SUMMARY_CSV"

  echo "Instance '$instance' terminée."
done

echo
echo "=========================================="
echo "Tous les tests sont terminés."
echo "Nombre total : $count"
echo "Succès tuner : $success"
echo "Échecs tuner : $failure"
echo "CSV résumé   : $SUMMARY_CSV"

if [[ ${#failed_instances[@]} -gt 0 ]]; then
  echo "Instances en échec :"
  for item in "${failed_instances[@]}"; do
    echo "  - $item"
  done
fi
echo "=========================================="
