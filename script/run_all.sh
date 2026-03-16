#!/bin/bash

# === VALEURS PAR DEFAUT ===
DEFAULT_MPI_PROCS=1
DEFAULT_SOLVER_TIME=5
DEFAULT_TUNING_OBJECTIVE="gap"
DEFAULT_ENABLE_TRACE=1

# === CHEMINS RACINES ===
REPO_DIR="/home/yorig/tuner/mpils"
RESULT_ROOT="/home/yorig/tuner/result"
INSTANCE_DIR="/home/yorig/miplib/mpils-miplib-test"
CPLEX_APP="/home/yorig/CPLEX_Studio2212/cplex/bin/x86-64_linux/cplex"

# === LISTE DES INSTANCES ===
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
  # "neos-3004026-krka.mps"
  "neos-4413714-turia.mps"
  "neos-4722843-widden.mps"
  "pk1.mps"
  "supportcase18.mps"
  "supportcase33.mps"
  "supportcase7.mps"
  "swath3.mps"
  "trento1.mps"
  "triptim1.mps"
)

usage() {
  cat <<EOF
Usage:
  $0 [--mpi N] [--solver-time SECONDS] [--tuning-objective gap|upper_bound] [--trace 0|1]
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

  if ! command -v "$CPLEX_APP" >/dev/null 2>&1; then
    echo "[!] CPLEX introuvable: $CPLEX_APP"
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

# === INITIALISATION DES PARAMETRES ===
MPI_PROCS="$DEFAULT_MPI_PROCS"
SOLVER_TIME="$DEFAULT_SOLVER_TIME"
TUNING_OBJECTIVE="$DEFAULT_TUNING_OBJECTIVE"
ENABLE_TRACE="$DEFAULT_ENABLE_TRACE"

# === PARSING DES ARGUMENTS ===
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mpi)
      [[ $# -ge 2 ]] || { echo "Erreur: --mpi nécessite une valeur"; usage; exit 1; }
      MPI_PROCS="$2"
      shift 2
      ;;
    --solver-time)
      [[ $# -ge 2 ]] || { echo "Erreur: --solver-time nécessite une valeur"; usage; exit 1; }
      SOLVER_TIME="$2"
      shift 2
      ;;
    --tuning-objective)
      [[ $# -ge 2 ]] || { echo "Erreur: --tuning-objective nécessite une valeur"; usage; exit 1; }
      TUNING_OBJECTIVE="$2"
      shift 2
      ;;
    --trace)
      [[ $# -ge 2 ]] || { echo "Erreur: --trace nécessite 0 ou 1"; usage; exit 1; }
      ENABLE_TRACE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Erreur: argument inconnu '$1'"
      usage
      exit 1
      ;;
  esac
done

# === VALIDATION ===
[[ "$MPI_PROCS" =~ ^[1-9][0-9]*$ ]] || { echo "Erreur: --mpi doit être un entier >= 1"; exit 1; }
[[ "$SOLVER_TIME" =~ ^[1-9][0-9]*$ ]] || { echo "Erreur: --solver-time doit être un entier >= 1"; exit 1; }
[[ "$ENABLE_TRACE" =~ ^[01]$ ]] || { echo "Erreur: --trace doit valoir 0 ou 1"; exit 1; }

case "$TUNING_OBJECTIVE" in
  gap|upper_bound)
    ;;
  *)
    echo "Erreur: --tuning-objective doit valoir 'gap' ou 'upper_bound'"
    exit 1
    ;;
esac

# === CHEMINS CALCULES AUTOMATIQUEMENT ===
RUN_TAG="${MPI_PROCS}proc-${SOLVER_TIME}s-${TUNING_OBJECTIVE}"
BASE_DIR="${REPO_DIR}/tmp/${RUN_TAG}"
SAVE_BASE="${RESULT_ROOT}/${RUN_TAG}"
TUNER_APP="${REPO_DIR}/build/mpils"
SUMMARY_CSV="${SAVE_BASE}/summary.csv"

# === OPTIONS SHELL ===
set -u
if [[ "$ENABLE_TRACE" -eq 1 ]]; then
  set -x
fi

# === VERIFICATIONS ===
mkdir -p "$BASE_DIR"
mkdir -p "$SAVE_BASE"

[[ -x "$TUNER_APP" ]] || { echo "Erreur: tuner non exécutable: $TUNER_APP"; exit 1; }
[[ -d "$INSTANCE_DIR" ]] || { echo "Erreur: dossier d'instances introuvable: $INSTANCE_DIR"; exit 1; }

if [[ "$MPI_PROCS" -gt 1 ]]; then
  command -v mpirun >/dev/null 2>&1 || {
    echo "Erreur: mpirun introuvable alors que MPI_PROCS=$MPI_PROCS"
    exit 1
  }
fi

# === INITIALISATION DU CSV ===
echo "instance,tuner_rc,best_prm_found,cplex_test_rc,cplex_elapsed_seconds,save_dir" >"$SUMMARY_CSV"

count=0
success=0
failure=0
failed_instances=()

for instance in "${INSTANCE_LIST[@]}"; do
  ((count++))

  echo
  echo "------------------------------------------"
  echo "Instance #$count : '$instance'"
  echo "Date / Heure      : $(date)"
  echo "MPI_PROCS         : $MPI_PROCS"
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
    ((failure++))
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

  if [[ "$MPI_PROCS" -eq 1 ]]; then
    "$TUNER_APP" \
      "$INSTANCE_PATH" \
      --solver-time "$SOLVER_TIME" \
      --tuning-objective "$TUNING_OBJECTIVE" \
      --working-dir "$BASE_DIR" \
      </dev/null >"$LOG_FILE" 2>&1
  else
    mpirun -np "$MPI_PROCS" \
      "$TUNER_APP" \
      "$INSTANCE_PATH" \
      --solver-time "$SOLVER_TIME" \
      --tuning-objective "$TUNING_OBJECTIVE" \
      --working-dir "$BASE_DIR" \
      </dev/null >"$LOG_FILE" 2>&1
  fi

  TUNER_RC=$?
  echo "[*] Code retour du tuner pour '$instance' : $TUNER_RC"

  if [[ $TUNER_RC -ne 0 ]]; then
    echo "[!] Erreur détectée pour '$instance', mais on continue"
    ((failure++))
    failed_instances+=("$instance (rc=$TUNER_RC)")
  else
    ((success++))
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
    run_cplex_test "$INSTANCE_PATH" "$BEST_PRM_FILE" "$SAVE_DIR"
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
