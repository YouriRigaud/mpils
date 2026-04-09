#!/bin/bash
set -u

# ============================================================================
# Run CPLEX with default parameters on the hardcoded instance list
# and save elapsed time for each instance.
# ============================================================================

# === CHEMINS RACINES ===
RESULT_ROOT="/home/yorig/tuner/result/default_cplex"
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

SUMMARY_CSV="${RESULT_ROOT}/summary_default.csv"

run_cplex_default_test() {
  local instance_path="$1"
  local save_dir="$2"

  local cplex_log="${save_dir}/cplex_default.log"
  local cplex_time_file="${save_dir}/cplex_default_time.txt"
  local cplex_status_file="${save_dir}/cplex_default_status.txt"

  CPLEX_TEST_RC=-1
  CPLEX_ELAPSED_SECONDS="NA"

  echo "[*] Test CPLEX par défaut sur: $instance_path"

  if [[ ! -x "$CPLEX_APP" ]]; then
    echo "[!] CPLEX introuvable ou non exécutable: $CPLEX_APP"
    echo "CPLEX executable not found: $CPLEX_APP" >"$cplex_status_file"
    return 1
  fi

  local start_ts
  local end_ts
  local elapsed

  start_ts=$(date +%s)

  "$CPLEX_APP" -c \
    "read $instance_path" \
    "read /home/yorig/tuner/mpils/default.prm" \
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
    echo "mode=default"
    echo "return_code=$CPLEX_TEST_RC"
    echo "elapsed_seconds=$CPLEX_ELAPSED_SECONDS"
  } >"$cplex_time_file"

  if [[ $CPLEX_TEST_RC -eq 0 ]]; then
    echo "CPLEX default test finished successfully" >"$cplex_status_file"
    echo "[*] Test CPLEX par défaut terminé en ${CPLEX_ELAPSED_SECONDS}s"
  else
    echo "CPLEX default test failed with return code $CPLEX_TEST_RC" >"$cplex_status_file"
    echo "[!] Test CPLEX par défaut échoué avec code retour $CPLEX_TEST_RC après ${CPLEX_ELAPSED_SECONDS}s"
  fi

  return $CPLEX_TEST_RC
}

mkdir -p "$RESULT_ROOT"

echo "instance,cplex_default_rc,cplex_default_elapsed_seconds,save_dir" >"$SUMMARY_CSV"

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
  echo "RESULT_ROOT       : $RESULT_ROOT"
  echo "------------------------------------------"

  CPLEX_TEST_RC="NA"
  CPLEX_ELAPSED_SECONDS="NA"

  INSTANCE_PATH="${INSTANCE_DIR}/${instance}"
  if [[ ! -f "$INSTANCE_PATH" ]]; then
    echo "[!] Instance introuvable : $INSTANCE_PATH"
    failed_instances+=("$instance (missing file)")
    ((failure++))
    echo "${instance},missing_file,NA,NA" >>"$SUMMARY_CSV"
    continue
  fi

  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  CLEAN_INSTANCE_NAME="${instance%.mps}"
  SAVE_DIR="${RESULT_ROOT}/${CLEAN_INSTANCE_NAME}_${TIMESTAMP}"
  mkdir -p "$SAVE_DIR"

  run_cplex_default_test "$INSTANCE_PATH" "$SAVE_DIR"
  RC=$?

  if [[ $RC -ne 0 ]]; then
    ((failure++))
    failed_instances+=("$instance (rc=$RC)")
  else
    ((success++))
  fi

  echo "${instance},${CPLEX_TEST_RC},${CPLEX_ELAPSED_SECONDS},${SAVE_DIR}" >>"$SUMMARY_CSV"

  echo "Instance '$instance' terminée."
done

echo
echo "=========================================="
echo "Tous les tests par défaut sont terminés."
echo "Nombre total : $count"
echo "Succès       : $success"
echo "Échecs       : $failure"
echo "CSV résumé   : $SUMMARY_CSV"

if [[ ${#failed_instances[@]} -gt 0 ]]; then
  echo "Instances en échec :"
  for item in "${failed_instances[@]}"; do
    echo "  - $item"
  done
fi
echo "=========================================="
