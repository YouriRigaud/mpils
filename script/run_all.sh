#!/bin/bash
#set -euo pipefail

set -x

# === CONFIGURATION ===
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
  "neos-3004026-krka.mps"
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

INSTANCE_DIR="/home/rigayour/miplib/mpils-miplib-test"
TMP_DIR="/home/rigayour/mpils/tuner_working_dir"
CLEAN_SCRIPT="/home/rigayour/mpils/script/clean_working_dir.sh"
TUNER_APP="/home/rigayour/mpils/build/mpils"
SAVE_BASE="/home/rigayour/MPILS_result_v2/tuner-test-14-fev-all-deterministic-2-taches"

# === CREATE SAVE DIR ===
mkdir -p "$SAVE_BASE"

count=0
for instance in "${INSTANCE_LIST[@]}"; do
  ((count++))
  echo
  echo "------------------------------------------"
  echo "Instance #$count : '$instance'"
  echo "Date / Heure : $(date)"
  echo "------------------------------------------"

  echo "[*] Nettoyage temporaire"
  bash "$CLEAN_SCRIPT" || echo "[!] Le nettoyage a échoué, mais on continue"

  INSTANCE_PATH="${INSTANCE_DIR}/${instance}"

  echo "[*] Lancement du tuner sur '$instance'"
  #"$TUNER_APP" "${INSTANCE_PATH}" </dev/null
  /usr/lib64/openmpi/bin/mpirun -np 2 "$TUNER_APP" "${INSTANCE_PATH}" </dev/null
  rc=$?
  echo "[*] Code retour du tuner pour '$instance' : $rc"

  if [[ $rc -ne 0 ]]; then
    echo "[!] Erreur détectée pour '$instance', mais on continue"
  fi

  # === Sauvegarde des résultats ===
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  CLEAN_INSTANCE_NAME=$(basename "$instance")
  SAVE_DIR="${SAVE_BASE}/${CLEAN_INSTANCE_NAME}_${TIMESTAMP}"

  echo "[*] Sauvegarde des résultats dans $SAVE_DIR"
  cp -r "$TMP_DIR" "$SAVE_DIR"  

  echo "Instance '$instance' terminée et sauvegardée."
done

echo
echo "Tous les tests sont terminés. Nombre total : $count"
