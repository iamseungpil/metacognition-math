#!/usr/bin/env bash
# scripts/local/make_data.sh — build local Countdown train/val parquets, once.
#
# Output: $WORK/data/countdown_{train,val}_4num_<variant>.parquet
#   train: n=8000 seed=1   (matches the "8000/500" scale used in prior rounds —
#          see docs/PREREGISTRATION_countdown_osd_round2.md §5 and FINDINGS)
#   val:   n=500  seed=2
#   n_nums=4 (per task spec; countdown_task.py's own default is 5 — DEFAULT_N_NUMS
#             — so --n_nums 4 must be passed explicitly every time this data is
#             regenerated or reused elsewhere)
#
# Variants requested: plain, new, p3. `p3` does not exist in the countdown_task.py
# checked out at the time this script was written (PROMPT_VARIANTS = new/old/shot/
# plain) — it is expected to land from a concurrent fix. This script probes for
# each variant's existence in PROMPT_VARIANTS before building and SKIPS (not
# fails) any variant that isn't defined yet, so re-running this script after the
# fix lands picks up p3 without edits.
#
# Idempotent: skips a file that already exists. Delete the file to force a rebuild.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/env.sh"

cd "${REPO_ROOT}"

TRAIN_N=8000
TRAIN_SEED=1
VAL_N=500
VAL_SEED=2
N_NUMS=4
VARIANTS=(plain new p3 opt)

mkdir -p "${WORK}/data"

variant_exists() {
  python -c "
from src.training.countdown_task import PROMPT_VARIANTS
import sys
sys.exit(0 if '$1' in PROMPT_VARIANTS else 1)
" 2>/dev/null
}

for variant in "${VARIANTS[@]}"; do
  if ! variant_exists "${variant}"; then
    echo "[make_data] SKIP variant='${variant}': not yet in src.training.countdown_task.PROMPT_VARIANTS"
    continue
  fi

  train_out="${WORK}/data/countdown_train_4num_${variant}.parquet"
  val_out="${WORK}/data/countdown_val_4num_${variant}.parquet"

  if [ -f "${train_out}" ]; then
    echo "[make_data] train exists, skip: ${train_out}"
  else
    echo "[make_data] building train (n=${TRAIN_N} seed=${TRAIN_SEED} variant=${variant}) -> ${train_out}"
    python -m src.training.countdown_task \
      --n "${TRAIN_N}" --seed "${TRAIN_SEED}" --variant "${variant}" \
      --n_nums "${N_NUMS}" --split train --out "${train_out}"
  fi

  if [ -f "${val_out}" ]; then
    echo "[make_data] val exists, skip: ${val_out}"
  else
    echo "[make_data] building val (n=${VAL_N} seed=${VAL_SEED} variant=${variant}) -> ${val_out}"
    python -m src.training.countdown_task \
      --n "${VAL_N}" --seed "${VAL_SEED}" --variant "${variant}" \
      --n_nums "${N_NUMS}" --split val --out "${val_out}"
  fi
done

echo "[make_data] done. Contents of ${WORK}/data:"
ls -la "${WORK}/data" || true
