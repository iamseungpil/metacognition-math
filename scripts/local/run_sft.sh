#!/usr/bin/env bash
# scripts/local/run_sft.sh CONFIG [--dry-run]
#
# Run ONE local SFT training job on the single GPU the caller has assigned
# (this script does not touch CUDA_VISIBLE_DEVICES — same convention as
# scripts/local/run_arm.sh: the gpu_queue.py worker sets it, this script just
# runs on whatever it is handed).
#
# CONFIG   path to a src/training/sft.py yaml config (e.g.
#          configs/sft_coupling_v1.yaml) — model_name_or_path/dataset_path/
#          output_dir/etc, same schema as every configs/sft_*.yaml.
#
# accelerate config: single-GPU DeepSpeed ZeRO-3 + CPU optimizer offload
# (configs/accelerate_sft_1gpu_cpuoff.yaml) — the same recipe already used for
# 8B SFT2 runs on one A100 (see that file's header comment). zero3_save_16bit_model:
# true means src/training/sft.py's trainer.save_model(output_dir) at the end
# already writes MERGED bf16 safetensors + tokenizer.save_pretrained alongside
# it, so there is no separate merge step here (unlike verl's FSDP checkpoints
# in run_arm.sh).
#
# Storage rule (root CLAUDE.md, 0905 incident): nothing here writes to / or
# /tmp except tiny files. env.sh points HF/XDG/Ray/Triton caches at
# /hdd_data/seungpil already; this script only adds the log file, which also
# goes under $WORK/logs.
set -euo pipefail

_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

usage() {
  echo "Usage: $0 CONFIG [--dry-run]" >&2
  exit 1
}

DRY_RUN=0
POSITIONAL=()
for a in "$@"; do
  case "$a" in
    --dry-run) DRY_RUN=1 ;;
    *) POSITIONAL+=("$a") ;;
  esac
done
set -- "${POSITIONAL[@]:-}"
[ -z "${1:-}" ] && usage
CONFIG="${1:?CONFIG required}"

# shellcheck disable=SC1091
source "${_SCRIPT_DIR}/env.sh"
cd "${REPO_ROOT}"

if [ ! -f "${CONFIG}" ]; then
  echo "[run_sft] FATAL: config not found: ${CONFIG}" >&2
  exit 1
fi

# Resolve run_name from the config (for logging/wandb only; sft.py reads the
# config itself, this is not passed through as a CLI flag).
RUN_NAME=$(python -c "
import yaml
with open('${CONFIG}') as f:
    cfg = yaml.safe_load(f)
print(cfg.get('run_name', 'metacot-sft'))
")

ACCEL_CONFIG="${ACCEL_CONFIG:-configs/accelerate_sft_1gpu_cpuoff.yaml}"
LOG_FILE="${WORK}/logs/sft_${RUN_NAME}.log"

TRAIN_CMD=(python -m accelerate.commands.launch --config_file "${ACCEL_CONFIG}" \
  src/training/sft.py --config "${CONFIG}")

echo "[run_sft] CONFIG=${CONFIG} RUN_NAME=${RUN_NAME} ACCEL_CONFIG=${ACCEL_CONFIG}"
echo "[run_sft] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset! queue should have set this>}"
echo "[run_sft] exact command:"
printf '  %q' "${TRAIN_CMD[@]}"; echo
echo "[run_sft] log file: ${LOG_FILE}"

if [ "${DRY_RUN}" = "1" ]; then
  echo "[run_sft] --dry-run: not executing."
  exit 0
fi

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  echo "[run_sft] WARNING: CUDA_VISIBLE_DEVICES is unset. This script expects the queue" >&2
  echo "[run_sft]          worker to have set it to a single GPU index (0-3)." >&2
fi

mkdir -p "${WORK}/logs"

echo "[run_sft] starting training, appending to ${LOG_FILE}" | tee -a "${LOG_FILE}"
set +e
"${TRAIN_CMD[@]}" >> "${LOG_FILE}" 2>&1
SFT_RC=$?
set -e

echo "[run_sft] RUN_NAME=${RUN_NAME} rc=${SFT_RC} (see ${LOG_FILE})"
if [ "${SFT_RC}" != "0" ]; then
  echo "[run_sft] FATAL: training exited ${SFT_RC} (see ${LOG_FILE})" >&2
  exit "${SFT_RC}"
fi
echo "[run_sft] training exited 0"
