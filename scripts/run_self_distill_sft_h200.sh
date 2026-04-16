#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml>" >&2
  exit 1
fi

CONFIG_PATH="$1"
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "FATAL: missing config $CONFIG_PATH" >&2
  exit 1
fi
NUM_PROCESSES="${NUM_PROCESSES:-4}"
ACCELERATE_CONFIG="${ACCELERATE_CONFIG:-configs/accelerate_sft.yaml}"

if [[ ! -f "$ACCELERATE_CONFIG" ]]; then
  echo "FATAL: missing accelerate config $ACCELERATE_CONFIG" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

VISIBLE_GPU_COUNT="$(python - <<'PY'
import os

visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
if not visible:
    print(0)
else:
    print(len([item for item in visible.split(",") if item.strip()]))
PY
)"
if [[ "$VISIBLE_GPU_COUNT" =~ ^[0-9]+$ ]] && [[ "$VISIBLE_GPU_COUNT" -gt 0 ]] && [[ "$NUM_PROCESSES" -gt "$VISIBLE_GPU_COUNT" ]]; then
  echo "[warn] num_processes=$NUM_PROCESSES but only $VISIBLE_GPU_COUNT visible GPU(s); lowering." >&2
  NUM_PROCESSES="$VISIBLE_GPU_COUNT"
fi

echo "[launch] config=$CONFIG_PATH"
echo "[launch] accelerate_config=$ACCELERATE_CONFIG"
echo "[launch] num_processes=$NUM_PROCESSES cuda_visible_devices=$CUDA_VISIBLE_DEVICES"

accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_PROCESSES" \
  src/training/sft.py \
  --config "$CONFIG_PATH"
