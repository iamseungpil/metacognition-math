#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config_path>" >&2
  exit 1
fi

CONFIG_PATH="$1"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$REMOTE_CONDA_ENV"
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="$(cat ~/.wandb_key 2>/dev/null || echo '${WANDB_API_KEY}')"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
if pgrep -fa "src/training/sft.py --config $CONFIG_PATH" >/dev/null; then
  echo "Refusing duplicate SFT launch for $CONFIG_PATH" >&2
  pgrep -fa "src/training/sft.py --config $CONFIG_PATH" >&2 || true
  exit 2
fi
"$PYTHON_BIN" -m accelerate.commands.launch --config_file "${ACCELERATE_CONFIG:-configs/accelerate_sft.yaml}" \
  --main_process_port 0 \
  src/training/sft.py --config "$CONFIG_PATH"
