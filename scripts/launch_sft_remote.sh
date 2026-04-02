#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <config_path>" >&2
  exit 1
fi

CONFIG_PATH="$1"

source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="$(cat ~/.wandb_key 2>/dev/null || echo '2f4e627868f1f9dad10bcb1a14fbf96817e6baa9')"

accelerate launch --config_file "${ACCELERATE_CONFIG:-configs/accelerate_sft.yaml}" \
  src/training/sft.py --config "$CONFIG_PATH"
