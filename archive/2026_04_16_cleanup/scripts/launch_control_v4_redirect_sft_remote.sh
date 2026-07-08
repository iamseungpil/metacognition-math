#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$SCRIPT_DIR/launch_sft_remote.sh" ]]; then
  exec "$SCRIPT_DIR/launch_sft_remote.sh" configs/sft_control_v4_redirect.yaml
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="$(cat ~/.wandb_key 2>/dev/null || echo '${WANDB_API_KEY}')"
accelerate launch --config_file "${ACCELERATE_CONFIG:-configs/accelerate_sft.yaml}" \
  src/training/sft.py --config configs/sft_control_v4_redirect.yaml
