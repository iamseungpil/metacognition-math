#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "$SCRIPT_DIR/launch_sft_remote.sh" ]]; then
  exec "$SCRIPT_DIR/launch_sft_remote.sh" configs/sft_control_v4_verify.yaml
fi

source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="$(cat ~/.wandb_key 2>/dev/null || echo '2f4e627868f1f9dad10bcb1a14fbf96817e6baa9')"
accelerate launch --config_file "${ACCELERATE_CONFIG:-configs/accelerate_sft.yaml}" \
  src/training/sft.py --config configs/sft_control_v4_verify.yaml
