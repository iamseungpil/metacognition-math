#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/conda/bin:$PATH"
source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="${WANDB_API_KEY}"

echo "$(date) Starting Slot C: E9 + 164 seed × 5 epochs"
accelerate launch --config_file configs/accelerate_grpo_z2.yaml \
    src/training/sft.py --config configs/sft_v6_e11_5ep.yaml \
    2>&1 | tee results/control_v6_SlotC/sft.log

echo "$(date) Slot C complete"
