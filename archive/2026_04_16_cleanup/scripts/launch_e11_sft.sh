#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/conda/bin:$PATH"
source /opt/conda/etc/profile.d/conda.sh
conda activate ptca

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"

rm -rf checkpoints/control_v6_E11
mkdir -p results/control_v6_E11

echo "$(date) Starting E11 SFT"
accelerate launch --config_file configs/accelerate_grpo_z2.yaml \
    src/training/sft.py --config configs/sft_v6_e11.yaml \
    2>&1 | tee results/control_v6_E11/sft.log

echo "$(date) E11 SFT complete"
