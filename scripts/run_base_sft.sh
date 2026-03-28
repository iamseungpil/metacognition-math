#!/bin/bash
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

echo "=== Base SFT (no meta, ZeRO-3) ==="
accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/sft.py --config configs/phase1_base_sft.yaml
echo "=== DONE ==="
