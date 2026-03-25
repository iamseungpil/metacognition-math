#!/bin/bash
# GRPO v2 with accelerate (4 GPU) + Flash Attention + Gnosis
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

echo "=== GRPO v2: 4 GPU accelerate + Flash Attention + Gnosis ==="
accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m src.training.grpo --config configs/phase2_grpo.yaml
