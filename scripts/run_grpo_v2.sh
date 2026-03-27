#!/bin/bash
# GRPO v2: Full FT + GDPO + modular rewards
# Usage: bash scripts/run_grpo_v2.sh E1
#        bash scripts/run_grpo_v2.sh E3 500
set -e
export OPENSSL_CONF=/dev/null
export OPENSSL_ia32cap="~0x200000200000000"
export LD_PRELOAD="/opt/conda/envs/ptca/lib/libcrypto.so.3:/opt/conda/envs/ptca/lib/libssl.so.3"

source /opt/conda/etc/profile.d/conda.sh
conda activate ptca

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

MODE=${1:-E1}
STEPS=${2:-200}

echo "=== GRPO v2: $MODE, $STEPS steps ==="

accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_v2.py \
    --mode $MODE \
    --max_steps $STEPS \
    --model_path checkpoints/qwen3_meta_sft \
    --data filtered

echo "=== DONE: $MODE ==="
