#!/bin/bash
# GRPO v2: Full FT + vLLM colocate + ZeRO-3 (4 GPU)
# Proven working: loss=0.067, grad_norm=1.4, 15s/step
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate grpo

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

MODE=${1:-E1}
STEPS=${2:-200}
MODEL_PATH=${3:-checkpoints/qwen3_meta_sft}

echo "=== GRPO v2: $MODE, $STEPS steps ==="
echo "Config: ZeRO-3 + vLLM colocate, 4xA100, Full FT"

accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/grpo_v2.py \
    --mode $MODE \
    --max_steps $STEPS \
    --model_path $MODEL_PATH \
    --data filtered

echo "=== DONE: $MODE ==="
