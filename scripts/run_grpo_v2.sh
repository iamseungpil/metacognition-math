#!/bin/bash
# GRPO v2: Full FT + FSDP + HF generate (4 GPU)
# Based on Open-R1 patterns, adapted for 4xA100 80GB
# Usage: bash scripts/run_grpo_v2.sh E1 200
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
echo "Config: FSDP FULL_SHARD + HF generate, 4xA100, Full FT"
echo "Batch: 4/GPU × 4GPU × 2accum = 32, num_gen=8 → 4 unique prompts/step"

accelerate launch --config_file configs/accelerate_fsdp.yaml \
    src/training/grpo_v2.py \
    --mode $MODE \
    --max_steps $STEPS \
    --model_path $MODEL_PATH \
    --data filtered

echo "=== DONE: $MODE ==="
