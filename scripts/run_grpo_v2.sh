#!/bin/bash
# GRPO v2: Full FT + vLLM server (GPU 0) + Training (GPU 1-3)
# Usage: bash scripts/run_grpo_v2.sh E1
#        bash scripts/run_grpo_v2.sh E3 500
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate grpo

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

MODE=${1:-E1}
STEPS=${2:-200}
MODEL_PATH=${3:-checkpoints/qwen3_meta_sft}

echo "=== GRPO v2: $MODE, $STEPS steps, vLLM server mode ==="
echo "GPU 0: vLLM server | GPU 1-3: Training"

# Step 1: Start vLLM server on GPU 0
echo "Starting vLLM server on GPU 0..."
CUDA_VISIBLE_DEVICES=0 trl vllm-serve \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --dtype bfloat16 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM to be ready
echo "Waiting for vLLM server..."
for i in $(seq 1 60); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "vLLM server ready!"
        break
    fi
    sleep 5
done

# Step 2: Training on GPU 1-3
echo "Starting training on GPU 1-3..."
CUDA_VISIBLE_DEVICES=1,2,3 accelerate launch --num_processes 3 --multi_gpu \
    src/training/grpo_v2.py \
    --mode $MODE \
    --max_steps $STEPS \
    --model_path $MODEL_PATH \
    --data filtered

echo "=== Training done, stopping vLLM ==="
kill $VLLM_PID 2>/dev/null
echo "=== DONE: $MODE ==="
