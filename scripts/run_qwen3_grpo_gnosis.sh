#!/bin/bash
# Phase 3: GRPO + Full Gnosis on Qwen3-8B Meta SFT
# Uses gnosis_repo's TRL GRPOTrainer with vLLM colocate, 4 GPU
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate verl
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

# Use gnosis_repo's transformers and TRL (with Gnosis integration)
export PYTHONPATH="/scratch/metacognition/gnosis_repo/transformers/src:/scratch/metacognition/gnosis_repo/trl:$PYTHONPATH"
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

# Fix OpenSSL FIPS for Ray workers
export OPENSSL_ia32cap="~0x200000200000000"

echo "=== Phase 3: GRPO + Full Gnosis (4 GPU) ==="
echo "Model: checkpoints/qwen3_meta_sft (Qwen3-8B + Meta-CoT SFT)"
echo "Gnosis: Full (attention + hidden + confidence feature extractors)"
echo "Reward: R_correct + R_calib + R_penalty + stepwise importance"
echo "Generation: vLLM colocate, 4 GPU"

# FIX A1: Use accelerate for multi-GPU launch
# TRL GRPOTrainer with vLLM colocate needs distributed environment
accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_gnosis.py \
    --model_path checkpoints/qwen3_meta_sft \
    --train_data verl_train.parquet \
    --output_dir checkpoints/qwen3_grpo_gnosis \
    --max_steps 1000

echo "=== Phase 3 DONE ==="
