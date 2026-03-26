#!/bin/bash
# Phase 3: GRPO + SimpleProbe on Qwen3-8B Meta SFT
set -e
export OPENSSL_CONF=/dev/null
export OPENSSL_ia32cap="~0x200000200000000"

source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
export OPENSSL_CONF=/dev/null

# Ensure standard transformers (no Gnosis patches)
pip install transformers==4.51.3 --force-reinstall --no-deps --quiet 2>/dev/null || true
pip install "trl==0.19.1" "peft>=0.10" --quiet 2>/dev/null || true

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

echo "=== Phase 3: GRPO + SimpleProbe ==="
echo "Model: Qwen3-8B Meta SFT + LoRA"
echo "Probe: SimpleCorrectnessProbe (AUROC ~0.95)"
echo "Rewards: R_correct + R_calib(probe) + R_penalty(meta)"
echo "4 GPU DDP"

accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_gnosis.py \
    --model_path checkpoints/qwen3_meta_sft \
    --probe_path checkpoints/simple_probe_qwen3/best_probe.pt \
    --train_data verl_train.parquet \
    --output_dir checkpoints/qwen3_grpo_probe \
    --max_completion_length 1024 \
    --max_steps 1000

echo "=== Phase 3 DONE ==="
