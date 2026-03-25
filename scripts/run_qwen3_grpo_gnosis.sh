#!/bin/bash
# Phase 3: GRPO + Full Gnosis on Qwen3-8B Meta SFT
set -e

# Fix OpenSSL FIPS
export OPENSSL_CONF=/dev/null
export OPENSSL_ia32cap="~0x200000200000000"
sudo sed -i 's/^\.include.*fips.*//g; s/^fips = fips_sect/# fips = fips_sect/g; s/^activate = 1/# activate = 1/g' /etc/ssl/openssl.cnf 2>/dev/null || true

source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
export OPENSSL_CONF=/dev/null

# Pin: torch 2.5.1 + transformers 4.51.3 (Qwen3) + trl 0.19.1 (GRPOTrainer)
# flash-attn 2.8.3 needs torch 2.5 (not 2.6)
pip install "transformers==4.51.3" "trl==0.19.1" "peft>=0.10" --quiet 2>/dev/null || true
echo "Installed: torch=$(python -c 'import torch;print(torch.__version__)'), trl=$(python -c 'import trl;print(trl.__version__)'), tf=$(python -c 'import transformers;print(transformers.__version__)')"

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

# Phase 1: Use standard Qwen3 + TRL (no Gnosis model modifications)
# Gnosis integration will be added in Phase 2 after basic GRPO works
echo "Using standard Qwen3 model + pip TRL"
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

echo "=== Phase 3: GRPO + Full Gnosis ==="
echo "Model: checkpoints/qwen3_meta_sft"
echo "Gnosis: Full (attention + hidden + confidence)"
echo "Stepwise: Agent Lightning style (R_correct to ALL steps)"

# Multi-GPU (4x A100) — single GPU OOMs on Qwen3-8B
accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_gnosis.py \
    --model_path checkpoints/qwen3_meta_sft \
    --train_data verl_train.parquet \
    --output_dir checkpoints/qwen3_grpo_gnosis \
    --max_steps 1000

echo "=== Phase 3 DONE ==="
