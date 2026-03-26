#!/bin/bash
# Phase 2: SimpleCorrectnessProbe Training
# No Gnosis patching needed — standard HF model + hidden states
set -e
export OPENSSL_CONF=/dev/null

source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
export OPENSSL_CONF=/dev/null

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

# Restore standard Qwen3 model (remove Gnosis patches from previous runs)
pip install transformers==4.51.3 --force-reinstall --no-deps --quiet 2>/dev/null || true
echo "Restored standard Qwen3 model"
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

echo "=== Phase 2: Simple Probe Training ==="
echo "Model: Qwen3-8B Meta SFT (frozen, for hidden state extraction)"
echo "Probe: MLP (4096 → 512 → 256 → 1)"
echo "Data: rollouts_final.parquet (balanced correct/incorrect)"

# Single GPU is enough — model is frozen, probe is tiny
CUDA_VISIBLE_DEVICES=0 python src/training/probe_sft.py \
    --model_path checkpoints/qwen3_meta_sft \
    --data_path rollouts/rollouts_final.parquet \
    --output_dir checkpoints/simple_probe_qwen3 \
    --max_length 2048 \
    --batch_size 64 \
    --lr 1e-3 \
    --epochs 10 \
    --max_samples 10000

echo "=== Phase 2 DONE ==="
