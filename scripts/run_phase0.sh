#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "========================================="
echo "Phase 0: Rollout Generation + Profiling"
echo "========================================="

export PYTHONPATH="${PWD}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Create output dirs
mkdir -p /scratch/metacognition/{rollouts,profiles,gnosis_data}

# Step 0.1-0.2: Generate rollouts with vLLM
echo "[Phase 0.1] Generating rollouts..."
python -m src.rollout.vllm_rollout --config configs/phase0_rollout.yaml

# Step 0.3: Build capability profile
echo "[Phase 0.3] Building capability profile..."
python -m src.rollout.vllm_rollout \
    --profile-only \
    --rollouts-path /scratch/metacognition/rollouts/rollouts_final.parquet \
    --profile-output /scratch/metacognition/profiles/capability_profile.json

# Step 0.4: Cache hidden states for probe training
echo "[Phase 0.4] Caching hidden states for probe training..."
python -m src.rollout.hidden_cache \
    --rollouts-path /scratch/metacognition/rollouts/rollouts_final.parquet \
    --output-dir /scratch/metacognition/gnosis_data \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --num-samples 30000

# Backup critical files to persistent storage
echo "[Backup] Copying to /mnt/input/metacognition/..."
mkdir -p /mnt/input/metacognition/
cp /scratch/metacognition/profiles/capability_profile.json /mnt/input/metacognition/
cp /scratch/metacognition/rollouts/rollouts_final.parquet /mnt/input/metacognition/

echo "Phase 0 complete!"
