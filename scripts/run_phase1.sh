#!/bin/bash
source "$(dirname "$0")/common.sh"

echo "========================================="
echo "Phase 1: Meta-CoT Data Generation + SFT"
echo "========================================="

# Step 1.1: Generate Meta-CoT chains via TRAPI
echo "[Phase 1.1] Generating Meta-CoT chains via GPT-5.4..."
python -m src.metacot.generator --config configs/phase1_metacot.yaml

# Step 1.1b: Build SFT dataset from chains
echo "[Phase 1.1b] Building SFT dataset..."
python -m src.metacot.generator \
    --build-sft \
    --metacot-path /scratch/metacognition/metacot_chains/metacot_final.parquet \
    --sft-output /scratch/metacognition/metacot_chains/metacot_sft.parquet

# Step 1.2: Run Meta-CoT SFT
echo "[Phase 1.2] Running Meta-CoT SFT..."
accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    -m src.training.sft \
    --config configs/phase1_sft.yaml

# Backup checkpoint
echo "[Backup] Copying SFT checkpoint to /mnt/input/..."
cp -r /scratch/metacognition/checkpoints/phase1_sft /mnt/input/metacognition/phase1_sft

echo "Phase 1 complete!"
