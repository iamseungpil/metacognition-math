#!/bin/bash
source "$(dirname "$0")/common.sh"

echo "========================================="
echo "Phase 2: Probe Training + GRPO"
echo "========================================="

# Step 2.0: Train simple probe (baseline control)
echo "[Phase 2.0] Training simple hidden-state probe (baseline)..."
python -m src.probes.simple_probe \
    --data-dir /scratch/metacognition/gnosis_data \
    --output-dir /scratch/metacognition/checkpoints/simple_probe \
    --hidden-dim 3584 \
    --epochs 10

# Step 2.2: Run GRPO training
echo "[Phase 2.2] Running GRPO with R_meta..."
python -m src.training.grpo --config configs/phase2_grpo.yaml

# Backup checkpoints
echo "[Backup] Copying checkpoints to /mnt/input/..."
cp -r /scratch/metacognition/checkpoints/phase2_grpo/final /mnt/input/metacognition/phase2_grpo

echo "Phase 2 complete!"
