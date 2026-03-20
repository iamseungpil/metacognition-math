#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "========================================="
echo "Phase 2: Probe Training + GRPO"
echo "========================================="

export PYTHONPATH="${PWD}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

# Step 2.0: Train simple probe (baseline control)
echo "[Phase 2.0] Training simple hidden-state probe (baseline)..."
python -c "
from src.probes.simple_probe import train_simple_probe
import wandb
run = wandb.init(project='metacot-math', name='simple-probe-baseline')
train_simple_probe(
    data_dir='/scratch/metacognition/gnosis_data',
    output_dir='/scratch/metacognition/checkpoints/simple_probe',
    hidden_dim=3584,
    epochs=10,
    batch_size=32,
    lr=1e-3,
    wandb_run=run,
)
wandb.finish()
"

# Step 2.1: Train Gnosis probe (if repo configured for Qwen2.5)
echo "[Phase 2.1] Training Gnosis probe..."
# Use Gnosis repo's pipeline
cd gnosis_repo/open-r1
if [ -f recipes/training/Qwen2.5/Qwen2.5-7B_gnosis.yaml ]; then
    accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
        src/open_r1/sft.py \
        --config recipes/training/Qwen2.5/Qwen2.5-7B_gnosis.yaml
else
    echo "  Gnosis Qwen2.5 config not found, using simple probe as fallback"
fi
cd ../..

# Step 2.2: Run GRPO training
echo "[Phase 2.2] Running GRPO with R_meta..."
python -m src.training.grpo --config configs/phase2_grpo.yaml

# Backup checkpoints
echo "[Backup] Copying GRPO checkpoint to /mnt/input/..."
cp -r /scratch/metacognition/checkpoints/phase2_grpo/final /mnt/input/metacognition/phase2_grpo

echo "Phase 2 complete!"
