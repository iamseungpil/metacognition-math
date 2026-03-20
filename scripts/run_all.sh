#!/bin/bash
# Full experiment pipeline — run all phases sequentially
# Each phase depends on the previous one's output
set -e

# Setup environment
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

eval "$(conda shell.bash hook 2>/dev/null || true)"
conda activate ptca 2>/dev/null || echo "Warning: ptca env not found"

export WANDB_API_KEY="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"
export WANDB_PROJECT="metacot-math"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

echo "Project dir: $PROJECT_DIR"
echo "Python: $(which python)"
python -c "import torch; print(f'torch={torch.__version__}, GPUs={torch.cuda.device_count()}')"

# =========================================
# Phase 0: Rollout Generation + Profiling
# =========================================
echo ""
echo "========================================="
echo "Phase 0: Rollout Generation + Profiling"
echo "========================================="

mkdir -p /scratch/metacognition/{rollouts,profiles,gnosis_data}

echo "[Phase 0.1] Generating rollouts..."
python -m src.rollout.vllm_rollout --config configs/phase0_rollout.yaml

echo "[Phase 0.3] Building capability profile..."
python -m src.rollout.vllm_rollout \
    --profile-only \
    --rollouts-path /scratch/metacognition/rollouts/rollouts_final.parquet \
    --profile-output /scratch/metacognition/profiles/capability_profile.json

echo "[Phase 0.4] Caching hidden states..."
python -m src.rollout.hidden_cache \
    --rollouts-path /scratch/metacognition/rollouts/rollouts_final.parquet \
    --output-dir /scratch/metacognition/gnosis_data \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --num-samples 30000

# Backup
mkdir -p /mnt/input/metacognition/
cp /scratch/metacognition/profiles/capability_profile.json /mnt/input/metacognition/

echo "Phase 0 complete!"

# =========================================
# Phase 1: Meta-CoT Data Generation + SFT
# =========================================
echo ""
echo "========================================="
echo "Phase 1: Meta-CoT Data Generation + SFT"
echo "========================================="

echo "[Phase 1.1] Generating Meta-CoT chains via GPT-5.4..."
python -m src.metacot.generator --config configs/phase1_metacot.yaml

echo "[Phase 1.1b] Building SFT dataset..."
python -m src.metacot.generator \
    --build-sft \
    --metacot-path /scratch/metacognition/metacot_chains/metacot_final.parquet \
    --sft-output /scratch/metacognition/metacot_chains/metacot_sft.parquet

echo "[Phase 1.2] Running Meta-CoT SFT..."
accelerate launch \
    --num_processes 4 \
    --mixed_precision bf16 \
    -m src.training.sft \
    --config configs/phase1_sft.yaml

cp -r /scratch/metacognition/checkpoints/phase1_sft /mnt/input/metacognition/phase1_sft

echo "Phase 1 complete!"

# =========================================
# Phase 2: Probe Training + GRPO
# =========================================
echo ""
echo "========================================="
echo "Phase 2: Probe Training + GRPO"
echo "========================================="

echo "[Phase 2.0] Training simple probe..."
python -m src.probes.simple_probe \
    --data-dir /scratch/metacognition/gnosis_data \
    --output-dir /scratch/metacognition/checkpoints/simple_probe \
    --hidden-dim 3584 \
    --epochs 10

echo "[Phase 2.2] Running GRPO..."
python -m src.training.grpo --config configs/phase2_grpo.yaml

cp -r /scratch/metacognition/checkpoints/phase2_grpo/final /mnt/input/metacognition/phase2_grpo 2>/dev/null || true

echo "Phase 2 complete!"

# =========================================
# Evaluation
# =========================================
echo ""
echo "========================================="
echo "Evaluation: All models on all benchmarks"
echo "========================================="

python -m src.eval.evaluator --config configs/eval.yaml

cp -r /scratch/metacognition/results /mnt/input/metacognition/ 2>/dev/null || true

echo ""
echo "========================================="
echo "ALL PHASES COMPLETE!"
echo "========================================="
