#!/bin/bash
# Run short GRPO training + AIME evaluation
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh && conda activate ptca
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

echo "=== GRPO Training (short) ==="
python -m src.training.grpo --config configs/phase2_grpo.yaml

echo "=== AIME Evaluation: Base model ==="
bash scripts/eval_aime_quick.sh Qwen/Qwen2.5-7B-Instruct 2>&1 | grep "aime2025"

echo "=== AIME Evaluation: SFT model ==="
bash scripts/eval_aime_quick.sh /scratch/metacognition/checkpoints/phase1_sft 2>&1 | grep "aime2025"

echo "=== AIME Evaluation: GRPO model ==="
bash scripts/eval_aime_quick.sh /scratch/metacognition/checkpoints/phase2_grpo/final 2>&1 | grep "aime2025"

echo "ALL_EVAL_DONE"
