#!/bin/bash
# Eval: compare Meta SFT vs GRPO E1
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

echo "=== Eval: Meta SFT ==="
python src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_meta_sft \
    --benchmarks gsm8k math_test \
    --max_problems 50 \
    --output_dir results 2>&1 | tee results/eval_meta_sft.log

echo "=== Eval: GRPO E1 ==="
python src/eval/eval_hf.py \
    --model_path checkpoints/grpo_v2_E1/checkpoint-200 \
    --benchmarks gsm8k math_test \
    --max_problems 50 \
    --output_dir results 2>&1 | tee results/eval_grpo_e1.log

echo "=== EVAL DONE ==="
