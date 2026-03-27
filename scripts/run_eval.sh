#!/bin/bash
# Eval: compare Meta SFT vs GRPO E1
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

BENCHMARKS="gsm8k math500 aime2024"
MAX=30

echo "=== Eval: Meta SFT ==="
python src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_meta_sft \
    --benchmarks $BENCHMARKS \
    --max_problems $MAX \
    --output_dir results 2>&1 | tee results/eval_meta_sft.log

echo "=== Eval: GRPO E3 ==="
python src/eval/eval_hf.py \
    --model_path checkpoints/grpo_v2_E3/checkpoint-200 \
    --benchmarks $BENCHMARKS \
    --max_problems $MAX \
    --output_dir results 2>&1 | tee results/eval_grpo_e3.log

echo "=== EVAL DONE ==="
