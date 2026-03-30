#!/bin/bash
# Large-scale eval: 4 models x ~1030 problems x max_tokens=4096
# GSM8K: 500 (of 1319), MATH-500: 500 (all), AIME2024: 30 (all) = 1030 total
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

BENCHMARKS="gsm8k math500 aime2024"

# GSM8K has 1319 test problems, MATH-500 has 500, AIME2024 has 30
# We use max_problems=500 for GSM8K and MATH, 30 for AIME (all available)
MAX=500

mkdir -p results

echo "=== Large-scale eval: 3 models x ~1030 problems ==="

CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_base_sft \
    --model_name 1030_base_sft \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir results > results/eval_1030_base.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_metacot_v2_sft \
    --model_name 1030_v2_sft \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir results > results/eval_1030_v2sft.log 2>&1 &

# E3 GRPO checkpoint — prefer /final, fallback to latest checkpoint
E3_PATH="checkpoints/grpo_v2_E3/final"
if [ ! -d "$E3_PATH" ]; then
    E3_PATH=$(ls -d checkpoints/grpo_v2_E3/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
if [ -z "$E3_PATH" ] || [ ! -d "$E3_PATH" ]; then
    echo "ERROR: E3 checkpoint not found at checkpoints/grpo_v2_E3/"
    exit 1
fi
echo "Using E3 checkpoint: $E3_PATH"
CUDA_VISIBLE_DEVICES=2 nohup python -u src/eval/eval_hf.py \
    --model_path $E3_PATH \
    --model_name 1030_e3 \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir results > results/eval_1030_e3.log 2>&1 &

echo "3 parallel evals started. Monitor with:"
echo "  tail -f results/eval_1030_*.log"
wait
echo "=== ALL EVAL DONE ==="
