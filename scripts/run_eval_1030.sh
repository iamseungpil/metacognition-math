#!/bin/bash
# Large-scale eval: 4 models x ~1030 problems x max_tokens=4096
# GSM8K: 500 (of 1319), MATH-500: 500 (all), AIME2024: 30 (all) = 1030 total
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

BENCHMARKS="gsm8k math500 aime2024"
MAX=500

mkdir -p results

echo "=== Large-scale eval: 4 models x ~1030 problems ==="

# GPU 0: Base SFT (no meta tokens — accuracy baseline)
CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_base_sft \
    --model_name 1030_base_sft \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir results > results/eval_1030_base.log 2>&1 &

# GPU 1: V2 Meta SFT (meta tokens, no GRPO)
CUDA_VISIBLE_DEVICES=1 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_metacot_v2_sft \
    --model_name 1030_v2_sft \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir results > results/eval_1030_v2sft.log 2>&1 &

# GPU 2: E3 checkpoint-100 (peak correctness phase)
CUDA_VISIBLE_DEVICES=2 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/grpo_v2_E3/checkpoint-100 \
    --model_name 1030_e3_ckpt100 \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir results > results/eval_1030_e3_ckpt100.log 2>&1 &

# GPU 3: E3 checkpoint-200 (more training, declining correctness)
CUDA_VISIBLE_DEVICES=3 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/grpo_v2_E3/checkpoint-200 \
    --model_name 1030_e3_ckpt200 \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir results > results/eval_1030_e3_ckpt200.log 2>&1 &

echo "4 parallel evals started. Monitor with:"
echo "  tail -f results/eval_1030_*.log"
wait
echo "=== ALL EVAL DONE ==="
