#!/bin/bash
# 1,030-problem eval on EVAL node: 6 models across 2 rounds
# Round 1: 4 models on 4 GPUs in parallel
# Round 2: 2 models on 2 GPUs in parallel
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

BENCHMARKS="gsm8k math500 aime2024"
MAX=500
OUTDIR="results/eval_1030_v5"
mkdir -p "$OUTDIR"

echo "=== EVAL NODE: Round 1 (4 models) ==="

CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_base_sft \
    --model_name 1030_base_sft \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir "$OUTDIR" > "$OUTDIR/base_sft.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_metacot_control_v5_all_sft \
    --model_name 1030_all_sft \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir "$OUTDIR" > "$OUTDIR/all_sft.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_metacot_control_v5_verify_sft \
    --model_name 1030_verify_sft \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir "$OUTDIR" > "$OUTDIR/verify_sft.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/control_v5_E3/final \
    --model_name 1030_E3 \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir "$OUTDIR" > "$OUTDIR/E3.log" 2>&1 &

echo "Round 1 started: base_sft, all_sft, verify_sft, E3"
wait
echo "Round 1 done."

echo "=== EVAL NODE: Round 2 (2 models) ==="

CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/control_v5_E5/final \
    --model_name 1030_E5 \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir "$OUTDIR" > "$OUTDIR/E5.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/control_v5_E9/final \
    --model_name 1030_E9 \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir "$OUTDIR" > "$OUTDIR/E9.log" 2>&1 &

echo "Round 2 started: E5, E9"
wait
echo "=== EVAL NODE: ALL 6 MODELS DONE ==="
ls -la "$OUTDIR"/eval_1030_*.json 2>/dev/null || echo "listing results..."
ls -la "$OUTDIR"/*.json 2>/dev/null
