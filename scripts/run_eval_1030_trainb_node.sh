#!/bin/bash
# Legacy v5 side-evidence eval script. Not the active-plan eval contract.
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

ALLOW_LEGACY_V5_EVAL="${ALLOW_LEGACY_V5_EVAL:-0}"
EVIDENCE_CLASS="${EVIDENCE_CLASS:-side_evidence}"

if [[ "${ALLOW_LEGACY_V5_EVAL}" != "1" ]]; then
    echo "This eval script is legacy v5 side evidence only." >&2
    echo "Set ALLOW_LEGACY_V5_EVAL=1 to acknowledge that it is not the active-plan eval entrypoint." >&2
    exit 1
fi

BENCHMARKS="gsm8k math500 aime2024"
MAX=500
OUTDIR="results/eval_1030_v5"
mkdir -p "$OUTDIR"

echo "=== LEGACY V5 SIDE-EVIDENCE EVAL ==="
echo "evidence_class=$EVIDENCE_CLASS"

echo "=== TRAIN_B NODE: Round 1 (4 models) ==="

CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_metacot_control_v5_redirect_sft \
    --model_name 1030_redirect_sft \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir "$OUTDIR" > "$OUTDIR/redirect_sft.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/control_v5_E8/final \
    --model_name 1030_E8 \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir "$OUTDIR" > "$OUTDIR/E8.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/control_v5_E9b/final \
    --model_name 1030_E9b \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir "$OUTDIR" > "$OUTDIR/E9b.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/control_v5_E9c/final \
    --model_name 1030_E9c \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir "$OUTDIR" > "$OUTDIR/E9c.log" 2>&1 &

echo "Round 1 started: redirect_sft, E8, E9b, E9c"
wait
echo "Round 1 done."

echo "=== TRAIN_B NODE: Round 2 (1 model) ==="

CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval/eval_hf.py \
    --model_path checkpoints/control_v5_E10/final \
    --model_name 1030_E10 \
    --benchmarks $BENCHMARKS --max_problems $MAX \
    --output_dir "$OUTDIR" > "$OUTDIR/E10.log" 2>&1 &

echo "Round 2 started: E10"
wait
echo "=== TRAIN_B NODE: ALL 5 MODELS DONE ==="
ls -la "$OUTDIR"/*.json 2>/dev/null
