#!/usr/bin/env bash
# Launch V2 experiments on EVAL and TRAIN_B nodes after eval completes.
# EVAL: E9v2, TRAIN_B: E9bv2
# E10v2 runs on E8 after E6+E7 complete.
set -euo pipefail

export PATH="/opt/conda/bin:$PATH"
source /opt/conda/etc/profile.d/conda.sh
conda activate ptca

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat ~/.wandb_key 2>/dev/null || echo ${WANDB_API_KEY})}"

MODEL_PATH="checkpoints/qwen3_metacot_control_v5_all_sft"
ACCEL_CONFIG="configs/accelerate_grpo.yaml"
MAX_STEPS=300
LOG_DIR="results/control_v5_v2_lane"
mkdir -p "$LOG_DIR"

# Determine which experiment to run based on $1
MODE="${1:?Usage: $0 <E9v2|E9bv2|E10v2>}"
OUTDIR="checkpoints/control_v5_${MODE}"
LOG="$LOG_DIR/${MODE}.log"

# Preflight
python --version || { echo "FATAL: python unavailable"; exit 1; }
for f in "$MODEL_PATH/config.json" "src/training/grpo_v2.py" "src/training/rewards.py" "$ACCEL_CONFIG"; do
    [[ -f "$f" ]] || { echo "FATAL: missing $f"; exit 1; }
done
nvidia-smi > /dev/null 2>&1 || { echo "FATAL: no GPU"; exit 1; }

echo "$(date '+%Y-%m-%d %H:%M:%S') Starting $MODE training ($MAX_STEPS steps)"
echo "  Output: $OUTDIR"
echo "  Log: $LOG"

if [[ -f "$OUTDIR/final/config.json" ]]; then
    echo "SKIP: $OUTDIR/final already exists"
    exit 0
fi

accelerate launch --config_file "$ACCEL_CONFIG" \
    src/training/grpo_v2.py \
    --mode "$MODE" \
    --max_steps "$MAX_STEPS" \
    --model_path "$MODEL_PATH" \
    --data mixed_train \
    --output_dir "$OUTDIR" \
    --num_generations 2 \
    --max_completion_length 1024 \
    --max_prompt_length 384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    2>&1 | tee "$LOG"

EXIT_CODE=${PIPESTATUS[0]}
if [[ "$EXIT_CODE" -ne 0 ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') $MODE FAILED (exit $EXIT_CODE)"
    exit 1
fi

if [[ ! -f "$OUTDIR/final/config.json" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') $MODE FAILED: no final checkpoint"
    exit 1
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') $MODE DONE: $OUTDIR/final"
