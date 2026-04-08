#!/bin/bash
# GRPO v2 launcher for control-v5 unified SFT initialisation.
set -euo pipefail
source /opt/conda/etc/profile.d/conda.sh
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"
conda activate "$REMOTE_CONDA_ENV"
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

MODE=${MODE:-${1:-E3}}
STEPS=${2:-1000}
MODEL_PATH="${MODEL_PATH:-checkpoints/qwen3_metacot_control_v5_all_sft}"
DATA="${DATA:-mixed_train}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/grpo_v2_${MODE}}"
PROBE_PATH="${PROBE_PATH:-checkpoints/simple_probe_control_v5_all_sft/best_probe.pt}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-1024}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-384}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

EXTRA_ARGS=()
if [[ "$MODE" == "E6" || "$MODE" == "E7" ]]; then
    EXTRA_ARGS+=(--probe_path "$PROBE_PATH")
fi

echo "=== GRPO v2: $MODE, $STEPS steps ==="
echo "  model_path=$MODEL_PATH"
echo "  data=$DATA"
echo "  output_dir=$OUTPUT_DIR"
echo "  num_generations=$NUM_GENERATIONS"
echo "  max_completion_length=$MAX_COMPLETION_LENGTH"
echo "  max_prompt_length=$MAX_PROMPT_LENGTH"
echo "  per_device_train_batch_size=$PER_DEVICE_TRAIN_BATCH_SIZE"
echo "  gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/grpo_v2.py --mode "$MODE" --max_steps "$STEPS" \
    --model_path "$MODEL_PATH" --data "$DATA" --output_dir "$OUTPUT_DIR" \
    --num_generations "$NUM_GENERATIONS" \
    --max_completion_length "$MAX_COMPLETION_LENGTH" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    "${EXTRA_ARGS[@]}"
echo "=== DONE ==="
