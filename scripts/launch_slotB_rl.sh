#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/conda/bin:$PATH"
source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"

MODE="${MODE:-E13}"
PLAN_ID="${PLAN_ID:-}"
EXPERIMENT_ROLE="${EXPERIMENT_ROLE:-mainline}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/control_v6_SlotB_RL}"
RUN_DIR="${RUN_DIR:-results/control_v6_SlotB}"
RUN_LABEL="${RUN_LABEL:-slotB_mainline}"
MAX_STEPS="${MAX_STEPS:-400}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-2048}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-384}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"

if [[ -z "${BASE_MODEL_PATH}" ]]; then
    echo "BASE_MODEL_PATH must be set explicitly for mainline RL launches." >&2
    echo "Example: BASE_MODEL_PATH=checkpoints/control_v6_SlotC_5ep/checkpoint-490 bash scripts/launch_slotB_rl.sh" >&2
    exit 1
fi

if [[ -z "${PLAN_ID}" ]]; then
    echo "PLAN_ID must be set explicitly." >&2
    echo "Example: PLAN_ID=plan_metacot_v6.4_active_2026_04_06" >&2
    exit 1
fi

if [[ "${EXPERIMENT_ROLE}" != "mainline" ]]; then
    echo "launch_slotB_rl.sh is reserved for mainline launches only." >&2
    echo "Set EXPERIMENT_ROLE=mainline or use the exploratory launcher instead." >&2
    exit 1
fi

mkdir -p "${RUN_DIR}"

echo "$(date) Starting Slot B mainline RL"
echo "  plan_id=${PLAN_ID}"
echo "  experiment_role=${EXPERIMENT_ROLE}"
echo "  run_label=${RUN_LABEL}"
echo "  mode=${MODE}"
echo "  base_model_path=${BASE_MODEL_PATH}"
echo "  output_dir=${OUTPUT_DIR}"
accelerate launch --config_file configs/accelerate_grpo_z2.yaml \
    src/training/grpo_v2.py \
    --mode "${MODE}" \
    --max_steps "${MAX_STEPS}" \
    --model_path "${BASE_MODEL_PATH}" \
    --data mixed_train \
    --output_dir "${OUTPUT_DIR}" \
    --num_generations "${NUM_GENERATIONS}" \
    --max_completion_length "${MAX_COMPLETION_LENGTH}" \
    --max_prompt_length "${MAX_PROMPT_LENGTH}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
    2>&1 | tee "${RUN_DIR}/rl.log"

echo "$(date) Slot B mainline RL complete"
