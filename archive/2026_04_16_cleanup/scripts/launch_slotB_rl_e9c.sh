#!/usr/bin/env bash
set -euo pipefail
export PATH="/opt/conda/bin:$PATH"
source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="${WANDB_API_KEY}"

PLAN_ID="${PLAN_ID:-}"
EXPERIMENT_ROLE="${EXPERIMENT_ROLE:-side_evidence}"
ALLOW_EXPLORATORY_E9C="${ALLOW_EXPLORATORY_E9C:-0}"

if [[ -z "${PLAN_ID}" ]]; then
    echo "PLAN_ID must be set explicitly for exploratory launches." >&2
    exit 1
fi

if [[ "${ALLOW_EXPLORATORY_E9C}" != "1" ]]; then
    echo "This launcher is legacy exploratory only." >&2
    echo "Set ALLOW_EXPLORATORY_E9C=1 to acknowledge that it is not mainline evidence." >&2
    exit 1
fi

mkdir -p results/control_v6_SlotB

echo "$(date) Starting Slot B exploratory sidecar: E9c + RL E13"
echo "  plan_id=${PLAN_ID}"
echo "  experiment_role=${EXPERIMENT_ROLE}"
echo "  mode=E13"
echo "  base_model_path=checkpoints/control_v5_E9c/final"
echo "  output_dir=checkpoints/control_v6_SlotB_RL"
echo "This launcher is legacy exploratory only."
echo "It is not the mainline RL launcher because it hard-codes the E9c base."
accelerate launch --config_file configs/accelerate_grpo_z2.yaml \
    src/training/grpo_v2.py \
    --mode E13 \
    --max_steps 400 \
    --model_path checkpoints/control_v5_E9c/final \
    --data mixed_train \
    --output_dir checkpoints/control_v6_SlotB_RL \
    --num_generations 4 \
    --max_completion_length 2048 \
    --max_prompt_length 384 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    2>&1 | tee results/control_v6_SlotB/rl.log

echo "$(date) Slot B complete"
