#!/bin/bash
# GRPO v2: Full FT + ZeRO-3 + HF generate (no vLLM, no veRL)
set -e
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

MODE=${1:-E1}
STEPS=${2:-200}

echo "=== GRPO v2: $MODE, $STEPS steps, ZeRO-3 + HF generate ==="
accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/grpo_v2.py --mode $MODE --max_steps $STEPS \
    --model_path checkpoints/qwen3_meta_sft --data filtered
echo "=== DONE ==="
