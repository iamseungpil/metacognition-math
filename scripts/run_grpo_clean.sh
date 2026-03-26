#!/bin/bash
# Clean GRPO: vanilla GRPOTrainer + reward_funcs only
# Usage: bash scripts/run_grpo_clean.sh meta filtered
#        bash scripts/run_grpo_clean.sh baseline gsm8k
set -e
export OPENSSL_CONF=/dev/null
export OPENSSL_ia32cap="~0x200000200000000"

source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
export OPENSSL_CONF=/dev/null

pip install "transformers==4.51.3" "trl==0.19.1" "peft>=0.10" --quiet 2>/dev/null || true

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")
export WANDB_RUN_ID="grpo-clean-${1:-meta}-${2:-filtered}-$(date +%m%d-%H%M)"

MODE=${1:-meta}
DATA=${2:-filtered}

# Filter data if needed
if [ "$DATA" = "filtered" ]; then
    python scripts/filter_by_passrate.py 2>/dev/null || true
fi

echo "=== GRPO Clean: $MODE mode, $DATA data ==="

accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_clean.py \
    --mode $MODE \
    --data $DATA \
    --model_path checkpoints/qwen3_meta_sft \
    --max_steps 200

echo "=== DONE ==="
