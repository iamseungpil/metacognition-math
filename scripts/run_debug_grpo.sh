#!/bin/bash
# Debug: run baseline vs meta GRPO to isolate problem
# Usage: bash scripts/run_debug_grpo.sh baseline
#        bash scripts/run_debug_grpo.sh meta
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
export WANDB_RUN_ID="debug-${1:-baseline}-$(date +%m%d-%H%M)"

MODE=${1:-baseline}
echo "=== DEBUG GRPO: $MODE ==="

# Filter data if meta mode
if [ "$MODE" = "meta" ]; then
    python scripts/filter_by_passrate.py 2>/dev/null || true
fi

accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_baseline.py \
    --mode $MODE \
    --model_path checkpoints/qwen3_meta_sft \
    --max_steps 50 \
    --num_generations 8

echo "=== DEBUG $MODE DONE ==="
