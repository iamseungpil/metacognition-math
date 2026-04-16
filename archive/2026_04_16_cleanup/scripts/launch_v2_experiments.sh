#!/usr/bin/env bash
# Launch V2 experiments on 3 nodes:
#   EVAL node:   E9v2 (verify quality)
#   TRAIN_B node: E9bv2 (redirect execution)
#   E8 node:     E10v2 (full controller)
#
# After each RL run, runs 1,030-problem eval on the final checkpoint.
# Usage: copy this script to each remote node and run the relevant section.
set -euo pipefail

REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"
MAX_STEPS="${MAX_STEPS:-300}"
MODEL_NAME="${MODEL_NAME:-qwen3_metacot_control_v5_all_sft}"
MODEL_PATH="${MODEL_PATH:-checkpoints/qwen3_metacot_control_v5_all_sft}"
LOG_DIR="${LOG_DIR:-results/control_v5_v2_lane}"

mkdir -p "$LOG_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$REMOTE_CONDA_ENV"
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export HF_TOKEN="${HF_TOKEN:-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}"
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat ~/.wandb_key 2>/dev/null || echo 2f4e627868f1f9dad10bcb1a14fbf96817e6baa9)}"

# ─── Determine which experiment to run based on NODE_ROLE env var ───
NODE_ROLE="${NODE_ROLE:-E9v2}"  # default, override per node

echo "=== V2 Experiment: $NODE_ROLE ==="
echo "Model: $MODEL_PATH"
echo "Max steps: $MAX_STEPS"

# ─��─ Ensure SFT model is available ───
python scripts/check_runtime_env.py --install-missing 2>&1 | tee "$LOG_DIR/runtime_env.txt"
until python scripts/ensure_hf_model.py \
  --model-name "$MODEL_NAME" \
  --output-dir "$MODEL_PATH" \
  --wait --poll-seconds 60 --timeout-seconds 604800 2>&1 | tee "$LOG_DIR/model_wait.log"; do
  echo "ensure_hf_model retry in 60s" | tee -a "$LOG_DIR/model_wait.log"
  sleep 60
done

# ─── Run RL training ───
OUTDIR="checkpoints/control_v5_${NODE_ROLE}"
if [[ -f "$OUTDIR/final/config.json" ]]; then
  echo "skip $NODE_ROLE: final checkpoint already exists"
else
  MODEL_PATH="$MODEL_PATH" DATA="mixed_train" OUTPUT_DIR="$OUTDIR" \
    bash scripts/run_grpo_v2.sh "$NODE_ROLE" "$MAX_STEPS" 2>&1 | tee "$LOG_DIR/${NODE_ROLE}_train.log"
fi

# ─── Push checkpoint to HF ───
python scripts/push_models_hf.py --model_path "$OUTDIR/final" --model_name "control_v5_${NODE_ROLE}" \
  2>&1 | tee "$LOG_DIR/${NODE_ROLE}_hf.log"

# ─── Run 1,030-problem eval on final checkpoint ��──
echo "=== 1,030-problem eval: $NODE_ROLE ==="
BENCHMARKS="gsm8k math500 aime2024"
MAX_PROBLEMS=500

# Use 2 GPUs: one for the V2 model, one for base_sft reference
CUDA_VISIBLE_DEVICES=0 nohup python -u src/eval/eval_hf.py \
  --model_path "$OUTDIR/final" \
  --model_name "1030_${NODE_ROLE}" \
  --benchmarks $BENCHMARKS --max_problems $MAX_PROBLEMS \
  --output_dir results > "$LOG_DIR/${NODE_ROLE}_eval_1030.log" 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python -u src/eval/eval_hf.py \
  --model_path checkpoints/qwen3_base_sft \
  --model_name "1030_base_sft_ref" \
  --benchmarks $BENCHMARKS --max_problems $MAX_PROBLEMS \
  --output_dir results > "$LOG_DIR/base_sft_eval_1030.log" 2>&1 &

wait
echo "=== ALL DONE: $NODE_ROLE training + eval ==="
