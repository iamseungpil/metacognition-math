#!/usr/bin/env bash
set -euo pipefail

REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"
MAX_STEPS="${MAX_STEPS:-200}"
MODEL_NAME="${MODEL_NAME:-qwen3_metacot_control_v5_all_sft}"
MODEL_PATH="${MODEL_PATH:-checkpoints/qwen3_metacot_control_v5_all_sft}"
LOG_DIR="${LOG_DIR:-results/control_v5_eval_lane}"

mkdir -p "$LOG_DIR"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$REMOTE_CONDA_ENV"
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export HF_TOKEN="${HF_TOKEN:-${HF_TOKEN}}"
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat ~/.wandb_key 2>/dev/null || echo ${WANDB_API_KEY})}"

python scripts/check_runtime_env.py --install-missing | tee "$LOG_DIR/runtime_env.txt"
until python scripts/ensure_hf_model.py \
  --model-name "$MODEL_NAME" \
  --output-dir "$MODEL_PATH" \
  --wait \
  --poll-seconds 60 \
  --timeout-seconds 604800 | tee "$LOG_DIR/model_wait.log"; do
  echo "ensure_hf_model retry in 60s" | tee -a "$LOG_DIR/model_wait.log"
  sleep 60
done

for MODE in E3 E5 E9; do
  OUTDIR="checkpoints/control_v5_${MODE}"
  if [[ -f "$OUTDIR/final/config.json" ]]; then
    echo "skip $MODE: final checkpoint already exists"
  else
    MODEL_PATH="$MODEL_PATH" \
    DATA="mixed_train" \
    OUTPUT_DIR="$OUTDIR" \
    bash scripts/run_grpo_v2.sh "$MODE" "$MAX_STEPS" | tee "$LOG_DIR/${MODE}.log"
  fi
  python scripts/push_models_hf.py --model_path "$OUTDIR/final" --model_name "control_v5_${MODE}" \
    | tee "$LOG_DIR/${MODE}_hf.log"
done
