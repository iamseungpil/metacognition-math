#!/usr/bin/env bash
set -euo pipefail

REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"
MAX_STEPS="${MAX_STEPS:-200}"
MODEL_NAME="${MODEL_NAME:-qwen3_metacot_control_v5_all_sft}"
MODEL_PATH="${MODEL_PATH:-checkpoints/qwen3_metacot_control_v5_all_sft}"
PROBE_ROLLOUTS="${PROBE_ROLLOUTS:-results/probe/control_v5_all_sft_rollouts.parquet}"
PROBE_DIR="${PROBE_DIR:-checkpoints/simple_probe_control_v5_all_sft}"
LOG_DIR="${LOG_DIR:-results/control_v5_probe_lane}"
PROBE_GSM_N="${PROBE_GSM_N:-1024}"
PROBE_MATH_N="${PROBE_MATH_N:-1024}"
PROBE_NUM_SAMPLES="${PROBE_NUM_SAMPLES:-2048}"
PROBE_EPOCHS="${PROBE_EPOCHS:-5}"
PROBE_MIN_ROLLOUT_ROWS="${PROBE_MIN_ROLLOUT_ROWS:-1800}"
PROBE_CONTINUATIONS_PER_PREFIX="${PROBE_CONTINUATIONS_PER_PREFIX:-8}"
PROBE_ROLLOUT_SHARDS="${PROBE_ROLLOUT_SHARDS:-4}"
PROBE_ROLLOUT_FLUSH_EVERY="${PROBE_ROLLOUT_FLUSH_EVERY:-8}"

mkdir -p "$LOG_DIR" "$(dirname "$PROBE_ROLLOUTS")"

source /opt/conda/etc/profile.d/conda.sh
conda activate "$REMOTE_CONDA_ENV"
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export HF_TOKEN="${HF_TOKEN:-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}"
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat ~/.wandb_key 2>/dev/null || echo 2f4e627868f1f9dad10bcb1a14fbf96817e6baa9)}"

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

PROBE_READY=0

quarantine_path() {
  local target="$1"
  if [[ -e "$target" ]]; then
    local stamp
    stamp="$(date +%Y%m%d_%H%M%S)"
    local base
    base="$(basename "$target")"
    mkdir -p results/stale_probe_artifacts
    mv "$target" "results/stale_probe_artifacts/${base}.${stamp}.stale"
  fi
}

probe_rollouts_are_current() {
  PROBE_ROLLOUTS_ENV="$PROBE_ROLLOUTS" \
  PROBE_MIN_ROLLOUT_ROWS_ENV="$PROBE_MIN_ROLLOUT_ROWS" \
  python3 - <<'PY'
from pathlib import Path
import os

import pandas as pd

path = Path(os.environ["PROBE_ROLLOUTS_ENV"])
min_rows = int(os.environ["PROBE_MIN_ROLLOUT_ROWS_ENV"])
if not path.exists():
    raise SystemExit(1)

df = pd.read_parquet(path)
required_cols = {"problem_id", "question", "gold_answer", "completion", "is_correct"}
required_cols |= {"meta_prefix_target_probs", "meta_prefix_count", "prompt_text"}
if not required_cols.issubset(df.columns):
    raise SystemExit(1)
if len(df) < min_rows:
    raise SystemExit(1)
if int(df["meta_prefix_count"].fillna(0).sum()) < min_rows:
    raise SystemExit(1)
PY
}

probe_artifact_is_current() {
  PROBE_DIR_ENV="$PROBE_DIR" python3 - <<'PY'
from pathlib import Path
import json
import os
import torch

probe_dir = Path(os.environ["PROBE_DIR_ENV"])
probe_path = probe_dir / "best_probe.pt"
metrics_path = probe_dir / "best_metrics.json"
cache_stats_path = probe_dir / "hidden_states_cache" / "manifest_stats.json"

if not probe_path.exists() or not metrics_path.exists() or not cache_stats_path.exists():
    raise SystemExit(1)

metrics = json.loads(metrics_path.read_text())
required_metric_keys = {"val_brier", "val_mae", "calibrated_val_brier", "temperature"}
if not required_metric_keys.issubset(metrics):
    raise SystemExit(1)

state = torch.load(probe_path, map_location="cpu", weights_only=False)
if not isinstance(state, dict) or "state_dict" not in state or "temperature" not in state:
    raise SystemExit(1)

stats = json.loads(cache_stats_path.read_text())
target_counts = stats.get("target_source_counts", {})
if "missing_prefix_target" in target_counts:
    # Old or conceptually invalid prefix supervision should not silently pass.
    raise SystemExit(1)
PY
}

if ! probe_rollouts_are_current; then
  quarantine_path "$PROBE_ROLLOUTS"
  pids=()
  for shard in $(seq 0 $((PROBE_ROLLOUT_SHARDS - 1))); do
    shard_log="$LOG_DIR/probe_rollouts_gpu${shard}.log"
    env CUDA_VISIBLE_DEVICES="$shard" \
      python scripts/build_probe_rollouts_hf.py \
        --model-path "$MODEL_PATH" \
        --output-path "$PROBE_ROLLOUTS" \
        --gsm-n "$PROBE_GSM_N" \
        --math-n "$PROBE_MATH_N" \
        --continuations-per-prefix "$PROBE_CONTINUATIONS_PER_PREFIX" \
        --num-shards "$PROBE_ROLLOUT_SHARDS" \
        --shard-index "$shard" \
        --flush-every "$PROBE_ROLLOUT_FLUSH_EVERY" \
        --resume > "$shard_log" 2>&1 &
    pids+=("$!")
  done

  probe_rollout_failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      probe_rollout_failed=1
    fi
  done
  if [[ "$probe_rollout_failed" -ne 0 ]]; then
    echo "probe rollout shard failure" | tee "$LOG_DIR/probe_rollouts.log"
    exit 1
  fi

  python scripts/build_probe_rollouts_hf.py \
    --output-path "$PROBE_ROLLOUTS" \
    --num-shards "$PROBE_ROLLOUT_SHARDS" \
    --merge-shards | tee "$LOG_DIR/probe_rollouts.log"
fi

if ! probe_artifact_is_current; then
  quarantine_path "$PROBE_DIR"
  if python src/probes/retrain.py \
    --model-path "$MODEL_PATH" \
    --rollouts-path "$PROBE_ROLLOUTS" \
    --output-dir "$PROBE_DIR" \
    --num-samples "$PROBE_NUM_SAMPLES" \
    --epochs "$PROBE_EPOCHS" | tee "$LOG_DIR/probe_retrain.log"; then
    PROBE_READY=1
  else
    echo "probe retrain gate failed; skipping E6/E7 and falling back to E10" | tee -a "$LOG_DIR/probe_retrain.log"
  fi
else
  PROBE_READY=1
fi

if [[ "$PROBE_READY" -eq 1 ]]; then
  if python scripts/smoke_probe_pipeline.py \
    --probe-dir "$PROBE_DIR" | tee "$LOG_DIR/probe_smoke.log"; then
    PROBE_READY=1
  else
    PROBE_READY=0
    echo "probe smoke failed; skipping E6/E7 and falling back to E10" | tee -a "$LOG_DIR/probe_smoke.log"
  fi
fi

if [[ "$PROBE_READY" -eq 1 ]]; then
  python scripts/push_models_hf.py \
    --model_path "$PROBE_DIR" \
    --model_name "simple_probe_control_v5_all_sft" | tee "$LOG_DIR/probe_hf.log"
fi

MODES=(E10)
if [[ "$PROBE_READY" -eq 1 ]]; then
  MODES=(E6 E7 E10)
fi

for MODE in "${MODES[@]}"; do
  OUTDIR="checkpoints/control_v5_${MODE}"
  if [[ -f "$OUTDIR/final/config.json" ]]; then
    echo "skip $MODE: final checkpoint already exists"
  else
    MODEL_PATH="$MODEL_PATH" \
    DATA="mixed_train" \
    OUTPUT_DIR="$OUTDIR" \
    PROBE_PATH="$PROBE_DIR/best_probe.pt" \
    bash scripts/run_grpo_v2.sh "$MODE" "$MAX_STEPS" | tee "$LOG_DIR/${MODE}.log"
  fi
  python scripts/push_models_hf.py --model_path "$OUTDIR/final" --model_name "control_v5_${MODE}" \
    | tee "$LOG_DIR/${MODE}_hf.log"
done
