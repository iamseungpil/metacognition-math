#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scratch/metacognition}"
cd "$ROOT"

E10_PATTERN='src/training/grpo_v2.py --mode E10 --max_steps 200 --model_path checkpoints/qwen3_metacot_control_v5_all_sft --data mixed_train --output_dir checkpoints/control_v5_E10'
LOG_DIR="${LOG_DIR:-results/control_v5_probe_lane}"
mkdir -p "$LOG_DIR"

while pgrep -f "$E10_PATTERN" >/dev/null; do
  sleep 120
done

if [[ -f checkpoints/control_v5_E6/final/config.json && -f checkpoints/control_v5_E7/final/config.json ]]; then
  echo "probe-gated lane already complete" >> "$LOG_DIR/retry_after_e10.log"
  exit 0
fi

export REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"
export MAX_STEPS="${MAX_STEPS:-200}"
nohup bash scripts/launch_control_v5_rl_probe_lane_remote.sh > "$LOG_DIR/probe_lane_rerun_after_e10.out" 2>&1 < /dev/null &
echo $! >> "$LOG_DIR/retry_after_e10.pid"
