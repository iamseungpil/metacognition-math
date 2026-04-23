#!/bin/bash
# N3 Contrastive RLSD — smoke launcher (plan §9.4, §9.5)
#
# Purpose: 50-step smoke run on 4×H200 to verify the contrastive signal
# hypothesis + leakage-isolation invariants before committing to a full run.
#
# Expected wall-time: ~25 min (§9.4 target). Hard cutoff: 45 min.
#
# Usage (on node):
#   bash scripts/smoke_n3.sh               # rule-based decoy (primary)
#   bash scripts/smoke_n3.sh n3-random     # random-noise decoy (H3 ablation)
#   DAEMONIZED=1 ...                        # internal: daemonized background
#
# Writes:
#   /scratch/meta/run/<run_id>/
#       stdout.log, stderr.log, metrics.jsonl,
#       smoke_acceptance.json, final_metrics.json,
#       checkpoint-*/ (at eval_interval)
#
# Mirror-pushed to HF by scripts/n3_monitor_push.sh (parallel).

set -u

VARIANT="${1:-n3}"
SEED="${2:-42}"
RUN_ID="n3_${VARIANT}_s${SEED}_$(date +%Y%m%d_%H%M%S)"
SMOKE_STEPS="${SMOKE_STEPS:-50}"
ROOT="${ROOT:-/scratch/meta/code}"
RUN_DIR="/scratch/meta/run/${RUN_ID}"
LOG="${RUN_DIR}/launch.log"

if [ "${DAEMONIZED:-0}" != "1" ]; then
  mkdir -p "$RUN_DIR"
  DAEMONIZED=1 RUN_ID="$RUN_ID" RUN_DIR="$RUN_DIR" \
      nohup setsid bash "$0" "$VARIANT" "$SEED" > "$LOG" 2>&1 </dev/null &
  disown
  echo "N3_SMOKE_STARTED pid=$! run_id=$RUN_ID log=$LOG"
  echo "$RUN_ID" > /tmp/_n3_last_run_id
  exit 0
fi

echo "=== $(date) N3 smoke START (variant=$VARIANT, seed=$SEED) ==="
echo "RUN_DIR=$RUN_DIR"
echo "ROOT=$ROOT"

cd "$ROOT" || { echo "ROOT not found: $ROOT"; exit 1; }

# ── PF1: env sanity ─────────────────────────────────────────────────────
echo "=== $(date) PF1: env sanity ==="
python3 -c "import torch, trl, transformers; print('torch=', torch.__version__, 'trl=', trl.__version__, 'tf=', transformers.__version__)" \
    | tee -a "$RUN_DIR/stdout.log"

# ── PF3: device budget ──────────────────────────────────────────────────
echo "=== $(date) PF3: GPU status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader \
    | tee -a "$RUN_DIR/stdout.log"

# ── PF6a: checkpoint path ───────────────────────────────────────────────
CKPT="${STUDENT_INIT:-checkpoints/self_distill_rebuilt_d2_epistemic_h200}"
if [ ! -d "$CKPT" ]; then
  echo "FAIL: student_init not found: $CKPT" | tee -a "$RUN_DIR/stdout.log"
  echo "run §9.5 staging first" | tee -a "$RUN_DIR/stdout.log"
  echo "FAILED" > "$RUN_DIR/FAILED"
  exit 2
fi

# ── PF6b: data path ─────────────────────────────────────────────────────
DATA="${TRAIN_DATA:-data/verl_train_redirect.parquet}"
if [ ! -f "$DATA" ]; then
  echo "FAIL: train_data not found: $DATA" | tee -a "$RUN_DIR/stdout.log"
  echo "FAILED" > "$RUN_DIR/FAILED"
  exit 2
fi

# ── PF6e: decoy self-test (CPU, fast) ───────────────────────────────────
echo "=== $(date) PF6e: decoy self-test ==="
python3 -m pytest tests/test_contrastive_meta_rlsd.py -x -q 2>&1 | tee -a "$RUN_DIR/stdout.log"
if [ "${PIPESTATUS[0]}" != "0" ]; then
  echo "FAIL: unit tests did not pass" | tee -a "$RUN_DIR/stdout.log"
  echo "FAILED" > "$RUN_DIR/FAILED"
  exit 2
fi

# ── Override smoke-specific YAML values via env-var sed ─────────────────
SMOKE_YAML="$RUN_DIR/smoke_config.yaml"
cp configs/contrastive_meta_rlsd.yaml "$SMOKE_YAML"
# Override total_steps, output_dir, seed, save/eval cadence.
python3 - "$SMOKE_YAML" "$SMOKE_STEPS" "$RUN_DIR" "$SEED" <<'PY'
import sys, yaml, pathlib
cfg_path, steps, run_dir, seed = sys.argv[1:5]
cfg = yaml.safe_load(pathlib.Path(cfg_path).read_text())
cfg["total_steps"] = int(steps)
cfg["output_dir"] = f"{run_dir}/checkpoints"
cfg["seed"] = int(seed)
cfg["save_interval"] = max(int(steps) // 2, 10)
cfg["eval_interval"] = max(int(steps) // 2, 10)
cfg["run_name"] = pathlib.Path(run_dir).name
pathlib.Path(cfg_path).write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[smoke] wrote {cfg_path}: total_steps={cfg['total_steps']} output_dir={cfg['output_dir']}")
PY

# ── Launch ──────────────────────────────────────────────────────────────
echo "=== $(date) launching accelerate ==="

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-metacot-meta-rlsd}"
export WANDB_RUN_NAME="$RUN_ID"
export HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var required}"

# Hard cutoff: 45 min via `timeout` wrapper.
timeout 2700 accelerate launch \
    --config_file configs/accelerate_ds3.yaml \
    src/training/contrastive_meta_rlsd_trainer.py \
    --config "$SMOKE_YAML" \
    --variant "$VARIANT" \
    --seed "$SEED" \
    >> "$RUN_DIR/stdout.log" 2>> "$RUN_DIR/stderr.log"

RC=$?
echo "=== $(date) training exit code=$RC ==="

# ── Post-run smoke acceptance (§4.2 + §9.3) ─────────────────────────────
python3 scripts/smoke_n3_acceptance.py "$RUN_DIR" "$RC" \
    >> "$RUN_DIR/stdout.log" 2>&1 || {
    echo "smoke acceptance FAILED" | tee -a "$RUN_DIR/stdout.log"
    echo "FAILED" > "$RUN_DIR/FAILED"
    exit 3
}

echo "$(date) smoke DONE"
echo "DONE" > "$RUN_DIR/DONE"
