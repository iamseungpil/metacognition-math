#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Robust launch script for E6 + E7 training on E8 node
#
# E6: probe-calibration (correctness + format + meta_quality + calibration + probe_calibration)
# E7: E6 + stepwise probe scoring
#
# Designed to survive SSH disconnection (runs under nohup).
# Explicitly sources conda to avoid the broken bash -lc login profile issue
# that killed the previous V2 launch.
#
# Usage (local): bash scripts/launch_e6_e7_on_e8.sh
# Or copy to E8 and run: nohup bash /scratch/metacognition/scripts/launch_e6_e7_on_e8.sh \
#                           > /scratch/metacognition/results/control_v5_v2_lane/e6_e7_master.log 2>&1 &
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ─── Environment ───
export PATH="/opt/conda/bin:$PATH"
source /opt/conda/etc/profile.d/conda.sh
conda activate ptca

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat ~/.wandb_key 2>/dev/null || echo ${WANDB_API_KEY})}"
export HF_TOKEN="${HF_TOKEN:-${HF_TOKEN}}"

# ─── Paths ───
MODEL_PATH="checkpoints/qwen3_metacot_control_v5_all_sft"
PROBE_PATH="checkpoints/simple_probe_control_v5_all_sft/best_probe.pt"
LOG_DIR="results/control_v5_v2_lane"
# ZeRO-2 for E6/E7: probe forward pass causes NCCL timeout with ZeRO-3
ACCEL_CONFIG="configs/accelerate_grpo_z2.yaml"
MAX_STEPS=300

mkdir -p "$LOG_DIR"

# ─── Preflight checks ───
echo "$(date '+%Y-%m-%d %H:%M:%S') [preflight] Starting E6+E7 launch"
echo "  Model: $MODEL_PATH"
echo "  Probe: $PROBE_PATH"
echo "  Max steps: $MAX_STEPS"

# 1) Verify conda env
python --version || { echo "FATAL: python not available in ptca env"; exit 1; }

# 2) Verify critical files
for f in "$MODEL_PATH/config.json" "$PROBE_PATH" "src/training/grpo_v2.py" "src/training/rewards.py" "$ACCEL_CONFIG"; do
    if [[ ! -f "$f" ]]; then
        echo "FATAL: missing required file: $f"
        exit 1
    fi
    echo "  OK: $f"
done

# 3) Verify CUDA
nvidia-smi > /dev/null 2>&1 || { echo "FATAL: nvidia-smi failed"; exit 1; }
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "  GPUs: $GPU_COUNT"
if [[ "$GPU_COUNT" -lt 4 ]]; then
    echo "FATAL: expected 4 GPUs, found $GPU_COUNT"
    exit 1
fi

# 4) Quick import test (catches ModuleNotFoundError before burning GPU time)
python -c "
import sys; sys.path.insert(0, '.')
from src.training.rewards import probe_calibration_reward, stepwise_probe_reward
from src.training.grpo_v2 import _ensure_vllm_stub
print('  Import check: OK')
" || { echo "FATAL: import check failed"; exit 1; }

# 5) Check no other training running on GPUs
PROCS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
if [[ "$PROCS" -gt 0 ]]; then
    echo "WARNING: $PROCS processes already using GPUs:"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv
    echo "Continuing anyway (they may be unrelated)..."
fi

echo "$(date '+%Y-%m-%d %H:%M:%S') [preflight] All checks passed"
echo ""

# ─── Clean up stale E6 directory from previous failed attempt ───
if [[ -d "checkpoints/control_v5_E6" ]]; then
    if [[ ! -f "checkpoints/control_v5_E6/final/config.json" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') Removing stale control_v5_E6 directory (no final checkpoint)"
        rm -rf "checkpoints/control_v5_E6"
    else
        echo "$(date '+%Y-%m-%d %H:%M:%S') control_v5_E6/final already exists, skipping E6 training"
    fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# E6: probe calibration (300 steps, 4 GPUs)
# ══════════════════════════════════════════════════════════════════════════════
E6_OUTDIR="checkpoints/control_v5_E6_v2"
E6_LOG="$LOG_DIR/E6.log"

if [[ -f "$E6_OUTDIR/final/config.json" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [E6] SKIP: final checkpoint already exists at $E6_OUTDIR/final"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [E6] Starting training ($MAX_STEPS steps)..."
    echo "  Output: $E6_OUTDIR"
    echo "  Log:    $E6_LOG"

    accelerate launch --config_file "$ACCEL_CONFIG" \
        src/training/grpo_v2.py \
        --mode E6 \
        --max_steps "$MAX_STEPS" \
        --model_path "$MODEL_PATH" \
        --data mixed_train \
        --output_dir "$E6_OUTDIR" \
        --probe_path "$PROBE_PATH" \
        --num_generations 2 \
        --max_completion_length 1024 \
        --max_prompt_length 384 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        2>&1 | tee "$E6_LOG"

    E6_EXIT=${PIPESTATUS[0]}
    if [[ "$E6_EXIT" -ne 0 ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [E6] FAILED with exit code $E6_EXIT"
        echo "Check $E6_LOG for details"
        exit 1
    fi

    # Verify E6 produced a final checkpoint
    if [[ ! -f "$E6_OUTDIR/final/config.json" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [E6] FAILED: no final checkpoint produced"
        exit 1
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') [E6] DONE: checkpoint at $E6_OUTDIR/final"
fi

echo ""

# ══════════════════════════════════════════════════════════════════════════════
# E7: E6 + stepwise probe scoring (300 steps, 4 GPUs)
#   Uses the SAME base model (not E6 output) -- E7 is an independent ablation
# ══════════════════════════════════════════════════════════════════════════════
E7_OUTDIR="checkpoints/control_v5_E7"
E7_LOG="$LOG_DIR/E7.log"

if [[ -f "$E7_OUTDIR/final/config.json" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') [E7] SKIP: final checkpoint already exists at $E7_OUTDIR/final"
else
    echo "$(date '+%Y-%m-%d %H:%M:%S') [E7] Starting training ($MAX_STEPS steps)..."
    echo "  Output: $E7_OUTDIR"
    echo "  Log:    $E7_LOG"

    accelerate launch --config_file "$ACCEL_CONFIG" \
        src/training/grpo_v2.py \
        --mode E7 \
        --max_steps "$MAX_STEPS" \
        --model_path "$MODEL_PATH" \
        --data mixed_train \
        --output_dir "$E7_OUTDIR" \
        --probe_path "$PROBE_PATH" \
        --num_generations 2 \
        --max_completion_length 1024 \
        --max_prompt_length 384 \
        --per_device_train_batch_size 1 \
        --gradient_accumulation_steps 2 \
        2>&1 | tee "$E7_LOG"

    E7_EXIT=${PIPESTATUS[0]}
    if [[ "$E7_EXIT" -ne 0 ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [E7] FAILED with exit code $E7_EXIT"
        echo "Check $E7_LOG for details"
        exit 1
    fi

    if [[ ! -f "$E7_OUTDIR/final/config.json" ]]; then
        echo "$(date '+%Y-%m-%d %H:%M:%S') [E7] FAILED: no final checkpoint produced"
        exit 1
    fi
    echo "$(date '+%Y-%m-%d %H:%M:%S') [E7] DONE: checkpoint at $E7_OUTDIR/final"
fi

echo ""
echo "$(date '+%Y-%m-%d %H:%M:%S') [COMPLETE] E6 + E7 training finished successfully"
echo "  E6: $E6_OUTDIR/final"
echo "  E7: $E7_OUTDIR/final"
