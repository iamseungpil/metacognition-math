#!/bin/bash
# veRL GDPO launcher for Meta-CoT training.
#
# Usage:
#   scripts/run_verl_gdpo.sh E13          # 4 rewards (default)
#   scripts/run_verl_gdpo.sh E12          # 2 rewards only
#   MODE=E13 MODEL_PATH=checkpoints/other scripts/run_verl_gdpo.sh
#
# Prerequisites:
#   1. conda activate ptca (or set REMOTE_CONDA_ENV)
#   2. pip install verl ray[default] vllm
#   3. python src/training/verl_gdpo_data.py --output data/verl_train.parquet --split train
#      python src/training/verl_gdpo_data.py --output data/verl_val.parquet --split val
set -euo pipefail

# ── Environment ──
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Conda activation (works on both local and AMLT)
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
    source /opt/conda/etc/profile.d/conda.sh
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-ptca}"
conda activate "$REMOTE_CONDA_ENV" 2>/dev/null || true

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

# WandB key (from file or fallback)
export WANDB_API_KEY="${WANDB_API_KEY:-$(cat ~/.wandb_key 2>/dev/null || echo '2f4e627868f1f9dad10bcb1a14fbf96817e6baa9')}"

# ── Arguments ──
MODE="${1:-${MODE:-E13}}"
MODEL_PATH="${MODEL_PATH:-checkpoints/v6_clean_10k_E19}"
TRAIN_DATA="${TRAIN_DATA:-data/verl_train.parquet}"
VAL_DATA="${VAL_DATA:-data/verl_val.parquet}"
EPOCHS="${EPOCHS:-15}"
N_GPUS="${N_GPUS:-4}"
SAVE_FREQ="${SAVE_FREQ:-100}"
TEST_FREQ="${TEST_FREQ:-50}"

# Map mode to config name
CONFIG_NAME="verl_gdpo_${MODE,,}"  # lowercase: E13 -> verl_gdpo_e13

echo "============================================"
echo "  veRL GDPO Training: ${MODE}"
echo "============================================"
echo "  config:     configs/${CONFIG_NAME}.yaml"
echo "  model:      ${MODEL_PATH}"
echo "  train_data: ${TRAIN_DATA}"
echo "  val_data:   ${VAL_DATA}"
echo "  epochs:     ${EPOCHS}"
echo "  n_gpus:     ${N_GPUS}"
echo "============================================"

# ── Step 1: Prepare data if not exists ──
if [ ! -f "$TRAIN_DATA" ]; then
    echo "[PREP] Generating training data: ${TRAIN_DATA}"
    python src/training/verl_gdpo_data.py --output "$TRAIN_DATA" --split train
fi
if [ ! -f "$VAL_DATA" ]; then
    echo "[PREP] Generating validation data: ${VAL_DATA}"
    python src/training/verl_gdpo_data.py --output "$VAL_DATA" --split val
fi

# ── Step 2: Launch veRL GDPO training ──
# Hydra overrides allow runtime configuration without editing YAML
python src/training/verl_gdpo.py \
    --config-name "${CONFIG_NAME}" \
    mode="${MODE}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    trainer.total_epochs="${EPOCHS}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.experiment_name="verl_gdpo_${MODE}"

echo "============================================"
echo "  Training complete: ${MODE}"
echo "============================================"
