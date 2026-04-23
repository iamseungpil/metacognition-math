#!/bin/bash
# Common preamble for all run scripts
set -e
cd "$(dirname "$0")/.."

# Conda activation
eval "$(conda shell.bash hook 2>/dev/null || true)"
conda activate ptca 2>/dev/null || echo "Warning: ptca env not found, using default"

# WandB
export WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY:-$(cat ~/.wandb_key 2>/dev/null || true)}}"
export WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
export WANDB_PROJECT="${WANDB_PROJECT:-metacot-math}"

# Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
