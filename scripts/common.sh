#!/bin/bash
# Common preamble for all run scripts
set -e
cd "$(dirname "$0")/.."

# Conda activation
eval "$(conda shell.bash hook 2>/dev/null || true)"
conda activate ptca 2>/dev/null || echo "Warning: ptca env not found, using default"

# WandB
export WANDB_API_KEY="2f4e627868f1f9dad10bcb1a14fbf96817e6baa9"
export WANDB_PROJECT="metacot-math"

# Python path
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
