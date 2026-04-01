#!/usr/bin/env bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda activate grpo
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
accelerate launch --config_file configs/accelerate_sft.yaml \
  src/training/sft.py --config configs/sft_behavior_all.yaml
