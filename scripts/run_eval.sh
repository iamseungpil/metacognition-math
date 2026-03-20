#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "========================================="
echo "Evaluation: All models on all benchmarks"
echo "========================================="

export PYTHONPATH="${PWD}:${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false

python -m src.eval.evaluator --config configs/eval.yaml

# Copy results to persistent storage
cp -r /scratch/metacognition/results /mnt/input/metacognition/

echo "Evaluation complete! Results in /mnt/input/metacognition/results/"
