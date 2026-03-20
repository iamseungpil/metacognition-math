#!/bin/bash
source "$(dirname "$0")/common.sh"

echo "========================================="
echo "Evaluation: All models on all benchmarks"
echo "========================================="

python -m src.eval.evaluator --config configs/eval.yaml

cp -r /scratch/metacognition/results /mnt/input/metacognition/
echo "Evaluation complete! Results in /mnt/input/metacognition/results/"
