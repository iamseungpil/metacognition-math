#!/bin/bash
# Full Meta-CoT pipeline: setup → SFT → E8 GRPO → eval
# Runs entirely on compute node after AMLT job starts
set -e

cd /scratch
echo "=== CLONING REPO ==="
git clone https://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@github.com/iamseungpil/metacognition-math.git metacognition
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

echo "=== SETTING UP CONDA ENV ==="
source /opt/conda/etc/profile.d/conda.sh
conda create -n grpo python=3.10 -y
conda activate grpo
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.52.3 trl==0.19.1 datasets accelerate deepspeed
pip install peft bitsandbytes sentencepiece protobuf pandas pyarrow
pip install math_verify latex2sympy2_extended wandb

echo "=== DOWNLOADING DATA FROM HF ==="
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('data', exist_ok=True)
hf_hub_download('iamseungpil/metacot', 'metacot_v2_trapi.parquet', repo_type='dataset', local_dir='data', token='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')
hf_hub_download('iamseungpil/metacot', 'base_sft.parquet', repo_type='dataset', local_dir='data', token='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')
print('Data downloaded')
"

echo "=== PHASE 1: V2 Meta SFT Training ==="
accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/sft.py \
    --data_path data/metacot_v2_trapi.parquet \
    --output_dir checkpoints/qwen3_metacot_v2_sft \
    --num_epochs 3 \
    --batch_size 2 \
    --lr 2e-5

echo "=== PHASE 2: Base SFT Training ==="
accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/sft.py \
    --data_path data/base_sft.parquet \
    --output_dir checkpoints/qwen3_base_sft \
    --num_epochs 3 \
    --batch_size 2 \
    --lr 2e-5

echo "=== PHASE 3: E8 GRPO Training (200 steps) ==="
accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/grpo_v2.py \
    --mode E8 \
    --max_steps 200 \
    --model_path checkpoints/qwen3_metacot_v2_sft \
    --data mixed

echo "=== PHASE 4: 1030-Problem Eval ==="
mkdir -p results

# Eval 3 models in parallel
CUDA_VISIBLE_DEVICES=0 python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_base_sft \
    --model_name 1030_base_sft \
    --benchmarks gsm8k math500 aime2024 --max_problems 500 \
    --output_dir results > results/eval_base.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_metacot_v2_sft \
    --model_name 1030_v2_sft \
    --benchmarks gsm8k math500 aime2024 --max_problems 500 \
    --output_dir results > results/eval_v2sft.log 2>&1 &

# E8 path
E8_PATH="checkpoints/grpo_v2_E8/final"
[ ! -d "$E8_PATH" ] && E8_PATH=$(ls -d checkpoints/grpo_v2_E8/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
CUDA_VISIBLE_DEVICES=2 python -u src/eval/eval_hf.py \
    --model_path $E8_PATH \
    --model_name 1030_e8 \
    --benchmarks gsm8k math500 aime2024 --max_problems 500 \
    --output_dir results > results/eval_e8.log 2>&1 &

wait
echo "=== ALL EVALS DONE ==="

echo "=== PHASE 5: Analysis ==="
python scripts/analyze_1030.py --results_dir results

echo "=== PIPELINE COMPLETE ==="
echo "Results saved to results/"
cat results/analysis_1030.json | python3 -c "
import json, sys
a = json.load(sys.stdin)
print()
print('=== FINAL RESULTS ===')
for model in a.get('models', []):
    acc = a['accuracy'].get(model, {}).get('overall', {})
    print(f\"  {model}: {acc.get('accuracy', 0)*100:.1f}% ({acc.get('n_correct', 0)}/{acc.get('n_total', 0)})\")
"

# Keep alive for SSH access
echo "Pipeline done. Keeping node alive for analysis..."
sleep infinity
