#!/bin/bash
# Full Meta-CoT pipeline: setup → SFT → E8 GRPO → eval → analysis
# Runs entirely on compute node after AMLT job starts
set -euo pipefail

LOG=/scratch/metacognition/pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "$(date): === PIPELINE START ==="

# ─── Step 0: Clone repo and install ───
cd /scratch
if [ ! -d metacognition ]; then
    echo "$(date): Cloning repo..."
    git clone https://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@github.com/iamseungpil/metacognition-math.git metacognition
else
    echo "$(date): Repo exists, pulling latest..."
    cd metacognition && git pull && cd /scratch
fi
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

echo "$(date): Setting up conda env..."
source /opt/conda/etc/profile.d/conda.sh
if conda env list | grep -q grpo; then
    echo "Env grpo exists, activating..."
    conda activate grpo
else
    conda create -n grpo python=3.10 -y
    conda activate grpo
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install transformers==4.52.3 trl==0.19.1 datasets accelerate deepspeed
    pip install peft bitsandbytes sentencepiece protobuf pandas pyarrow
    pip install math_verify latex2sympy2_extended wandb
fi

# Login to wandb
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

# ─── Step 1: Download data from HuggingFace ───
echo "$(date): Downloading data from HF..."
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('data', exist_ok=True)
for f in ['metacot_v2_trapi.parquet', 'base_sft.parquet']:
    hf_hub_download('iamseungpil/metacot', f, repo_type='dataset',
                    local_dir='data', token='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')
    print(f'Downloaded {f}')
"

# ─── Step 2: V2 Meta SFT Training ───
if [ -d checkpoints/qwen3_metacot_v2_sft ]; then
    echo "$(date): V2 Meta SFT already exists, skipping..."
else
    echo "$(date): === PHASE 1: V2 Meta SFT Training ==="
    accelerate launch --config_file configs/accelerate_sft.yaml \
        src/training/sft.py --config configs/sft_v2_meta.yaml
    echo "$(date): V2 Meta SFT done."
fi

# ─── Step 3: Base SFT Training ───
if [ -d checkpoints/qwen3_base_sft ]; then
    echo "$(date): Base SFT already exists, skipping..."
else
    echo "$(date): === PHASE 2: Base SFT Training ==="
    accelerate launch --config_file configs/accelerate_sft.yaml \
        src/training/sft.py --config configs/sft_base.yaml
    echo "$(date): Base SFT done."
fi

# ─── Step 4: E8 GRPO Training ───
echo "$(date): === PHASE 3: E8 GRPO Training (200 steps) ==="
accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/grpo_v2.py \
    --mode E8 \
    --max_steps 200 \
    --model_path checkpoints/qwen3_metacot_v2_sft \
    --data mixed
echo "$(date): GRPO E8 done."

# ─── Step 5: 1030-Problem Eval (3 models in parallel) ───
echo "$(date): === PHASE 4: 1030-Problem Eval ==="
mkdir -p results

CUDA_VISIBLE_DEVICES=0 python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_base_sft \
    --model_name 1030_base_sft \
    --benchmarks gsm8k math500 aime2024 --max_problems 500 \
    --output_dir results > results/eval_base.log 2>&1 &
PID_BASE=$!

CUDA_VISIBLE_DEVICES=1 python -u src/eval/eval_hf.py \
    --model_path checkpoints/qwen3_metacot_v2_sft \
    --model_name 1030_v2_sft \
    --benchmarks gsm8k math500 aime2024 --max_problems 500 \
    --output_dir results > results/eval_v2sft.log 2>&1 &
PID_V2=$!

# Find best E8 checkpoint
E8_PATH="checkpoints/grpo_v2_E8/final"
if [ ! -d "$E8_PATH" ]; then
    E8_PATH=$(ls -d checkpoints/grpo_v2_E8/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
fi
echo "Using E8 checkpoint: $E8_PATH"
CUDA_VISIBLE_DEVICES=2 python -u src/eval/eval_hf.py \
    --model_path "$E8_PATH" \
    --model_name 1030_e8 \
    --benchmarks gsm8k math500 aime2024 --max_problems 500 \
    --output_dir results > results/eval_e8.log 2>&1 &
PID_E8=$!

echo "Waiting for 3 evals (PIDs: $PID_BASE $PID_V2 $PID_E8)..."
wait $PID_BASE $PID_V2 $PID_E8
echo "$(date): All evals done."

# ─── Step 6: Analysis ───
echo "$(date): === PHASE 5: Analysis ==="
python scripts/analyze_1030.py --results_dir results

# Print summary
python3 -c "
import json
with open('results/analysis_1030.json') as f:
    a = json.load(f)
print()
print('=' * 60)
print('  FINAL RESULTS')
print('=' * 60)
for model in a.get('models', []):
    acc = a['accuracy'].get(model, {}).get('overall', {})
    ece_data = a.get('ece', {}).get(model, {})
    ece = ece_data.get('overall', 'N/A')
    print(f'  {model}: acc={acc.get(\"accuracy\", 0)*100:.1f}% ({acc.get(\"n_correct\", 0)}/{acc.get(\"n_total\", 0)}) ECE={ece}')

# Check success criterion
base_acc = a['accuracy'].get('1030_base_sft', {}).get('overall', {}).get('accuracy', 0)
e8_acc = a['accuracy'].get('1030_e8', {}).get('overall', {}).get('accuracy', 0)
print()
if e8_acc >= base_acc:
    print('SUCCESS: Meta-CoT E8 >= Base SFT!')
else:
    gap = (base_acc - e8_acc) * 100
    print(f'GAP: Meta-CoT E8 is {gap:.1f}%p behind Base SFT')
    print('Next: Try autoresearch hypotheses H2-H6')
print('=' * 60)
"

echo "$(date): === PIPELINE COMPLETE ==="
echo "Keeping node alive for SSH access..."
sleep infinity
