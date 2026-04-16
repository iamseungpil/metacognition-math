#!/bin/bash
# Full Meta-CoT pipeline: setup → SFT → GRPO (E3,E5,E7,E8) → eval → analysis
# Each model uploaded to HuggingFace immediately after training.
# Background keepalive ensures node survives individual task failures.

LOG=/scratch/metacognition/pipeline.log

log() { echo "$(date '+%Y-%m-%d %H:%M:%S'): $*" | tee -a "$LOG"; }
run_phase() {
    local name="$1"; shift
    log "=== START: $name ==="
    if "$@" >> "$LOG" 2>&1; then
        log "=== DONE: $name ==="
        return 0
    else
        log "=== FAILED: $name (exit $?) — continuing ==="
        return 1
    fi
}
push_hf() {
    local path="$1" name="$2"
    log "Uploading $name to HuggingFace..."
    python scripts/push_models_hf.py \
        --model_path "$path" --model_name "$name" >> "$LOG" 2>&1 \
        && log "HF upload done: $name" \
        || log "HF upload FAILED: $name"
}

# ─── Phase 0: Keepalive + Setup ───
cd /scratch

# CRITICAL: background keepalive so node survives any task failure
nohup bash -c 'while true; do sleep 3600; echo "keepalive $(date)" >> /scratch/keepalive.log; done' &
KEEPALIVE_PID=$!
echo "Keepalive PID: $KEEPALIVE_PID" > /scratch/keepalive.pid
log "Keepalive started (PID $KEEPALIVE_PID)"

if [ ! -d metacognition ]; then
    log "Cloning repo..."
    git clone https://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@github.com/iamseungpil/metacognition-math.git metacognition
else
    log "Repo exists, pulling latest..."
    cd metacognition && git pull && cd /scratch
fi
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

log "Setting up conda env..."
source /opt/conda/etc/profile.d/conda.sh
if conda env list | grep -q "^grpo "; then
    conda activate grpo
else
    conda create -n grpo python=3.10 -y
    conda activate grpo
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    pip install transformers==4.52.3 trl==0.19.1 datasets accelerate deepspeed
    pip install peft bitsandbytes sentencepiece protobuf pandas pyarrow
    pip install math_verify latex2sympy2_extended wandb huggingface_hub
fi

export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9
export HF_TOKEN=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

# Download data
log "Downloading data from HF..."
python3 -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('data', exist_ok=True)
for f in ['metacot_v2_trapi.parquet', 'base_sft.parquet']:
    hf_hub_download('iamseungpil/metacot', f, repo_type='dataset',
                    local_dir='data', token='$HF_TOKEN')
    print(f'Downloaded {f}')
"

# ─── Phase 1: V2 Meta SFT ───
if [ -d checkpoints/qwen3_metacot_v2_sft ]; then
    log "V2 Meta SFT exists, skipping training."
else
    run_phase "V2 Meta SFT" \
        accelerate launch --config_file configs/accelerate_sft.yaml \
            src/training/sft.py --config configs/sft_v2_meta.yaml
fi
push_hf checkpoints/qwen3_metacot_v2_sft qwen3_metacot_v2_sft

# ─── Phase 2: Base SFT ───
if [ -d checkpoints/qwen3_base_sft ]; then
    log "Base SFT exists, skipping training."
else
    run_phase "Base SFT" \
        accelerate launch --config_file configs/accelerate_sft.yaml \
            src/training/sft.py --config configs/sft_base.yaml
fi
push_hf checkpoints/qwen3_base_sft qwen3_base_sft

# ─── Phase 3: GRPO experiments (E3, E5, E7, E8) ───
for MODE in E3 E5 E7 E8; do
    CKPT_DIR="checkpoints/grpo_v2_${MODE}"
    if [ -d "${CKPT_DIR}/final" ]; then
        log "GRPO $MODE already done, skipping."
    else
        run_phase "GRPO $MODE (200 steps)" \
            accelerate launch --config_file configs/accelerate_grpo.yaml \
                src/training/grpo_v2.py \
                --mode "$MODE" \
                --max_steps 200 \
                --model_path checkpoints/qwen3_metacot_v2_sft \
                --data mixed
    fi
    # Upload best checkpoint
    if [ -d "${CKPT_DIR}/final" ]; then
        push_hf "${CKPT_DIR}/final" "grpo_v2_${MODE}_final"
    elif ls -d ${CKPT_DIR}/checkpoint-* >/dev/null 2>&1; then
        BEST=$(ls -d ${CKPT_DIR}/checkpoint-* | sort -t- -k2 -n | tail -1)
        push_hf "$BEST" "grpo_v2_${MODE}_best"
    fi
done

# ─── Phase 4: 1030-Problem Eval (6 models, 2 rounds of 3) ───
log "=== Phase 4: 1030-Problem Eval ==="
mkdir -p results

# Round 1: Base SFT, V2 SFT, E3
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

E3_PATH="checkpoints/grpo_v2_E3/final"
[ ! -d "$E3_PATH" ] && E3_PATH=$(ls -d checkpoints/grpo_v2_E3/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
CUDA_VISIBLE_DEVICES=2 python -u src/eval/eval_hf.py \
    --model_path "${E3_PATH:-checkpoints/qwen3_metacot_v2_sft}" \
    --model_name 1030_e3 \
    --benchmarks gsm8k math500 aime2024 --max_problems 500 \
    --output_dir results > results/eval_e3.log 2>&1 &

wait
log "Round 1 eval done (base, v2_sft, e3)"

# Round 2: E5, E7, E8
for MODE in E5 E7 E8; do
    GPU_IDX=$(($(echo $MODE | tr -dc '0-9') % 4))
    [ "$MODE" = "E5" ] && GPU_IDX=0
    [ "$MODE" = "E7" ] && GPU_IDX=1
    [ "$MODE" = "E8" ] && GPU_IDX=2

    M_PATH="checkpoints/grpo_v2_${MODE}/final"
    [ ! -d "$M_PATH" ] && M_PATH=$(ls -d checkpoints/grpo_v2_${MODE}/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -n "$M_PATH" ] && [ -d "$M_PATH" ]; then
        CUDA_VISIBLE_DEVICES=$GPU_IDX python -u src/eval/eval_hf.py \
            --model_path "$M_PATH" \
            --model_name "1030_${MODE,,}" \
            --benchmarks gsm8k math500 aime2024 --max_problems 500 \
            --output_dir results > "results/eval_${MODE,,}.log" 2>&1 &
    else
        log "SKIP eval $MODE: no checkpoint found"
    fi
done
wait
log "Round 2 eval done (e5, e7, e8)"

# ─── Phase 5: Analysis ───
log "=== Phase 5: Analysis ==="
python scripts/analyze_1030.py --results_dir results

python3 -c "
import json
with open('results/analysis_1030.json') as f:
    a = json.load(f)
print()
print('=' * 60)
print('  FINAL RESULTS')
print('=' * 60)
base_acc = 0
for model in sorted(a.get('models', [])):
    acc = a['accuracy'].get(model, {}).get('overall', {})
    ece_data = a.get('ece', {}).get(model, {})
    ece = ece_data.get('overall', 'N/A')
    acc_val = acc.get('accuracy', 0)
    if 'base' in model:
        base_acc = acc_val
    print(f'  {model}: acc={acc_val*100:.1f}% ({acc.get(\"n_correct\", 0)}/{acc.get(\"n_total\", 0)}) ECE={ece}')
print()
best_meta = 0
best_name = ''
for model in a.get('models', []):
    if 'base' not in model:
        acc_val = a['accuracy'].get(model, {}).get('overall', {}).get('accuracy', 0)
        if acc_val > best_meta:
            best_meta = acc_val
            best_name = model
if best_meta >= base_acc:
    print(f'SUCCESS: {best_name} ({best_meta*100:.1f}%) >= Base SFT ({base_acc*100:.1f}%)!')
else:
    gap = (base_acc - best_meta) * 100
    print(f'BEST META: {best_name} ({best_meta*100:.1f}%), Base SFT ({base_acc*100:.1f}%)')
    print(f'GAP: {gap:.1f}%p — autoresearch needed')
print('=' * 60)
"

log "=== PIPELINE COMPLETE ==="
log "Node alive. SSH in for analysis or autoresearch."
sleep infinity
