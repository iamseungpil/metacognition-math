#!/bin/bash
# Sequential pipeline: Base SFT → E5 → E7 → E8 (500 steps each)
# Runs after E3 completes. Each model uploaded to HF immediately.
# E3 is already running separately (200 steps).

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
source /opt/conda/etc/profile.d/conda.sh
conda activate grpo

export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9
export HF_TOKEN=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

LOG=/scratch/metacognition/sequential.log
log() { echo "$(date '+%Y-%m-%d %H:%M:%S'): $*" | tee -a "$LOG"; }

push_hf() {
    local path="$1" name="$2"
    log "HF upload: $name"
    python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')
api.upload_folder(
    repo_id='iamseungpil/metacot', folder_path='$path',
    path_in_repo='models/$name', repo_type='dataset',
    commit_message='Upload $name',
    ignore_patterns=['checkpoint-*','optimizer*','scheduler*','trainer_state*','training_args*','wandb/*','runs/*'],
)
print('Upload done: $name')
" >> "$LOG" 2>&1 && log "HF done: $name" || log "HF FAILED: $name"
}

# ─── Wait for E3 to finish ───
log "Waiting for E3 to finish..."
while ps aux | grep "grpo_v2.*E3" | grep -v grep > /dev/null 2>&1; do
    sleep 60
done
log "E3 finished."

# Upload E3
if [ -d checkpoints/grpo_v2_E3/final ]; then
    push_hf checkpoints/grpo_v2_E3/final grpo_v2_E3_final
else
    BEST=$(ls -d checkpoints/grpo_v2_E3/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -n "$BEST" ] && push_hf "$BEST" grpo_v2_E3_best
fi

# ─── Base SFT (re-train, previous save failed) ───
if [ -z "$(ls checkpoints/qwen3_base_sft/*.safetensors 2>/dev/null)" ]; then
    log "=== Base SFT training ==="
    git pull
    accelerate launch --config_file configs/accelerate_sft.yaml \
        src/training/sft.py --config configs/sft_base.yaml >> "$LOG" 2>&1
    log "Base SFT done."
    push_hf checkpoints/qwen3_base_sft qwen3_base_sft
else
    log "Base SFT already exists, skipping."
fi

# ─── GRPO E5 (500 steps) ───
log "=== GRPO E5 (500 steps) ==="
git pull
accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/grpo_v2.py --mode E5 --max_steps 500 \
    --model_path checkpoints/qwen3_metacot_v2_sft --data mixed >> "$LOG" 2>&1
log "E5 done."
if [ -d checkpoints/grpo_v2_E5/final ]; then
    push_hf checkpoints/grpo_v2_E5/final grpo_v2_E5_final
else
    BEST=$(ls -d checkpoints/grpo_v2_E5/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -n "$BEST" ] && push_hf "$BEST" grpo_v2_E5_best
fi

# ─── GRPO E7 (500 steps) ───
log "=== GRPO E7 (500 steps) ==="
git pull
accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/grpo_v2.py --mode E7 --max_steps 500 \
    --model_path checkpoints/qwen3_metacot_v2_sft --data mixed >> "$LOG" 2>&1
log "E7 done."
if [ -d checkpoints/grpo_v2_E7/final ]; then
    push_hf checkpoints/grpo_v2_E7/final grpo_v2_E7_final
else
    BEST=$(ls -d checkpoints/grpo_v2_E7/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -n "$BEST" ] && push_hf "$BEST" grpo_v2_E7_best
fi

# ─── GRPO E8 (500 steps) ───
log "=== GRPO E8 (500 steps) ==="
git pull
accelerate launch --config_file configs/accelerate_grpo.yaml \
    src/training/grpo_v2.py --mode E8 --max_steps 500 \
    --model_path checkpoints/qwen3_metacot_v2_sft --data mixed >> "$LOG" 2>&1
log "E8 done."
if [ -d checkpoints/grpo_v2_E8/final ]; then
    push_hf checkpoints/grpo_v2_E8/final grpo_v2_E8_final
else
    BEST=$(ls -d checkpoints/grpo_v2_E8/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    [ -n "$BEST" ] && push_hf "$BEST" grpo_v2_E8_best
fi

log "=== ALL SEQUENTIAL TRAINING COMPLETE ==="
log "Models on HF: E3, Base SFT, E5, E7, E8"
