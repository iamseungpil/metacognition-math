#!/bin/bash
# Phase 3: GRPO + Full Gnosis on Qwen3-8B Meta SFT
# Uses gnosis_repo's TRL GRPOTrainer with vLLM colocate, 4 GPU
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate verl
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

# Patch gnosis_repo's Qwen3 model into installed transformers (avoid full override)
# This symlinks only the Gnosis-modified Qwen3 model files, not the entire transformers package
INSTALLED_TRANSFORMERS=$(conda run -n verl python -c "import transformers, os; print(os.path.dirname(transformers.__file__))" 2>/dev/null)
echo "Installed transformers at: $INSTALLED_TRANSFORMERS"

# Backup and symlink Qwen3 model with Gnosis
if [ -d "$INSTALLED_TRANSFORMERS/models/qwen3" ]; then
    cp -r "$INSTALLED_TRANSFORMERS/models/qwen3" "$INSTALLED_TRANSFORMERS/models/qwen3_backup" 2>/dev/null || true
fi
# Copy gnosis Qwen3 files over installed ones
cp /scratch/metacognition/gnosis_repo/transformers/src/transformers/models/qwen3/modeling_qwen3.py "$INSTALLED_TRANSFORMERS/models/qwen3/modeling_qwen3.py" 2>/dev/null
cp /scratch/metacognition/gnosis_repo/transformers/src/transformers/models/qwen3/feature_extractors.py "$INSTALLED_TRANSFORMERS/models/qwen3/feature_extractors.py" 2>/dev/null
echo "Patched Qwen3 model with Gnosis feature extractors"

# Use gnosis_repo's TRL only (not transformers)
export PYTHONPATH="/scratch/metacognition/gnosis_repo/trl:$PYTHONPATH"
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

# Fix OpenSSL FIPS for Ray workers
export OPENSSL_ia32cap="~0x200000200000000"

echo "=== Phase 3: GRPO + Full Gnosis (4 GPU) ==="
echo "Model: checkpoints/qwen3_meta_sft (Qwen3-8B + Meta-CoT SFT)"
echo "Gnosis: Full (attention + hidden + confidence feature extractors)"
echo "Reward: R_correct + R_calib + R_penalty + stepwise importance"
echo "Generation: vLLM colocate, 4 GPU"

# FIX A1: Use accelerate for multi-GPU launch
# TRL GRPOTrainer with vLLM colocate needs distributed environment
accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_gnosis.py \
    --model_path checkpoints/qwen3_meta_sft \
    --train_data verl_train.parquet \
    --output_dir checkpoints/qwen3_grpo_gnosis \
    --max_steps 1000

echo "=== Phase 3 DONE ==="
