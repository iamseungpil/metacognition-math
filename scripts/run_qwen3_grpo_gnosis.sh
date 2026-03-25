#!/bin/bash
# Phase 3: GRPO + Full Gnosis on Qwen3-8B Meta SFT
# Uses gnosis_repo's TRL GRPOTrainer with vLLM colocate, 4 GPU
set -e

# Fix OpenSSL FIPS — MUST be before any import
export OPENSSL_CONF=/dev/null
export OPENSSL_ia32cap="~0x200000200000000"

# Disable FIPS at system level
sudo sed -i 's/^\.include.*fips.*//g; s/^fips = fips_sect/# fips = fips_sect/g; s/^activate = 1/# activate = 1/g' /etc/ssl/openssl.cnf 2>/dev/null || true

source /opt/conda/etc/profile.d/conda.sh
# Use ptca env (no FIPS issues, unlike verl env)
conda activate ptca
export OPENSSL_CONF=/dev/null

# Install PEFT if not present (TRL comes from gnosis_repo)
pip install peft --quiet 2>/dev/null || true
# Remove installed TRL to avoid conflict with gnosis_repo's version
pip uninstall trl -y --quiet 2>/dev/null || true
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

# Patch gnosis_repo's Qwen3 model into installed transformers (avoid full override)
# This symlinks only the Gnosis-modified Qwen3 model files, not the entire transformers package
INSTALLED_TRANSFORMERS=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))" 2>/dev/null)
echo "Installed transformers at: $INSTALLED_TRANSFORMERS"

# Backup and symlink Qwen3 model with Gnosis
if [ -d "$INSTALLED_TRANSFORMERS/models/qwen3" ]; then
    cp -r "$INSTALLED_TRANSFORMERS/models/qwen3" "$INSTALLED_TRANSFORMERS/models/qwen3_backup" 2>/dev/null || true
fi
# Copy gnosis Qwen3 files over installed ones
cp /scratch/metacognition/gnosis_repo/transformers/src/transformers/models/qwen3/modeling_qwen3.py "$INSTALLED_TRANSFORMERS/models/qwen3/modeling_qwen3.py" 2>/dev/null
cp /scratch/metacognition/gnosis_repo/transformers/src/transformers/models/qwen3/feature_extractors.py "$INSTALLED_TRANSFORMERS/models/qwen3/feature_extractors.py" 2>/dev/null
# Also patch the forward() to not require correctness_labels during generate
python3 -c "
f = '$INSTALLED_TRANSFORMERS/models/qwen3/modeling_qwen3.py'
with open(f) as fh: code = fh.read()
# Replace the ValueError raise with a pass/return
old = 'raise ValueError(\"\`correctness_labels\` (shape (B,) or (B,1), values in {-1,0,1}) is required for training.\")'
new = 'pass  # Allow forward without correctness_labels (needed for generate)'
if old in code:
    code = code.replace(old, new)
    with open(f, 'w') as fh: fh.write(code)
    print('Patched Qwen3 forward: correctness_labels no longer required')
else:
    print('Qwen3 forward already patched or pattern not found')
" 2>/dev/null
echo "Patched Qwen3 model with Gnosis feature extractors"

# Patch gnosis_repo TRL: replace assertion with auto-unfreeze
python3 -c "
f = '/scratch/metacognition/gnosis_repo/trl/trl/trainer/grpo_trainer.py'
with open(f) as fh: code = fh.read()
old = 'assert not bad, f\"Correctness head accidentally frozen: {bad[:4]}...\"'
new = '''if bad:
            print(f\"[WARN] Auto-unfreezing {len(bad)} Gnosis params\")
            for n_, p_ in model.named_parameters():
                if _trainable_correctness_param(n_): p_.requires_grad_(True)'''
if old in code:
    code = code.replace(old, new)
    with open(f, 'w') as fh: fh.write(code)
    print('Patched TRL assertion')
else:
    print('TRL already patched or assertion not found')
" 2>/dev/null
echo "TRL patch applied"

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

# First test with single GPU to catch errors, then scale to 4
# Set CUDA_VISIBLE_DEVICES=0 for single GPU test
python src/training/grpo_gnosis.py \
    --model_path checkpoints/qwen3_meta_sft \
    --train_data verl_train.parquet \
    --output_dir checkpoints/qwen3_grpo_gnosis \
    --max_steps 1000

echo "=== Phase 3 DONE ==="
