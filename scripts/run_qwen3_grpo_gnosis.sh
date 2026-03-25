#!/bin/bash
# Phase 3: GRPO + Full Gnosis on Qwen3-8B Meta SFT
set -e

# Fix OpenSSL FIPS
export OPENSSL_CONF=/dev/null
export OPENSSL_ia32cap="~0x200000200000000"
sudo sed -i 's/^\.include.*fips.*//g; s/^fips = fips_sect/# fips = fips_sect/g; s/^activate = 1/# activate = 1/g' /etc/ssl/openssl.cnf 2>/dev/null || true

source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
export OPENSSL_CONF=/dev/null

# Install PEFT if not present
pip install peft --quiet 2>/dev/null || true
# Remove installed TRL to use gnosis_repo version
pip uninstall trl -y --quiet 2>/dev/null || true

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

# Step 1: Copy Gnosis-modified Qwen3 files into installed transformers
INSTALLED_TRANSFORMERS=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
echo "Installed transformers at: $INSTALLED_TRANSFORMERS"

cp /scratch/metacognition/gnosis_repo/transformers/src/transformers/models/qwen3/modeling_qwen3.py \
   "$INSTALLED_TRANSFORMERS/models/qwen3/modeling_qwen3.py"
cp /scratch/metacognition/gnosis_repo/transformers/src/transformers/models/qwen3/feature_extractors.py \
   "$INSTALLED_TRANSFORMERS/models/qwen3/feature_extractors.py"
echo "Copied Gnosis Qwen3 model files"

# Step 2: Patch Qwen3 forward to skip Gnosis head during generate()
python scripts/patch_qwen3_forward.py

# Step 3: Patch TRL assertion (auto-unfreeze instead of assert)
python scripts/patch_trl_assertion.py

# Step 4: Set PYTHONPATH for gnosis TRL
export PYTHONPATH="/scratch/metacognition/gnosis_repo/trl:$PYTHONPATH"
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

echo "=== Phase 3: GRPO + Full Gnosis ==="
echo "Model: checkpoints/qwen3_meta_sft"
echo "Gnosis: Full (attention + hidden + confidence)"
echo "Stepwise: Agent Lightning style (R_correct to ALL steps)"

# Single GPU first to verify, then scale to multi-GPU
python src/training/grpo_gnosis.py \
    --model_path checkpoints/qwen3_meta_sft \
    --train_data verl_train.parquet \
    --output_dir checkpoints/qwen3_grpo_gnosis \
    --max_steps 1000

echo "=== Phase 3 DONE ==="
