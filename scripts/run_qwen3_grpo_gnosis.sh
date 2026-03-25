#!/bin/bash
# Phase 3: GRPO + Full Gnosis + Stepwise (Agent Lightning Transition Mode)
set -e

# Fix OpenSSL FIPS
export OPENSSL_CONF=/dev/null
export OPENSSL_ia32cap="~0x200000200000000"
sudo sed -i 's/^\.include.*fips.*//g; s/^fips = fips_sect/# fips = fips_sect/g; s/^activate = 1/# activate = 1/g' /etc/ssl/openssl.cnf 2>/dev/null || true

source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
export OPENSSL_CONF=/dev/null

# Pin verified compatible versions
pip install "transformers==4.51.3" "trl==0.19.1" "peft>=0.10" --quiet 2>/dev/null || true
echo "Installed: torch=$(python -c 'import torch;print(torch.__version__)'), trl=$(python -c 'import trl;print(trl.__version__)'), tf=$(python -c 'import transformers;print(transformers.__version__)')"

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition

# Step 1: Patch Gnosis into installed Qwen3 model
INSTALLED_TRANSFORMERS=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
echo "Patching Gnosis into: $INSTALLED_TRANSFORMERS/models/qwen3/"
cp /scratch/metacognition/gnosis_repo/transformers/src/transformers/models/qwen3/modeling_qwen3.py \
   "$INSTALLED_TRANSFORMERS/models/qwen3/modeling_qwen3.py"
cp /scratch/metacognition/gnosis_repo/transformers/src/transformers/models/qwen3/feature_extractors.py \
   "$INSTALLED_TRANSFORMERS/models/qwen3/feature_extractors.py"

# Step 2: Patch forward to skip Gnosis head during generate()
python scripts/patch_qwen3_forward.py

# Step 3: Patch TRL assertion for PEFT compat
python scripts/patch_trl_assertion.py

export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

echo "=== Phase 3: GRPO + Full Gnosis + Stepwise ==="
echo "Model: Qwen3-8B Meta SFT"
echo "Gnosis: Full (attn + hidden + conf extractors)"
echo "Stepwise: Agent Lightning transition mode"
echo "Step splitting: each <|meta|> step = separate training example"
echo "R_correct: same for ALL steps | R_calib: per-step"

# Multi-GPU (4x A100)
accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_gnosis.py \
    --model_path checkpoints/qwen3_meta_sft \
    --train_data verl_train.parquet \
    --output_dir checkpoints/qwen3_grpo_gnosis \
    --max_steps 1000

echo "=== Phase 3 DONE ==="
