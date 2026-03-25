#!/bin/bash
# Phase 2: Gnosis Head SFT Training (backbone frozen)
set -e
export OPENSSL_CONF=/dev/null
export OPENSSL_ia32cap="~0x200000200000000"
sudo sed -i 's/^\.include.*fips.*//g; s/^fips = fips_sect/# fips = fips_sect/g; s/^activate = 1/# activate = 1/g' /etc/ssl/openssl.cnf 2>/dev/null || true

source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
export OPENSSL_CONF=/dev/null

pip install "transformers==4.51.3" "trl==0.19.1" "peft>=0.10" --quiet 2>/dev/null || true

cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=$(cat ~/.wandb_key 2>/dev/null || echo "2f4e627868f1f9dad10bcb1a14fbf96817e6baa9")

# Patch Gnosis into installed transformers
INSTALLED_TRANSFORMERS=$(python -c "import transformers, os; print(os.path.dirname(transformers.__file__))")
cp /scratch/metacognition/gnosis_repo/transformers/src/transformers/models/qwen3/modeling_qwen3.py \
   "$INSTALLED_TRANSFORMERS/models/qwen3/modeling_qwen3.py"
cp /scratch/metacognition/gnosis_repo/transformers/src/transformers/models/qwen3/feature_extractors.py \
   "$INSTALLED_TRANSFORMERS/models/qwen3/feature_extractors.py"
echo "Patched Gnosis model files"

# NO need to patch forward — for Gnosis SFT, correctness_label IS provided

echo "=== Phase 2: Gnosis Head SFT ==="
echo "Model: Qwen3-8B Meta SFT (backbone frozen)"
echo "Data: rollouts_final.parquet"
echo "Head: AttnExtractor + HiddenExtractor + ConfExtractor + StopHead"

accelerate launch --num_processes 4 --multi_gpu \
    src/training/gnosis_sft.py \
    --model_path checkpoints/qwen3_meta_sft \
    --data_path rollouts/rollouts_final.parquet \
    --output_dir checkpoints/gnosis_head \
    --max_length 2048 \
    --batch_size 2 \
    --lr 1e-4 \
    --epochs 2

echo "=== Phase 2 DONE ==="
