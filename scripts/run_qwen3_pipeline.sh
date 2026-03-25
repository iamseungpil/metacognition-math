#!/bin/bash
# Full Qwen3-8B Meta-CoT pipeline: SFT → Probe → GRPO+Gnosis
# Phase 1: SFT (this script)
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh
conda activate ptca
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

echo "=== Phase 1: Qwen3-8B Meta-CoT SFT ==="
echo "Model: Qwen/Qwen3-8B"
echo "Data: 7,371 Meta-CoT chains (GSM8K + MATH + NuminaMath)"
echo "GPUs: 4 x A100 80GB, DeepSpeed ZeRO-3"

# Download Qwen3-8B if not cached
python << 'PYEOF'
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_id = "Qwen/Qwen3-8B"
cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

print(f"Downloading/caching {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
print(f"Tokenizer loaded: vocab_size={tokenizer.vocab_size}")

# Just trigger download, don't load into memory
from huggingface_hub import snapshot_download
snapshot_download(model_id, ignore_patterns=["*.gguf"])
print(f"Model files cached at {cache_dir}")
PYEOF

echo "=== Starting SFT training ==="
accelerate launch --config_file configs/accelerate_ds3.yaml \
    -m src.training.sft --config configs/phase1_qwen3_sft.yaml

echo "=== Phase 1 DONE ==="
echo "Checkpoint: /scratch/metacognition/checkpoints/qwen3_meta_sft"
