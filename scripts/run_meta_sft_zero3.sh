#!/bin/bash
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh && conda activate ptca
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

echo "=== Meta SFT with ZeRO-3 (4x A100, full 4096 seq length) ==="
rm -rf checkpoints/meta_sft

python -c "
import yaml
cfg = {
    'model_name_or_path': 'Qwen/Qwen2.5-7B-Instruct',
    'dataset_path': '/scratch/metacognition/sft_data/metacot_sft.parquet',
    'output_dir': '/scratch/metacognition/checkpoints/meta_sft',
    'num_train_epochs': 5,
    'per_device_train_batch_size': 1,
    'gradient_accumulation_steps': 8,
    'learning_rate': 2e-5,
    'save_steps': 200,
    'max_length': 4096,
    'deepspeed': 'configs/ds_zero3.json',
    'wandb_project': 'metacot-math',
    'run_name': 'meta-sft-zero3-1551ex-5ep',
}
with open('/tmp/meta_sft_zero3.yaml', 'w') as f:
    yaml.dump(cfg, f)
"

export WANDB_NAME=meta-sft-zero3
accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m src.training.sft --config /tmp/meta_sft_zero3.yaml

echo "META_SFT_ZERO3_DONE"
