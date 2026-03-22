#!/bin/bash
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh && conda activate ptca
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

echo "=== Training Base SFT (standard CoT) ==="
rm -rf checkpoints/base_sft
export WANDB_NAME=base-sft
accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m src.training.sft --config configs/phase1_sft.yaml

echo "=== Training Meta SFT (3-phase metacognitive) ==="
rm -rf checkpoints/meta_sft

# Create meta SFT config
python -c "
import yaml
with open('configs/phase1_sft.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['dataset_path'] = '/scratch/metacognition/sft_data/metacot_sft.parquet'
cfg['output_dir'] = '/scratch/metacognition/checkpoints/meta_sft'
cfg['run_name'] = 'meta-sft'
with open('/tmp/meta_sft_config.yaml', 'w') as f:
    yaml.dump(cfg, f)
"
export WANDB_NAME=meta-sft
accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m src.training.sft --config /tmp/meta_sft_config.yaml

echo "BOTH_SFT_DONE"
