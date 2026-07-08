#!/bin/bash
# Rebuild EVAL node after /scratch wipe
# Step 1: Create venv
# Step 2: Copy code + data
# Step 3: Pull checkpoint from HF
# Step 4: Resume training
set -euo pipefail

echo "$(date) === EVAL Node Rebuild ==="

# Step 1: Create venv
echo "$(date) Creating venv..."
python3 -m venv /scratch/simplerl_venv
source /scratch/simplerl_venv/bin/activate

echo "$(date) Installing packages..."
pip install --upgrade pip
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.7.1
pip install verl==0.3.0.post1
pip install wandb peft huggingface_hub transformers accelerate datasets pandas pyarrow

echo "$(date) Venv ready."

# Step 2: Code + data already copied via amlt ssh scp

# Step 3: Pull checkpoint from HF
echo "$(date) Pulling checkpoint from HF..."
python -c "
from huggingface_hub import HfApi, snapshot_download
import os

api = HfApi(token='${HF_TOKEN}')

# Download latest E21R-v2 checkpoint
ckpt_dir = '/scratch/metacognition/checkpoints/metacot-math/verl_e21r_v2_0413/global_step_250'
os.makedirs(ckpt_dir, exist_ok=True)

snapshot_download(
    repo_id='iamseungpil/metacot',
    repo_type='dataset',
    local_dir='/tmp/hf_download',
    allow_patterns='checkpoints/verl_e21r_v2_0413/global_step_250/*',
)

# Move to correct location
import shutil
src = '/tmp/hf_download/checkpoints/verl_e21r_v2_0413/global_step_250'
if os.path.exists(src):
    shutil.copytree(src, ckpt_dir, dirs_exist_ok=True)
    print(f'Checkpoint restored to {ckpt_dir}')
else:
    print('ERROR: Checkpoint not found in HF download')
    exit(1)

# Write latest step marker
with open('/scratch/metacognition/checkpoints/metacot-math/verl_e21r_v2_0413/latest_checkpointed_iteration.txt', 'w') as f:
    f.write('250')
print('Done')
"

# Also pull SFT model (needed as ref model)
echo "$(date) Pulling SFT model from HF..."
python -c "
from huggingface_hub import HfApi, snapshot_download
import os, shutil

snapshot_download(
    repo_id='iamseungpil/metacot',
    repo_type='dataset',
    local_dir='/tmp/hf_download_sft',
    allow_patterns='checkpoints/v8_meta_inside_strict_sft/*',
)
src = '/tmp/hf_download_sft/checkpoints/v8_meta_inside_strict_sft'
dst = '/scratch/metacognition/checkpoints/v8_meta_inside_strict_sft'
if os.path.exists(src):
    os.makedirs(dst, exist_ok=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
    print(f'SFT model restored to {dst}')
else:
    print('WARNING: SFT model not found, may need manual copy')
"

echo "$(date) === Rebuild complete ==="
