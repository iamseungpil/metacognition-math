#!/bin/bash
set -e

echo "=== Setting up metacognition environment ==="

# Create conda env if not exists
if ! conda env list | grep -q "metacot"; then
    conda create -n metacot python=3.11 -y
fi

eval "$(conda shell.bash hook)"
conda activate metacot

# Core dependencies
pip install --upgrade pip wheel setuptools
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install vllm==0.8.5.post1
pip install flash-attn --no-build-isolation

# Install Gnosis forked transformers + TRL
cd "$(dirname "$0")/../gnosis_repo"
pip install -e ./transformers
pip install -e "./trl[vllm]"
cd open-r1
GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]" --no-deps
cd ../..

# Project dependencies
pip install azure-identity openai datasets wandb accelerate deepspeed
pip install pandas scikit-learn pyyaml tqdm

# Wandb login
wandb login 2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

# Verify
python -c "
import transformers, trl, torch, vllm
print(f'transformers: {transformers.__version__}')
print(f'trl: {trl.__version__}')
print(f'torch: {torch.__version__}')
print(f'vllm: {vllm.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
"

echo "=== Setup complete ==="
