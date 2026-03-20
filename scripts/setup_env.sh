#!/bin/bash
set -e
echo "=== Setting up metacognition environment (in ptca conda env) ==="

eval "$(conda shell.bash hook)"
conda activate ptca

# Core deps (torch already in ptca)
pip install vllm datasets accelerate wandb scikit-learn tqdm
pip install flash-attn --no-build-isolation

# Install Gnosis forked transformers + TRL
cd "$(dirname "$0")/../gnosis_repo"
pip install -e ./transformers
pip install -e "./trl[vllm]"
cd open-r1 && GIT_LFS_SKIP_SMUDGE=1 pip install -e ".[dev]" --no-deps && cd ../..

# WandB login
wandb login 2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

# Verify
python -c "
import transformers, torch
print(f'torch={torch.__version__} transformers={transformers.__version__}')
print(f'CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')
"
echo "=== Setup complete ==="
