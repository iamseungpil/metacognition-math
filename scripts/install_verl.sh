#!/bin/bash
# Install veRL + vLLM + Ray on EVAL node for Meta-CoT RL
# Creates new conda env 'simplerl' to avoid breaking ptca
# Based on simpleRL-reason requirements

set -euo pipefail

echo "=== Creating simplerl conda env ==="
conda create -n simplerl python=3.10 -y
conda activate simplerl

echo "=== Installing PyTorch ==="
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124

echo "=== Installing vLLM ==="
pip install vllm==0.6.3

echo "=== Installing Ray ==="
pip install "ray[default]==2.10.0"

echo "=== Installing veRL dependencies ==="
pip install accelerate datasets dill hydra-core omegaconf numpy pybind11 \
    tensordict "transformers<4.48" peft liger-kernel word2number \
    "math-verify[antlr4_11_0]==0.6.0" deepspeed wandb pandas pyarrow pyyaml

echo "=== Installing veRL from simpleRL-reason ==="
cd /scratch/metacognition
# Copy veRL source from RandomSoftPrompt if available, else install from pip
if [ -d "/home/v-seungplee/RandomSoftPrompt/grpo/simpleRL-reason" ]; then
    cp -r /home/v-seungplee/RandomSoftPrompt/grpo/simpleRL-reason/verl /scratch/metacognition/verl_src
    cd /home/v-seungplee/RandomSoftPrompt/grpo/simpleRL-reason
    pip install -e .
    echo "Installed veRL from simpleRL-reason source"
else
    pip install verl
    echo "Installed veRL from pip"
fi

echo "=== Verification ==="
python -c "import verl; print('veRL: OK')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import ray; print(f'Ray: {ray.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

echo "=== Done ==="
