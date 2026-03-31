#!/bin/bash
set -e
cd /scratch/metacognition
source /opt/conda/etc/profile.d/conda.sh
conda create -n grpo python=3.10 -y
conda activate grpo
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.52.3 trl==0.19.1 datasets accelerate deepspeed
pip install peft bitsandbytes sentencepiece protobuf pandas pyarrow
pip install math_verify latex2sympy2_extended wandb huggingface_hub
python3 -c "
from huggingface_hub import hf_hub_download
import os, shutil
os.makedirs('checkpoints/qwen3_metacot_v2_sft', exist_ok=True)
for f in ['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','generation_config.json','merges.txt','vocab.json','model.safetensors.index.json','chat_template.jinja','model-00001-of-00004.safetensors','model-00002-of-00004.safetensors','model-00003-of-00004.safetensors','model-00004-of-00004.safetensors']:
    print(f'DL {f}')
    hf_hub_download('iamseungpil/metacot', f'models/qwen3_metacot_v2_sft/{f}', repo_type='dataset', local_dir='.', token='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')
shutil.copytree('models/qwen3_metacot_v2_sft', 'checkpoints/qwen3_metacot_v2_sft', dirs_exist_ok=True)
print('MODEL_READY')
"
echo "ALL_SETUP_DONE"
