#!/bin/bash
# Quick AIME-only evaluation for autoresearch iterations
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh && conda activate ptca
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

MODEL_PATH="${1:-/scratch/metacognition/checkpoints/phase1_sft}"

echo "Evaluating: $MODEL_PATH on AIME 2025"
python -c "
import sys
sys.path.insert(0, '/scratch/metacognition')
from vllm import LLM, SamplingParams
from src.data.dataset_loader import load_aime, extract_boxed_answer
from src.rollout.vllm_rollout import check_correctness, build_chat_messages

ds = load_aime('2025')
print(f'AIME 2025: {len(ds)} problems')

llm = LLM(model='${MODEL_PATH}', tensor_parallel_size=1, gpu_memory_utilization=0.85, max_model_len=2048, dtype='bfloat16', trust_remote_code=True)
tokenizer = llm.get_tokenizer()
sp = SamplingParams(temperature=0.0, max_tokens=1024)

prompts = []
for row in ds:
    msgs = build_chat_messages(row['question'])
    prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

outputs = llm.generate(prompts, sampling_params=sp)
correct = 0
for i, out in enumerate(outputs):
    text = out.outputs[0].text
    if check_correctness(text, ds[i]['answer']):
        correct += 1

print(f'aime2025_pass_at_1: {correct}')
print(f'aime2025_accuracy: {correct/len(ds):.3f}')
" 2>&1
