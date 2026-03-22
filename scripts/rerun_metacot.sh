#!/bin/bash
# Re-run Meta-CoT experiment from SFT onwards (rollouts + Meta-CoT chains already exist)
set -e
export OPENSSL_CONF=/dev/null
source /opt/conda/etc/profile.d/conda.sh && conda activate ptca
export OPENSSL_CONF=/dev/null
cd /scratch/metacognition
export PYTHONPATH=/scratch/metacognition
export WANDB_API_KEY=2f4e627868f1f9dad10bcb1a14fbf96817e6baa9

echo "========================================="
echo "Meta-CoT Experiment Re-run"
echo "========================================="

# Step 1: Re-build SFT dataset with fixed validation
echo ""
echo "=== Step 1: Re-build SFT dataset ==="
python -m src.metacot.generator \
    --build-sft \
    --metacot-path /scratch/metacognition/metacot_chains/metacot_final.parquet \
    --sft-output /scratch/metacognition/metacot_chains/metacot_sft.parquet

# Step 2: Re-run SFT (full fine-tune on 4x A100)
echo ""
echo "=== Step 2: SFT Training (full fine-tune) ==="
rm -rf /scratch/metacognition/checkpoints/phase1_sft
accelerate launch --num_processes 4 --mixed_precision bf16 \
    -m src.training.sft --config configs/phase1_sft.yaml

# Verify SFT model generates Meta-CoT
echo ""
echo "=== Step 2b: Verify SFT produces Meta-CoT ==="
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained('/scratch/metacognition/checkpoints/phase1_sft')
model = AutoModelForCausalLM.from_pretrained('/scratch/metacognition/checkpoints/phase1_sft', torch_dtype=torch.bfloat16, device_map='cuda:0')
messages = [
    {'role': 'system', 'content': 'You are a math problem solver with metacognitive awareness. For each problem, solve it step by step, then analyze your solution quality, plan what to study next, select practice problems, and predict your improvement.'},
    {'role': 'user', 'content': 'What is the remainder when 17^2023 is divided by 5?'}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
ids = tokenizer(prompt, return_tensors='pt').to('cuda:0')
out = model.generate(**ids, max_new_tokens=500, temperature=0.7, do_sample=True)
text = tokenizer.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)
stages = ['solve', 'diagnose', 'strategize', 'select', 'predict']
found = [s for s in stages if s in text.lower()]
print(f'Meta-CoT stages found: {found} ({len(found)}/5)')
print(f'Output preview: {text[:300]}...')
del model; torch.cuda.empty_cache()
" 2>&1

# Step 3: GRPO with Meta-CoT system prompt
echo ""
echo "=== Step 3: GRPO Training ==="
rm -rf /scratch/metacognition/checkpoints/phase2_grpo
python -m src.training.grpo --config configs/phase2_grpo.yaml

# Step 4: 3-model comparison on MATH test
echo ""
echo "=== Step 4: 3-Model Evaluation ==="
python << 'PYEOF'
import sys, json
sys.path.insert(0, "/scratch/metacognition")
from vllm import LLM, SamplingParams
from src.data.dataset_loader import load_math_test
from src.rollout.vllm_rollout import check_correctness, build_chat_messages, MATH_SYSTEM_PROMPT, METACOT_SYSTEM_PROMPT

ds = load_math_test()
ds = ds.select(range(min(200, len(ds))))
print(f"Evaluating {len(ds)} MATH test problems", flush=True)

sp = SamplingParams(temperature=0.0, max_tokens=512)

models = [
    ("base", "Qwen/Qwen2.5-7B-Instruct", MATH_SYSTEM_PROMPT),
    ("sft", "/scratch/metacognition/checkpoints/phase1_sft", METACOT_SYSTEM_PROMPT),
    ("grpo", "/scratch/metacognition/checkpoints/phase2_grpo/final", METACOT_SYSTEM_PROMPT),
]

results = {}
for name, path, sys_prompt in models:
    print(f"\n--- {name}: {path} ---", flush=True)
    try:
        llm = LLM(model=path, tensor_parallel_size=1, gpu_memory_utilization=0.85, max_model_len=2048, dtype="bfloat16", trust_remote_code=True)
        tokenizer = llm.get_tokenizer()
        prompts = []
        for row in ds:
            msgs = build_chat_messages(row["question"], system_prompt=sys_prompt)
            prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
        outputs = llm.generate(prompts, sampling_params=sp)
        correct = sum(1 for i, out in enumerate(outputs) if check_correctness(out.outputs[0].text, ds[i]["answer"]))
        results[name] = correct
        print(f"{name}_pass_at_1: {correct}/{len(ds)} = {correct/len(ds):.3f}", flush=True)
        del llm
        import torch; torch.cuda.empty_cache()
    except Exception as e:
        print(f"{name} FAILED: {e}", flush=True)
        results[name] = -1

print(f"\n=== FINAL RESULTS ===", flush=True)
for name, score in results.items():
    print(f"  {name}: {score}/200", flush=True)

base = results.get("base", 0)
sft = results.get("sft", 0)
grpo = results.get("grpo", 0)
print(f"\nResearch Question: Does Meta-CoT improve performance?", flush=True)
print(f"  SFT vs Base: {sft - base:+d} problems ({'+' if sft > base else ''}{(sft-base)/max(base,1)*100:.1f}%)", flush=True)
print(f"  GRPO vs SFT: {grpo - sft:+d} problems ({'+' if grpo > sft else ''}{(grpo-sft)/max(sft,1)*100:.1f}%)", flush=True)
print(f"  GRPO vs Base: {grpo - base:+d} problems ({'+' if grpo > base else ''}{(grpo-base)/max(base,1)*100:.1f}%)", flush=True)

with open("/scratch/metacognition/eval_results.json", "w") as f:
    json.dump(results, f)
PYEOF

# Step 5: Self-directed curriculum learning (Phase 3)
echo ""
echo "=== Step 5: Self-Directed Curriculum Learning ==="
python -m src.training.curriculum \
    --model-path /scratch/metacognition/checkpoints/phase2_grpo/final \
    --data-pool /scratch/metacognition/rollouts/rollouts_final.parquet \
    --output-dir /scratch/metacognition/curriculum \
    --n-cycles 1 \
    --problems-per-cycle 500

echo ""
echo "RERUN_COMPLETE"
