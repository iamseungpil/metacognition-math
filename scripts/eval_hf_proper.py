"""Proper evaluation using HF generate (not vLLM) to preserve <|meta|> tokens."""
import sys, json, torch, time
sys.path.insert(0, "/scratch/metacognition")

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.data.dataset_loader import load_math_test
from src.rollout.vllm_rollout import check_correctness
from src.metacot.prompt import META_START, META_END, parse_meta_blocks

# Load test data
ds = load_math_test()
ds = ds.select(range(min(200, len(ds))))
print(f"Eval on {len(ds)} MATH test problems", flush=True)

models = [
    ("base", "Qwen/Qwen2.5-7B-Instruct", False),
    ("base_sft", "/scratch/metacognition/checkpoints/base_sft", False),
    ("meta_sft", "/scratch/metacognition/checkpoints/meta_sft", True),
    ("meta_grpo", "/scratch/metacognition/checkpoints/meta_grpo/final", True),
]

results = {}

for name, path, needs_meta_tokens in models:
    print(f"\n{'='*50}", flush=True)
    print(f"=== {name}: {path} ===", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if needs_meta_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": [META_START, META_END]})
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
    )
    if needs_meta_tokens:
        model.resize_token_embeddings(len(tokenizer))
    model.eval()

    correct = 0
    meta_count = 0
    meta_with_conf = 0
    t0 = time.time()

    for i, row in enumerate(ds):
        msgs = [{"role": "user", "content": row["question"]}]
        prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **ids, max_new_tokens=2048, temperature=0.7, do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        text = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=False)
        clean = tokenizer.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

        if check_correctness(clean, row["answer"]):
            correct += 1

        # Detect meta behavior (both token and text pattern)
        has_meta_token = META_START in text
        has_meta_text = any(kw in clean.lower() for kw in [
            "can i solve", "probability of solving", "watch out",
            "is this correct", "is this right", "let me check",
            "what did i learn",
        ])
        if has_meta_token or has_meta_text:
            meta_count += 1

        parsed = parse_meta_blocks(text)
        if parsed["confidences"]:
            meta_with_conf += 1

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(ds)}] correct={correct}, meta={meta_count}, "
                  f"conf={meta_with_conf}, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    results[name] = {
        "correct": correct,
        "total": len(ds),
        "accuracy": correct / len(ds),
        "meta_usage": meta_count / len(ds),
        "meta_with_confidence": meta_with_conf / len(ds),
        "time_sec": elapsed,
    }
    print(f"{name}: {correct}/{len(ds)} ({correct/len(ds):.1%}), "
          f"meta={meta_count/len(ds):.0%}, conf={meta_with_conf/len(ds):.0%}, "
          f"{elapsed:.0f}s", flush=True)

    del model
    torch.cuda.empty_cache()

print(f"\n{'='*50}", flush=True)
print("FINAL RESULTS (HF generate, temp=0.7, max_tokens=2048)", flush=True)
print(f"{'='*50}", flush=True)
for name, r in results.items():
    print(f"  {name}: {r['correct']}/{r['total']} ({r['accuracy']:.1%}) "
          f"meta={r['meta_usage']:.0%} conf={r['meta_with_confidence']:.0%}", flush=True)

with open("/scratch/metacognition/eval_results_hf.json", "w") as f:
    json.dump(results, f, indent=2)
print("EVAL_HF_DONE", flush=True)
