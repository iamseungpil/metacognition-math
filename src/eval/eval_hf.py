"""HF generate eval (vLLM 0.6.6 doesn't support Qwen3).

Evaluates models using HF generate. Slower but works with any model.

Usage:
  python src/eval/eval_hf.py --model_path checkpoints/qwen3_meta_sft --benchmarks gsm8k math_test --max_problems 30
  python src/eval/eval_hf.py --model_path checkpoints/grpo_clean_meta_filtered/checkpoint-200 --is_lora --base_model checkpoints/qwen3_meta_sft
"""
import argparse
import json
import os
import re

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from src.metacot.prompt import parse_meta_blocks


from src.training.rewards import _check_correctness, _extract_answer_fallback


def extract_answer(text):
    return _extract_answer_fallback(text)


def check_correctness(pred, gold):
    return _check_correctness(pred, gold)


def load_benchmarks(names, max_problems=30):
    benchmarks = {
        "gsm8k": ("openai/gsm8k", "main", "test", "question", "answer"),
        "math_test": ("hendrycks/competition_math", None, "test", "problem", "solution"),
        "aime2024": ("HuggingFaceH4/aime_2024", None, "train", "problem", "answer"),
    }
    all_problems = []
    for name in names:
        if name not in benchmarks:
            print(f"  Unknown benchmark: {name}")
            continue
        ds_id, config, split, q_col, a_col = benchmarks[name]
        try:
            if config:
                ds = load_dataset(ds_id, config, split=split)
            else:
                ds = load_dataset(ds_id, split=split)
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            continue

        count = 0
        for row in ds:
            if count >= max_problems:
                break
            q = str(row.get(q_col, ""))
            a = str(row.get(a_col, ""))
            if q:
                all_problems.append({"question": q, "gold_answer": a, "benchmark": name})
                count += 1
        print(f"  {name}: {count} problems")

    return all_problems


def evaluate(model, tokenizer, problems, num_samples=1, max_tokens=1024):
    results = []
    for idx, prob in enumerate(problems):
        messages = [{"role": "user", "content": prob["question"]}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

        for _ in range(num_samples):
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=True, temperature=0.7, top_p=0.95,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=False)

            is_correct = check_correctness(gen, prob["gold_answer"])
            parsed = parse_meta_blocks(gen)
            confs = parsed["confidences"]
            avg_conf = sum(confs) / len(confs) if confs else None

            results.append({
                "benchmark": prob["benchmark"],
                "question": prob["question"][:80],
                "is_correct": is_correct,
                "num_meta_blocks": parsed["num_blocks"],
                "avg_confidence": avg_conf,
                "answer_extracted": extract_answer(gen),
                "gold_answer": prob["gold_answer"][:50],
                "completion": gen[:500],  # for qualitative analysis
            })

        if (idx + 1) % 10 == 0:
            n_correct = sum(r["is_correct"] for r in results)
            print(f"  {idx+1}/{len(problems)}: {n_correct}/{len(results)} correct")

    return results


def print_results(model_name, results):
    df = pd.DataFrame(results)
    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  {'Benchmark':<12} {'Acc':>6} {'ECE':>6} {'Meta':>5} {'ConfRate':>8}")
    print(f"  {'-'*42}")

    for bench in sorted(df["benchmark"].unique()):
        bdf = df[df["benchmark"] == bench]
        acc = bdf["is_correct"].mean()
        avg_meta = bdf["num_meta_blocks"].mean()
        conf_rate = bdf["avg_confidence"].notna().mean()

        calib_errors = []
        for _, r in bdf.iterrows():
            if r["avg_confidence"] is not None:
                actual = 1.0 if r["is_correct"] else 0.0
                calib_errors.append(abs(r["avg_confidence"] - actual))
        ece = sum(calib_errors) / len(calib_errors) if calib_errors else None
        ece_str = f"{ece:.3f}" if ece else "  N/A"

        print(f"  {bench:<12} {acc*100:5.1f}% {ece_str} {avg_meta:5.1f} {conf_rate*100:6.1f}%")

    acc = df["is_correct"].mean()
    print(f"  {'-'*42}")
    print(f"  {'OVERALL':<12} {acc*100:5.1f}%")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default=None, help="Base model for LoRA")
    parser.add_argument("--is_lora", action="store_true")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math_test"])
    parser.add_argument("--max_problems", type=int, default=30)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    if args.is_lora:
        base_path = args.base_model or "checkpoints/qwen3_meta_sft"
        print(f"Loading base: {base_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        print(f"Loading LoRA: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    else:
        print(f"Loading: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.cuda().eval()

    model_name = args.model_path.split("/")[-1]
    if args.is_lora:
        model_name = f"grpo-{model_name}"

    # Load benchmarks
    print(f"\nLoading benchmarks: {args.benchmarks}")
    problems = load_benchmarks(args.benchmarks, args.max_problems)
    print(f"Total: {len(problems)} problems\n")

    # Evaluate
    results = evaluate(model, tokenizer, problems, args.num_samples)
    df = print_results(model_name, results)

    # Save
    save_path = os.path.join(args.output_dir, f"eval_{model_name}.json")
    with open(save_path, "w") as f:
        json.dump({"model": model_name, "results": results}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {save_path}")


if __name__ == "__main__":
    main()
