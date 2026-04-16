"""HF generate eval (vLLM 0.6.6 doesn't support Qwen3).

Evaluates models using HF generate. Slower but works with any model.

Usage:
  python src/eval/eval_hf.py --model_path checkpoints/qwen3_meta_sft --benchmarks gsm8k math_test --max_problems 30
  python src/eval/eval_hf.py --model_path checkpoints/grpo_clean_meta_filtered/checkpoint-200 --is_lora --base_model checkpoints/qwen3_meta_sft
"""
import argparse
import datetime as dt
import json
import os
import re
import socket

import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.metacot.prompt import META_END, META_START, parse_meta_blocks
from src.curriculum.control_rag import (
    TfidfExampleRetriever,
    build_model_inputs,
    load_example_bank,
    run_redirect_rag_pass,
)
from src.training.tokenizer_utils import ensure_meta_tokens_not_special


from src.training.rewards import _check_correctness, _extract_answer_fallback


def extract_answer(text):
    return _extract_answer_fallback(text)


def check_correctness(pred, gold):
    return _check_correctness(pred, gold)


def load_benchmarks(names, max_problems=30):
    benchmarks = {
        "gsm8k": ("openai/gsm8k", "main", "test", "question", "answer"),
        "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem", "answer"),
        "aime2024": ("HuggingFaceH4/aime_2024", None, "train", "problem", "answer"),
        "omni_math": ("KbsdJames/Omni-MATH", None, "test", "problem", "answer"),
        "openmath_cot": ("nvidia/OpenMathReasoning", "cot", "train", "problem", "expected_answer"),
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
                # For GSM8K, extract just the final answer after ####
                if name == "gsm8k" and "####" in a:
                    a = a.split("####")[-1].strip()
                all_problems.append({"question": q, "gold_answer": a, "benchmark": name})
                count += 1
        print(f"  {name}: {count} problems")

    return all_problems


def evaluate(
    model,
    tokenizer,
    problems,
    num_samples=1,
    max_tokens=4096,
    retriever=None,
    rag_top_k=1,
    *,
    max_prompt_tokens=2048,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
):
    results = []
    for idx, prob in enumerate(problems):
        messages = [{"role": "user", "content": prob["question"]}]
        text, inputs = build_model_inputs(
            tokenizer,
            messages,
            device=model.device,
            add_generation_prompt=True,
            max_prompt_tokens=max_prompt_tokens,
        )
        prompt_tokenized_full = tokenizer(text, return_tensors="pt", truncation=False)
        prompt_total_tokens = int(prompt_tokenized_full["input_ids"].shape[1])
        prompt_len_tokens = int(inputs["input_ids"].shape[1])
        prompt_was_truncated = prompt_total_tokens > prompt_len_tokens

        for _ in range(num_samples):
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=max_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion_ids = output[0][prompt_len_tokens:]
            completion_len_tokens = int(completion_ids.shape[0])
            hit_max_new_tokens = completion_len_tokens >= int(max_tokens)
            gen = tokenizer.decode(completion_ids, skip_special_tokens=False)
            first_completion = gen
            rag_run = None
            if retriever is not None:
                rag_run = run_redirect_rag_pass(
                    model,
                    tokenizer,
                    prob["question"],
                    first_completion,
                    retriever,
                    top_k=rag_top_k,
                    max_new_tokens=max_tokens,
                )
                if rag_run["rag_used"]:
                    gen = rag_run["rag_completion"]
                    completion_len_tokens = len(tokenizer(gen, return_tensors="pt")["input_ids"][0])
                    hit_max_new_tokens = completion_len_tokens >= int(max_tokens)

            is_correct = check_correctness(gen, prob["gold_answer"])
            parsed = parse_meta_blocks(gen)
            confs = parsed["confidences"]
            avg_conf = sum(confs) / len(confs) if confs else None

            results.append({
                "benchmark": prob["benchmark"],
                "question": prob["question"][:80],
                "full_question": prob["question"],
                "is_correct": is_correct,
                "num_meta_blocks": parsed["num_blocks"],
                "meta_confidences": confs,
                "avg_confidence": avg_conf,
                "answer_extracted": extract_answer(gen),
                "gold_answer": prob["gold_answer"][:50],
                "full_gold_answer": prob["gold_answer"],
                "prompt_length_chars": len(text),
                "prompt_length_tokens": prompt_len_tokens,
                "prompt_total_tokens_before_truncation": prompt_total_tokens,
                "prompt_was_truncated": prompt_was_truncated,
                "completion_length_chars": len(gen),
                "completion_length_tokens": completion_len_tokens,
                "hit_max_new_tokens": hit_max_new_tokens,
                "rag_used": bool(rag_run and rag_run["rag_used"]),
                "retrieved_questions": [item["question"] for item in (rag_run["retrieved"] if rag_run else [])],
                "retrieval_scores": [item["score"] for item in (rag_run["retrieved"] if rag_run else [])],
                "rag_diagnosis": rag_run["analysis"]["diagnosis_text"] if rag_run else "",
                "first_completion": first_completion,
                "completion": gen,  # full completion for qualitative analysis
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


def build_run_metadata(args, model_name, problems, tokenizer, *, resolved_do_sample: bool):
    additional_special = getattr(tokenizer, "additional_special_tokens", None)
    return {
        "model": model_name,
        "model_path": args.model_path,
        "base_model": args.base_model,
        "is_lora": args.is_lora,
        "benchmarks": args.benchmarks,
        "max_problems_per_benchmark": args.max_problems,
        "num_samples": args.num_samples,
        "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens,
        "do_sample_requested": args.do_sample,
        "do_sample_resolved": resolved_do_sample,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "device_map": args.device_map,
        "total_problems": len(problems),
        "hostname": socket.gethostname(),
        "utc_timestamp": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
        "meta_token_ids": {
            META_START: tokenizer.convert_tokens_to_ids(META_START),
            META_END: tokenizer.convert_tokens_to_ids(META_END),
        },
        "additional_special_tokens": list(additional_special or []),
    }


def save_results_bundle(output_dir, model_name, run_metadata, results):
    json_path = os.path.join(output_dir, f"eval_{model_name}.json")
    metadata_path = os.path.join(output_dir, f"eval_{model_name}.metadata.json")
    parquet_path = os.path.join(output_dir, f"eval_{model_name}.parquet")

    payload = {
        "model": model_name,
        "run_metadata": run_metadata,
        "results": results,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    with open(metadata_path, "w") as f:
        json.dump(run_metadata, f, indent=2, ensure_ascii=False)

    parquet_saved = False
    try:
        pd.DataFrame(results).to_parquet(parquet_path, index=False)
        parquet_saved = True
    except Exception as e:
        print(f"Warning: failed to save parquet ({type(e).__name__}: {e})")

    print(f"\nSaved JSON to {json_path}")
    print(f"Saved metadata to {metadata_path}")
    if parquet_saved:
        print(f"Saved parquet to {parquet_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--base_model", default=None, help="Base model for LoRA")
    parser.add_argument("--is_lora", action="store_true")
    parser.add_argument("--benchmarks", nargs="+", default=["gsm8k", "math500"])
    parser.add_argument("--max_problems", type=int, default=30)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--max_prompt_tokens", type=int, default=2048)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device_map", default="single", choices=["single", "auto"])
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--model_name", default=None, help="Override model name for output file")
    parser.add_argument("--rag_example_bank", nargs="*", default=None,
                        help="Optional parquet/json/jsonl paths for redirect-time retrieval")
    parser.add_argument("--rag_top_k", type=int, default=1)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    use_cuda = torch.cuda.is_available()
    load_dtype = torch.bfloat16 if use_cuda else torch.float32
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    do_sample = bool(args.do_sample)
    if args.temperature <= 0.0:
        do_sample = False

    model_load_kwargs = {
        "torch_dtype": load_dtype,
        "trust_remote_code": True,
    }
    if use_cuda and args.device_map == "auto":
        model_load_kwargs["device_map"] = "auto"

    # Load model
    if args.is_lora:
        from peft import PeftModel

        base_path = args.base_model or "checkpoints/qwen3_meta_sft"
        print(f"Loading base: {base_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_path, **model_load_kwargs,
        )
        print(f"Loading LoRA: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(base_path, trust_remote_code=True)
    else:
        print(f"Loading: {args.model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, **model_load_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    ensure_meta_tokens_not_special(tokenizer, [META_START, META_END])

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if not (use_cuda and args.device_map == "auto"):
        model = model.to("cuda" if use_cuda else "cpu")
    model = model.eval()

    if args.model_name:
        model_name = args.model_name
    else:
        model_name = args.model_path.split("/")[-1]
        if args.is_lora:
            model_name = f"grpo-{model_name}"

    # Load benchmarks
    print(f"\nLoading benchmarks: {args.benchmarks}")
    problems = load_benchmarks(args.benchmarks, args.max_problems)
    print(f"Total: {len(problems)} problems\n")

    retriever = None
    if args.rag_example_bank:
        bank_records = load_example_bank(args.rag_example_bank)
        if bank_records:
            retriever = TfidfExampleRetriever(bank_records)
            print(f"Loaded retrieval bank with {len(bank_records)} solved examples")
        else:
            print("Warning: retrieval bank paths were provided but no examples were loaded")

    # Evaluate
    results = evaluate(
        model,
        tokenizer,
        problems,
        args.num_samples,
        args.max_new_tokens,
        retriever=retriever,
        rag_top_k=args.rag_top_k,
        max_prompt_tokens=args.max_prompt_tokens,
        do_sample=do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    df = print_results(model_name, results)
    run_metadata = build_run_metadata(
        args,
        model_name,
        problems,
        tokenizer,
        resolved_do_sample=do_sample,
    )

    # Save
    save_results_bundle(args.output_dir, model_name, run_metadata, results)


if __name__ == "__main__":
    main()
