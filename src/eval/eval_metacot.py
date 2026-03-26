"""Metacognitive evaluation with vLLM (fast).

Compares: Base vs Base SFT vs Meta SFT vs Meta GRPO
Benchmarks: GSM8K, MATH test, AIME 2024/2025, OlymMATH
Metrics: accuracy, calibration (ECE), meta block quality

Usage:
  python src/eval/eval_metacot.py --config configs/eval_qwen3.yaml
  python src/eval/eval_metacot.py --config configs/eval_qwen3.yaml --model_name qwen3-meta-sft
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import load_dataset
from vllm import LLM, SamplingParams

from src.metacot.prompt import parse_meta_blocks


def extract_answer(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    m = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'(?:the answer is|answer:\s*)\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def check_correctness(model_answer, gold_answer):
    model_final = extract_answer(model_answer)
    gold_str = str(gold_answer).strip()
    gold_final = extract_answer(gold_str) or gold_str
    if not model_final:
        return False
    if model_final == gold_final:
        return True
    try:
        if abs(float(model_final) - float(gold_final)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    return model_final.lower().strip() == gold_final.lower().strip()


def load_benchmark(name, dataset_id, split, max_problems=None, config=None):
    """Load benchmark dataset."""
    try:
        if config:
            ds = load_dataset(dataset_id, config, split=split, trust_remote_code=True)
        else:
            ds = load_dataset(dataset_id, split=split, trust_remote_code=True)
    except Exception:
        try:
            ds = load_dataset(dataset_id, trust_remote_code=True)
            ds = ds.get("test", ds.get("train", list(ds.values())[0]))
        except Exception as e:
            print(f"  Failed to load {name}: {e}")
            return []

    problems = []
    for row in ds:
        q = row.get("problem", row.get("question", row.get("input", "")))
        a = row.get("solution", row.get("answer", row.get("expected_output", "")))
        if q:
            problems.append({"question": str(q), "gold_answer": str(a or ""), "benchmark": name})

    if max_problems:
        problems = problems[:max_problems]
    print(f"  {name}: {len(problems)} problems")
    return problems


def evaluate_model(model_name, model_path, problems, config):
    """Evaluate a model using vLLM on all problems."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name} ({model_path})")
    print(f"{'='*60}")

    try:
        llm = LLM(
            model=model_path,
            tensor_parallel_size=config.get("tensor_parallel_size", 4),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.90),
            max_model_len=config.get("max_model_len", 4096),
            dtype="bfloat16",
            trust_remote_code=True,
        )
    except Exception as e:
        print(f"  Failed to load model: {e}")
        return []

    tokenizer = llm.get_tokenizer()
    num_samples = config.get("num_samples", 4)
    sampling_params = SamplingParams(
        temperature=config.get("temperature", 0.7),
        top_p=config.get("top_p", 0.95),
        max_tokens=config.get("max_tokens", 2048),
        n=num_samples,
    )

    # Build prompts
    prompts = []
    for prob in problems:
        messages = [{"role": "user", "content": prob["question"]}]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = f"Question: {prob['question']}\n\nAnswer: "
        prompts.append(prompt)

    # Generate
    print(f"  Generating {len(prompts)} × {num_samples} = {len(prompts)*num_samples} completions...")
    outputs = llm.generate(prompts, sampling_params)

    # Evaluate
    results = []
    for prob, output in zip(problems, outputs):
        for completion in output.outputs:
            text = completion.text
            is_correct = check_correctness(text, prob["gold_answer"])

            parsed = parse_meta_blocks(text)
            confidences = parsed["confidences"]
            num_meta = parsed["num_blocks"]

            avg_conf = sum(confidences) / len(confidences) if confidences else None
            actual = 1.0 if is_correct else 0.0
            calib_error = abs(avg_conf - actual) if avg_conf is not None else None

            results.append({
                "benchmark": prob["benchmark"],
                "is_correct": is_correct,
                "num_meta_blocks": num_meta,
                "avg_confidence": avg_conf,
                "calibration_error": calib_error,
            })

    # Free GPU
    del llm
    torch.cuda.empty_cache()

    return results


def print_metrics(model_name, results):
    """Print formatted metrics table."""
    df = pd.DataFrame(results)
    if df.empty:
        print("  No results")
        return {}

    metrics = {}
    print(f"\n  {'Benchmark':<15} {'Acc':>6} {'ECE':>6} {'Meta':>5} {'ConfRate':>8}")
    print(f"  {'-'*45}")

    for bench in sorted(df["benchmark"].unique()):
        bdf = df[df["benchmark"] == bench]
        acc = bdf["is_correct"].mean()
        ece = bdf["calibration_error"].dropna().mean() if bdf["calibration_error"].notna().any() else None
        avg_meta = bdf["num_meta_blocks"].mean()
        conf_rate = bdf["avg_confidence"].notna().mean()

        ece_str = f"{ece:.3f}" if ece is not None else "  N/A"
        print(f"  {bench:<15} {acc*100:5.1f}% {ece_str} {avg_meta:5.1f} {conf_rate*100:6.1f}%")

        metrics[bench] = {"accuracy": acc, "ece": ece, "avg_meta_blocks": avg_meta, "confidence_rate": conf_rate}

    # Overall
    acc = df["is_correct"].mean()
    ece = df["calibration_error"].dropna().mean() if df["calibration_error"].notna().any() else None
    ece_str = f"{ece:.3f}" if ece is not None else "  N/A"
    print(f"  {'-'*45}")
    print(f"  {'OVERALL':<15} {acc*100:5.1f}% {ece_str}")

    metrics["overall"] = {"accuracy": acc, "ece": ece}
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_name", default=None, help="Evaluate only this model")
    parser.add_argument("--max_problems", type=int, default=50)
    parser.add_argument("--output_dir", default="results")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load benchmarks
    print("Loading benchmarks...")
    all_problems = []
    for bench in config["benchmarks"]:
        problems = load_benchmark(bench["name"], bench["dataset"], bench.get("split", "test"), args.max_problems, bench.get("config"))
        all_problems.extend(problems)
    print(f"Total: {len(all_problems)} problems\n")

    # Evaluate models
    models = config["models"]
    if args.model_name:
        models = [m for m in models if m["name"] == args.model_name]

    all_metrics = {}
    for model_cfg in models:
        results = evaluate_model(model_cfg["name"], model_cfg["path"], all_problems, config)
        if results:
            metrics = print_metrics(model_cfg["name"], results)
            all_metrics[model_cfg["name"]] = metrics

            save_path = os.path.join(args.output_dir, f"eval_{model_cfg['name']}.json")
            with open(save_path, "w") as f:
                json.dump({"model": model_cfg["name"], "metrics": metrics, "n_results": len(results)}, f, indent=2)

    # Comparison table
    if len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        print(f"  {'Model':<25} {'Accuracy':>8} {'ECE':>6} {'Meta':>5}")
        print(f"  {'-'*50}")
        for name, m in all_metrics.items():
            if "overall" in m:
                acc = m["overall"]["accuracy"]
                ece = m["overall"].get("ece")
                ece_str = f"{ece:.3f}" if ece else " N/A"
                print(f"  {name:<25} {acc*100:6.1f}%  {ece_str}")


if __name__ == "__main__":
    main()
