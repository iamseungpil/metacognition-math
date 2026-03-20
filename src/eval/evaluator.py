"""Unified evaluation for AIME and MATH benchmarks."""
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
import wandb

from src.data.dataset_loader import (
    load_aime, load_math_test, load_omni_math,
    extract_boxed_answer, extract_numeric_answer,
)
from src.rollout.vllm_rollout import build_chat_messages, check_correctness


def evaluate_model(config_path: str):
    """Run full evaluation suite on a model."""
    from vllm import LLM, SamplingParams

    with open(config_path) as f:
        config = yaml.safe_load(f)

    wandb_project = config.get("wandb_project", "metacot-math")

    results_all = {}

    for model_cfg in config["models"]:
        model_name = model_cfg["name"]
        model_path = model_cfg["path"]
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} ({model_path})")
        print(f"{'='*60}")

        llm = LLM(
            model=model_path,
            tensor_parallel_size=config.get("tensor_parallel_size", 4),
            gpu_memory_utilization=config.get("gpu_memory_utilization", 0.90),
            max_model_len=config.get("max_model_len", 4096),
            dtype="bfloat16",
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()

        num_samples = config.get("num_samples", 8)
        sampling_params = SamplingParams(
            temperature=config.get("temperature", 0.7),
            top_p=config.get("top_p", 0.95),
            max_tokens=config.get("max_tokens", 2048),
            n=num_samples,
        )

        model_results = {}

        for bench_cfg in config["benchmarks"]:
            bench_name = bench_cfg["name"]
            print(f"\n--- {bench_name} ---")

            dataset = _load_benchmark(bench_name, bench_cfg)
            if dataset is None:
                print(f"  Skipping {bench_name}: failed to load")
                continue

            # Build prompts
            prompts = []
            for row in dataset:
                messages = build_chat_messages(row["question"])
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompts.append(prompt)

            # Generate
            outputs = llm.generate(prompts, sampling_params=sampling_params)

            # Score
            bench_results = _score_benchmark(
                bench_name, dataset, outputs, num_samples
            )
            model_results[bench_name] = bench_results

            print(f"  pass@1: {bench_results['pass_at_1']:.4f}")
            if "pass_at_k" in bench_results:
                print(f"  pass@{num_samples}: {bench_results['pass_at_k']:.4f}")

        results_all[model_name] = model_results

        # Cleanup vLLM
        del llm
        torch.cuda.empty_cache()

    # Save and log results
    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results_all, f, indent=2, default=str)

    # Log to wandb
    wandb.init(project=wandb_project, name="evaluation", config=config, reinit=True)
    _log_comparison_table(results_all)
    wandb.finish()

    _print_comparison_table(results_all, config["benchmarks"])
    return results_all


def _load_benchmark(name: str, cfg: dict):
    """Load a benchmark dataset."""
    try:
        if "aime" in name.lower():
            year = "2025" if "2025" in name else "2024"
            return load_aime(year)
        elif "math" in name.lower() and "test" in name.lower():
            return load_math_test()
        elif "olym" in name.lower():
            return load_omni_math()
        else:
            from datasets import load_dataset
            ds = load_dataset(cfg["dataset"], split=cfg.get("split", "test"), trust_remote_code=True)
            return ds
    except Exception as e:
        print(f"  Error loading {name}: {e}")
        return None


def _score_benchmark(
    bench_name: str,
    dataset,
    outputs,
    num_samples: int,
) -> dict:
    """Score model outputs against gold answers."""
    correct_at_1 = []
    correct_at_k = []
    per_problem = []

    for i, output in enumerate(outputs):
        gold = dataset[i]["answer"]
        is_aime = "aime" in bench_name.lower()

        sample_correct = []
        for j, completion in enumerate(output.outputs):
            text = completion.text
            if is_aime:
                model_ans = extract_numeric_answer(text)
                try:
                    gold_int = int(gold)
                    correct = model_ans == gold_int
                except ValueError:
                    correct = check_correctness(text, gold)
            else:
                correct = check_correctness(text, gold)
            sample_correct.append(correct)

        correct_at_1.append(np.mean(sample_correct))  # unbiased pass@1
        correct_at_k.append(any(sample_correct))

        per_problem.append({
            "problem_idx": i,
            "question": dataset[i]["question"][:100],
            "category": dataset[i].get("category", ""),
            "correct_at_1": sample_correct[0],
            "correct_at_k": any(sample_correct),
            "num_correct": sum(sample_correct),
            "pass_rate": sum(sample_correct) / len(sample_correct),
        })

    results = {
        "pass_at_1": np.mean(correct_at_1),
        "pass_at_k": np.mean(correct_at_k),
        "num_problems": len(outputs),
        "num_correct_at_1": sum(correct_at_1),
    }

    # Category breakdown
    categories = {}
    for p in per_problem:
        cat = p["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if p["correct_at_1"]:
            categories[cat]["correct"] += 1

    results["category_accuracy"] = {
        cat: v["correct"] / max(v["total"], 1)
        for cat, v in categories.items()
    }
    results["per_problem"] = per_problem

    return results


def _log_comparison_table(results_all: dict):
    """Log comparison table to wandb."""
    table_data = []
    for model_name, benchmarks in results_all.items():
        row = {"model": model_name}
        for bench_name, bench_results in benchmarks.items():
            row[f"{bench_name}/pass@1"] = bench_results["pass_at_1"]
            if "pass_at_k" in bench_results:
                row[f"{bench_name}/pass@k"] = bench_results["pass_at_k"]
        table_data.append(row)

    wandb.log({"eval/comparison": wandb.Table(dataframe=pd.DataFrame(table_data))})

    # Also log individual metrics
    for model_name, benchmarks in results_all.items():
        for bench_name, bench_results in benchmarks.items():
            wandb.log({
                f"eval/{model_name}/{bench_name}/pass_at_1": bench_results["pass_at_1"],
            })


def _print_comparison_table(results_all: dict, benchmarks: list):
    """Print comparison table to console."""
    bench_names = [b["name"] for b in benchmarks]
    header = f"{'Model':<30}" + "".join(f"{b:<20}" for b in bench_names)
    print(f"\n{'='*len(header)}")
    print(header)
    print(f"{'='*len(header)}")

    for model_name, model_results in results_all.items():
        row = f"{model_name:<30}"
        for bench in bench_names:
            if bench in model_results:
                p1 = model_results[bench]["pass_at_1"]
                pk = model_results[bench].get("pass_at_k", 0)
                row += f"{p1:.3f}/{pk:.3f}{'':>8}"
            else:
                row += f"{'N/A':<20}"
        print(row)
    print(f"{'='*len(header)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    evaluate_model(args.config)
