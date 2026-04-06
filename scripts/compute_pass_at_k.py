"""Compute pass@k to determine upper bound of route switching value.

If pass@8 >> pass@1, the model has alternative solution paths that meta
could learn to switch to. If pass@8 ~ pass@1, switching is futile.

Usage:
  python scripts/compute_pass_at_k.py \
    --model_path checkpoints/qwen3_base_sft \
    --benchmark math500 \
    --k_values 1,4,8,16 \
    --output_dir results/pass_at_k

  # Quick test (5 problems, k=4):
  python scripts/compute_pass_at_k.py \
    --model_path checkpoints/qwen3_base_sft \
    --benchmark math500 \
    --k_values 1,4 \
    --max_problems 5 \
    --output_dir results/pass_at_k_test
"""
import argparse
import datetime as dt
import json
import math
import os
import socket
import sys
import time

sys.path.insert(0, ".")

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.training.rewards import _check_correctness
from src.curriculum.control_rag import build_model_inputs


# ---------------------------------------------------------------------------
# Pass@k formula: pass@k = 1 - C(n-c, k) / C(n, k)
# Uses the numerically stable log-space computation from the Chen et al.
# (Codex) paper to avoid overflow in large combinatorics.
# ---------------------------------------------------------------------------

def _log_comb(n: int, k: int) -> float:
    """Compute log(C(n, k)) using lgamma for numerical stability."""
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def pass_at_k(n: int, c: int, k: int) -> float:
    """Compute pass@k = 1 - C(n-c, k) / C(n, k).

    Args:
        n: Total number of samples generated for a problem.
        c: Number of correct samples among n.
        k: The k in pass@k.

    Returns:
        Probability (0.0 to 1.0) that at least one of k randomly chosen
        samples is correct.
    """
    if n < k:
        # Not enough samples; fall back to simple estimate.
        return 1.0 if c > 0 else 0.0
    if c == 0:
        return 0.0
    if c >= n:
        return 1.0
    # 1 - C(n-c, k) / C(n, k)  in log space
    log_ratio = _log_comb(n - c, k) - _log_comb(n, k)
    return 1.0 - math.exp(log_ratio)


# ---------------------------------------------------------------------------
# Benchmark loader (mirrors eval_hf.py load_benchmarks)
# ---------------------------------------------------------------------------

BENCHMARKS = {
    "gsm8k": ("openai/gsm8k", "main", "test", "question", "answer"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", "problem", "answer"),
    "aime2024": ("HuggingFaceH4/aime_2024", None, "train", "problem", "answer"),
}


def load_problems(benchmark_name: str, max_problems: int) -> list[dict]:
    """Load problems from a single benchmark."""
    if benchmark_name not in BENCHMARKS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {list(BENCHMARKS.keys())}"
        )

    ds_id, config, split, q_col, a_col = BENCHMARKS[benchmark_name]
    print(f"Loading {benchmark_name} from {ds_id} ...")

    if config:
        ds = load_dataset(ds_id, config, split=split)
    else:
        ds = load_dataset(ds_id, split=split)

    problems = []
    for row in ds:
        if len(problems) >= max_problems:
            break
        q = str(row.get(q_col, ""))
        a = str(row.get(a_col, ""))
        if not q:
            continue
        # For GSM8K, extract just the final answer after ####
        if benchmark_name == "gsm8k" and "####" in a:
            a = a.split("####")[-1].strip()
        problems.append({
            "problem_id": len(problems),
            "question": q,
            "gold_answer": a,
            "benchmark": benchmark_name,
        })

    print(f"  Loaded {len(problems)} problems")
    return problems


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_samples(
    model,
    tokenizer,
    problems: list[dict],
    n_samples: int,
    max_new_tokens: int = 4096,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> list[dict]:
    """Generate n_samples completions per problem and check correctness.

    Returns a list of per-problem dicts with fields:
        problem_id, question (truncated), gold_answer (truncated),
        benchmark, n_correct, n_total, samples (list of {completion, is_correct})
    """
    results = []
    total_problems = len(problems)
    t0 = time.time()

    for idx, prob in enumerate(problems):
        messages = [{"role": "user", "content": prob["question"]}]
        _, inputs = build_model_inputs(
            tokenizer,
            messages,
            device=model.device,
            add_generation_prompt=True,
            max_prompt_tokens=2048,
        )
        prompt_len = int(inputs["input_ids"].shape[1])

        samples = []
        n_correct = 0

        for s_idx in range(n_samples):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    pad_token_id=tokenizer.pad_token_id,
                )

            completion_ids = output[0][prompt_len:]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=False)
            is_correct = _check_correctness(completion, prob["gold_answer"])
            if is_correct:
                n_correct += 1

            samples.append({
                "sample_idx": s_idx,
                "is_correct": is_correct,
                "completion": completion,
            })

        results.append({
            "problem_id": prob["problem_id"],
            "question": prob["question"][:120],
            "full_question": prob["question"],
            "gold_answer": prob["gold_answer"][:80],
            "full_gold_answer": prob["gold_answer"],
            "benchmark": prob["benchmark"],
            "n_correct": n_correct,
            "n_total": n_samples,
            "samples": samples,
        })

        # Progress reporting
        if (idx + 1) % 5 == 0 or idx == 0 or idx + 1 == total_problems:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed if elapsed > 0 else 0
            eta_s = (total_problems - idx - 1) / rate if rate > 0 else 0
            eta_str = f"{eta_s / 3600:.1f}h" if eta_s > 3600 else f"{eta_s / 60:.1f}m"
            # Quick pass@1 running estimate
            running_p1 = sum(r["n_correct"] > 0 for r in results) / len(results)
            print(
                f"  [{idx+1}/{total_problems}] "
                f"this={n_correct}/{n_samples} correct | "
                f"running pass@1={running_p1:.1%} | "
                f"elapsed={elapsed:.0f}s eta={eta_str}"
            )

    return results


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def compute_pass_at_k_table(
    results: list[dict],
    k_values: list[int],
) -> dict:
    """Compute pass@k for each problem and aggregate.

    Returns:
        {
            "per_problem": [{problem_id, ..., "pass@1": x, "pass@4": y, ...}],
            "summary": {"pass@1": agg, "pass@4": agg, ...},
            "gap_analysis": {...},
        }
    """
    per_problem = []
    # Accumulators for aggregate pass@k
    k_sums = {k: 0.0 for k in k_values}

    for r in results:
        n = r["n_total"]
        c = r["n_correct"]

        row = {
            "problem_id": r["problem_id"],
            "question": r["question"],
            "benchmark": r["benchmark"],
            "n_correct": c,
            "n_total": n,
        }
        for k in k_values:
            pk = pass_at_k(n, c, k)
            row[f"pass@{k}"] = pk
            k_sums[k] += pk

        per_problem.append(row)

    total = len(results)
    summary = {}
    for k in k_values:
        summary[f"pass@{k}"] = k_sums[k] / total if total > 0 else 0.0

    # Gap analysis
    k_sorted = sorted(k_values)
    gap_analysis = {}
    if len(k_sorted) >= 2:
        p1_key = f"pass@{k_sorted[0]}"
        for k in k_sorted[1:]:
            pk_key = f"pass@{k}"
            gap = summary[pk_key] - summary[p1_key]
            gap_analysis[f"gap_{p1_key}_vs_{pk_key}"] = gap

    # Per-benchmark breakdown
    benchmarks = sorted(set(r["benchmark"] for r in results))
    per_benchmark = {}
    for bench in benchmarks:
        bench_results = [r for r in per_problem if r["benchmark"] == bench]
        bench_summary = {}
        for k in k_values:
            col = f"pass@{k}"
            bench_summary[col] = (
                sum(r[col] for r in bench_results) / len(bench_results)
                if bench_results
                else 0.0
            )
        per_benchmark[bench] = bench_summary

    return {
        "per_problem": per_problem,
        "summary": summary,
        "gap_analysis": gap_analysis,
        "per_benchmark": per_benchmark,
    }


def print_summary(table: dict, k_values: list[int]) -> None:
    """Print a readable summary of pass@k results."""
    summary = table["summary"]
    per_benchmark = table["per_benchmark"]
    gap_analysis = table["gap_analysis"]

    print("\n" + "=" * 60)
    print("  Pass@k Results")
    print("=" * 60)

    # Header
    k_headers = "  ".join(f"pass@{k:>2}" for k in sorted(k_values))
    print(f"  {'Benchmark':<12}  {k_headers}")
    print(f"  {'-' * (14 + 9 * len(k_values))}")

    # Per-benchmark rows
    for bench in sorted(per_benchmark.keys()):
        bench_data = per_benchmark[bench]
        vals = "  ".join(
            f"{bench_data[f'pass@{k}'] * 100:6.1f}%"
            for k in sorted(k_values)
        )
        print(f"  {bench:<12}  {vals}")

    # Overall
    print(f"  {'-' * (14 + 9 * len(k_values))}")
    overall_vals = "  ".join(
        f"{summary[f'pass@{k}'] * 100:6.1f}%"
        for k in sorted(k_values)
    )
    print(f"  {'OVERALL':<12}  {overall_vals}")

    # Gap analysis
    if gap_analysis:
        print(f"\n  Gap Analysis:")
        for gap_name, gap_val in gap_analysis.items():
            print(f"    {gap_name}: {gap_val * 100:+.1f}pp")

    # Decision guide (per V6 plan thresholds)
    k_sorted = sorted(k_values)
    if len(k_sorted) >= 2:
        p1 = summary.get(f"pass@{k_sorted[0]}", 0)
        # Find a k >= 8 if available, else use the largest k
        target_k = None
        for k in k_sorted:
            if k >= 8:
                target_k = k
                break
        if target_k is None:
            target_k = k_sorted[-1]
        pk = summary.get(f"pass@{target_k}", 0)
        gap_pp = (pk - p1) * 100

        print(f"\n  Decision (V6 Plan thresholds on pass@{k_sorted[0]} vs pass@{target_k}):")
        if gap_pp >= 10:
            print(f"    Gap = {gap_pp:.1f}pp >= 10pp --> PROCEED to Phase 1 (route switching has value)")
        elif gap_pp >= 5:
            print(f"    Gap = {gap_pp:.1f}pp (5-10pp) --> Partial value; consider meta-lite overhead reduction first")
        else:
            print(f"    Gap = {gap_pp:.1f}pp < 5pp --> Route switching unlikely to help; use inference-time methods")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pass@k for a model on a math benchmark.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to the HF model checkpoint (e.g., checkpoints/qwen3_base_sft)",
    )
    parser.add_argument(
        "--benchmark",
        default="math500",
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to evaluate (default: math500)",
    )
    parser.add_argument(
        "--k_values",
        default="1,4,8,16",
        help="Comma-separated k values for pass@k (default: 1,4,8,16)",
    )
    parser.add_argument(
        "--max_problems",
        type=int,
        default=500,
        help="Max problems to evaluate (default: 500, use smaller for testing)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="Max new tokens per generation (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling (default: 0.95)",
    )
    parser.add_argument(
        "--output_dir",
        default="results/pass_at_k",
        help="Directory for output files (default: results/pass_at_k)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    k_values = sorted(set(int(k) for k in args.k_values.split(",")))
    n_samples = max(k_values)  # Generate max(k) samples per problem

    print(f"Model:       {args.model_path}")
    print(f"Benchmark:   {args.benchmark}")
    print(f"k values:    {k_values}")
    print(f"n_samples:   {n_samples} (= max k)")
    print(f"max_problems:{args.max_problems}")
    print(f"temperature: {args.temperature}")
    print(f"top_p:       {args.top_p}")
    print(f"output_dir:  {args.output_dir}")

    # Load model
    use_cuda = torch.cuda.is_available()
    load_dtype = torch.bfloat16 if use_cuda else torch.float32
    device = "cuda" if use_cuda else "cpu"

    print(f"\nLoading model from {args.model_path} (dtype={load_dtype}, device={device}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=load_dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device).eval()
    print("  Model loaded.")

    # Load problems
    problems = load_problems(args.benchmark, args.max_problems)

    # Generate samples
    print(f"\nGenerating {n_samples} samples per problem ({len(problems)} problems) ...")
    t_start = time.time()
    results = generate_samples(
        model,
        tokenizer,
        problems,
        n_samples=n_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    t_elapsed = time.time() - t_start
    print(f"Generation complete in {t_elapsed:.0f}s ({t_elapsed / 3600:.1f}h)")

    # Compute pass@k
    table = compute_pass_at_k_table(results, k_values)

    # Print summary
    print_summary(table, k_values)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    model_short = args.model_path.rstrip("/").split("/")[-1]

    # Save full results (with completions for downstream trajectory mining)
    full_path = os.path.join(
        args.output_dir,
        f"pass_at_k_{args.benchmark}_{model_short}_{timestamp}.json",
    )
    payload = {
        "run_metadata": {
            "model_path": args.model_path,
            "benchmark": args.benchmark,
            "k_values": k_values,
            "n_samples": n_samples,
            "max_problems": args.max_problems,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "hostname": socket.gethostname(),
            "utc_timestamp": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "elapsed_seconds": round(t_elapsed, 1),
        },
        "summary": table["summary"],
        "gap_analysis": table["gap_analysis"],
        "per_benchmark": table["per_benchmark"],
        "per_problem": table["per_problem"],
        "raw_results": [
            {
                "problem_id": r["problem_id"],
                "question": r["full_question"],
                "gold_answer": r["full_gold_answer"],
                "benchmark": r["benchmark"],
                "n_correct": r["n_correct"],
                "n_total": r["n_total"],
                "samples": r["samples"],
            }
            for r in results
        ],
    }
    with open(full_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved full results to {full_path}")

    # Save summary-only (lighter, for quick review)
    summary_path = os.path.join(
        args.output_dir,
        f"pass_at_k_{args.benchmark}_{model_short}_{timestamp}_summary.json",
    )
    summary_payload = {
        "run_metadata": payload["run_metadata"],
        "summary": table["summary"],
        "gap_analysis": table["gap_analysis"],
        "per_benchmark": table["per_benchmark"],
        "per_problem": table["per_problem"],
    }
    with open(summary_path, "w") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
