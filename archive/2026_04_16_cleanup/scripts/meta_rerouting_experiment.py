"""Identify meta re-routing candidates and compute oracle upper bound.

Reads existing eval JSONs to find problems where:
1. A meta-CoT model says "route is weak" + confidence < 0.4 (re-route signal)
2. The same problem has different correct/wrong status across models (oracle potential)

This gives the theoretical upper bound of re-routing without new generation.

Usage:
  python scripts/meta_rerouting_experiment.py \
    --results_dir results/eval_1030_v5 \
    --signal_model all_sft \
    --output_dir results/rerouting_analysis

  # Use specific confidence threshold:
  python scripts/meta_rerouting_experiment.py \
    --results_dir results/eval_1030_v5 \
    --signal_model all_sft \
    --confidence_threshold 0.3 \
    --output_dir results/rerouting_analysis
"""
import argparse
import datetime as dt
import json
import os
import re
import socket
import sys

sys.path.insert(0, ".")

from src.training.rewards import _check_correctness, _extract_answer_fallback


# ---------------------------------------------------------------------------
# Eval JSON loading
# ---------------------------------------------------------------------------

def load_eval_json(path: str) -> dict | None:
    """Load a single eval JSON file, returning None on failure."""
    try:
        with open(path) as f:
            data = json.load(f)
        if "results" not in data or not data["results"]:
            print(f"  Warning: {path} has no results, skipping")
            return None
        return data
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Warning: failed to load {path}: {e}")
        return None


def discover_eval_files(results_dir: str) -> dict[str, str]:
    """Find eval JSON files and map model names to file paths.

    Prefers the most-rescored version (most _v2 suffixes) for each model.
    Returns: {model_name: file_path}
    """
    if not os.path.isdir(results_dir):
        print(f"Error: results_dir does not exist: {results_dir}")
        return {}

    json_files = sorted(
        f for f in os.listdir(results_dir)
        if f.startswith("eval_") and f.endswith(".json") and "metadata" not in f
    )

    if not json_files:
        print(f"No eval_*.json files found in {results_dir}")
        return {}

    # Group by base model name, prefer the most-rescored version
    # e.g., eval_1030_all_sft_v2_v2_v2.json > eval_1030_all_sft_v2.json > eval_1030_all_sft.json
    model_files: dict[str, tuple[int, str]] = {}
    for jf in json_files:
        # Extract model name: remove eval_1030_ prefix and .json suffix,
        # then count _v2 suffixes to determine version depth
        name = jf.replace(".json", "")
        v2_count = name.count("_v2")
        # Strip all _v2 suffixes to get the base model name
        base_name = name
        while base_name.endswith("_v2"):
            base_name = base_name[:-3]
        # Remove the eval_1030_ prefix to get a clean model name
        # Pattern: eval_NNNN_MODELNAME
        parts = base_name.split("_", 2)
        if len(parts) >= 3:
            model_name = parts[2]  # e.g., "all_sft", "base_sft", "E8"
        else:
            model_name = base_name

        current = model_files.get(model_name)
        if current is None or v2_count > current[0]:
            model_files[model_name] = (v2_count, jf)

    result = {name: os.path.join(results_dir, info[1]) for name, info in model_files.items()}
    print(f"Discovered {len(result)} model eval files:")
    for name, path in sorted(result.items()):
        print(f"  {name}: {os.path.basename(path)}")
    return result


# ---------------------------------------------------------------------------
# Re-route signal detection
# ---------------------------------------------------------------------------

def has_route_weak_signal(completion: str) -> bool:
    """Detect if the completion contains a 'route is weak' signal.

    Looks for language indicating the model diagnosed its approach as failing.
    """
    return bool(re.search(
        r'\b(route is weak|this approach fails|this route is weak|'
        r'current route fails|switch_method|switch to|'
        r'something feels off|this feels off|that seems off|'
        r'contradiction|inconsistent|mismatch|'
        r'too large|too small|cannot be|can\'t be|'
        r'forcing|overcommitted|I committed too early|'
        r'approach is not working|this method fails|'
        r'wrong approach|failed approach|weak route)\b',
        completion,
        re.IGNORECASE,
    ))


def extract_min_confidence(result: dict) -> float | None:
    """Extract the minimum confidence from a result's meta blocks.

    Tries meta_confidences field first (from eval_hf.py output),
    then falls back to parsing the completion text.
    """
    # Try the pre-parsed field first
    confs = result.get("meta_confidences", [])
    if confs:
        return min(confs)

    # Fallback: parse from completion text
    completion = result.get("completion", "")
    matches = re.findall(
        r'(?:probability|confidence)[:\s\w]*?(\d+\.\d+|\d+)\s*%?',
        completion,
        re.IGNORECASE,
    )
    parsed = []
    for m in matches:
        v = float(m)
        if v > 1:
            v /= 100
        v = max(0.0, min(1.0, v))
        if v > 0.001:
            parsed.append(v)
    return min(parsed) if parsed else None


def get_correctness(result: dict) -> bool:
    """Get correctness, preferring the rescored is_correct_v2 field.

    Falls back to re-checking with _check_correctness for maximum accuracy.
    """
    # Prefer is_correct_v2 (rescored with fixed answer extraction)
    if "is_correct_v2" in result:
        return bool(result["is_correct_v2"])

    # Fall back to re-checking
    completion = result.get("completion", "")
    gold = result.get("full_gold_answer", result.get("gold_answer", ""))
    if completion and gold:
        return _check_correctness(completion, gold)

    return bool(result.get("is_correct", False))


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def identify_reroute_candidates(
    signal_results: list[dict],
    confidence_threshold: float = 0.4,
) -> list[dict]:
    """Find problems where the signal model flagged as hard AND got wrong.

    A re-route candidate meets ALL of:
    1. The model got it wrong (is_correct_v2 = False)
    2. Meta blocks exist (num_meta_blocks > 0)
    3. Route-weak signal detected OR min confidence < threshold
    """
    candidates = []
    for idx, r in enumerate(signal_results):
        is_correct = get_correctness(r)
        if is_correct:
            continue  # Only consider wrong answers

        num_meta = r.get("num_meta_blocks", 0)
        if num_meta == 0:
            continue  # No meta blocks -> no signal

        min_conf = extract_min_confidence(r)
        has_weak = has_route_weak_signal(r.get("completion", ""))

        # Re-route signal: low confidence OR explicit route-weak language
        is_candidate = False
        signal_reasons = []

        if min_conf is not None and min_conf < confidence_threshold:
            is_candidate = True
            signal_reasons.append(f"low_confidence={min_conf:.2f}")

        if has_weak:
            is_candidate = True
            signal_reasons.append("route_weak_language")

        if is_candidate:
            candidates.append({
                "problem_idx": idx,
                "question": r.get("question", "")[:120],
                "full_question": r.get("full_question", r.get("question", "")),
                "benchmark": r.get("benchmark", "unknown"),
                "gold_answer": r.get("gold_answer", ""),
                "full_gold_answer": r.get("full_gold_answer", r.get("gold_answer", "")),
                "signal_model_answer": _extract_answer_fallback(r.get("completion", "")),
                "min_confidence": min_conf,
                "has_route_weak": has_weak,
                "signal_reasons": signal_reasons,
                "num_meta_blocks": num_meta,
            })

    return candidates


def compute_oracle_recovery(
    candidates: list[dict],
    all_model_results: dict[str, list[dict]],
    signal_model_name: str,
) -> list[dict]:
    """For each candidate, check if ANY other model got it correct.

    This gives the oracle upper bound: if we had a perfect router that could
    pick the best model's answer, how many problems could we recover?
    """
    enriched = []
    other_models = [m for m in all_model_results if m != signal_model_name]

    # Build question-based index for each model (CRITICAL: don't use positional index)
    model_q_index = {}
    for model_name in other_models:
        model_q_index[model_name] = {
            r.get("full_question", r.get("question", "")): r
            for r in all_model_results[model_name]
            if r.get("full_question", r.get("question", ""))
        }

    for cand in candidates:
        question = cand.get("full_question", "")
        recovery_models = []

        for model_name in other_models:
            other_r = model_q_index[model_name].get(question)
            if other_r is None:
                continue
            if get_correctness(other_r):
                recovery_models.append(model_name)

        cand_enriched = {
            **cand,
            "oracle_recoverable": len(recovery_models) > 0,
            "recovery_models": recovery_models,
            "n_recovery_models": len(recovery_models),
        }
        enriched.append(cand_enriched)

    return enriched


def compute_analysis(
    signal_model_name: str,
    signal_results: list[dict],
    candidates_enriched: list[dict],
    all_model_results: dict[str, list[dict]],
) -> dict:
    """Compute full analysis with per-benchmark breakdown."""
    total = len(signal_results)
    n_correct_original = sum(1 for r in signal_results if get_correctness(r))
    n_candidates = len(candidates_enriched)
    n_oracle_recoverable = sum(1 for c in candidates_enriched if c["oracle_recoverable"])

    original_acc = n_correct_original / total if total > 0 else 0
    oracle_acc = (n_correct_original + n_oracle_recoverable) / total if total > 0 else 0

    # Per-benchmark breakdown
    benchmarks = sorted(set(r.get("benchmark", "unknown") for r in signal_results))
    per_benchmark = {}
    for bench in benchmarks:
        bench_indices = [i for i, r in enumerate(signal_results) if r.get("benchmark") == bench]
        bench_total = len(bench_indices)
        bench_correct = sum(1 for i in bench_indices if get_correctness(signal_results[i]))
        bench_candidates = [c for c in candidates_enriched if c["benchmark"] == bench]
        bench_recoverable = sum(1 for c in bench_candidates if c["oracle_recoverable"])

        per_benchmark[bench] = {
            "total": bench_total,
            "original_correct": bench_correct,
            "original_accuracy": bench_correct / bench_total if bench_total > 0 else 0,
            "n_candidates": len(bench_candidates),
            "candidate_rate": len(bench_candidates) / bench_total if bench_total > 0 else 0,
            "n_oracle_recoverable": bench_recoverable,
            "oracle_accuracy": (bench_correct + bench_recoverable) / bench_total if bench_total > 0 else 0,
            "oracle_gain_pp": bench_recoverable / bench_total * 100 if bench_total > 0 else 0,
        }

    # Per-other-model contribution
    other_models = sorted(m for m in all_model_results if m != signal_model_name)
    model_contribution = {}
    for model_name in other_models:
        n_recovers = sum(
            1 for c in candidates_enriched
            if model_name in c.get("recovery_models", [])
        )
        model_results = all_model_results[model_name]
        model_acc = (
            sum(1 for r in model_results if get_correctness(r)) / len(model_results)
            if model_results else 0
        )
        model_contribution[model_name] = {
            "n_recovers_from_candidates": n_recovers,
            "overall_accuracy": model_acc,
        }

    return {
        "signal_model": signal_model_name,
        "total_problems": total,
        "original_correct": n_correct_original,
        "original_accuracy": original_acc,
        "n_candidates": n_candidates,
        "candidate_rate": n_candidates / total if total > 0 else 0,
        "n_oracle_recoverable": n_oracle_recoverable,
        "oracle_accuracy": oracle_acc,
        "oracle_gain_pp": (oracle_acc - original_acc) * 100,
        "recovery_rate": n_oracle_recoverable / n_candidates if n_candidates > 0 else 0,
        "per_benchmark": per_benchmark,
        "model_contribution": model_contribution,
    }


def print_analysis(analysis: dict) -> None:
    """Print a clear, structured analysis report."""
    print("\n" + "=" * 70)
    print("  Meta Re-routing Oracle Analysis")
    print("=" * 70)

    print(f"\n  Signal model: {analysis['signal_model']}")
    print(f"  Total problems: {analysis['total_problems']}")
    print(f"  Original accuracy: {analysis['original_accuracy']:.1%} ({analysis['original_correct']}/{analysis['total_problems']})")
    print(f"  Re-route candidates: {analysis['n_candidates']} ({analysis['candidate_rate']:.1%} of total)")
    print(f"  Oracle recoverable: {analysis['n_oracle_recoverable']}/{analysis['n_candidates']} ({analysis['recovery_rate']:.1%} of candidates)")
    print(f"  Oracle accuracy: {analysis['oracle_accuracy']:.1%} (gain: {analysis['oracle_gain_pp']:+.1f}pp)")

    # Per-benchmark table
    per_bench = analysis["per_benchmark"]
    if per_bench:
        print(f"\n  {'Benchmark':<12} {'Orig':>6} {'Cands':>6} {'Recov':>6} {'Oracle':>7} {'Gain':>7}")
        print(f"  {'-' * 50}")
        for bench in sorted(per_bench.keys()):
            b = per_bench[bench]
            print(
                f"  {bench:<12} "
                f"{b['original_accuracy'] * 100:5.1f}% "
                f"{b['n_candidates']:>5} "
                f"{b['n_oracle_recoverable']:>5} "
                f"{b['oracle_accuracy'] * 100:6.1f}% "
                f"{b['oracle_gain_pp']:+6.1f}pp"
            )

    # Model contribution
    contrib = analysis.get("model_contribution", {})
    if contrib:
        print(f"\n  Other model contributions (recoveries from candidates):")
        for model_name in sorted(contrib.keys()):
            mc = contrib[model_name]
            print(
                f"    {model_name:<20} "
                f"recovers={mc['n_recovers_from_candidates']:>3} "
                f"(own acc={mc['overall_accuracy']:.1%})"
            )

    # Decision guide
    gain = analysis["oracle_gain_pp"]
    print(f"\n  Decision (V6 Plan thresholds):")
    if gain >= 5:
        print(f"    Oracle gain = {gain:+.1f}pp >= +5pp --> Meta detection IS a useful routing signal")
        print(f"    Proceed with inference-time re-routing (Phase 3)")
    elif gain >= 2:
        print(f"    Oracle gain = {gain:+.1f}pp (2-5pp) --> Marginal value; may help on specific benchmarks")
    else:
        print(f"    Oracle gain = {gain:+.1f}pp < +2pp --> Meta detection NOT useful for routing; try other signals")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Identify meta re-routing candidates and compute oracle upper bound.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results_dir",
        default="results/eval_1030_v5",
        help="Directory containing eval_*.json files (default: results/eval_1030_v5)",
    )
    parser.add_argument(
        "--signal_model",
        default="all_sft",
        help="Model name whose meta signals drive candidate detection (default: all_sft)",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.4,
        help="Confidence threshold below which a problem is flagged (default: 0.4)",
    )
    parser.add_argument(
        "--output_dir",
        default="results/rerouting_analysis",
        help="Directory for output files (default: results/rerouting_analysis)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Results dir:     {args.results_dir}")
    print(f"Signal model:    {args.signal_model}")
    print(f"Confidence thr:  {args.confidence_threshold}")
    print(f"Output dir:      {args.output_dir}")

    # Discover eval files
    model_files = discover_eval_files(args.results_dir)
    if not model_files:
        print("Error: no eval files found. Exiting.")
        sys.exit(1)

    if args.signal_model not in model_files:
        available = ", ".join(sorted(model_files.keys()))
        print(f"Error: signal_model '{args.signal_model}' not found. Available: {available}")
        sys.exit(1)

    # Load all eval data
    print(f"\nLoading eval data ...")
    all_model_results: dict[str, list[dict]] = {}
    for model_name, file_path in sorted(model_files.items()):
        data = load_eval_json(file_path)
        if data is not None:
            all_model_results[model_name] = data["results"]
            n = len(data["results"])
            acc = sum(1 for r in data["results"] if get_correctness(r)) / n if n > 0 else 0
            print(f"  {model_name}: {n} results, accuracy={acc:.1%}")

    signal_results = all_model_results[args.signal_model]
    print(f"\nSignal model '{args.signal_model}': {len(signal_results)} results")

    # Identify re-route candidates
    print(f"\nIdentifying re-route candidates (conf < {args.confidence_threshold}, route-weak signals) ...")
    candidates = identify_reroute_candidates(
        signal_results,
        confidence_threshold=args.confidence_threshold,
    )
    print(f"  Found {len(candidates)} re-route candidates")

    if not candidates:
        print("No re-route candidates found. This could mean:")
        print("  - The signal model has no meta blocks with low confidence")
        print("  - The signal model never uses route-weak language")
        print("  - All wrong answers have high confidence (overconfidence)")
        print("Consider lowering --confidence_threshold or checking a different --signal_model.")

        # Still save empty results for recordkeeping
        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(args.output_dir, f"rerouting_{args.signal_model}_{timestamp}.json")
        payload = {
            "run_metadata": {
                "results_dir": args.results_dir,
                "signal_model": args.signal_model,
                "confidence_threshold": args.confidence_threshold,
                "hostname": socket.gethostname(),
                "utc_timestamp": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            },
            "analysis": {
                "signal_model": args.signal_model,
                "total_problems": len(signal_results),
                "n_candidates": 0,
                "note": "No re-route candidates found",
            },
            "candidates": [],
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved (empty) results to {out_path}")
        return

    # Compute oracle recovery
    print(f"Computing oracle recovery across {len(all_model_results) - 1} other models ...")
    candidates_enriched = compute_oracle_recovery(
        candidates,
        all_model_results,
        args.signal_model,
    )

    # Full analysis
    analysis = compute_analysis(
        args.signal_model,
        signal_results,
        candidates_enriched,
        all_model_results,
    )

    # Print report
    print_analysis(analysis)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(
        args.output_dir,
        f"rerouting_{args.signal_model}_{timestamp}.json",
    )

    payload = {
        "run_metadata": {
            "results_dir": args.results_dir,
            "signal_model": args.signal_model,
            "confidence_threshold": args.confidence_threshold,
            "n_models_loaded": len(all_model_results),
            "models_loaded": sorted(all_model_results.keys()),
            "hostname": socket.gethostname(),
            "utc_timestamp": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        },
        "analysis": analysis,
        "candidates": [
            {
                "problem_idx": c["problem_idx"],
                "question": c["question"],
                "benchmark": c["benchmark"],
                "gold_answer": c["gold_answer"],
                "signal_model_answer": c["signal_model_answer"],
                "min_confidence": c["min_confidence"],
                "has_route_weak": c["has_route_weak"],
                "signal_reasons": c["signal_reasons"],
                "num_meta_blocks": c["num_meta_blocks"],
                "oracle_recoverable": c["oracle_recoverable"],
                "recovery_models": c["recovery_models"],
                "n_recovery_models": c["n_recovery_models"],
            }
            for c in candidates_enriched
        ],
    }

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved results to {out_path}")

    # Also save a brief per-candidate CSV-like summary for quick review
    summary_path = out_path.replace(".json", "_candidates.txt")
    with open(summary_path, "w") as f:
        f.write(f"# Re-route candidates for signal_model={args.signal_model}\n")
        f.write(f"# confidence_threshold={args.confidence_threshold}\n")
        f.write(f"# {len(candidates_enriched)} candidates, "
                f"{sum(1 for c in candidates_enriched if c['oracle_recoverable'])} oracle-recoverable\n\n")
        f.write(f"{'idx':>4} {'bench':<10} {'conf':>5} {'weak':>5} {'recov':>5} {'by_models'}\n")
        f.write("-" * 70 + "\n")
        for c in candidates_enriched:
            conf_str = f"{c['min_confidence']:.2f}" if c["min_confidence"] is not None else "  N/A"
            f.write(
                f"{c['problem_idx']:>4} "
                f"{c['benchmark']:<10} "
                f"{conf_str:>5} "
                f"{'Y' if c['has_route_weak'] else 'N':>5} "
                f"{'Y' if c['oracle_recoverable'] else 'N':>5} "
                f"{','.join(c['recovery_models']) if c['recovery_models'] else '-'}\n"
            )
    print(f"Saved candidate summary to {summary_path}")


if __name__ == "__main__":
    main()
