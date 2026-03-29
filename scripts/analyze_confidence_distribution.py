"""Analyze confidence distributions across evaluated models.

Loads all eval_*.json from results/ and produces:
  - Confidence histogram (5 buckets)
  - ECE per bucket
  - Accuracy per confidence bucket
  - Comparison table across models

Usage:
  python scripts/analyze_confidence_distribution.py
  python scripts/analyze_confidence_distribution.py --results_dir results
  python scripts/analyze_confidence_distribution.py --output results/confidence_report.txt
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


# Confidence bucket boundaries
BUCKET_EDGES = [0.0, 0.3, 0.5, 0.7, 0.9, 1.01]
BUCKET_LABELS = ["<0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", ">0.9"]


def load_eval_files(results_dir: str) -> Dict[str, List[dict]]:
    """Load all eval_*.json files from the results directory.

    Returns:
        Dict mapping model name to list of result dicts.
    """
    pattern = os.path.join(results_dir, "eval_*.json")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"No eval_*.json files found in {results_dir}")
        sys.exit(1)

    models = {}
    for fpath in files:
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            model_name = data.get("model", os.path.basename(fpath))
            results = data.get("results", [])
            if results:
                models[model_name] = results
                print(f"  Loaded {model_name}: {len(results)} samples")
            else:
                print(f"  Skipped {model_name}: no results")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"  Error loading {fpath}: {e}")

    return models


def bucket_index(confidence: float) -> Optional[int]:
    """Return the bucket index for a given confidence value."""
    for i in range(len(BUCKET_EDGES) - 1):
        if BUCKET_EDGES[i] <= confidence < BUCKET_EDGES[i + 1]:
            return i
    return None


def compute_bucket_stats(results: List[dict]) -> dict:
    """Compute per-bucket statistics for a single model.

    Returns:
        Dict with keys: bucket_counts, bucket_acc, bucket_ece, total_with_conf,
                        total_without_conf, overall_ece.
    """
    bucket_correct = defaultdict(int)
    bucket_total = defaultdict(int)
    bucket_conf_sum = defaultdict(float)
    total_with_conf = 0
    total_without_conf = 0

    for r in results:
        conf = r.get("avg_confidence")
        if conf is None:
            total_without_conf += 1
            continue

        total_with_conf += 1
        idx = bucket_index(conf)
        if idx is None:
            continue

        bucket_total[idx] += 1
        bucket_conf_sum[idx] += conf
        if r.get("is_correct", False):
            bucket_correct[idx] += 1

    # Per-bucket accuracy and ECE
    bucket_acc = {}
    bucket_ece = {}
    bucket_counts = {}
    weighted_ece_sum = 0.0
    weighted_ece_n = 0

    for i in range(len(BUCKET_LABELS)):
        n = bucket_total[i]
        bucket_counts[i] = n
        if n > 0:
            acc = bucket_correct[i] / n
            avg_conf = bucket_conf_sum[i] / n
            ece = abs(avg_conf - acc)
            bucket_acc[i] = acc
            bucket_ece[i] = ece
            weighted_ece_sum += ece * n
            weighted_ece_n += n
        else:
            bucket_acc[i] = None
            bucket_ece[i] = None

    overall_ece = weighted_ece_sum / weighted_ece_n if weighted_ece_n > 0 else None

    return {
        "bucket_counts": bucket_counts,
        "bucket_acc": bucket_acc,
        "bucket_ece": bucket_ece,
        "total_with_conf": total_with_conf,
        "total_without_conf": total_without_conf,
        "overall_ece": overall_ece,
    }


def format_pct(val: Optional[float], width: int = 6) -> str:
    """Format a float as percentage string, or N/A."""
    if val is None:
        return "N/A".center(width)
    return f"{val * 100:5.1f}%"


def print_model_table(model_name: str, stats: dict, file=None) -> None:
    """Print a detailed table for one model."""
    def out(s):
        print(s, file=file)

    out(f"\n{'=' * 70}")
    out(f"  {model_name}")
    out(f"  Samples with confidence: {stats['total_with_conf']}, "
        f"without: {stats['total_without_conf']}")
    if stats["overall_ece"] is not None:
        out(f"  Overall ECE: {stats['overall_ece']:.4f}")
    out(f"{'=' * 70}")
    out(f"  {'Bucket':<10} {'Count':>6} {'Pct':>7} {'Accuracy':>9} {'ECE':>7}")
    out(f"  {'-' * 42}")

    total = stats["total_with_conf"] or 1
    for i, label in enumerate(BUCKET_LABELS):
        n = stats["bucket_counts"][i]
        pct = n / total * 100 if total > 0 else 0
        acc_str = format_pct(stats["bucket_acc"][i])
        ece_str = f"{stats['bucket_ece'][i]:.4f}" if stats["bucket_ece"][i] is not None else "  N/A"
        out(f"  {label:<10} {n:>6} {pct:>6.1f}% {acc_str:>9} {ece_str:>7}")


def print_comparison_table(
    all_stats: Dict[str, dict],
    file=None,
) -> None:
    """Print a side-by-side comparison table across all models."""
    def out(s):
        print(s, file=file)

    models = list(all_stats.keys())
    if not models:
        return

    # --- Confidence distribution comparison ---
    out(f"\n{'=' * 80}")
    out("  CONFIDENCE DISTRIBUTION COMPARISON (% of samples in each bucket)")
    out(f"{'=' * 80}")

    # Header
    header = f"  {'Model':<25}"
    for label in BUCKET_LABELS:
        header += f" {label:>7}"
    header += f" {'ECE':>7} {'ConfRate':>8}"
    out(header)
    out(f"  {'-' * (25 + 8 * len(BUCKET_LABELS) + 18)}")

    for model in models:
        s = all_stats[model]
        total = s["total_with_conf"] or 1
        total_all = s["total_with_conf"] + s["total_without_conf"]
        conf_rate = s["total_with_conf"] / total_all * 100 if total_all > 0 else 0

        # Truncate model name if too long
        display_name = model[:24] if len(model) > 24 else model
        line = f"  {display_name:<25}"
        for i in range(len(BUCKET_LABELS)):
            n = s["bucket_counts"][i]
            pct = n / total * 100 if total > 0 else 0
            line += f" {pct:>6.1f}%"

        ece_str = f"{s['overall_ece']:.4f}" if s["overall_ece"] is not None else "  N/A"
        line += f" {ece_str:>7} {conf_rate:>6.1f}%"
        out(line)

    # --- Accuracy per bucket comparison ---
    out(f"\n{'=' * 80}")
    out("  ACCURACY PER CONFIDENCE BUCKET")
    out(f"{'=' * 80}")

    header = f"  {'Model':<25}"
    for label in BUCKET_LABELS:
        header += f" {label:>7}"
    header += f" {'Overall':>8}"
    out(header)
    out(f"  {'-' * (25 + 8 * len(BUCKET_LABELS) + 10)}")

    for model in models:
        s = all_stats[model]
        display_name = model[:24] if len(model) > 24 else model
        line = f"  {display_name:<25}"
        for i in range(len(BUCKET_LABELS)):
            acc = s["bucket_acc"][i]
            if acc is not None:
                line += f" {acc * 100:>6.1f}%"
            else:
                line += f" {'N/A':>7}"

        # Overall accuracy (across all samples, including those without confidence)
        total_correct = sum(1 for r in _model_results[model] if r.get("is_correct", False))
        total_n = len(_model_results[model])
        overall_acc = total_correct / total_n * 100 if total_n > 0 else 0
        line += f" {overall_acc:>6.1f}%"
        out(line)

    # --- ECE per bucket comparison ---
    out(f"\n{'=' * 80}")
    out("  ECE PER CONFIDENCE BUCKET")
    out(f"{'=' * 80}")

    header = f"  {'Model':<25}"
    for label in BUCKET_LABELS:
        header += f" {label:>7}"
    header += f" {'Overall':>8}"
    out(header)
    out(f"  {'-' * (25 + 8 * len(BUCKET_LABELS) + 10)}")

    for model in models:
        s = all_stats[model]
        display_name = model[:24] if len(model) > 24 else model
        line = f"  {display_name:<25}"
        for i in range(len(BUCKET_LABELS)):
            ece = s["bucket_ece"][i]
            if ece is not None:
                line += f" {ece:>6.4f}"
            else:
                line += f" {'N/A':>7}"
        ece_str = f" {s['overall_ece']:.4f}" if s["overall_ece"] is not None else f" {'N/A':>7}"
        line += f" {ece_str:>7}"
        out(line)


# Module-level reference for comparison table to access raw results
_model_results: Dict[str, List[dict]] = {}


def main():
    parser = argparse.ArgumentParser(description="Analyze confidence distributions across models")
    parser.add_argument("--results_dir", default="results",
                        help="Directory containing eval_*.json files")
    parser.add_argument("--output", default=None,
                        help="Save report to file (in addition to stdout)")
    args = parser.parse_args()

    global _model_results

    print(f"Loading eval results from: {args.results_dir}")
    _model_results = load_eval_files(args.results_dir)

    if not _model_results:
        print("No valid eval files found. Run eval first.")
        sys.exit(1)

    # Compute stats for each model
    all_stats = {}
    for model_name, results in _model_results.items():
        all_stats[model_name] = compute_bucket_stats(results)

    # Print per-model tables
    for model_name, stats in all_stats.items():
        print_model_table(model_name, stats)

    # Print comparison tables
    print_comparison_table(all_stats)

    # Save to file if requested
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            for model_name, stats in all_stats.items():
                print_model_table(model_name, stats, file=f)
            print_comparison_table(all_stats, file=f)
        print(f"\nReport saved to {args.output}")


if __name__ == "__main__":
    main()
