"""Comprehensive analysis for large-scale 1030-problem evaluation.

Produces:
  - Accuracy table (4 models x 3 benchmarks)
  - ECE per model per benchmark
  - Selective abstention at confidence thresholds
  - Bootstrap 95% CI for accuracy
  - Confidence distribution histogram data
  - Saves everything to results/analysis_1030.json

Usage:
  python scripts/analyze_1030.py
  python scripts/analyze_1030.py --results_dir /scratch/metacognition/results
"""
import argparse
import json
import os
import glob
import random
from typing import Dict, List, Tuple, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CONF_THRESHOLDS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8]
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 42
CONF_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
EXPECTED_PREFIX = "eval_1030_"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_results(results_dir: str) -> Dict[str, List[dict]]:
    """Load all eval_1030_*.json result files.

    Returns:
        dict mapping model_name -> list of result dicts.
    """
    models: Dict[str, List[dict]] = {}
    pattern = os.path.join(results_dir, EXPECTED_PREFIX + "*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        # Fallback: try the naming convention from --model_name (eval_1030_*.json)
        pattern = os.path.join(results_dir, "eval_1030*.json")
        files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No eval_1030_*.json files found in {results_dir}. "
            f"Run scripts/run_eval_1030.sh first."
        )

    for fpath in files:
        with open(fpath) as fh:
            data = json.load(fh)
        name = data.get("model", os.path.basename(fpath).replace(".json", ""))
        models[name] = data["results"]
        print(f"  Loaded {name}: {len(data['results'])} results from {os.path.basename(fpath)}")

    return models


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------
def compute_accuracy(results: List[dict], benchmark: Optional[str] = None) -> Tuple[int, int, float]:
    """Compute accuracy, optionally filtered by benchmark.

    Returns:
        (n_correct, n_total, accuracy)
    """
    subset = results if benchmark is None else [r for r in results if r["benchmark"] == benchmark]
    if not subset:
        return 0, 0, 0.0
    n_correct = sum(1 for r in subset if r["is_correct"])
    return n_correct, len(subset), n_correct / len(subset)


# ---------------------------------------------------------------------------
# ECE (Expected Calibration Error)
# ---------------------------------------------------------------------------
def compute_ece(results: List[dict], n_bins: int = 10) -> Optional[float]:
    """Compute Expected Calibration Error with equal-width bins.

    Returns None if no confidence data is available.
    """
    pairs = [
        (r["avg_confidence"], 1.0 if r["is_correct"] else 0.0)
        for r in results
        if r.get("avg_confidence") is not None
    ]
    if not pairs:
        return None

    total = len(pairs)
    ece = 0.0
    for i in range(n_bins):
        lo = i / n_bins
        hi = (i + 1) / n_bins
        bucket = [(c, a) for c, a in pairs if lo <= c < hi]
        if not bucket:
            continue
        avg_conf = sum(c for c, _ in bucket) / len(bucket)
        avg_acc = sum(a for _, a in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(avg_conf - avg_acc)

    return ece


# ---------------------------------------------------------------------------
# Selective abstention
# ---------------------------------------------------------------------------
def selective_abstention(
    results: List[dict], thresholds: List[float]
) -> List[dict]:
    """Compute accuracy and coverage at each confidence threshold.

    Returns a list of dicts with keys: threshold, n_answered, n_correct,
    accuracy, coverage.
    """
    has_conf = [(r["avg_confidence"], r["is_correct"]) for r in results if r.get("avg_confidence") is not None]
    total_with_conf = len(has_conf)

    rows = []
    for thr in thresholds:
        answered = [(c, corr) for c, corr in has_conf if c >= thr]
        n_answered = len(answered)
        n_correct = sum(1 for _, corr in answered if corr)
        accuracy = n_correct / n_answered if n_answered > 0 else None
        coverage = n_answered / total_with_conf if total_with_conf > 0 else 0.0
        rows.append({
            "threshold": thr,
            "n_answered": n_answered,
            "n_correct": n_correct,
            "accuracy": accuracy,
            "coverage": coverage,
        })
    return rows


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def bootstrap_ci(
    results: List[dict],
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Tuple[float, float, float]:
    """Compute bootstrap 95% CI for accuracy.

    Returns:
        (mean_accuracy, ci_lower, ci_upper)
    """
    rng = random.Random(seed)
    correctness = [1 if r["is_correct"] else 0 for r in results]
    n = len(correctness)
    if n == 0:
        return 0.0, 0.0, 0.0

    means = []
    for _ in range(n_bootstrap):
        sample = [correctness[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    lo_idx = int(0.025 * n_bootstrap)
    hi_idx = int(0.975 * n_bootstrap)
    mean_acc = sum(correctness) / n
    return mean_acc, means[lo_idx], means[hi_idx]


# ---------------------------------------------------------------------------
# Confidence distribution
# ---------------------------------------------------------------------------
def confidence_distribution(results: List[dict]) -> List[dict]:
    """Compute confidence histogram: count and accuracy per bin.

    Returns list of dicts with keys: bin_lo, bin_hi, count, accuracy.
    """
    rows = []
    for lo, hi in CONF_BINS:
        bucket = [r for r in results if r.get("avg_confidence") is not None and lo <= r["avg_confidence"] < hi]
        count = len(bucket)
        if count > 0:
            acc = sum(1 for r in bucket if r["is_correct"]) / count
        else:
            acc = None
        label = f"{lo:.1f}-{hi:.2f}" if hi > 1.0 else f"{lo:.1f}-{hi:.1f}"
        rows.append({"bin_lo": lo, "bin_hi": min(hi, 1.0), "count": count, "accuracy": acc})
    return rows


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------
def print_accuracy_table(
    models: Dict[str, List[dict]], benchmarks: List[str]
) -> Dict:
    """Print and return the accuracy table."""
    print("\n" + "=" * 70)
    print("  ACCURACY TABLE (4 models x 3 benchmarks)")
    print("=" * 70)
    header = f"  {'Model':<25}"
    for b in benchmarks:
        header += f" {b:>10}"
    header += f" {'OVERALL':>10}"
    print(header)
    print("  " + "-" * (25 + 11 * (len(benchmarks) + 1)))

    table = {}
    for name, data in models.items():
        row_str = f"  {name:<25}"
        row_data = {}
        for b in benchmarks:
            nc, nt, acc = compute_accuracy(data, b)
            row_str += f" {acc*100:9.1f}%"
            row_data[b] = {"n_correct": nc, "n_total": nt, "accuracy": round(acc, 4)}
        nc, nt, acc = compute_accuracy(data)
        row_str += f" {acc*100:9.1f}%"
        row_data["overall"] = {"n_correct": nc, "n_total": nt, "accuracy": round(acc, 4)}
        print(row_str)
        table[name] = row_data

    return table


def print_ece_table(
    models: Dict[str, List[dict]], benchmarks: List[str]
) -> Dict:
    """Print and return the ECE table."""
    print("\n" + "=" * 70)
    print("  ECE TABLE (Expected Calibration Error)")
    print("=" * 70)
    header = f"  {'Model':<25}"
    for b in benchmarks:
        header += f" {b:>10}"
    header += f" {'OVERALL':>10}"
    print(header)
    print("  " + "-" * (25 + 11 * (len(benchmarks) + 1)))

    table = {}
    for name, data in models.items():
        row_str = f"  {name:<25}"
        row_data = {}
        for b in benchmarks:
            subset = [r for r in data if r["benchmark"] == b]
            ece = compute_ece(subset)
            if ece is not None:
                row_str += f" {ece:9.3f} "
                row_data[b] = round(ece, 4)
            else:
                row_str += f" {'N/A':>10}"
                row_data[b] = None
        ece_all = compute_ece(data)
        if ece_all is not None:
            row_str += f" {ece_all:9.3f} "
            row_data["overall"] = round(ece_all, 4)
        else:
            row_str += f" {'N/A':>10}"
            row_data["overall"] = None
        print(row_str)
        table[name] = row_data

    return table


def print_selective_abstention(models: Dict[str, List[dict]]) -> Dict:
    """Print and return selective abstention results."""
    print("\n" + "=" * 70)
    print("  SELECTIVE ABSTENTION (accuracy at confidence thresholds)")
    print("=" * 70)

    table = {}
    for name, data in models.items():
        rows = selective_abstention(data, CONF_THRESHOLDS)
        has_conf = any(r.get("avg_confidence") is not None for r in data)
        print(f"\n  {name}:")
        if not has_conf:
            total = len(data)
            correct = sum(1 for r in data if r["is_correct"])
            print(f"    No confidence data. Always answers: {correct}/{total} = {correct/total:.1%}")
            table[name] = {"no_confidence": True, "accuracy": round(correct / total, 4)}
            continue

        print(f"    {'Threshold':>10} {'Answered':>10} {'Correct':>10} {'Accuracy':>10} {'Coverage':>10}")
        abstention_data = []
        for row in rows:
            acc_str = f"{row['accuracy']:.1%}" if row["accuracy"] is not None else "N/A"
            print(f"    {row['threshold']:>10.1f} {row['n_answered']:>10} {row['n_correct']:>10} {acc_str:>10} {row['coverage']:.1%}")
            abstention_data.append({
                "threshold": row["threshold"],
                "n_answered": row["n_answered"],
                "n_correct": row["n_correct"],
                "accuracy": round(row["accuracy"], 4) if row["accuracy"] is not None else None,
                "coverage": round(row["coverage"], 4),
            })
        table[name] = abstention_data

    return table


def print_bootstrap_ci(models: Dict[str, List[dict]], benchmarks: List[str]) -> Dict:
    """Print and return bootstrap 95% CI for accuracy."""
    print("\n" + "=" * 70)
    print("  BOOTSTRAP 95% CI FOR ACCURACY")
    print("=" * 70)
    print(f"  {'Model':<25} {'Benchmark':<12} {'Accuracy':>10} {'95% CI':>20}")
    print("  " + "-" * 68)

    table = {}
    for name, data in models.items():
        model_ci = {}
        for b in benchmarks + ["overall"]:
            subset = data if b == "overall" else [r for r in data if r["benchmark"] == b]
            if not subset:
                continue
            mean_acc, ci_lo, ci_hi = bootstrap_ci(subset)
            print(f"  {name:<25} {b:<12} {mean_acc*100:9.1f}%  [{ci_lo*100:.1f}%, {ci_hi*100:.1f}%]")
            model_ci[b] = {
                "accuracy": round(mean_acc, 4),
                "ci_lower": round(ci_lo, 4),
                "ci_upper": round(ci_hi, 4),
            }
        table[name] = model_ci

    return table


def print_confidence_distribution(models: Dict[str, List[dict]]) -> Dict:
    """Print and return confidence distribution histogram data."""
    print("\n" + "=" * 70)
    print("  CONFIDENCE DISTRIBUTION")
    print("=" * 70)

    table = {}
    for name, data in models.items():
        dist = confidence_distribution(data)
        has_conf = any(d["count"] > 0 for d in dist)
        if not has_conf:
            print(f"\n  {name}: No confidence data")
            table[name] = None
            continue

        print(f"\n  {name}:")
        print(f"    {'Bin':>12} {'Count':>8} {'Accuracy':>10}")
        for d in dist:
            label = f"[{d['bin_lo']:.1f}, {d['bin_hi']:.1f})"
            acc_str = f"{d['accuracy']:.1%}" if d["accuracy"] is not None else "N/A"
            print(f"    {label:>12} {d['count']:>8} {acc_str:>10}")
        table[name] = dist

    return table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Analyze large-scale 1030-problem eval results")
    parser.add_argument("--results_dir", default="results", help="Directory containing eval_1030_*.json files")
    args = parser.parse_args()

    print("Loading results...")
    models = load_results(args.results_dir)

    # Determine benchmarks present
    all_benchmarks = set()
    for data in models.values():
        for r in data:
            all_benchmarks.add(r["benchmark"])
    benchmarks = sorted(all_benchmarks)
    print(f"Benchmarks found: {benchmarks}")
    print(f"Models: {list(models.keys())}")

    # Run all analyses
    acc_table = print_accuracy_table(models, benchmarks)
    ece_table = print_ece_table(models, benchmarks)
    abstention_table = print_selective_abstention(models)
    ci_table = print_bootstrap_ci(models, benchmarks)
    dist_table = print_confidence_distribution(models)

    # Combine and save
    analysis = {
        "n_models": len(models),
        "n_benchmarks": len(benchmarks),
        "benchmarks": benchmarks,
        "models": list(models.keys()),
        "accuracy": acc_table,
        "ece": ece_table,
        "selective_abstention": abstention_table,
        "bootstrap_ci": ci_table,
        "confidence_distribution": dist_table,
    }

    save_path = os.path.join(args.results_dir, "analysis_1030.json")
    with open(save_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 70}")
    print(f"  All analysis saved to {save_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
