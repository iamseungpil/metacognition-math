"""Comprehensive analysis for 1,030-problem evaluation (GSM8K 500 + MATH500 500 + AIME 30).

Produces:
  1. Quantitative metrics per model per benchmark (accuracy, ECE, overconfidence, etc.)
  2. Formatted comparison table with highlights
  3. Failure mode classification for wrong answers
  4. Pilot (n=90) vs full (n=1030) comparison
  5. JSON summaries saved to output_dir

Usage:
  python scripts/analyze_1030_eval.py --results_dir results/eval_1030_v5 --output_dir results/eval_1030_v5
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Signal detection patterns (from rewards.py)
# ---------------------------------------------------------------------------
VERIFY_RE = re.compile(
    r"\b(verify|check|confirm|validate|double.check|re.?check)\b", re.I
)
DIAGNOSIS_RE = re.compile(
    r"\b(study_need|error.*found|mistake|incorrect.*approach|wrong.*method)\b", re.I
)
ROUTE_SWITCH_RE = re.compile(
    r"\b(alternative|different.*approach|try.*instead|switch.*method|let me try)\b",
    re.I,
)

# Pilot n=90 results (hardcoded from readout)
PILOT_ACCURACY: Dict[str, float] = {
    "base_sft": 42.2,
    "all_sft": 33.3,
    "verify_sft": 36.7,
    "redirect_sft": 35.6,
    "E3": 30.0,
    "E5": 40.0,
    "E8": 38.9,
    "E9": 41.1,
    "E9b": 40.0,
    "E9c": 36.7,
    "E10": 35.6,
}

# Canonical benchmark order
BENCHMARK_ORDER = ["gsm8k", "math500", "aime2024"]

# ECE bin count
ECE_N_BINS = 15


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_all_results(results_dir: str) -> Dict[str, dict]:
    """Load all eval_<model>.json files from results_dir.

    Returns:
        dict mapping model_name -> full JSON payload (with 'model',
        'run_metadata', 'results' keys).
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    files = sorted(
        p
        for p in results_path.glob("eval_*.json")
        if not p.name.endswith(".metadata.json")
    )
    if not files:
        raise FileNotFoundError(
            f"No eval_*.json files found in {results_dir}. "
            "Run the evaluation first."
        )

    models: Dict[str, dict] = {}
    for fpath in files:
        try:
            payload = json.loads(fpath.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: skipping {fpath.name} ({exc})")
            continue

        name = payload.get("model", fpath.stem.replace("eval_", ""))
        results_list = payload.get("results", [])
        if not results_list:
            print(f"  WARNING: {fpath.name} has no results, skipping")
            continue

        models[name] = payload
        print(f"  Loaded {name}: {len(results_list)} results from {fpath.name}")

    if not models:
        raise FileNotFoundError(
            f"No valid eval results loaded from {results_dir}"
        )

    return models


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _subset(results: List[dict], benchmark: Optional[str] = None) -> List[dict]:
    """Filter results by benchmark if specified."""
    if benchmark is None:
        return results
    return [r for r in results if r.get("benchmark") == benchmark]


def _safe_mean(values: List[Optional[float]]) -> Optional[float]:
    """Mean of non-None values, or None if empty."""
    cleaned = [v for v in values if v is not None]
    return float(np.mean(cleaned)) if cleaned else None


# ---------------------------------------------------------------------------
# 1. Quantitative metrics
# ---------------------------------------------------------------------------
def compute_accuracy(results: List[dict]) -> Tuple[int, int, float]:
    """Return (n_correct, n_total, accuracy_fraction)."""
    if not results:
        return 0, 0, 0.0
    n_correct = sum(1 for r in results if r.get("is_correct"))
    return n_correct, len(results), n_correct / len(results)


def compute_ece(results: List[dict], n_bins: int = ECE_N_BINS) -> Optional[float]:
    """Expected Calibration Error with equal-width bins.

    Returns None when no confidence data is available.
    """
    pairs = [
        (float(r["avg_confidence"]), 1.0 if r["is_correct"] else 0.0)
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
        avg_conf = np.mean([c for c, _ in bucket])
        avg_acc = np.mean([a for _, a in bucket])
        ece += (len(bucket) / total) * abs(avg_conf - avg_acc)

    return float(ece)


def compute_confidence_coverage(results: List[dict]) -> float:
    """Fraction of problems with avg_confidence != None."""
    if not results:
        return 0.0
    covered = sum(1 for r in results if r.get("avg_confidence") is not None)
    return covered / len(results)


def compute_overconfidence_rate(
    results: List[dict], threshold: float = 0.7
) -> Optional[float]:
    """Among WRONG answers, fraction with avg_confidence > threshold."""
    wrong = [r for r in results if not r.get("is_correct")]
    if not wrong:
        return None
    flagged = sum(
        1
        for r in wrong
        if r.get("avg_confidence") is not None and r["avg_confidence"] > threshold
    )
    return flagged / len(wrong)


def compute_mean_meta_blocks(results: List[dict]) -> Optional[float]:
    """Mean number of meta blocks per completion."""
    vals = [r.get("num_meta_blocks") for r in results]
    return _safe_mean(vals)


def compute_mean_completion_length(results: List[dict]) -> Optional[float]:
    """Mean completion length in tokens."""
    vals = [r.get("completion_length_tokens") for r in results]
    return _safe_mean(vals)


def compute_model_benchmark_metrics(
    results: List[dict], benchmark: Optional[str] = None
) -> Dict[str, Any]:
    """Compute all quantitative metrics for a model on a benchmark slice."""
    sub = _subset(results, benchmark)
    nc, nt, acc = compute_accuracy(sub)
    return {
        "n_correct": nc,
        "n_total": nt,
        "accuracy": round(acc, 4),
        "accuracy_pct": round(acc * 100, 2),
        "ece": _round_opt(compute_ece(sub), 4),
        "confidence_coverage": round(compute_confidence_coverage(sub), 4),
        "confidence_coverage_pct": round(compute_confidence_coverage(sub) * 100, 1),
        "overconfidence_rate": _round_opt(compute_overconfidence_rate(sub), 4),
        "overconfidence_rate_pct": _round_opt(
            compute_overconfidence_rate(sub), 4, scale=100
        ),
        "mean_meta_blocks": _round_opt(compute_mean_meta_blocks(sub), 2),
        "mean_completion_tokens": _round_opt(compute_mean_completion_length(sub), 1),
    }


def _round_opt(
    val: Optional[float], digits: int, scale: float = 1.0
) -> Optional[float]:
    if val is None:
        return None
    return round(val * scale, digits)


# ---------------------------------------------------------------------------
# 2. Comparison table with highlights
# ---------------------------------------------------------------------------
def find_highlights(
    per_model: Dict[str, Dict[str, Dict[str, Any]]],
    benchmarks: List[str],
) -> Dict[str, Dict[str, str]]:
    """Identify best model per benchmark for accuracy, ECE, overconfidence.

    Returns nested dict: metric -> benchmark -> model_name.
    """
    highlights: Dict[str, Dict[str, str]] = {
        "best_accuracy": {},
        "best_ece": {},
        "lowest_overconfidence": {},
    }
    for bm in benchmarks + ["overall"]:
        best_acc_model, best_acc = None, -1.0
        best_ece_model, best_ece = None, float("inf")
        best_oc_model, best_oc = None, float("inf")

        for model_name, bm_metrics in per_model.items():
            m = bm_metrics.get(bm)
            if m is None:
                continue

            acc = m.get("accuracy", 0.0)
            if acc > best_acc:
                best_acc = acc
                best_acc_model = model_name

            ece_val = m.get("ece")
            if ece_val is not None and ece_val < best_ece:
                best_ece = ece_val
                best_ece_model = model_name

            oc_val = m.get("overconfidence_rate")
            if oc_val is not None and oc_val < best_oc:
                best_oc = oc_val
                best_oc_model = model_name

        if best_acc_model:
            highlights["best_accuracy"][bm] = best_acc_model
        if best_ece_model:
            highlights["best_ece"][bm] = best_ece_model
        if best_oc_model:
            highlights["lowest_overconfidence"][bm] = best_oc_model

    return highlights


def print_comparison_table(
    per_model: Dict[str, Dict[str, Dict[str, Any]]],
    benchmarks: List[str],
    highlights: Dict[str, Dict[str, str]],
) -> None:
    """Print formatted comparison tables to stdout."""
    models_sorted = sorted(per_model.keys())
    bm_cols = benchmarks + ["overall"]

    # --- Accuracy table ---
    _print_section_header("ACCURACY (%)")
    col_w = 10
    name_w = 22
    header = f"  {'Model':<{name_w}}"
    for bm in bm_cols:
        header += f" {bm:>{col_w}}"
    print(header)
    print("  " + "-" * (name_w + (col_w + 1) * len(bm_cols)))

    for model in models_sorted:
        row = f"  {model:<{name_w}}"
        for bm in bm_cols:
            m = per_model[model].get(bm)
            val = m["accuracy_pct"] if m else 0.0
            marker = " *" if highlights["best_accuracy"].get(bm) == model else "  "
            row += f" {val:>{col_w - 2}.1f}{marker}"
        print(row)

    print("  (* = best per column)")

    # --- ECE table ---
    _print_section_header("ECE (Expected Calibration Error, 15 bins)")
    header = f"  {'Model':<{name_w}}"
    for bm in bm_cols:
        header += f" {bm:>{col_w}}"
    print(header)
    print("  " + "-" * (name_w + (col_w + 1) * len(bm_cols)))

    for model in models_sorted:
        row = f"  {model:<{name_w}}"
        for bm in bm_cols:
            m = per_model[model].get(bm)
            ece_val = m["ece"] if m else None
            marker = " *" if highlights["best_ece"].get(bm) == model else "  "
            if ece_val is not None:
                row += f" {ece_val:>{col_w - 2}.3f}{marker}"
            else:
                row += f" {'N/A':>{col_w - 2}}{marker}"
        print(row)

    print("  (* = best per column)")

    # --- Overconfidence table ---
    _print_section_header("OVERCONFIDENCE RATE (% of wrong answers with conf > 0.7)")
    header = f"  {'Model':<{name_w}}"
    for bm in bm_cols:
        header += f" {bm:>{col_w}}"
    print(header)
    print("  " + "-" * (name_w + (col_w + 1) * len(bm_cols)))

    for model in models_sorted:
        row = f"  {model:<{name_w}}"
        for bm in bm_cols:
            m = per_model[model].get(bm)
            oc_val = m["overconfidence_rate_pct"] if m else None
            marker = " *" if highlights["lowest_overconfidence"].get(bm) == model else "  "
            if oc_val is not None:
                row += f" {oc_val:>{col_w - 2}.1f}{marker}"
            else:
                row += f" {'N/A':>{col_w - 2}}{marker}"
        print(row)

    print("  (* = lowest per column)")

    # --- Full detail table ---
    _print_section_header("FULL METRICS (Conf Coverage, Mean Meta Blocks, Mean Tokens)")
    header = f"  {'Model':<{name_w}} {'Bench':<10} {'ConfCov%':>8} {'MetaBlk':>8} {'Tokens':>8}"
    print(header)
    print("  " + "-" * (name_w + 40))

    for model in models_sorted:
        first_line = True
        for bm in bm_cols:
            m = per_model[model].get(bm)
            if m is None:
                continue
            model_col = model if first_line else ""
            cc = m["confidence_coverage_pct"]
            mb = m["mean_meta_blocks"]
            tk = m["mean_completion_tokens"]
            cc_str = f"{cc:.1f}" if cc is not None else "N/A"
            mb_str = f"{mb:.2f}" if mb is not None else "N/A"
            tk_str = f"{tk:.0f}" if tk is not None else "N/A"
            print(
                f"  {model_col:<{name_w}} {bm:<10} {cc_str:>8} {mb_str:>8} {tk_str:>8}"
            )
            first_line = False


# ---------------------------------------------------------------------------
# 3. Failure mode analysis
# ---------------------------------------------------------------------------
def classify_failure(row: dict) -> str:
    """Classify a wrong answer into a failure mode category.

    Categories:
      - no_meta: wrong answer with 0 meta blocks
      - overconfident_verify: wrong + has verification signal + avg_confidence > 0.7
      - diagnosis_no_recovery: wrong + has diagnosis signal + no route switch
      - single_intervention: wrong + exactly 1 meta block on hard problem (AIME)
      - confidence_omission: wrong + has meta blocks but no confidence values
      - other: doesn't match above
    """
    completion = row.get("completion", "") or ""
    num_meta = row.get("num_meta_blocks") or 0
    avg_conf = row.get("avg_confidence")
    benchmark = row.get("benchmark", "")
    confs = row.get("meta_confidences") or []

    # no_meta: wrong answer with 0 meta blocks
    if num_meta == 0:
        return "no_meta"

    # confidence_omission: has meta blocks but no confidence values
    # (meta_confidences is empty AND avg_confidence is None)
    if num_meta > 0 and not confs and avg_conf is None:
        return "confidence_omission"

    # overconfident_verify: wrong + has verification signal + avg_confidence > 0.7
    has_verify = bool(VERIFY_RE.search(completion))
    if has_verify and avg_conf is not None and avg_conf > 0.7:
        return "overconfident_verify"

    # diagnosis_no_recovery: wrong + has diagnosis signal + no route switch
    has_diagnosis = bool(DIAGNOSIS_RE.search(completion))
    has_route_switch = bool(ROUTE_SWITCH_RE.search(completion))
    if has_diagnosis and not has_route_switch:
        return "diagnosis_no_recovery"

    # single_intervention: wrong + exactly 1 meta block on hard problem (AIME)
    if num_meta == 1 and benchmark == "aime2024":
        return "single_intervention"

    return "other"


def failure_mode_analysis(
    model_name: str, results: List[dict]
) -> Dict[str, Any]:
    """Classify all wrong answers and return counts + samples."""
    wrong = [r for r in results if not r.get("is_correct")]
    if not wrong:
        return {
            "model": model_name,
            "n_wrong": 0,
            "n_total": len(results),
            "categories": {},
            "category_counts": {},
            "samples": {},
        }

    category_counts: Counter = Counter()
    samples_by_cat: Dict[str, List[dict]] = defaultdict(list)

    for row in wrong:
        cat = classify_failure(row)
        category_counts[cat] += 1
        if len(samples_by_cat[cat]) < 3:
            samples_by_cat[cat].append(
                {
                    "benchmark": row.get("benchmark"),
                    "confidence": row.get("avg_confidence"),
                    "num_meta_blocks": row.get("num_meta_blocks"),
                    "answer_extracted": row.get("answer_extracted"),
                    "gold_answer": row.get("gold_answer"),
                    "question": (
                        row.get("full_question") or row.get("question", "")
                    )[:200],
                }
            )

    # Build per-benchmark breakdown
    by_benchmark: Dict[str, Dict[str, int]] = defaultdict(lambda: Counter())
    for row in wrong:
        cat = classify_failure(row)
        by_benchmark[row.get("benchmark", "unknown")][cat] += 1

    return {
        "model": model_name,
        "n_wrong": len(wrong),
        "n_total": len(results),
        "category_counts": dict(category_counts.most_common()),
        "category_pcts": {
            cat: round(cnt / len(wrong) * 100, 1)
            for cat, cnt in category_counts.most_common()
        },
        "by_benchmark": {
            bm: dict(counts.most_common()) for bm, counts in sorted(by_benchmark.items())
        },
        "samples": {cat: samps for cat, samps in samples_by_cat.items()},
    }


def print_failure_modes(
    all_failures: Dict[str, Dict[str, Any]]
) -> None:
    """Print failure mode analysis to stdout."""
    _print_section_header("FAILURE MODE ANALYSIS")

    categories_legend = [
        ("no_meta", "Wrong with 0 meta blocks"),
        ("overconfident_verify", "Wrong + verification signal + conf > 0.7"),
        ("diagnosis_no_recovery", "Wrong + diagnosis signal + no route switch"),
        ("single_intervention", "Wrong + 1 meta block on AIME"),
        ("confidence_omission", "Wrong + meta blocks but no confidence values"),
        ("other", "Does not match above categories"),
    ]
    print("  Category definitions:")
    for cat, desc in categories_legend:
        print(f"    {cat:<28} {desc}")
    print()

    for model_name in sorted(all_failures.keys()):
        fm = all_failures[model_name]
        print(f"  {model_name}: {fm['n_wrong']} wrong / {fm['n_total']} total")
        if fm["n_wrong"] == 0:
            print("    (no wrong answers)")
            print()
            continue

        # Category counts
        for cat, cnt in fm["category_counts"].items():
            pct = fm["category_pcts"][cat]
            print(f"    {cat:<28} {cnt:>4}  ({pct:5.1f}%)")

        # Per-benchmark breakdown
        if fm.get("by_benchmark"):
            print("    Per-benchmark:")
            for bm, counts in fm["by_benchmark"].items():
                parts = ", ".join(f"{c}={n}" for c, n in counts.items())
                print(f"      {bm:<12} {parts}")
        print()


# ---------------------------------------------------------------------------
# 4. Pilot vs Full comparison
# ---------------------------------------------------------------------------
def print_pilot_comparison(
    per_model: Dict[str, Dict[str, Dict[str, Any]]]
) -> Dict[str, Dict[str, Any]]:
    """Compare pilot n=90 accuracy to full n=1030 overall accuracy."""
    _print_section_header("PILOT (n=90) vs FULL (n=1030) COMPARISON")

    comparison: Dict[str, Dict[str, Any]] = {}
    name_w = 22

    print(
        f"  {'Model':<{name_w}} {'Pilot%':>8} {'Full%':>8} {'Delta':>8} {'Direction':>10}"
    )
    print("  " + "-" * (name_w + 38))

    for model_name in sorted(per_model.keys()):
        pilot_acc = PILOT_ACCURACY.get(model_name)
        full_metrics = per_model[model_name].get("overall")
        full_acc = full_metrics["accuracy_pct"] if full_metrics else None

        pilot_str = f"{pilot_acc:.1f}" if pilot_acc is not None else "N/A"
        full_str = f"{full_acc:.1f}" if full_acc is not None else "N/A"

        if pilot_acc is not None and full_acc is not None:
            delta = full_acc - pilot_acc
            delta_str = f"{delta:+.1f}"
            direction = "UP" if delta > 0.5 else ("DOWN" if delta < -0.5 else "SAME")
        else:
            delta = None
            delta_str = "N/A"
            direction = "---"

        print(
            f"  {model_name:<{name_w}} {pilot_str:>8} {full_str:>8} {delta_str:>8} {direction:>10}"
        )

        comparison[model_name] = {
            "pilot_accuracy_pct": pilot_acc,
            "full_accuracy_pct": full_acc,
            "delta": round(delta, 2) if delta is not None else None,
            "direction": direction,
        }

    # Models in pilot but not in full
    missing_from_full = set(PILOT_ACCURACY.keys()) - set(per_model.keys())
    if missing_from_full:
        print()
        print(f"  Models in pilot but not in full eval: {sorted(missing_from_full)}")

    return comparison


# ---------------------------------------------------------------------------
# Printing helpers
# ---------------------------------------------------------------------------
def _print_section_header(title: str) -> None:
    width = 78
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis for 1,030-problem evaluation"
    )
    parser.add_argument(
        "--results_dir",
        required=True,
        help="Directory containing eval_<model>.json files",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save analysis JSON outputs",
    )
    args = parser.parse_args()

    # Load
    print("Loading evaluation results...")
    all_payloads = load_all_results(args.results_dir)
    model_names = sorted(all_payloads.keys())
    print(f"\nModels loaded ({len(model_names)}): {model_names}")

    # Discover benchmarks present
    all_benchmarks: set = set()
    for payload in all_payloads.values():
        for r in payload["results"]:
            all_benchmarks.add(r.get("benchmark"))
    # Use canonical order, then any extras alphabetically
    benchmarks = [b for b in BENCHMARK_ORDER if b in all_benchmarks]
    extras = sorted(all_benchmarks - set(benchmarks))
    benchmarks.extend(extras)
    print(f"Benchmarks found: {benchmarks}")

    # -----------------------------------------------------------------------
    # 1. Quantitative metrics
    # -----------------------------------------------------------------------
    per_model: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for model_name in model_names:
        results = all_payloads[model_name]["results"]
        bm_metrics: Dict[str, Dict[str, Any]] = {}

        for bm in benchmarks:
            bm_metrics[bm] = compute_model_benchmark_metrics(results, bm)

        bm_metrics["overall"] = compute_model_benchmark_metrics(results, None)
        per_model[model_name] = bm_metrics

    # -----------------------------------------------------------------------
    # 2. Comparison tables with highlights
    # -----------------------------------------------------------------------
    highlights = find_highlights(per_model, benchmarks)
    print_comparison_table(per_model, benchmarks, highlights)

    # -----------------------------------------------------------------------
    # 3. Failure mode analysis
    # -----------------------------------------------------------------------
    all_failures: Dict[str, Dict[str, Any]] = {}
    for model_name in model_names:
        results = all_payloads[model_name]["results"]
        all_failures[model_name] = failure_mode_analysis(model_name, results)

    print_failure_modes(all_failures)

    # -----------------------------------------------------------------------
    # 4. Pilot vs full comparison
    # -----------------------------------------------------------------------
    pilot_comparison = print_pilot_comparison(per_model)

    # -----------------------------------------------------------------------
    # 5. Save outputs
    # -----------------------------------------------------------------------
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # analysis_summary.json: high-level overview
    summary = {
        "n_models": len(model_names),
        "models": model_names,
        "benchmarks": benchmarks,
        "ece_n_bins": ECE_N_BINS,
        "overconfidence_threshold": 0.7,
        "highlights": highlights,
        "pilot_comparison": pilot_comparison,
    }
    summary_path = output_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n  Saved summary -> {summary_path}")

    # analysis_per_model.json: full metrics + failure modes
    per_model_output: Dict[str, Any] = {}
    for model_name in model_names:
        per_model_output[model_name] = {
            "metrics": per_model[model_name],
            "failure_modes": all_failures[model_name],
        }

    per_model_path = output_dir / "analysis_per_model.json"
    per_model_path.write_text(
        json.dumps(per_model_output, indent=2, ensure_ascii=False)
    )
    print(f"  Saved per-model -> {per_model_path}")

    _print_section_header("DONE")
    print(f"  Results dir:  {args.results_dir}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Models:       {len(model_names)}")
    print(f"  Benchmarks:   {benchmarks}")
    print(f"  Summary JSON: {summary_path}")
    print(f"  Detail JSON:  {per_model_path}")
    print()


if __name__ == "__main__":
    main()
