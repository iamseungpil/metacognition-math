#!/usr/bin/env python3
"""Comprehensive entropy + behavioral analysis of meta-CoT eval completions.

Determines whether <|meta|> blocks contribute to reasoning quality or merely
consume tokens.  Adapted from metacognition-behavior-uncertainty methodology.

Analyses performed:
  1. Behavioral marker extraction (verify, redirect, diagnosis, epistemic, confidence revision)
  2. Meta block impact (accuracy, length, confidence by has_meta vs no_meta)
  3. Behavioral marker x accuracy correlation (Fisher exact test)
  4. Meta position analysis (early / middle / late)
  5. Confidence calibration deep dive (reliability diagram data)
  6. Token efficiency analysis (useful tokens vs accuracy)

Usage:
  PYTHONPATH=. python scripts/analyze_meta_behavior.py --results_dir results/eval_1030_v5
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Signal detection regexes (from behavior-uncertainty codebase)
# ---------------------------------------------------------------------------
VERIFY_RE = re.compile(
    r"\b(verify|check|confirm|validate|double.check|re.?check|sanity check)\b", re.I
)
REDIRECT_RE = re.compile(
    r"\b(instead|alternative|try another|different approach|backtrack|switch|"
    r"let me try|new plan)\b",
    re.I,
)
DIAGNOSIS_RE = re.compile(
    r"\b(mistake|error|wrong|incorrect|problem is|issue is|not right|missed|overlooked)\b",
    re.I,
)
EPISTEMIC_RE = re.compile(
    r"\b(not sure|uncertain|maybe|perhaps|might be|could be|feels off|hmm|wait)\b",
    re.I,
)

# Meta block detection
META_PATTERN = re.compile(r"<\|meta\|>(.*?)<\|/meta\|>", re.DOTALL | re.IGNORECASE)

# Confidence value inside a meta block (e.g. "confidence: 0.84")
CONFIDENCE_VAL_RE = re.compile(r"confidence\s*:\s*([\d.]+)", re.I)

# Benchmark display order
BENCHMARK_ORDER = ["gsm8k", "math500", "aime2024"]
BENCHMARK_LABELS = {"gsm8k": "GSM8K", "math500": "MATH500", "aime2024": "AIME"}

# Confidence bins for calibration analysis
CONFIDENCE_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]


# =========================================================================
# Helpers
# =========================================================================
def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return a / b if b else default


def _pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "N/A"


def _fmt_float(v: Optional[float], decimals: int = 3) -> str:
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _fisher_exact_2x2(a: int, b: int, c: int, d: int) -> Tuple[float, float]:
    """Compute Fisher exact test p-value for a 2x2 contingency table.

    Table layout:
                Correct   Wrong
    Present       a         b
    Absent        c         d

    Returns (odds_ratio, p_value).
    Uses exact hypergeometric calculation to avoid scipy dependency.
    """
    # Odds ratio
    odds = _safe_div(a * d, b * c, default=float("inf"))

    # Two-sided Fisher exact test via hypergeometric distribution
    # P(X = k) = C(a+b, k) * C(c+d, a+c-k) / C(n, a+c)
    n = a + b + c + d
    row1 = a + b
    row2 = c + d
    col1 = a + c

    def _log_factorial(x: int) -> float:
        return sum(math.log(i) for i in range(1, x + 1))

    def _log_comb(n_: int, k_: int) -> float:
        if k_ < 0 or k_ > n_:
            return float("-inf")
        return _log_factorial(n_) - _log_factorial(k_) - _log_factorial(n_ - k_)

    def _log_hyper(k: int) -> float:
        return (
            _log_comb(row1, k)
            + _log_comb(row2, col1 - k)
            - _log_comb(n, col1)
        )

    # Observed log-probability
    log_p_obs = _log_hyper(a)

    # Sum probabilities of all outcomes as extreme or more extreme
    k_min = max(0, col1 - row2)
    k_max = min(row1, col1)

    p_value = 0.0
    for k in range(k_min, k_max + 1):
        log_p_k = _log_hyper(k)
        if log_p_k <= log_p_obs + 1e-10:  # as extreme or more
            p_value += math.exp(log_p_k)

    p_value = min(p_value, 1.0)
    return odds, p_value


# =========================================================================
# Data loading
# =========================================================================
def load_eval_files(results_dir: str) -> Dict[str, List[dict]]:
    """Load original (non-_v2) eval JSON files.

    Returns dict: model_name -> list of result records.
    """
    results_path = Path(results_dir)
    if not results_path.is_dir():
        print(f"ERROR: results directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    models: Dict[str, List[dict]] = {}

    for fpath in sorted(results_path.glob("eval_1030_*.json")):
        # Skip _v2 / _v2_v2 rescored variants
        stem = fpath.stem  # e.g. "eval_1030_base_sft"
        after_prefix = stem.replace("eval_1030_", "")
        if "_v2" in after_prefix:
            continue

        try:
            payload = json.loads(fpath.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  WARNING: skipping {fpath.name} ({exc})")
            continue

        results_list = payload.get("results", [])
        if not results_list:
            print(f"  WARNING: {fpath.name} has no results, skipping")
            continue

        model_name = after_prefix  # e.g. "base_sft", "E8", "all_sft"
        models[model_name] = results_list
        print(f"  Loaded {fpath.name}: {len(results_list)} records ({model_name})")

    if not models:
        print("ERROR: no eval files found.", file=sys.stderr)
        sys.exit(1)

    return models


# =========================================================================
# 1. Behavioral marker extraction
# =========================================================================
def extract_markers(record: dict) -> dict:
    """Extract behavioral markers from a single completion record."""
    completion = record.get("completion", "") or ""
    meta_blocks = META_PATTERN.findall(completion)
    meta_text = " ".join(meta_blocks)
    non_meta_text = META_PATTERN.sub("", completion)

    # Markers -- search in FULL completion (meta + non-meta)
    has_verify = bool(VERIFY_RE.search(completion))
    has_redirect = bool(REDIRECT_RE.search(completion))
    has_diagnosis = bool(DIAGNOSIS_RE.search(completion))
    has_epistemic = bool(EPISTEMIC_RE.search(completion))

    # Markers specifically inside meta blocks
    meta_verify = bool(VERIFY_RE.search(meta_text)) if meta_text else False
    meta_redirect = bool(REDIRECT_RE.search(meta_text)) if meta_text else False
    meta_diagnosis = bool(DIAGNOSIS_RE.search(meta_text)) if meta_text else False

    # Confidence revision: did confidence change across meta blocks?
    confidences = record.get("meta_confidences", []) or []
    has_confidence_revision = False
    confidence_delta = 0.0
    if len(confidences) >= 2:
        has_confidence_revision = True
        confidence_delta = confidences[-1] - confidences[0]

    # Meta token ratio (character-based approximation: chars / 4 ~ tokens)
    total_chars = len(completion)
    meta_chars = sum(len(b) for b in meta_blocks) + len(meta_blocks) * 20  # tag overhead
    meta_token_ratio = _safe_div(meta_chars, total_chars)

    # Correctness: prefer is_correct_v2 if available
    is_correct = record.get("is_correct_v2", record.get("is_correct", False))

    return {
        "has_verify": has_verify,
        "has_redirect": has_redirect,
        "has_diagnosis": has_diagnosis,
        "has_epistemic": has_epistemic,
        "meta_verify": meta_verify,
        "meta_redirect": meta_redirect,
        "meta_diagnosis": meta_diagnosis,
        "has_confidence_revision": has_confidence_revision,
        "confidence_delta": confidence_delta,
        "meta_token_ratio": meta_token_ratio,
        "num_meta_blocks": record.get("num_meta_blocks", 0) or 0,
        "meta_confidences": confidences,
        "avg_confidence": record.get("avg_confidence"),
        "is_correct": is_correct,
        "benchmark": record.get("benchmark", "unknown"),
        "completion_length_tokens": record.get("completion_length_tokens", 0) or 0,
        "completion_length_chars": total_chars,
        "meta_chars": meta_chars,
    }


# =========================================================================
# 2. Meta block impact analysis
# =========================================================================
def analyze_meta_impact(
    enriched: List[dict],
) -> Dict[str, Any]:
    """Compare has_meta vs no_meta groups."""
    results: Dict[str, Any] = {}

    for bm in BENCHMARK_ORDER + ["overall"]:
        if bm == "overall":
            subset = enriched
        else:
            subset = [r for r in enriched if r["benchmark"] == bm]
        if not subset:
            continue

        has_meta = [r for r in subset if r["num_meta_blocks"] > 0]
        no_meta = [r for r in subset if r["num_meta_blocks"] == 0]

        def _group_stats(group: List[dict], label: str) -> dict:
            n = len(group)
            if n == 0:
                return {"n": 0, "accuracy": None, "avg_tokens": None, "avg_confidence": None}
            correct = sum(1 for r in group if r["is_correct"])
            avg_tokens = np.mean([r["completion_length_tokens"] for r in group])
            confs = [r["avg_confidence"] for r in group if r["avg_confidence"] is not None]
            avg_conf = float(np.mean(confs)) if confs else None
            return {
                "n": n,
                "accuracy": correct / n,
                "n_correct": correct,
                "avg_tokens": float(avg_tokens),
                "avg_confidence": avg_conf,
            }

        results[bm] = {
            "has_meta": _group_stats(has_meta, "has_meta"),
            "no_meta": _group_stats(no_meta, "no_meta"),
            "total_n": len(subset),
        }

    return results


# =========================================================================
# 3. Behavioral marker x accuracy correlation
# =========================================================================
def analyze_marker_accuracy(
    enriched: List[dict],
) -> Dict[str, Any]:
    """For each marker, compare accuracy when present vs absent."""
    markers = ["has_verify", "has_redirect", "has_diagnosis", "has_epistemic"]
    results: Dict[str, Any] = {}

    for bm in BENCHMARK_ORDER + ["overall"]:
        if bm == "overall":
            subset = enriched
        else:
            subset = [r for r in enriched if r["benchmark"] == bm]
        if not subset:
            continue

        bm_results: Dict[str, Any] = {}
        for marker in markers:
            present = [r for r in subset if r[marker]]
            absent = [r for r in subset if not r[marker]]

            n_present = len(present)
            n_absent = len(absent)
            correct_present = sum(1 for r in present if r["is_correct"])
            correct_absent = sum(1 for r in absent if r["is_correct"])
            wrong_present = n_present - correct_present
            wrong_absent = n_absent - correct_absent

            acc_present = _safe_div(correct_present, n_present)
            acc_absent = _safe_div(correct_absent, n_absent)

            # Fisher exact test
            odds, p_val = _fisher_exact_2x2(
                correct_present, wrong_present,
                correct_absent, wrong_absent,
            )

            bm_results[marker] = {
                "n_present": n_present,
                "n_absent": n_absent,
                "acc_present": acc_present,
                "acc_absent": acc_absent,
                "acc_delta": acc_present - acc_absent,
                "odds_ratio": odds,
                "p_value": p_val,
                "significant": p_val < 0.05,
            }

        results[bm] = bm_results

    return results


# =========================================================================
# 4. Meta position analysis
# =========================================================================
def _meta_positions(completion: str) -> List[float]:
    """Return normalized positions [0,1] of each meta block in the completion."""
    total_len = len(completion)
    if total_len == 0:
        return []
    positions = []
    for m in META_PATTERN.finditer(completion):
        mid = (m.start() + m.end()) / 2
        positions.append(mid / total_len)
    return positions


def _position_label(pos: float) -> str:
    if pos < 0.33:
        return "early"
    elif pos < 0.67:
        return "middle"
    else:
        return "late"


def analyze_meta_positions(
    enriched: List[dict], all_records: List[dict]
) -> Dict[str, Any]:
    """Analyze where meta blocks appear and correlation with accuracy."""
    # We need the raw completions for position detection
    # Build a parallel structure
    results: Dict[str, Any] = {}

    # Combine enriched data with raw completion text
    records_with_meta = [
        (e, r) for e, r in zip(enriched, all_records) if e["num_meta_blocks"] > 0
    ]

    if not records_with_meta:
        return {"note": "no records with meta blocks"}

    # Position distribution
    position_counts = {"early": 0, "middle": 0, "late": 0}
    position_accuracy = {"early": {"correct": 0, "total": 0},
                         "middle": {"correct": 0, "total": 0},
                         "late": {"correct": 0, "total": 0}}

    # Second meta usefulness
    second_meta_stats = {"n": 0, "correct": 0}

    for enriched_rec, raw_rec in records_with_meta:
        completion = raw_rec.get("completion", "") or ""
        positions = _meta_positions(completion)

        if not positions:
            continue

        # Use first meta block position for primary analysis
        first_pos = positions[0]
        label = _position_label(first_pos)
        position_counts[label] += 1
        position_accuracy[label]["total"] += 1
        if enriched_rec["is_correct"]:
            position_accuracy[label]["correct"] += 1

        # Second meta block analysis
        if len(positions) >= 2:
            second_meta_stats["n"] += 1
            if enriched_rec["is_correct"]:
                second_meta_stats["correct"] += 1

    # Compute accuracies
    position_acc_summary = {}
    for label in ["early", "middle", "late"]:
        t = position_accuracy[label]["total"]
        c = position_accuracy[label]["correct"]
        position_acc_summary[label] = {
            "n": t,
            "accuracy": _safe_div(c, t),
            "n_correct": c,
        }

    results["position_distribution"] = position_counts
    results["position_accuracy"] = position_acc_summary
    results["second_meta"] = {
        "n": second_meta_stats["n"],
        "accuracy": _safe_div(
            second_meta_stats["correct"], second_meta_stats["n"]
        ),
    }

    return results


# =========================================================================
# 5. Confidence calibration deep dive
# =========================================================================
def analyze_calibration(enriched: List[dict]) -> Dict[str, Any]:
    """Bin by confidence value and compute actual accuracy per bin."""
    results: Dict[str, Any] = {}

    for bm in BENCHMARK_ORDER + ["overall"]:
        if bm == "overall":
            subset = enriched
        else:
            subset = [r for r in enriched if r["benchmark"] == bm]

        # Only records with confidence
        with_conf = [r for r in subset if r["avg_confidence"] is not None]
        if not with_conf:
            results[bm] = {"note": "no confidence data"}
            continue

        bins_data: List[Dict[str, Any]] = []
        for lo, hi in CONFIDENCE_BINS:
            in_bin = [
                r for r in with_conf if lo <= r["avg_confidence"] < hi
                or (hi == 1.0 and r["avg_confidence"] == 1.0 and lo <= 1.0)
            ]
            n = len(in_bin)
            correct = sum(1 for r in in_bin if r["is_correct"])
            avg_conf_bin = float(np.mean([r["avg_confidence"] for r in in_bin])) if in_bin else None

            bins_data.append({
                "bin": f"{lo:.1f}-{hi:.1f}",
                "n": n,
                "accuracy": _safe_div(correct, n) if n > 0 else None,
                "avg_confidence": avg_conf_bin,
                "n_correct": correct,
            })

        # Overall calibration metrics
        all_confs = np.array([r["avg_confidence"] for r in with_conf])
        all_correct = np.array([1 if r["is_correct"] else 0 for r in with_conf])
        ece = _compute_ece(all_confs, all_correct, n_bins=10)
        overconf_rate = float(np.mean(all_confs > all_correct))

        results[bm] = {
            "bins": bins_data,
            "n_with_confidence": len(with_conf),
            "ece": ece,
            "avg_confidence": float(np.mean(all_confs)),
            "avg_accuracy": float(np.mean(all_correct)),
            "overconfidence_gap": float(np.mean(all_confs) - np.mean(all_correct)),
        }

    return results


def _compute_ece(confs: np.ndarray, corrects: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confs)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        mask = (confs >= bin_edges[i]) & (confs < bin_edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (confs == bin_edges[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = corrects[mask].mean()
        bin_conf = confs[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)


# =========================================================================
# 6. Token efficiency analysis
# =========================================================================
def analyze_token_efficiency(enriched: List[dict]) -> Dict[str, Any]:
    """Compare useful tokens (total - meta overhead) vs accuracy."""
    results: Dict[str, Any] = {}

    for bm in BENCHMARK_ORDER + ["overall"]:
        if bm == "overall":
            subset = enriched
        else:
            subset = [r for r in enriched if r["benchmark"] == bm]
        if not subset:
            continue

        has_meta = [r for r in subset if r["num_meta_blocks"] > 0]
        no_meta = [r for r in subset if r["num_meta_blocks"] == 0]

        def _eff_stats(group: List[dict]) -> dict:
            n = len(group)
            if n == 0:
                return {"n": 0}
            total_tokens = [r["completion_length_tokens"] for r in group]
            # Estimate meta tokens from meta_chars / 4 (rough tokenizer approx)
            meta_tokens_est = [r["meta_chars"] / 4 for r in group]
            useful_tokens = [t - m for t, m in zip(total_tokens, meta_tokens_est)]
            correct = sum(1 for r in group if r["is_correct"])
            meta_ratios = [r["meta_token_ratio"] for r in group]

            return {
                "n": n,
                "accuracy": _safe_div(correct, n),
                "avg_total_tokens": float(np.mean(total_tokens)),
                "avg_meta_tokens_est": float(np.mean(meta_tokens_est)),
                "avg_useful_tokens": float(np.mean(useful_tokens)),
                "avg_meta_ratio": float(np.mean(meta_ratios)),
                "tokens_per_correct": _safe_div(
                    float(np.sum(total_tokens)), correct
                ) if correct > 0 else None,
            }

        results[bm] = {
            "has_meta": _eff_stats(has_meta),
            "no_meta": _eff_stats(no_meta),
        }

    return results


# =========================================================================
# Printing helpers
# =========================================================================
def _hline(width: int = 90) -> str:
    return "-" * width


def print_section(title: str) -> None:
    print(f"\n{'=' * 90}")
    print(f"  {title}")
    print(f"{'=' * 90}")


def print_meta_impact(impact: Dict[str, Any], model: str) -> None:
    print(f"\n  Model: {model}")
    print(f"  {'Benchmark':<12} {'Group':<10} {'N':>6} {'Accuracy':>10} "
          f"{'AvgTokens':>10} {'AvgConf':>10}")
    print(f"  {_hline(68)}")
    for bm in BENCHMARK_ORDER + ["overall"]:
        if bm not in impact:
            continue
        data = impact[bm]
        for group_name in ["has_meta", "no_meta"]:
            g = data[group_name]
            if g["n"] == 0:
                continue
            label = BENCHMARK_LABELS.get(bm, bm)
            print(
                f"  {label:<12} {group_name:<10} {g['n']:>6} "
                f"{_pct(g.get('n_correct', 0), g['n']):>10} "
                f"{_fmt_float(g['avg_tokens'], 0):>10} "
                f"{_fmt_float(g['avg_confidence']):>10}"
            )


def print_marker_accuracy(marker_data: Dict[str, Any], model: str) -> None:
    print(f"\n  Model: {model}")
    print(f"  {'Benchmark':<12} {'Marker':<18} {'N(+)':>6} {'Acc(+)':>8} "
          f"{'N(-)':>6} {'Acc(-)':>8} {'Delta':>8} {'p-val':>8} {'Sig':>5}")
    print(f"  {_hline(85)}")
    for bm in BENCHMARK_ORDER + ["overall"]:
        if bm not in marker_data:
            continue
        for marker, vals in marker_data[bm].items():
            label = BENCHMARK_LABELS.get(bm, bm)
            short_marker = marker.replace("has_", "")
            sig_mark = " *" if vals["significant"] else ""
            print(
                f"  {label:<12} {short_marker:<18} "
                f"{vals['n_present']:>6} {vals['acc_present']:>7.1%} "
                f"{vals['n_absent']:>6} {vals['acc_absent']:>7.1%} "
                f"{vals['acc_delta']:>+7.1%} "
                f"{vals['p_value']:>8.4f}{sig_mark:>5}"
            )


def print_position_analysis(pos_data: Dict[str, Any], model: str) -> None:
    print(f"\n  Model: {model}")
    if "note" in pos_data:
        print(f"  {pos_data['note']}")
        return

    print(f"  Position distribution: {pos_data['position_distribution']}")
    print(f"  {'Position':<12} {'N':>6} {'Accuracy':>10}")
    print(f"  {_hline(32)}")
    for pos in ["early", "middle", "late"]:
        pa = pos_data["position_accuracy"].get(pos, {})
        n = pa.get("n", 0)
        acc = pa.get("accuracy", 0)
        print(f"  {pos:<12} {n:>6} {acc:>9.1%}")

    sm = pos_data.get("second_meta", {})
    if sm.get("n", 0) > 0:
        print(f"\n  Second meta block: n={sm['n']}, accuracy={sm['accuracy']:.1%}")
    else:
        print(f"\n  Second meta block: n=0")


def print_calibration(cal_data: Dict[str, Any], model: str) -> None:
    print(f"\n  Model: {model}")
    for bm in BENCHMARK_ORDER + ["overall"]:
        if bm not in cal_data:
            continue
        data = cal_data[bm]
        if "note" in data:
            print(f"  {BENCHMARK_LABELS.get(bm, bm)}: {data['note']}")
            continue

        label = BENCHMARK_LABELS.get(bm, bm)
        print(f"\n  {label} (n={data['n_with_confidence']}, ECE={data['ece']:.3f}, "
              f"AvgConf={data['avg_confidence']:.3f}, AvgAcc={data['avg_accuracy']:.3f}, "
              f"Gap={data['overconfidence_gap']:+.3f})")
        print(f"  {'Bin':<12} {'N':>6} {'Accuracy':>10} {'AvgConf':>10}")
        print(f"  {_hline(42)}")
        for b in data["bins"]:
            acc_str = f"{b['accuracy']:.1%}" if b["accuracy"] is not None else "N/A"
            conf_str = _fmt_float(b["avg_confidence"])
            print(f"  {b['bin']:<12} {b['n']:>6} {acc_str:>10} {conf_str:>10}")


def print_token_efficiency(eff_data: Dict[str, Any], model: str) -> None:
    print(f"\n  Model: {model}")
    print(f"  {'Benchmark':<12} {'Group':<10} {'N':>5} {'Acc':>7} "
          f"{'TotalTok':>9} {'MetaTok':>9} {'UsefulTok':>10} "
          f"{'MetaRatio':>10} {'Tok/Correct':>12}")
    print(f"  {_hline(88)}")
    for bm in BENCHMARK_ORDER + ["overall"]:
        if bm not in eff_data:
            continue
        for group_name in ["has_meta", "no_meta"]:
            g = eff_data[bm][group_name]
            if g["n"] == 0:
                continue
            label = BENCHMARK_LABELS.get(bm, bm)
            tpc = _fmt_float(g.get("tokens_per_correct"), 0)
            print(
                f"  {label:<12} {group_name:<10} {g['n']:>5} "
                f"{g.get('accuracy', 0):>6.1%} "
                f"{g.get('avg_total_tokens', 0):>9.0f} "
                f"{g.get('avg_meta_tokens_est', 0):>9.0f} "
                f"{g.get('avg_useful_tokens', 0):>10.0f} "
                f"{g.get('avg_meta_ratio', 0):>9.1%} "
                f"{tpc:>12}"
            )


# =========================================================================
# Cross-model summary table
# =========================================================================
def print_cross_model_summary(all_results: Dict[str, Dict[str, Any]]) -> None:
    """Print a compact cross-model comparison table."""
    print_section("CROSS-MODEL SUMMARY")

    # Overall accuracy and meta usage
    print(f"\n  {'Model':<18} {'N':>5} {'Acc':>7} {'HasMeta':>8} "
          f"{'MetaAcc':>8} {'NoMetaAcc':>10} {'AvgMetaRatio':>13} "
          f"{'AvgTokens':>10}")
    print(f"  {_hline(82)}")

    for model_name in sorted(all_results.keys()):
        data = all_results[model_name]
        impact = data["meta_impact"].get("overall", {})
        eff = data["token_efficiency"].get("overall", {})

        hm = impact.get("has_meta", {})
        nm = impact.get("no_meta", {})
        total_n = impact.get("total_n", 0)

        total_correct = hm.get("n_correct", 0) + nm.get("n_correct", 0)
        total_acc = _safe_div(total_correct, total_n)

        hm_acc = hm.get("accuracy")
        nm_acc = nm.get("accuracy")
        hm_n = hm.get("n", 0)

        # Average meta ratio from efficiency data
        meta_ratio = eff.get("has_meta", {}).get("avg_meta_ratio", 0) if hm_n > 0 else 0

        all_tokens = []
        for g in [hm, nm]:
            if g.get("n", 0) > 0 and g.get("avg_tokens") is not None:
                all_tokens.extend([g["avg_tokens"]] * g["n"])
        avg_tok = np.mean(all_tokens) if all_tokens else 0

        print(
            f"  {model_name:<18} {total_n:>5} {total_acc:>6.1%} "
            f"{hm_n:>8} "
            f"{_pct(hm.get('n_correct', 0), hm_n) if hm_n > 0 else 'N/A':>8} "
            f"{_pct(nm.get('n_correct', 0), nm.get('n', 0)) if nm.get('n', 0) > 0 else 'N/A':>10} "
            f"{meta_ratio:>12.1%} "
            f"{avg_tok:>10.0f}"
        )


# =========================================================================
# Main
# =========================================================================
def run_analysis(results_dir: str) -> Dict[str, Any]:
    """Run all analyses and return combined results dict."""
    print(f"Loading eval files from: {results_dir}")
    models = load_eval_files(results_dir)

    all_model_results: Dict[str, Any] = {}

    for model_name, records in sorted(models.items()):
        print(f"\n{'#' * 90}")
        print(f"  Analyzing model: {model_name} ({len(records)} records)")
        print(f"{'#' * 90}")

        # 1. Extract markers
        enriched = [extract_markers(r) for r in records]

        # Quick stats
        n_with_meta = sum(1 for e in enriched if e["num_meta_blocks"] > 0)
        n_correct = sum(1 for e in enriched if e["is_correct"])
        print(f"\n  Records with meta blocks: {n_with_meta}/{len(enriched)}")
        print(f"  Overall accuracy: {_pct(n_correct, len(enriched))}")

        # Marker prevalence
        for marker in ["has_verify", "has_redirect", "has_diagnosis", "has_epistemic"]:
            count = sum(1 for e in enriched if e[marker])
            print(f"  {marker}: {count}/{len(enriched)} ({_pct(count, len(enriched))})")

        # 2. Meta block impact
        print_section("2. META BLOCK IMPACT ANALYSIS")
        meta_impact = analyze_meta_impact(enriched)
        print_meta_impact(meta_impact, model_name)

        # 3. Marker x accuracy correlation
        print_section("3. BEHAVIORAL MARKER x ACCURACY CORRELATION")
        marker_acc = analyze_marker_accuracy(enriched)
        print_marker_accuracy(marker_acc, model_name)

        # 4. Meta position analysis
        print_section("4. META POSITION ANALYSIS")
        position_data = analyze_meta_positions(enriched, records)
        print_position_analysis(position_data, model_name)

        # 5. Confidence calibration
        print_section("5. CONFIDENCE CALIBRATION DEEP DIVE")
        calibration = analyze_calibration(enriched)
        print_calibration(calibration, model_name)

        # 6. Token efficiency
        print_section("6. TOKEN EFFICIENCY ANALYSIS")
        token_eff = analyze_token_efficiency(enriched)
        print_token_efficiency(token_eff, model_name)

        all_model_results[model_name] = {
            "n_records": len(records),
            "n_with_meta": n_with_meta,
            "overall_accuracy": _safe_div(n_correct, len(enriched)),
            "marker_prevalence": {
                marker: sum(1 for e in enriched if e[marker])
                for marker in ["has_verify", "has_redirect", "has_diagnosis",
                               "has_epistemic", "has_confidence_revision"]
            },
            "meta_impact": meta_impact,
            "marker_accuracy": marker_acc,
            "position_analysis": position_data,
            "calibration": calibration,
            "token_efficiency": token_eff,
        }

    # Cross-model summary
    print_cross_model_summary(all_model_results)

    return all_model_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze meta-CoT behavioral markers and their impact on accuracy."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/eval_1030_v5",
        help="Directory containing eval_1030_*.json files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <results_dir>/behavior_analysis.json)",
    )
    args = parser.parse_args()

    output_path = args.output or str(
        Path(args.results_dir) / "behavior_analysis.json"
    )

    all_results = run_analysis(args.results_dir)

    # Save JSON output
    # Convert numpy/inf values for JSON serialization
    def _json_safe(obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            v = float(obj)
            if math.isinf(v):
                return "Inf"
            if math.isnan(v):
                return None
            return v
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float):
            if math.isinf(obj):
                return "Inf"
            if math.isnan(obj):
                return None
        return obj

    class _SafeEncoder(json.JSONEncoder):
        def default(self, o: Any) -> Any:
            result = _json_safe(o)
            if result is not o:
                return result
            return super().default(o)

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, cls=_SafeEncoder)

    print(f"\n\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
