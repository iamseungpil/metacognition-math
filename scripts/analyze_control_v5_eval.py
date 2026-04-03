"""Summarize control-v5 eval results for the current metacognition plan."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import pandas as pd

from src.curriculum.control_rag import analyze_completion_for_rag
from src.training.rewards import (
    _has_effective_verification_signal,
    _meta_joined_text,
    _text_after_last_meta,
)


def _safe_mean(values):
    values = [v for v in values if v is not None]
    return float(mean(values)) if values else None


def _fmt(value):
    return "n/a" if value is None else f"{value:.3f}"


def _confidence_pairs(rows):
    pairs = []
    for row in rows:
        conf = row.get("avg_confidence")
        if conf is None:
            continue
        actual = 1.0 if row.get("is_correct") else 0.0
        pairs.append((float(conf), actual))
    return pairs


def _ece(rows, n_bins=10):
    pairs = _confidence_pairs(rows)
    if not pairs:
        return None
    total = len(pairs)
    ece = 0.0
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        bucket = [
            (conf, actual) for conf, actual in pairs
            if (lo <= conf < hi) or (idx == n_bins - 1 and lo <= conf <= hi)
        ]
        if not bucket:
            continue
        conf_mean = mean(conf for conf, _ in bucket)
        acc_mean = mean(actual for _, actual in bucket)
        ece += len(bucket) / total * abs(conf_mean - acc_mean)
    return float(ece)


def _brier(rows):
    pairs = _confidence_pairs(rows)
    if not pairs:
        return None
    return float(mean((conf - actual) ** 2 for conf, actual in pairs))


def _confidence_coverage(rows):
    if not rows:
        return None
    covered = sum(1 for row in rows if row.get("avg_confidence") is not None)
    return covered / len(rows)


def _wrong_high_conf_rate(rows, threshold=0.8):
    wrong = [r for r in rows if not r.get("is_correct")]
    if not wrong:
        return None
    flagged = [
        r for r in wrong
        if r.get("avg_confidence") is not None and r["avg_confidence"] >= threshold
    ]
    return len(flagged) / len(wrong)


def _behavior_metrics(rows):
    verify, redirect, diagnosis, study_need = [], [], [], []
    overconf_wrong = []
    redirect_cases = []
    aime_cases = []

    for row in rows:
        completion = row.get("completion", "")
        meta_text = _meta_joined_text(completion)
        tail = _text_after_last_meta(completion)
        rag_info = analyze_completion_for_rag(completion)

        has_verify = bool(meta_text) and _has_effective_verification_signal(tail)
        has_redirect = rag_info["has_switch"] and rag_info["has_low_confidence"]
        has_diagnosis = rag_info["has_diagnosis"] or rag_info["has_decomposition"]
        has_study_need = bool(rag_info["study_need"])

        verify.append(has_verify)
        redirect.append(has_redirect)
        diagnosis.append(has_diagnosis)
        study_need.append(has_study_need)

        conf = row.get("avg_confidence")
        if conf is not None and conf >= 0.8 and not row.get("is_correct"):
            overconf_wrong.append(
                {
                    "benchmark": row.get("benchmark"),
                    "confidence": conf,
                    "question": row.get("full_question", ""),
                    "completion": completion[:1000],
                }
            )
        if has_redirect:
            redirect_cases.append(
                {
                    "benchmark": row.get("benchmark"),
                    "confidence": conf,
                    "question": row.get("full_question", ""),
                    "diagnosis": rag_info["diagnosis_text"],
                    "study_need": rag_info["study_need"],
                    "completion": completion[:1000],
                }
            )
        if row.get("benchmark") == "aime2024":
            aime_cases.append(
                {
                    "is_correct": bool(row.get("is_correct")),
                    "confidence": conf,
                    "question": row.get("full_question", ""),
                    "completion": completion[:1000],
                    "has_verify": has_verify,
                    "has_redirect": has_redirect,
                    "has_diagnosis": has_diagnosis,
                }
            )

    n = max(len(rows), 1)
    return {
        "verify_rate": sum(verify) / n,
        "redirect_rate": sum(redirect) / n,
        "diagnosis_rate": sum(diagnosis) / n,
        "study_need_rate": sum(study_need) / n,
        "overconfident_wrong_samples": overconf_wrong[:5],
        "redirect_samples": redirect_cases[:5],
        "aime_samples": aime_cases[:5],
    }


def summarize_file(path: Path):
    payload = json.loads(path.read_text())
    results = payload["results"]
    model = payload.get("model", path.stem.replace("eval_", ""))

    overall = {
        "model": model,
        "n": len(results),
        "accuracy": _safe_mean([1.0 if r["is_correct"] else 0.0 for r in results]),
        "ece": _ece(results),
        "brier": _brier(results),
        "wrong_high_conf_rate": _wrong_high_conf_rate(results),
        "wrong_avg_conf": _safe_mean(
            [r.get("avg_confidence") for r in results if not r.get("is_correct")]
        ),
        "avg_meta_blocks": _safe_mean([r.get("num_meta_blocks") for r in results]),
        "confidence_coverage": _confidence_coverage(results),
    }
    overall.update(_behavior_metrics(results))

    by_benchmark = {}
    for benchmark in sorted({r["benchmark"] for r in results}):
        rows = [r for r in results if r["benchmark"] == benchmark]
        metric_row = {
            "n": len(rows),
            "accuracy": _safe_mean([1.0 if r["is_correct"] else 0.0 for r in rows]),
            "ece": _ece(rows),
            "brier": _brier(rows),
            "wrong_high_conf_rate": _wrong_high_conf_rate(rows),
            "wrong_avg_conf": _safe_mean(
                [r.get("avg_confidence") for r in rows if not r.get("is_correct")]
            ),
            "avg_meta_blocks": _safe_mean([r.get("num_meta_blocks") for r in rows]),
            "confidence_coverage": _confidence_coverage(rows),
        }
        metric_row.update(_behavior_metrics(rows))
        by_benchmark[benchmark] = metric_row

    return {"overall": overall, "by_benchmark": by_benchmark}


def build_markdown(summary):
    lines = ["# Control-V5 Eval Summary", ""]
    rows = []
    for model_name, item in summary.items():
        overall = item["overall"]
        rows.append(
            {
                "model": model_name,
                "acc": overall["accuracy"],
                "conf_cov": overall["confidence_coverage"],
                "ece": overall["ece"],
                "brier": overall["brier"],
                "wrong_hi": overall["wrong_high_conf_rate"],
                "meta": overall["avg_meta_blocks"],
                "verify": overall["verify_rate"],
                "redirect": overall["redirect_rate"],
                "diagnosis": overall["diagnosis_rate"],
                "study_need": overall["study_need_rate"],
            }
        )
    df = pd.DataFrame(rows).sort_values("model")
    lines.append(df.to_markdown(index=False, floatfmt=".3f"))
    lines.append("")

    for model_name, item in summary.items():
        lines.append(f"## {model_name}")
        for benchmark, metrics in item["by_benchmark"].items():
            lines.append(
                f"- `{benchmark}`: acc={_fmt(metrics['accuracy'])}, "
                f"conf_cov={_fmt(metrics['confidence_coverage'])}, "
                f"ece={_fmt(metrics['ece'])}, "
                f"brier={_fmt(metrics['brier'])}, "
                f"wrong_hi={_fmt(metrics['wrong_high_conf_rate'])}, "
                f"meta={_fmt(metrics['avg_meta_blocks'])}, "
                f"verify={_fmt(metrics['verify_rate'])}, "
                f"redirect={_fmt(metrics['redirect_rate'])}, "
                f"diagnosis={_fmt(metrics['diagnosis_rate'])}"
            )
        lines.append("")
        samples = item["overall"]["overconfident_wrong_samples"]
        if samples:
            lines.append("### Overconfident Wrong Samples")
            for sample in samples:
                lines.append(
                    f"- `{sample['benchmark']}` conf={sample['confidence']:.3f} | {sample['question'][:120]}"
                )
            lines.append("")
        aime_samples = item["overall"]["aime_samples"]
        if aime_samples:
            lines.append("### AIME Behavior Samples")
            for sample in aime_samples:
                conf_text = "n/a" if sample["confidence"] is None else f"{sample['confidence']:.3f}"
                lines.append(
                    f"- correct={sample['is_correct']} conf={conf_text} "
                    f"verify={sample['has_verify']} redirect={sample['has_redirect']} "
                    f"diagnosis={sample['has_diagnosis']} | {sample['question'][:120]}"
                )
            lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    files = sorted(results_dir.glob("eval_*.json"))
    if not files:
        raise FileNotFoundError(f"no eval_*.json in {results_dir}")

    summary = {}
    for path in files:
        item = summarize_file(path)
        summary[item["overall"]["model"]] = item

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    md_path = output_prefix.with_suffix(".md")
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    md_path.write_text(build_markdown(summary))

    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "models": sorted(summary)}, indent=2))


if __name__ == "__main__":
    main()
