"""Heuristic failure-mode analysis for control-v5 eval bundles.

This complements aggregate accuracy/calibration summaries with a lightweight
taxonomy over actual completions so we can inspect why Meta-CoT did not turn
into reliable adaptation on held-out evals.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from src.curriculum.control_rag import analyze_completion_for_rag

META_PATTERN = re.compile(r"<\|meta\|>(.*?)<\|/meta\|>", re.IGNORECASE | re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")


def _load_eval_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return [
        p for p in sorted(path.glob("eval_*.json"))
        if not p.name.endswith(".metadata.json")
    ]


def _extract_boxed_answer(text: str) -> str | None:
    if not text:
        return None
    matches = BOXED_PATTERN.findall(text)
    return matches[-1].strip() if matches else None


def _has_any(text: str, keywords: list[str]) -> bool:
    lowered = (text or "").lower()
    return any(keyword in lowered for keyword in keywords)


def _first_meta_ratio(text: str) -> float | None:
    if not text:
        return None
    idx = text.find("<|meta|>")
    if idx < 0 or not text:
        return None
    return idx / max(len(text), 1)


def _failure_reason(row: dict) -> str:
    completion = row.get("completion", "") or ""
    conf = row.get("avg_confidence")
    num_meta = row.get("num_meta_blocks") or 0
    rag = analyze_completion_for_rag(completion)
    first_meta = _first_meta_ratio(completion)

    if num_meta == 0:
        return "no_meta_signal"
    if conf is None:
        return "missing_confidence_signal"
    if num_meta == 1 and row.get("benchmark") == "aime2024":
        if conf < 0.5 and (rag["study_need"] or rag["has_switch"]):
            return "single_redirect_without_recovery"
        if conf >= 0.8 and _has_any(completion, ["verify", "independent check", "check"]):
            return "single_verify_without_correction"
    if first_meta is not None and first_meta > 0.55:
        return "late_meta_append"
    if conf >= 0.8 and _has_any(completion, ["verify", "independent check", "check"]):
        return "overconfident_verify_failed"
    if conf < 0.5 and rag["study_need"]:
        return "diagnosis_without_recovery"
    if conf < 0.5 and rag["has_switch"]:
        return "redirect_without_recovery"
    if num_meta <= 1:
        return "single_intervention_only"
    return "reasoning_error_after_control"


def _sample_payload(row: dict) -> dict:
    completion = row.get("completion", "") or ""
    meta_blocks = [
        " ".join(block.strip().split())
        for block in META_PATTERN.findall(completion)
    ]
    return {
        "benchmark": row.get("benchmark"),
        "confidence": row.get("avg_confidence"),
        "num_meta_blocks": row.get("num_meta_blocks"),
        "gold_answer": row.get("gold_answer"),
        "pred_answer": row.get("answer_extracted") or _extract_boxed_answer(completion),
        "question": (row.get("full_question") or row.get("question") or "")[:240],
        "meta_blocks": meta_blocks[:2],
        "tail": " ".join(completion[-600:].split())[:600],
    }


def summarize_file(path: Path) -> dict:
    payload = json.loads(path.read_text())
    rows = payload["results"]
    model = payload.get("model", path.stem.replace("eval_", ""))

    wrong_rows = [row for row in rows if not row.get("is_correct")]
    failure_counts = Counter()
    samples_by_reason: dict[str, list[dict]] = defaultdict(list)
    first_meta_wrong = []
    first_meta_aime_wrong = []
    repeated_aime_wrong = 0

    for row in wrong_rows:
        reason = _failure_reason(row)
        failure_counts[reason] += 1
        if len(samples_by_reason[reason]) < 3:
            samples_by_reason[reason].append(_sample_payload(row))
        ratio = _first_meta_ratio(row.get("completion", "") or "")
        if ratio is not None:
            first_meta_wrong.append(ratio)
            if row.get("benchmark") == "aime2024":
                first_meta_aime_wrong.append(ratio)
                if (row.get("num_meta_blocks") or 0) > 1:
                    repeated_aime_wrong += 1

    overall = {
        "model": model,
        "n": len(rows),
        "wrong": len(wrong_rows),
        "wrong_with_meta": sum(1 for row in wrong_rows if (row.get("num_meta_blocks") or 0) > 0),
        "wrong_with_repeated_meta": sum(1 for row in wrong_rows if (row.get("num_meta_blocks") or 0) > 1),
        "wrong_overconfident": sum(
            1 for row in wrong_rows
            if row.get("avg_confidence") is not None and row["avg_confidence"] >= 0.8
        ),
        "wrong_low_confidence": sum(
            1 for row in wrong_rows
            if row.get("avg_confidence") is not None and row["avg_confidence"] < 0.5
        ),
        "wrong_missing_confidence": sum(1 for row in wrong_rows if row.get("avg_confidence") is None),
        "avg_first_meta_pos_wrong": float(mean(first_meta_wrong)) if first_meta_wrong else None,
        "avg_first_meta_pos_aime_wrong": (
            float(mean(first_meta_aime_wrong)) if first_meta_aime_wrong else None
        ),
        "aime_wrong_repeated_meta": repeated_aime_wrong,
        "failure_reasons": dict(failure_counts.most_common()),
        "failure_samples": dict(samples_by_reason),
    }
    return {"overall": overall}


def build_markdown(summary: dict[str, dict]) -> str:
    lines = ["# Control-V5 Failure Mode Analysis", ""]
    lines.append(
        "아래 taxonomy는 heuristic이다. 목적은 실제 completion을 읽고 "
        "Meta-CoT가 왜 adaptation으로 이어지지 않는지 빠르게 분해하는 것이다."
    )
    lines.append("")
    for model_name, item in sorted(summary.items()):
        overall = item["overall"]
        lines.append(f"## {model_name}")
        lines.append(
            f"- wrong={overall['wrong']}/{overall['n']}, "
            f"wrong_with_meta={overall['wrong_with_meta']}, "
            f"repeated_meta_wrong={overall['wrong_with_repeated_meta']}, "
            f"overconf_wrong={overall['wrong_overconfident']}, "
            f"lowconf_wrong={overall['wrong_low_confidence']}, "
            f"missing_conf_wrong={overall['wrong_missing_confidence']}"
        )
        lines.append(
            f"- avg_first_meta_pos_wrong={overall['avg_first_meta_pos_wrong']}, "
            f"avg_first_meta_pos_aime_wrong={overall['avg_first_meta_pos_aime_wrong']}, "
            f"aime_wrong_repeated_meta={overall['aime_wrong_repeated_meta']}"
        )
        lines.append("- failure reasons:")
        for reason, count in overall["failure_reasons"].items():
            lines.append(f"  - {reason}: {count}")
        lines.append("")
        for reason, samples in overall["failure_samples"].items():
            lines.append(f"### {reason}")
            for sample in samples:
                conf = sample["confidence"]
                conf_text = "n/a" if conf is None else f"{conf:.2f}"
                lines.append(
                    f"- `{sample['benchmark']}` conf={conf_text} meta={sample['num_meta_blocks']} "
                    f"gold={sample['gold_answer']} pred={sample['pred_answer']} | {sample['question']}"
                )
                for meta_block in sample["meta_blocks"]:
                    lines.append(f"  meta: {meta_block}")
                lines.append(f"  tail: {sample['tail']}")
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()

    files: list[Path] = []
    for raw in args.inputs:
        files.extend(_load_eval_files(Path(raw)))
    files = sorted({path.resolve() for path in files})
    if not files:
        raise FileNotFoundError("no eval json files found")

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
    print(
        json.dumps(
            {
                "json": str(json_path),
                "markdown": str(md_path),
                "models": sorted(summary),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
