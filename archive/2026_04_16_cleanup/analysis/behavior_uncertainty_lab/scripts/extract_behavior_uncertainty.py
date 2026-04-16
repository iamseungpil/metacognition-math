#!/usr/bin/env python3
"""Extract behavior and uncertainty summaries from eval JSON outputs."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


META_START = "<|meta|>"
META_END = "<|/meta|>"

VERIFY_PATTERNS = [
    r"\bverify\b",
    r"\bverification\b",
    r"\bindependent check\b",
    r"\bdouble-check\b",
    r"\bsanity check\b",
    r"\bcheck again\b",
]

BACKTRACK_PATTERNS = [
    r"\bbacktrack\b",
    r"\bredirect\b",
    r"\bdifferent approach\b",
    r"\btry another\b",
    r"\bswitch(?:ing)? to\b",
    r"\broute is weak\b",
    r"\bsomething feels off\b",
    r"\bthis feels off\b",
    r"\bcontradiction\b",
    r"\binconsistent\b",
    r"\bnot making progress\b",
]

SUBGOAL_PATTERNS = [
    r"\bsubgoal\b",
    r"\bfirst I need to\b",
    r"\bwhat I need first\b",
    r"\bidentify the missing piece\b",
    r"\bbreak this into\b",
    r"\breduce this to\b",
]

BACKWARD_PATTERNS = [
    r"\bwork backward\b",
    r"\bworking backward\b",
    r"\bfrom the target\b",
    r"\bmust be true\b",
    r"\bif the answer were\b",
    r"\breverse[- ]engineer\b",
]

UNCERTAINTY_PATTERNS = [
    r"\bconfidence\b",
    r"\bnot sure\b",
    r"\buncertain\b",
    r"\bdoubt\b",
    r"\btoo quickly\b",
    r"\boverconfiden",
    r"\bcalibration gap\b",
    r"\bfeels off\b",
    r"\banomaly\b",
]

DIAGNOSIS_PATTERNS = [
    r"\broute is weak\b",
    r"\bcurrent route\b",
    r"\bmissing\b",
    r"\bI am not tracking\b",
    r"\bthis does not explain\b",
    r"\bnot exposing the constraint\b",
]


def load_targets(path: Path) -> list[dict]:
    with path.open() as f:
        payload = json.load(f)
    return payload["models"]


def meta_blocks(text: str) -> list[str]:
    return re.findall(rf"{re.escape(META_START)}(.*?){re.escape(META_END)}", text, re.DOTALL)


def count_any(patterns: list[str], text: str) -> int:
    return sum(bool(re.search(pattern, text, re.IGNORECASE)) for pattern in patterns)


def extract_confidences(text: str) -> list[float]:
    values = []
    for raw in re.findall(r"confidence[:\s]+(\d+\.\d+|\d+)", text, re.IGNORECASE):
        value = float(raw)
        if value > 1.0:
            value /= 100.0
        values.append(value)
    return values


def first_match(patterns: list[str], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def benchmark_group(results: list[dict]) -> dict[tuple[str, str], list[dict]]:
    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in results:
        grouped.setdefault((row["model"], row["benchmark"]), []).append(row)
    return grouped


def analyze_result(model_name: str, row: dict) -> dict:
    completion = row.get("completion", "") or ""
    blocks = meta_blocks(completion)
    meta_text = "\n".join(blocks)
    confidences = extract_confidences(meta_text)
    conf_drop = False
    if len(confidences) >= 2:
        conf_drop = confidences[-1] < confidences[0] - 1e-6
    verify = first_match(VERIFY_PATTERNS, meta_text)
    backtrack = first_match(BACKTRACK_PATTERNS, meta_text)
    subgoal = first_match(SUBGOAL_PATTERNS, meta_text)
    backward = first_match(BACKWARD_PATTERNS, meta_text)
    uncertainty = first_match(UNCERTAINTY_PATTERNS, meta_text)
    diagnosis = first_match(DIAGNOSIS_PATTERNS, meta_text)
    epistemic_behavior = uncertainty and (verify or backtrack or subgoal or backward)
    procedural_behavior = (verify or backtrack or subgoal or backward) and not uncertainty
    return {
        "model": model_name,
        "benchmark": row.get("benchmark"),
        "is_correct": bool(row.get("is_correct")),
        "avg_confidence": row.get("avg_confidence"),
        "num_meta_blocks": int(row.get("num_meta_blocks", 0)),
        "verify": int(verify),
        "backtrack_redirect": int(backtrack),
        "subgoal": int(subgoal),
        "backward": int(backward),
        "uncertainty": int(uncertainty),
        "diagnosis": int(diagnosis),
        "confidence_drop": int(conf_drop),
        "epistemic_behavior": int(epistemic_behavior),
        "procedural_only_behavior": int(procedural_behavior),
        "question": row.get("question", ""),
        "completion": completion,
    }


def summarize(rows: list[dict]) -> list[dict]:
    summaries = []
    for (model, benchmark), group in benchmark_group(rows).items():
        n = len(group)
        correct = sum(r["is_correct"] for r in group)
        confs = [r["avg_confidence"] for r in group if r["avg_confidence"] is not None]
        verify_rate = sum(r["verify"] for r in group) / n
        backtrack_rate = sum(r["backtrack_redirect"] for r in group) / n
        subgoal_rate = sum(r["subgoal"] for r in group) / n
        backward_rate = sum(r["backward"] for r in group) / n
        uncertainty_rate = sum(r["uncertainty"] for r in group) / n
        diagnosis_rate = sum(r["diagnosis"] for r in group) / n
        confidence_drop_rate = sum(r["confidence_drop"] for r in group) / n
        epistemic_behavior_rate = sum(r["epistemic_behavior"] for r in group) / n
        procedural_only_rate = sum(r["procedural_only_behavior"] for r in group) / n
        acc = correct / n if n else 0.0
        mean_conf = sum(confs) / len(confs) if confs else None
        calibration_gap = abs(mean_conf - acc) if mean_conf is not None else None
        summaries.append(
            {
                "model": model,
                "benchmark": benchmark,
                "n": n,
                "accuracy": round(acc, 4),
                "mean_confidence": None if mean_conf is None else round(mean_conf, 4),
                "calibration_gap": None if calibration_gap is None else round(calibration_gap, 4),
                "verify_rate": round(verify_rate, 4),
                "backtrack_redirect_rate": round(backtrack_rate, 4),
                "subgoal_rate": round(subgoal_rate, 4),
                "backward_rate": round(backward_rate, 4),
                "uncertainty_rate": round(uncertainty_rate, 4),
                "diagnosis_rate": round(diagnosis_rate, 4),
                "confidence_drop_rate": round(confidence_drop_rate, 4),
                "epistemic_behavior_rate": round(epistemic_behavior_rate, 4),
                "procedural_only_rate": round(procedural_only_rate, 4),
            }
        )
    return sorted(summaries, key=lambda x: (x["model"], x["benchmark"]))


def write_csv(rows: list[dict], path: Path) -> None:
    import csv

    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_examples(rows: list[dict], path: Path) -> None:
    interesting = [
        r
        for r in rows
        if r["benchmark"] == "aime2024" and (r["epistemic_behavior"] or r["backtrack_redirect"] or r["verify"])
    ]
    by_model = Counter(r["model"] for r in interesting)
    selected = []
    for model, _ in by_model.most_common():
        picked = next((r for r in interesting if r["model"] == model), None)
        if picked is not None:
            selected.append(picked)

    lines = ["# Behavior-Uncertainty Examples", ""]
    for row in selected:
        lines.extend(
            [
                f"## {row['model']} / {row['benchmark']}",
                f"- correct: {row['is_correct']}",
                f"- verify: {row['verify']}",
                f"- backtrack_redirect: {row['backtrack_redirect']}",
                f"- subgoal: {row['subgoal']}",
                f"- backward: {row['backward']}",
                f"- uncertainty: {row['uncertainty']}",
                f"- diagnosis: {row['diagnosis']}",
                f"- confidence_drop: {row['confidence_drop']}",
                "",
                "Question:",
                row["question"][:500],
                "",
                "Completion:",
                row["completion"][:2500],
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for target in load_targets(Path(args.targets)):
        with Path(target["eval_json"]).open() as f:
            payload = json.load(f)
        for row in payload["results"]:
            all_rows.append(analyze_result(target["name"], row))

    summaries = summarize(all_rows)
    (outdir / "behavior_uncertainty_items.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(summaries, outdir / "behavior_uncertainty_summary.csv")
    write_examples(all_rows, outdir / "behavior_uncertainty_examples.md")
    print(f"items={len(all_rows)} summaries={len(summaries)} outdir={outdir}")


if __name__ == "__main__":
    main()
