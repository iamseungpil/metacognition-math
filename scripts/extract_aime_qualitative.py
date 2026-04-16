#!/usr/bin/env python3
"""Extract qualitative AIME cases from an eval JSON bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize_case(row: dict) -> list[str]:
    confs = row.get("meta_confidences", []) or []
    conf_str = ", ".join(f"{x:.2f}" for x in confs) if confs else "N/A"
    return [
        f"- correct: {row.get('is_correct')}",
        f"- num_meta_blocks: {row.get('num_meta_blocks')}",
        f"- avg_confidence: {row.get('avg_confidence')}",
        f"- meta_confidences: {conf_str}",
        f"- answer_extracted: {row.get('answer_extracted', '')}",
        f"- gold_answer: {row.get('full_gold_answer', row.get('gold_answer', ''))}",
        "",
        "Question:",
        str(row.get("full_question", row.get("question", "")))[:1500],
        "",
        "Completion:",
        str(row.get("completion", ""))[:5000],
        "",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_wrong", type=int, default=8)
    parser.add_argument("--max_correct", type=int, default=4)
    args = parser.parse_args()

    payload = json.loads(Path(args.eval_json).read_text())
    results = payload.get("results", [])
    aime = [r for r in results if r.get("benchmark") == "aime2024"]

    wrong = [r for r in aime if not r.get("is_correct", False)][: args.max_wrong]
    correct = [r for r in aime if r.get("is_correct", False)][: args.max_correct]

    lines = [
        f"# AIME Qualitative Cases: {payload.get('model', Path(args.eval_json).stem)}",
        "",
        f"- total_aime_cases: {len(aime)}",
        f"- wrong_selected: {len(wrong)}",
        f"- correct_selected: {len(correct)}",
        "",
        "## Wrong Cases",
        "",
    ]
    for idx, row in enumerate(wrong, start=1):
        lines.append(f"### Wrong {idx}")
        lines.extend(summarize_case(row))

    lines.extend(["## Correct Cases", ""])
    for idx, row in enumerate(correct, start=1):
        lines.append(f"### Correct {idx}")
        lines.extend(summarize_case(row))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
