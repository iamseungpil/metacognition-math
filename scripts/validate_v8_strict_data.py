#!/usr/bin/env python3
"""Validate strict paired Meta/Base SFT data."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v8_strict_paired_data import (
    BLANK_INLINE_MATH_RE,
    extract_last_boxed,
    is_strong_redirect,
    is_strong_verify,
    load_messages,
    parse_assistant,
)

BASE_FORBIDDEN_SNIPPETS = [
    "A first thought is",
    "A tempting first thought is",
    "At first glance",
    "I might try",
    "one might try",
    "I should switch",
    "I should redirect",
    "What is missing is",
    "study_need:",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate strict paired V8 data")
    parser.add_argument("--meta", default="data/v8_meta_inside_strict.parquet")
    parser.add_argument("--base", default="data/v8_base_matched_strict.parquet")
    parser.add_argument("--summary-json", default="results/strict_data/v8_strict_validation_summary.json")
    args = parser.parse_args()

    root = ROOT
    meta_path = root / args.meta
    base_path = root / args.base
    summary_path = root / args.summary_json

    meta_df = pd.read_parquet(meta_path)
    base_df = pd.read_parquet(base_path)
    if len(meta_df) != len(base_df):
        raise ValueError(f"paired length mismatch: {len(meta_df)} vs {len(base_df)}")

    scenario_counts = Counter()
    failures = Counter()

    for idx in range(len(meta_df)):
        meta_row = meta_df.iloc[idx]
        base_row = base_df.iloc[idx]
        scenario = str(meta_row.get("scenario", "")).lower()
        scenario_counts[scenario] += 1

        meta_msgs = load_messages(meta_row["messages"])
        base_msgs = load_messages(base_row["messages"])
        if meta_msgs[0]["content"] != base_msgs[0]["content"]:
            failures["user_mismatch"] += 1
            continue

        meta_assistant = str(meta_msgs[1]["content"])
        base_assistant = str(base_msgs[1]["content"])
        meta_boxed = extract_last_boxed(parse_assistant(meta_assistant)[1])
        base_boxed = extract_last_boxed(base_assistant)

        if meta_boxed != base_boxed:
            failures["boxed_mismatch"] += 1
        if scenario == "verify" and not is_strong_verify(meta_assistant):
            failures["verify_meta_invalid"] += 1
        if scenario == "redirect" and not is_strong_redirect(meta_assistant):
            failures["redirect_meta_invalid"] += 1
        if BLANK_INLINE_MATH_RE.search(meta_assistant):
            failures["meta_blank_inline_math"] += 1
        if "<|meta|>" in base_assistant or "<|/meta|>" in base_assistant:
            failures["base_meta_leak"] += 1
        if base_assistant.count("<think>") != 1 or base_assistant.count("</think>") != 1:
            failures["base_think_envelope"] += 1
        if "The answer is $\\boxed{" not in base_assistant:
            failures["base_answer_format"] += 1
        if any(snippet in base_assistant for snippet in BASE_FORBIDDEN_SNIPPETS):
            failures["base_route_leak"] += 1
        if BLANK_INLINE_MATH_RE.search(base_assistant):
            failures["base_blank_inline_math"] += 1

    summary = {
        "meta_rows": int(len(meta_df)),
        "base_rows": int(len(base_df)),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "failures": dict(sorted(failures.items())),
        "passed": not failures,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
