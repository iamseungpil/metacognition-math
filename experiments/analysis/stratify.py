#!/usr/bin/env python3
"""Difficulty-quartile stratified accuracy + meta-emission per arm (RQ3 / T3).

Simpson guard: this stratification is what keeps aggregate accuracy honest.

The 1030 benchmark has no per-problem difficulty labels, so difficulty is
estimated EMPIRICALLY and POOLED ACROSS ALL ARMS: for each question, difficulty
= 1 - mean(correct) over every sample of every arm passed on the CLI. Questions
are then rank-split into equal-count quartiles Q1 (easiest) .. Q4 (hardest).
Pooling is what prevents the documented Simpson/selection artifact — binning by
one arm's own accuracy makes that arm look artificially good in its hard bin.

Per arm x quartile the script reports:
  - accuracy: re-graded with math_verify (metric rule 2; pass --no-regrade to
    trust the stored is_correct, which is known-broken — smoke tests only)
  - meta emission rate: fraction of responses containing at least one CLOSED
    <|meta|>...<|/meta|> block (metric rule 5; the stored num_meta_blocks may
    count a free-text fallback and is ignored)

Usage:
  python experiments/analysis/stratify.py \\
      --arm base=results/eval_1030_base_grpo_step300_16k/base_grpo_step300_16k.parquet \\
      --arm pmishift=results/eval_1030_pmishift_gs300/pmishift_gs300.parquet \\
      --out results/analysis/stratify_t3.md

Accepts parquet / json / jsonl outputs of scripts/eval_vllm_1030.py and
src/eval/eval_hf.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.analysis.analysis_common import (  # noqa: E402
    Grader,
    fmt_pct,
    load_eval_frame,
    md_table,
    parse_arm_specs,
    resolve_correct,
)


def assign_quartiles(df: pd.DataFrame, n_bins: int) -> pd.Series:
    """Map qid -> quartile label from POOLED per-question difficulty.

    Rank-based equal-count binning (ties broken deterministically by qid) so
    every bin is populated even when difficulty is heavily tied (e.g. many
    questions solved by every sample of every arm).
    """
    per_q = (
        df.groupby("qid")["correct"].mean().rename("acc").reset_index()
    )
    per_q["difficulty"] = 1.0 - per_q["acc"]
    per_q = per_q.sort_values(["difficulty", "qid"]).reset_index(drop=True)
    per_q["bin"] = (np.arange(len(per_q)) * n_bins // len(per_q)).astype(int)
    per_q["quartile"] = per_q["bin"].map(lambda b: f"Q{b + 1}")
    return per_q.set_index("qid")["quartile"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm", action="append", required=True, metavar="NAME=PATH",
        help="arm name and its eval output file (repeatable)",
    )
    parser.add_argument("--quartiles", type=int, default=4)
    parser.add_argument(
        "--no-regrade", action="store_true",
        help="trust stored is_correct instead of math_verify regrade (NOT for paper)",
    )
    parser.add_argument("--out", default=None, help="write the markdown table here")
    args = parser.parse_args()

    grader = Grader()
    frames = []
    for name, path in parse_arm_specs(args.arm):
        df = load_eval_frame(path)
        df["arm"] = name
        df["correct"] = resolve_correct(df, args.no_regrade, "stratify", grader)
        df["has_meta"] = df["num_meta_blocks_closed"] > 0
        frames.append(df)
    pooled = pd.concat(frames, ignore_index=True)

    quartile_of = assign_quartiles(pooled, args.quartiles)
    pooled["quartile"] = pooled["qid"].map(quartile_of)

    arms = [name for name, _ in parse_arm_specs(args.arm)]
    labels = [f"Q{i + 1}" for i in range(args.quartiles)]

    header = ["quartile", "n questions"]
    for arm in arms:
        header += [f"{arm} acc", f"{arm} meta%"]

    rows = []
    for label in labels + ["ALL"]:
        sub = pooled if label == "ALL" else pooled[pooled["quartile"] == label]
        row = [label, int(sub["qid"].nunique())]
        for arm in arms:
            asub = sub[sub["arm"] == arm]
            acc = asub["correct"].mean() if len(asub) else float("nan")
            meta = asub["has_meta"].mean() if len(asub) else float("nan")
            row += [fmt_pct(acc), fmt_pct(meta)]
        rows.append(row)

    lines = [
        "## Difficulty-stratified accuracy and meta emission",
        "",
        f"- difficulty = 1 - pooled accuracy over all arms/samples "
        f"({pooled['qid'].nunique()} questions, arms: {', '.join(arms)})",
        f"- quartiles = equal-count rank bins, Q1 easiest .. Q{args.quartiles} hardest",
        f"- grading = {'STORED is_correct (smoke only)' if args.no_regrade else 'math_verify regrade'}",
        "- meta emission = closed <|meta|>...<|/meta|> blocks only",
        "",
        md_table(header, rows),
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
        print(f"Wrote markdown: {args.out}")


if __name__ == "__main__":
    main()
