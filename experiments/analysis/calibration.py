#!/usr/bin/env python3
"""Calibration metrics from STATED confidence in meta blocks (RQ4 / T4).

Pure calibration only — the OOD-vs-ID split belongs to RQ3 / T3 (stratify.py
and the T3 section of aggregate_tables.py). Accuracy stays the primary paper
metric; calibration is secondary.

Confidence is extracted ONLY from what the model itself stated inside CLOSED
<|meta|>...<|/meta|> blocks (regex on "confidence"/"probability" phrases, e.g.
"Confidence: 0.90", "probability of solving it correctly is about 0.40",
"confidence is 85%"). The gold answer is used for exactly one thing: grading
correctness with math_verify. It is NEVER used to impute, select, or repair a
confidence value — rows without a stated confidence are excluded from
ECE/Brier/overconfidence and reported as (1 - coverage).

Metrics per arm, per benchmark and pooled:
  - accuracy        mean correctness over ALL rows (math_verify regrade)
  - coverage        fraction of rows with >= 1 stated confidence
  - mean conf       mean stated confidence (rows with confidence)
  - ECE (15-bin)    equal-width-bin expected calibration error
  - Brier           mean (confidence - correct)^2
  - overconf rate   P(wrong | confidence >= --high-conf, default 0.8)

--which picks which stated value represents the response: `last` (default;
the confidence closest to the final answer), `first` (pre-solve assessment),
or `mean`.

Usage:
  python experiments/analysis/calibration.py \\
      --arm base=results/eval_1030_base_grpo_step300_16k/base_grpo_step300_16k.parquet \\
      --arm pmishift=results/eval_1030_pmishift_gs300/pmishift_gs300.parquet \\
      --out results/analysis/calibration_t6.md \\
      --json-out results/analysis/calibration_t6.json

Accepts parquet / json / jsonl outputs of scripts/eval_vllm_1030.py and
src/eval/eval_hf.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.analysis.analysis_common import (  # noqa: E402
    Grader,
    brier,
    ece_15bin,
    fmt_pct,
    load_eval_frame,
    md_table,
    overconfidence_rate,
    parse_arm_specs,
    resolve_correct,
    stated_confidences,
)


def pick_confidence(vals: list[float], which: str) -> float | None:
    if not vals:
        return None
    if which == "first":
        return vals[0]
    if which == "mean":
        return float(np.mean(vals))
    return vals[-1]  # last


def scope_metrics(sub, bins: int, high_conf: float) -> dict:
    """Compute the metric block for one (arm, benchmark-or-pooled) slice."""
    stated = sub[sub["conf"].notna()]
    conf = stated["conf"].to_numpy(dtype=float)
    corr = stated["correct"].to_numpy(dtype=bool)
    return {
        "n": int(len(sub)),
        "accuracy": float(sub["correct"].mean()) if len(sub) else float("nan"),
        "coverage": float(len(stated) / len(sub)) if len(sub) else float("nan"),
        "mean_confidence": float(conf.mean()) if len(conf) else float("nan"),
        "ece": ece_15bin(conf, corr, n_bins=bins),
        "brier": brier(conf, corr),
        "overconfidence_rate": overconfidence_rate(conf, corr, high_conf),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--arm", action="append", required=True, metavar="NAME=PATH",
        help="arm name and its eval output file (repeatable)",
    )
    parser.add_argument("--bins", type=int, default=15, help="ECE bin count")
    parser.add_argument(
        "--which", choices=["first", "last", "mean"], default="last",
        help="which stated confidence represents the response",
    )
    parser.add_argument(
        "--high-conf", type=float, default=0.8,
        help="threshold for the overconfidence rate P(wrong | conf >= t)",
    )
    parser.add_argument(
        "--no-regrade", action="store_true",
        help="trust stored is_correct instead of math_verify regrade (NOT for paper)",
    )
    parser.add_argument("--out", default=None, help="write the markdown report here")
    parser.add_argument("--json-out", default=None, help="write metrics JSON here")
    args = parser.parse_args()

    grader = Grader()
    header = ["arm", "scope", "n", "acc", "coverage", "mean conf",
              f"ECE({args.bins})", "Brier", f"overconf@{args.high_conf:g}"]
    rows: list[list[str]] = []
    payload: dict[str, dict] = {}

    for name, path in parse_arm_specs(args.arm):
        df = load_eval_frame(path)
        df["correct"] = resolve_correct(df, args.no_regrade, "calibration", grader)
        df["conf"] = df["completion"].map(
            lambda t: pick_confidence(stated_confidences(t), args.which)
        )
        scopes = [(b, df[df["benchmark"] == b]) for b in sorted(df["benchmark"].unique())]
        scopes.append(("POOLED", df))
        payload[name] = {}
        for scope, sub in scopes:
            m = scope_metrics(sub, args.bins, args.high_conf)
            payload[name][scope] = m
            rows.append([
                name, scope, m["n"], fmt_pct(m["accuracy"]),
                fmt_pct(m["coverage"]),
                "n/a" if np.isnan(m["mean_confidence"]) else f"{m['mean_confidence']:.3f}",
                "n/a" if np.isnan(m["ece"]) else f"{m['ece']:.3f}",
                "n/a" if np.isnan(m["brier"]) else f"{m['brier']:.3f}",
                fmt_pct(m["overconfidence_rate"]),
            ])

    lines = [
        "## Calibration from stated meta-block confidence",
        "",
        f"- confidence source = CLOSED meta blocks only, `--which {args.which}`; "
        "gold is used only to grade correctness",
        f"- grading = {'STORED is_correct (smoke only)' if args.no_regrade else 'math_verify regrade'}",
        "- rows without a stated confidence are excluded from ECE/Brier/overconf "
        "(see coverage)",
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
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Wrote JSON: {args.json_out}")


if __name__ == "__main__":
    main()
