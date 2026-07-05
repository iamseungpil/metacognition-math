#!/usr/bin/env python3
"""SAVE / DERAIL flip analysis over eval outputs (RQ2 / T2 mechanism part).

For every response that contains at least one CLOSED <|meta|>...<|/meta|> block,
the completion is split at the START of each meta block into a sequence of
checkpoints:

    prefix_0 (text before block 1) -> prefix_1 (before block 2) -> ... -> full

Each checkpoint is graded with math_verify against the gold answer (the stored
is_correct column is never used here — it comes from the legacy broken
check_correctness path). A checkpoint's state is one of:

    no_answer  — no \\boxed{...} answer emitted yet
    right      — last boxed answer verifies against gold
    wrong      — last boxed answer does not verify

Row-level classification (prefix before FIRST meta block vs full response):
    SAVE       wrong  -> right   (meta segment rescued a committed wrong answer)
    DERAIL     right  -> wrong   (meta segment destroyed a committed right answer)
    KEPT_RIGHT right  -> right
    KEPT_WRONG wrong  -> wrong
    RESOLVED   no_answer -> right   (no committed answer before meta; not a SAVE)
    UNRESOLVED no_answer -> wrong

Block-level transitions (checkpoint i -> checkpoint i+1) are also counted so
multi-block responses attribute the flip to the specific meta segment.

Gold answers are used ONLY to grade correctness at each checkpoint.

Usage:
  python experiments/analysis/flip.py \\
      results/eval_1030_meta_grpo_e21r_v2_step300_16k/meta_grpo_e21r_v2_step300_16k.parquet \\
      --out results/analysis/flip_meta_gs300.json \\
      --per-row-out results/analysis/flip_meta_gs300.rows.jsonl

Accepts parquet / json / jsonl outputs of scripts/eval_vllm_1030.py and
src/eval/eval_hf.py, and the placebo jsonl from experiments/analysis/placebo.py
(whose `completion` column holds the shuffled-meta variant), so running flip.py
on real vs placebo files gives the T2 mechanism-vs-artifact comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.analysis.analysis_common import (  # noqa: E402
    Grader,
    extract_last_boxed,
    fmt_pct,
    load_eval_frame,
    md_table,
    meta_block_spans,
    warn_if_no_math_verify,
)

ROW_CATEGORIES = [
    "save", "derail", "kept_right", "kept_wrong", "resolved", "unresolved",
]


def checkpoint_state(text: str, gold: str, grader: Grader) -> str:
    """Grade one checkpoint: no_answer / right / wrong."""
    if extract_last_boxed(str(text)) is None:
        return "no_answer"
    return "right" if grader.grade(text, gold) else "wrong"


def classify(before: str, after: str) -> str:
    """Map a (state_before, state_after) transition to a flip category."""
    if before == "no_answer":
        return "resolved" if after == "right" else "unresolved"
    if before == "wrong":
        return "save" if after == "right" else "kept_wrong"
    # before == "right"
    return "kept_right" if after == "right" else "derail"


def analyze_row(completion: str, gold: str, grader: Grader) -> dict | None:
    """Return flip info for one response, or None if it has no closed block."""
    spans = meta_block_spans(completion)
    if not spans:
        return None
    checkpoints = [completion[: s[0]] for s in spans] + [completion]
    states = [checkpoint_state(t, gold, grader) for t in checkpoints]
    transitions = [
        classify(states[i], states[i + 1]) for i in range(len(states) - 1)
    ]
    return {
        "num_meta_blocks": len(spans),
        "states": states,
        "row_category": classify(states[0], states[-1]),
        "block_transitions": transitions,
        "final_correct": states[-1] == "right",
    }


def summarize(rows: list[dict]) -> dict:
    """Aggregate row/block category counts into rates."""
    n = len(rows)
    row_counts = {c: 0 for c in ROW_CATEGORIES}
    block_counts = {c: 0 for c in ROW_CATEGORIES}
    n_blocks = 0
    for r in rows:
        row_counts[r["row_category"]] += 1
        for t in r["block_transitions"]:
            block_counts[t] += 1
            n_blocks += 1
    committed = row_counts["save"] + row_counts["derail"] + \
        row_counts["kept_right"] + row_counts["kept_wrong"]
    return {
        "n_meta_rows": n,
        "n_blocks": n_blocks,
        "row_counts": row_counts,
        "block_counts": block_counts,
        # Rates over rows that had a COMMITTED (boxed) answer before the first
        # meta block — the population where save/derail is well-defined.
        "n_committed_prefix": committed,
        "save_rate": row_counts["save"] / committed if committed else None,
        "derail_rate": row_counts["derail"] / committed if committed else None,
        "net_save_rate": (
            (row_counts["save"] - row_counts["derail"]) / committed
            if committed else None
        ),
        "final_accuracy_meta_rows": (
            sum(r["final_correct"] for r in rows) / n if n else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("inputs", nargs="+", help="eval output file(s)")
    parser.add_argument("--out", default=None, help="write JSON summary here")
    parser.add_argument(
        "--per-row-out", default=None, help="write per-row flip records (jsonl)"
    )
    args = parser.parse_args()

    warn_if_no_math_verify("flip")
    grader = Grader()
    all_summaries: dict[str, dict] = {}
    per_row_f = open(args.per_row_out, "w") if args.per_row_out else None

    for path in args.inputs:
        df = load_eval_frame(path)
        per_bench_rows: dict[str, list[dict]] = {}
        n_no_meta = 0
        for _, row in df.iterrows():
            info = analyze_row(row["completion"], row["gold_answer"], grader)
            if info is None:
                n_no_meta += 1
                continue
            per_bench_rows.setdefault(row["benchmark"], []).append(info)
            if per_row_f is not None:
                per_row_f.write(json.dumps({
                    "source_file": str(path),
                    "benchmark": row["benchmark"],
                    "qid": row["qid"],
                    "sample_idx": int(row["sample_idx"]),
                    **info,
                }) + "\n")

        pooled = [r for rows in per_bench_rows.values() for r in rows]
        summary = {
            "file": str(path),
            "n_rows": int(len(df)),
            "n_rows_without_meta": int(n_no_meta),
            "pooled": summarize(pooled),
            "per_benchmark": {
                b: summarize(rows) for b, rows in sorted(per_bench_rows.items())
            },
        }
        all_summaries[str(path)] = summary

        # markdown report for this file
        header = ["scope", "meta rows", "committed", "SAVE", "DERAIL",
                  "net SAVE", "resolved", "unresolved"]
        table_rows = []
        scopes = list(sorted(per_bench_rows.items())) + [("POOLED", pooled)]
        for scope, rows in scopes:
            s = summarize(rows)
            table_rows.append([
                scope, s["n_meta_rows"], s["n_committed_prefix"],
                fmt_pct(s["save_rate"]), fmt_pct(s["derail_rate"]),
                fmt_pct(s["net_save_rate"]),
                s["row_counts"]["resolved"], s["row_counts"]["unresolved"],
            ])
        print(f"\n### Flip analysis — {path}")
        print(f"(rows without a closed meta block: {n_no_meta}/{len(df)})\n")
        print(md_table(header, table_rows))

    if per_row_f is not None:
        per_row_f.close()
        print(f"\nWrote per-row records: {args.per_row_out}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"Wrote summary JSON: {args.out}")


if __name__ == "__main__":
    main()
