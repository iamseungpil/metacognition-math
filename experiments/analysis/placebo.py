#!/usr/bin/env python3
"""Build shuffled-meta PLACEBO variants of eval responses (RQ2 / T2 control).

PROTOCOL
  Question: are the SAVE/DERAIL flips measured by flip.py caused by the
  CONTENT of the meta blocks, or are they an artifact of position / answer
  identity (a documented confound: PMI-toward-own-answer makes any block near
  the answer look causal)?

  Construction (generation-free):
    1. Collect every response that contains >= 1 CLOSED <|meta|>...<|/meta|>
       block. Non-meta responses pass through unchanged (kept for bookkeeping,
       flagged placebo_applied = false).
    2. Shuffle donor assignment: meta-bearing responses are randomly permuted
       (fixed --seed) and each response i receives meta content from a DONOR
       response at a rotated position, guaranteeing donor != self. With
       --swap-scope benchmark the donor comes from the same benchmark (keeps
       domain-plausible content); with --swap-scope global any problem can
       donate.
    3. Block j of response i is replaced by donor block j (donor blocks are
       reused cyclically when the donor has fewer blocks). All non-meta text —
       including the final boxed answer — is byte-identical to the original.

  Analysis (all generation-free re-grading, no model calls):
    a. Sanity: re-grade `completion` (the placebo text) with math_verify —
       accuracy must be IDENTICAL to the original file, because grading only
       reads the boxed answer outside meta blocks. Any difference exposes a
       grader artifact, not a mechanism.
    b. Main control: run experiments/analysis/flip.py on this jsonl and on the
       original file. flip.py grades prefixes that END where meta blocks start,
       so prefix states are unchanged; what the placebo tests downstream is the
       ATTRIBUTION analysis (e.g. conditioning flips on meta-content features,
       confidence statements, verification phrases). If a content-conditioned
       effect survives content shuffling, it is positional artifact, not
       mechanism.
    c. Optional continuation re-generation (NOT generation-free): each record
       carries `placebo_prefix` = original text up to and including the LAST
       swapped meta block; feeding prompt + placebo_prefix back into the model
       measures whether shuffled meta content derails the continuation. That
       step needs GPU generation and is out of scope for this script.

  The output jsonl keeps the normalized eval schema (`completion` holds the
  PLACEBO text) so flip.py / stratify.py / calibration.py consume it directly.

Usage:
  python experiments/analysis/placebo.py \\
      --input results/eval_1030_meta_grpo_e21r_v2_step300_16k/meta_grpo_e21r_v2_step300_16k.parquet \\
      --output results/analysis/placebo_meta_gs300.jsonl \\
      --seed 1234 --swap-scope benchmark
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.analysis.analysis_common import (  # noqa: E402
    META_END,
    META_START,
    load_eval_frame,
    meta_block_spans,
)


def rebuild_with_meta(completion: str, donor_blocks: list[str]) -> str:
    """Replace each closed meta block's content with donor content (cyclic)."""
    spans = meta_block_spans(completion)
    out, cursor = [], 0
    for j, (start, end, _inner) in enumerate(spans):
        out.append(completion[cursor:start])
        out.append(META_START + donor_blocks[j % len(donor_blocks)] + META_END)
        cursor = end
    out.append(completion[cursor:])
    return "".join(out)


def derangement_donor(indices: list[int], rng: random.Random) -> dict[int, int]:
    """Map each index to a donor index with no fixed points (shuffle + rotate)."""
    if len(indices) < 2:
        return {i: i for i in indices}  # degenerate; caller flags it
    order = indices[:]
    rng.shuffle(order)
    return {order[k]: order[(k + 1) % len(order)] for k in range(len(order))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="eval output file")
    parser.add_argument("--output", required=True, help="placebo jsonl path")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--swap-scope", choices=["benchmark", "global"], default="benchmark",
        help="donor pool: same benchmark (default) or the whole file",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    df = load_eval_frame(args.input)
    completions = df["completion"].tolist()
    blocks = [
        [inner for _, _, inner in meta_block_spans(c)] for c in completions
    ]
    meta_idx = [i for i, b in enumerate(blocks) if b]

    # Build the no-self-donor map inside each swap scope.
    donor_of: dict[int, int] = {}
    if args.swap_scope == "benchmark":
        for bench in sorted(df["benchmark"].unique()):
            scope = [i for i in meta_idx if df.iloc[i]["benchmark"] == bench]
            donor_of.update(derangement_donor(scope, rng))
    else:
        donor_of.update(derangement_donor(meta_idx, rng))

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    n_swapped = n_self = 0
    with open(args.output, "w") as f:
        for i, row in df.reset_index(drop=True).iterrows():
            donor = donor_of.get(i)
            applied = donor is not None and donor != i and bool(blocks[donor])
            if donor is not None and donor == i:
                n_self += 1  # scope had a single meta response; nothing to swap
            placebo = (
                rebuild_with_meta(row["completion"], blocks[donor])
                if applied else row["completion"]
            )
            spans = meta_block_spans(placebo)
            record = {
                "benchmark": row["benchmark"],
                "qid": row["qid"],
                "question": row["question"],
                "gold_answer": row["gold_answer"],
                "sample_idx": int(row["sample_idx"]),
                # `completion` = placebo text so downstream scripts read it as-is.
                "completion": placebo,
                "original_completion": row["completion"],
                "is_correct": bool(row["is_correct"]),  # stored grade, original text
                "placebo_applied": bool(applied),
                "donor_qid": df.iloc[donor]["qid"] if applied else None,
                "num_meta_blocks": len(spans),
                # prefix through the LAST swapped block, for optional regeneration
                "placebo_prefix": placebo[: spans[-1][1]] if spans else "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            n_swapped += int(applied)

    sidecar = {
        "input": str(args.input),
        "output": str(args.output),
        "seed": args.seed,
        "swap_scope": args.swap_scope,
        "n_rows": int(len(df)),
        "n_meta_rows": len(meta_idx),
        "n_swapped": n_swapped,
        "n_unswappable_singleton_scope": n_self,
    }
    sidecar_path = args.output + ".meta.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2)

    print(json.dumps(sidecar, indent=2))
    print(f"Wrote placebo jsonl: {args.output}")
    print(f"Wrote sidecar:       {sidecar_path}")
    print("Next: run flip.py on both the original file and this jsonl and "
          "compare SAVE/DERAIL — content-driven effects must vanish here.")


if __name__ == "__main__":
    main()
