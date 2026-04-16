#!/usr/bin/env python3
"""Compare eval bundles with self-distill-focused epistemic metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.self_distill.eval_metrics import load_eval_table, summarize_eval_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baseline = summarize_eval_table(load_eval_table(args.baseline))
    candidate = summarize_eval_table(load_eval_table(args.candidate))
    delta = {}
    for key, base_value in baseline.items():
        cand_value = candidate.get(key)
        if isinstance(base_value, (int, float)) and isinstance(cand_value, (int, float)):
            delta[key] = cand_value - base_value

    payload = {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
