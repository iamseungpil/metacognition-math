#!/usr/bin/env python3
"""Build a self-distillation parquet for naive or epistemic-preserving lanes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.self_distill_data import (
    SUPPORTED_SELF_DISTILL_MODES,
    build_self_distill_dataset,
    summarize_self_distill_dataframe,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input parquet/json/jsonl artifact")
    parser.add_argument("--output", required=True, help="Output parquet path")
    parser.add_argument("--mode", choices=list(SUPPORTED_SELF_DISTILL_MODES), required=True)
    parser.add_argument("--source-tag", default=None)
    parser.add_argument("--allow-missing-boxed", action="store_true")
    parser.add_argument("--summary-json", default=None)
    args = parser.parse_args()

    df = build_self_distill_dataset(
        args.input,
        mode=args.mode,
        source_tag=args.source_tag,
        require_boxed=not args.allow_missing_boxed,
    )
    if df.empty:
        raise ValueError(
            "Self-distill build produced 0 rows. Refusing to write an unusable parquet."
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    summary = summarize_self_distill_dataframe(df)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.summary_json:
        summary_path = Path(args.summary_json)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
