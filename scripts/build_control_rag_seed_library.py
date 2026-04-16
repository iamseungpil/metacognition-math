#!/usr/bin/env python3
"""Build a typed stable seed library for control-RAG from prior eval artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curriculum.control_rag import ExampleRecord, load_example_bank  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="artifact paths to scan")
    parser.add_argument("--output", required=True, help="output JSON file")
    parser.add_argument("--require_correct", action="store_true", help="keep only correct exemplars")
    parser.add_argument("--require_study_need", action="store_true", help="keep only typed exemplars")
    parser.add_argument("--min_per_family", type=int, default=1, help="drop families with fewer records than this")
    args = parser.parse_args()

    records = load_example_bank([Path(p) for p in args.inputs], require_solution=True)
    filtered: list[ExampleRecord] = []
    for record in records:
        meta = record.metadata or {}
        if args.require_correct and not meta.get("is_correct", True):
            continue
        if args.require_study_need and not meta.get("study_need"):
            continue
        filtered.append(
            ExampleRecord(
                question=record.question,
                solution=record.solution,
                answer=record.answer,
                source="stable_seed_library",
                metadata={
                    **meta,
                    "source": "stable_seed_library",
                    "source_artifact": meta.get("source", record.source),
                    "source_role": "stable_seed",
                },
            )
        )

    deduped: list[ExampleRecord] = []
    seen: set[tuple[str, str]] = set()
    family_counts: dict[str, int] = {}
    for record in filtered:
        meta = record.metadata or {}
        key = (record.question.strip(), str(meta.get("study_need", "")).strip())
        if key in seen:
            continue
        seen.add(key)
        family = str(meta.get("study_need_family", "")).strip() or "untyped"
        family_counts[family] = family_counts.get(family, 0) + 1
        deduped.append(record)

    kept = [
        record for record in deduped
        if family_counts.get(str((record.metadata or {}).get("study_need_family", "")).strip() or "untyped", 0) >= args.min_per_family
    ]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_dict() for record in kept]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "num_loaded": len(records),
        "num_filtered": len(filtered),
        "num_deduped": len(deduped),
        "num_written": len(kept),
        "families": family_counts,
        "output": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
