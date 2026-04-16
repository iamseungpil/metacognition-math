#!/usr/bin/env python3
"""Build a dynamic success library from saved RQ3 trace artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curriculum.rq3_pipeline import build_dynamic_library_from_trace_dicts  # noqa: E402


def load_rows(path: Path) -> list[dict]:
    if path.suffix == ".parquet":
        return pd.read_parquet(path).to_dict(orient="records")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "results" in payload:
            return list(payload["results"])
        if isinstance(payload, list):
            return payload
    raise ValueError(f"unsupported trace format: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", nargs="+", required=True, help="RQ3 trace artifact(s)")
    parser.add_argument("--output", required=True, help="output JSON file")
    parser.add_argument("--min_confidence_gain", type=float, default=0.10)
    args = parser.parse_args()

    rows: list[dict] = []
    for trace_path in args.traces:
        rows.extend(load_rows(Path(trace_path)))

    records = build_dynamic_library_from_trace_dicts(
        rows,
        min_confidence_gain=args.min_confidence_gain,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [record.to_dict() for record in records]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    by_lane: dict[str, int] = {}
    for record in records:
        lane = str((record.metadata or {}).get("from_lane", ""))
        by_lane[lane] = by_lane.get(lane, 0) + 1
    summary = {
        "num_rows_loaded": len(rows),
        "num_records_written": len(records),
        "by_lane": by_lane,
        "output": str(output_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
