#!/usr/bin/env python3
"""Run RQ3 side-evidence evaluation over saved root completions.

This script is intentionally offline-first. It expects precomputed root
completions and optional intervention completions so that RQ3 can be audited
without coupling it to a single generation stack.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curriculum.control_rag import load_example_bank, TfidfExampleRetriever
from src.curriculum.rq3_pipeline import evaluate_rq3_case, summarize_rq3_results


def load_cases(path: Path) -> list[dict[str, Any]]:
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
    raise ValueError(f"unsupported case format: {path}")


def parse_branch_completions(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
    return None


def extract_plain_retry_completion(row: dict[str, Any]) -> str | None:
    value = row.get("plain_retry_completion")
    if isinstance(value, str) and value.strip():
        return value
    nested = row.get("plain_retry", {}) or {}
    nested_value = nested.get("completion")
    if isinstance(nested_value, str) and nested_value.strip():
        return nested_value
    return None


def extract_curriculum_retry_completion(row: dict[str, Any]) -> str | None:
    value = row.get("curriculum_retry_completion")
    if isinstance(value, str) and value.strip():
        return value
    nested = row.get("curriculum_retry", {}) or {}
    nested_value = nested.get("retry_completion")
    if isinstance(nested_value, str) and nested_value.strip():
        return nested_value
    return None


def extract_branch_completions(row: dict[str, Any]) -> list[str] | None:
    direct = parse_branch_completions(row.get("branch_completions"))
    if direct:
        return direct
    nested = row.get("selective_branching", {}) or {}
    branches = nested.get("branches", [])
    if not isinstance(branches, list) or not branches:
        return None
    completions = []
    for branch in branches:
        if not isinstance(branch, dict):
            continue
        completion = branch.get("completion")
        if isinstance(completion, str) and completion.strip():
            completions.append(completion)
    return completions or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="json/jsonl/parquet with root completions")
    parser.add_argument("--example_bank", nargs="+", default=[], help="example-bank files for retrieval provenance")
    parser.add_argument("--output_dir", default="results/rq3_side_eval")
    args = parser.parse_args()

    cases_path = Path(args.cases)
    rows = load_cases(cases_path)
    retriever = None
    bank_summary = {"num_records": 0, "by_source": {}, "by_role": {}}
    if args.example_bank:
        records = load_example_bank([Path(p) for p in args.example_bank], require_solution=True)
        retriever = TfidfExampleRetriever(records)
        bank_summary["num_records"] = len(records)
        for record in records:
            meta = record.metadata or {}
            source = str(record.source or meta.get("source", ""))
            role = str(meta.get("source_role", "unspecified"))
            bank_summary["by_source"][source] = bank_summary["by_source"].get(source, 0) + 1
            bank_summary["by_role"][role] = bank_summary["by_role"].get(role, 0) + 1

    traces = []
    for row in rows:
        question = str(row.get("question") or row.get("problem") or row.get("full_question") or "").strip()
        gold_answer = str(row.get("gold_answer") or row.get("answer") or row.get("full_gold_answer") or "").strip()
        root_completion = str(row.get("root_completion") or row.get("completion") or "").strip()
        if not question or not gold_answer or not root_completion:
            continue

        trace = evaluate_rq3_case(
            question=question,
            gold_answer=gold_answer,
            root_completion=root_completion,
            retriever=retriever,
            plain_retry_completion=extract_plain_retry_completion(row),
            curriculum_retry_completion=extract_curriculum_retry_completion(row),
            branch_completions=extract_branch_completions(row),
        )
        traces.append(trace)

    summary = summarize_rq3_results(traces)

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "rq3_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with (outdir / "rq3_traces.jsonl").open("w", encoding="utf-8") as f:
        for trace in traces:
            f.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    manifest = {
        "cases": str(cases_path),
        "example_bank": args.example_bank,
        "bank_summary": bank_summary,
        "num_rows_loaded": len(rows),
        "num_traces_written": len(traces),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(outdir), "num_traces": len(traces)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
