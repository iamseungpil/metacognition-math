#!/usr/bin/env python3
"""Audit current retrieval behavior on real local artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curriculum.control_rag import (  # noqa: E402
    TfidfExampleRetriever,
    analyze_completion_for_rag,
    build_retrieval_query_bundle,
    load_example_bank,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval_parquet",
        default=str(ROOT / "results/eval_v8_meta_inside_strict_sft/eval_v8_meta_inside_strict_sft.remote.parquet"),
    )
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--sample_cases", type=int, default=8)
    parser.add_argument("--output_json", default=str(ROOT / "results/control_rag_real_audit.json"))
    parser.add_argument("--bank_paths", nargs="*", default=[], help="additional stable/dynamic banks to prepend")
    parser.add_argument("--skip_eval_correct_bank", action="store_true", help="do not build the temporary bank from correct eval rows")
    args = parser.parse_args()

    df = pd.read_parquet(args.eval_parquet)
    correct_df = df[df["is_correct"]].copy()
    wrong_df = df[~df["is_correct"]].copy()

    bank_paths: list[Path] = [Path(p) for p in args.bank_paths]
    if not args.skip_eval_correct_bank:
        bank_path = ROOT / "tmp/control_rag_real_bank.json"
        bank_path.parent.mkdir(parents=True, exist_ok=True)
        correct_df.to_json(bank_path, orient="records", force_ascii=False)
        bank_paths.append(bank_path)

    records = load_example_bank(bank_paths)
    retriever = TfidfExampleRetriever(records)

    triggered = []
    aggregate = {
        "num_rows": int(len(df)),
        "num_correct_bank_records": int(len(records)),
        "triggered_wrong_cases": 0,
        "with_study_need": 0,
        "top1_positive_study_need_score": 0,
        "top1_family_match": 0,
        "top1_dynamic_source": 0,
    }
    component_sums = {
        "total": 0.0,
        "problem_similarity": 0.0,
        "diagnosis_to_solution": 0.0,
        "study_need_to_strategy": 0.0,
        "strategy_hint": 0.0,
        "study_need_family_match": 0.0,
        "dynamic_bonus": 0.0,
        "typed_strategy_bonus": 0.0,
        "easy_bonus": 0.0,
        "generic_penalty": 0.0,
    }
    by_family: dict[str, dict[str, float]] = {}

    for _, row in wrong_df.iterrows():
        analysis = analyze_completion_for_rag(str(row["completion"]))
        if not analysis.get("should_retrieve"):
            continue
        aggregate["triggered_wrong_cases"] += 1
        if analysis.get("study_need"):
            aggregate["with_study_need"] += 1
        query = build_retrieval_query_bundle(str(row["full_question"]), analysis)
        hits = retriever.search(query, top_k=args.top_k)
        if not hits:
            continue
        top = hits[0]
        bd = top["score_breakdown"]
        for key in component_sums:
            component_sums[key] += float(bd.get(key, 0.0))
        if bd.get("study_need_to_strategy", 0.0) > 0:
            aggregate["top1_positive_study_need_score"] += 1
        if bd.get("study_need_family_match", 0.0) > 0:
            aggregate["top1_family_match"] += 1
        if bd.get("dynamic_bonus", 0.0) > 0:
            aggregate["top1_dynamic_source"] += 1
        family_key = query.study_need_family or "untyped"
        family_bucket = by_family.setdefault(
            family_key,
            {
                "count": 0,
                "top1_positive_study_need_score": 0,
                "top1_family_match": 0,
                "top1_missing_strategy_signal": 0,
                "top1_generic_penalty": 0,
            },
        )
        family_bucket["count"] += 1
        if bd.get("study_need_to_strategy", 0.0) > 0:
            family_bucket["top1_positive_study_need_score"] += 1
        if bd.get("study_need_family_match", 0.0) > 0:
            family_bucket["top1_family_match"] += 1
        if bd.get("typed_strategy_bonus", 0.0) <= 0:
            family_bucket["top1_missing_strategy_signal"] += 1
        if bd.get("generic_penalty", 0.0) > 0:
            family_bucket["top1_generic_penalty"] += 1
        triggered.append(
            {
                "benchmark": row["benchmark"],
                "question": row["full_question"],
                "study_need": analysis.get("study_need", ""),
                "diagnosis_text": analysis.get("diagnosis_text", ""),
                "query_family": query.study_need_family,
                "prefer_easy": query.prefer_easy,
                "top_hits": [
                    {
                        "score": float(hit["score"]),
                        "score_breakdown": hit["score_breakdown"],
                        "question": hit["record"].question,
                        "source": hit["record"].source,
                        "metadata": hit["record"].metadata or {},
                    }
                    for hit in hits
                ],
            }
        )

    n = max(1, len(triggered))
    summary = {
        **aggregate,
        "mean_top1_components": {key: value / n for key, value in component_sums.items()},
        "by_family": by_family,
        "sample_cases": triggered[: args.sample_cases],
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
