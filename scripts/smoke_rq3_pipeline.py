#!/usr/bin/env python3
"""Smoke test for the separated RQ3 retry and branching lanes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curriculum.control_rag import ExampleRecord, TfidfExampleRetriever
from src.curriculum.rq3_pipeline import evaluate_rq3_case, summarize_rq3_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/rq3_pipeline_smoke.json")
    args = parser.parse_args()

    retriever = TfidfExampleRetriever(
        [
            ExampleRecord(
                question="Solve x + 4 = 9.",
                solution="Subtract 4 from both sides to isolate x, so x = 5. \\boxed{5}",
                answer="5",
                source="synthetic",
                metadata={"topic": "linear equation", "difficulty": "easy"},
            ),
            ExampleRecord(
                question="Find y if 2y = 14.",
                solution="Divide by 2 to get y = 7. \\boxed{7}",
                answer="7",
                source="synthetic",
                metadata={"topic": "linear equation", "difficulty": "easy"},
            ),
        ]
    )

    root_completion = (
        "<|meta|>\n"
        "confidence: 0.32\n"
        "The issue is that the current route is weak because I am not isolating the variable cleanly.\n"
        "study_need: direct isolation of the variable\n"
        "switch to a cleaner direct equation-solving method.\n"
        "<|/meta|>\n"
        "I guessed x=4. \\boxed{4}"
    )
    curriculum_retry_completion = (
        "<|meta|>\n"
        "confidence: 0.78\n"
        "The retrieved example suggests isolating the variable directly.\n"
        "<|/meta|>\n"
        "Subtract 7 from both sides, so x = 5. \\boxed{5}"
    )
    branch_completions = [
        "I will retry but still guess x=4. \\boxed{4}",
        "<|meta|>\nconfidence: 0.74\nI can now isolate the variable directly.\n<|/meta|>\nSubtract 7 from both sides, so x=5. \\boxed{5}",
        "<|meta|>\nconfidence: 0.66\nLet me verify by substitution.\n<|/meta|>\nWe get x=5 and 5+7=12, so \\boxed{5}",
    ]

    case = evaluate_rq3_case(
        question="Solve x + 7 = 12.",
        gold_answer="5",
        root_completion=root_completion,
        retriever=retriever,
        plain_retry_completion="I will retry but still guess x=4. \\boxed{4}",
        curriculum_retry_completion=curriculum_retry_completion,
        branch_completions=branch_completions,
    )
    summary = summarize_rq3_results([case])

    if not case.curriculum_retry.applied:
        raise RuntimeError("curriculum retry lane was not applied")
    if not case.curriculum_retry.retry_judgment or not case.curriculum_retry.retry_judgment["is_correct"]:
        raise RuntimeError("curriculum retry lane did not recover the correct answer")
    if not case.selective_branching.applied:
        raise RuntimeError("branching lane was not applied")
    if case.selective_branching.policy_label != "confidence_bucket_side_evidence":
        raise RuntimeError("branching lane policy label drifted")
    if not case.selective_branching.best_branch_judgment or not case.selective_branching.best_branch_judgment["is_correct"]:
        raise RuntimeError("best branch was not selected correctly")
    if summary["curriculum_retry"]["improvement_rate_over_root"] != 1.0:
        raise RuntimeError("curriculum summary improvement rate should be 1.0 in the smoke case")
    if summary["curriculum_retry"]["beats_plain_retry_rate"] != 1.0:
        raise RuntimeError("curriculum lane should beat plain retry in the smoke case")
    if case.winner != "retrieval_retry":
        raise RuntimeError(f"unexpected winner label: {case.winner}")

    payload = {
        "summary": summary,
        "results": [case.to_dict()],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "root_correct": case.root_judgment["is_correct"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
