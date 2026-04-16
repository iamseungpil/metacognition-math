#!/usr/bin/env python3
"""Smoke test for confidence-bucket search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curriculum.control_rag import ExampleRecord, TfidfExampleRetriever
from src.curriculum.mcts_lite import run_mcts_lite_pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/mcts_lite_smoke.json")
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

    question = "Solve x + 7 = 12."
    root_completion = (
        "<|meta|>\n"
        "confidence: 0.32\n"
        "The issue is that the current route is weak because I am not isolating the variable cleanly.\n"
        "study_need: direct isolation of the variable\n"
        "switch to a cleaner direct equation-solving method.\n"
        "<|/meta|>\n"
        "I am stuck."
    )
    branch_completions = [
        "I will retry but I am still unsure. \\boxed{4}",
        "<|meta|>\nconfidence: 0.78\nI can now solve it directly.\n<|/meta|>\nSubtract 7 from both sides, so x = 5. \\boxed{5}",
        "<|meta|>\nconfidence: 0.65\nLet me check by substitution.\n<|/meta|>\nWe get x=5 and 5+7=12, so \\boxed{5}",
    ]

    result = run_mcts_lite_pass(
        question=question,
        root_completion=root_completion,
        retriever=retriever,
        branch_completions=branch_completions,
    )

    if result.confidence_bucket != "low":
        raise RuntimeError(f"expected low bucket, got {result.confidence_bucket}")
    if result.branch_budget != 3:
        raise RuntimeError(f"expected branch budget 3, got {result.branch_budget}")
    if len(result.branches) != 3:
        raise RuntimeError(f"expected 3 branches, got {len(result.branches)}")
    if result.best_branch is None or "\\boxed{5}" not in result.best_branch.completion:
        raise RuntimeError("best branch selection failed")

    payload = {
        "question": result.question,
        "policy_label": result.policy_label,
        "confidence_bucket": result.confidence_bucket,
        "branch_budget": result.branch_budget,
        "best_branch_index": result.best_branch_index,
        "branches": [
            {
                "label": branch.label,
                "source": branch.source,
                "retrieved_questions": branch.retrieved_questions,
                "value": branch.value,
                "score_breakdown": branch.score_breakdown,
                "completion": branch.completion,
            }
            for branch in result.branches
        ],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(out), "best_branch_index": result.best_branch_index}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
