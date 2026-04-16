#!/usr/bin/env python3
"""Smoke the full RQ3 -> dynamic library -> self-distill feedback path."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.curriculum.control_rag import ExampleRecord, TfidfExampleRetriever
from src.curriculum.rq3_pipeline import evaluate_rq3_case
from src.training.self_distill_data import build_self_distill_dataframe


def check(name: str, condition: bool) -> None:
    if not condition:
        raise RuntimeError(f"RQ3 self-distill smoke failed: {name}")
    print(f"PASS: {name}")


def main() -> None:
    retriever = TfidfExampleRetriever(
        [
            ExampleRecord(
                question="Solve x + 4 = 9.",
                solution="Subtract 4 from both sides to isolate x, so x = 5. \\boxed{5}",
                answer="5",
                source="stable_seed_library",
                metadata={"topic": "linear equation", "difficulty": "easy", "source_role": "stable_seed"},
            ),
        ]
    )
    case = evaluate_rq3_case(
        question="Solve x + 7 = 12.",
        gold_answer="5",
        root_completion=(
            "<|meta|>\n"
            "confidence: 0.32\n"
            "The issue is that the current route is weak because I am not isolating the variable cleanly.\n"
            "study_need: direct isolation of the variable\n"
            "switch to a cleaner direct equation-solving method.\n"
            "<|/meta|>\n"
            "I guessed x=4. \\boxed{4}"
        ),
        retriever=retriever,
        curriculum_retry_completion=(
            "<|meta|>\n"
            "confidence: 0.78\n"
            "The retrieved example suggests isolating the variable directly.\n"
            "<|/meta|>\n"
            "Subtract 7 from both sides, so x = 5. \\boxed{5}"
        ),
        branch_completions=[
            "I still guess x=4. \\boxed{4}",
            "<|meta|>\nconfidence: 0.74\nI can now isolate the variable directly.\n<|/meta|>\nSubtract 7 from both sides, so x=5. \\boxed{5}",
        ],
    )
    row = case.to_dict()
    row["benchmark"] = "aime2024"
    df = build_self_distill_dataframe([row], mode="epistemic")
    feedback_df = build_self_distill_dataframe([row], mode="sdpo_regen")

    check("one epistemic row should be built", len(df) == 1)
    built = df.iloc[0]
    check("feedback should be marked available", bool(built["teacher_feedback_available"]) is True)
    check("feedback kind should be teacher_only_rag", built["teacher_feedback_kind"] == "teacher_only_rag")
    payload = json.loads(built["teacher_feedback_context_json"])
    check("feedback payload should carry retrieved evidence", payload["teacher_feedback_context"]["evidence_items"][0]["question"] == "Solve x + 4 = 9.")
    check("benchmark should survive into self-distill row", built["benchmark"] == "aime2024")
    check("one feedback-conditioned row should be built", len(feedback_df) == 1)
    feedback_messages = json.loads(feedback_df.iloc[0]["messages"])
    check("feedback-conditioned prompt should include root attempt", "unsuccessful earlier attempt" in feedback_messages[0]["content"])
    check("feedback-conditioned prompt should include retrieved evidence", "Solve x + 4 = 9." in feedback_messages[0]["content"])
    check("feedback-conditioned row should be marked sdpo_regen", feedback_df.iloc[0]["self_distill_mode"] == "sdpo_regen")

    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False) as handle:
        json.dump({"rows": len(df), "feedback_rows": len(feedback_df), "feedback_kind": built["teacher_feedback_kind"]}, handle)
        print(json.dumps({"output": handle.name, "rows": len(df), "feedback_rows": len(feedback_df)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
