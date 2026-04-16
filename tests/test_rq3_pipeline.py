"""Unit checks for the separated RQ3 retry and branching lanes."""
import sys

sys.path.insert(0, ".")

from src.curriculum.control_rag import ExampleRecord, TfidfExampleRetriever
from src.curriculum.rq3_pipeline import (
    build_dynamic_library_candidates,
    build_dynamic_library_from_trace_dicts,
    evaluate_rq3_case,
    summarize_rq3_results,
)


passed = 0
failed = 0


def check(name, condition):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        print(f"  FAIL: {name}")


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
            question="Compute 3 + 3.",
            solution="3 + 3 = 6. \\boxed{6}",
            answer="6",
            source="synthetic",
            metadata={"topic": "arithmetic", "difficulty": "easy"},
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
    plain_retry_completion="I keep going and return \\boxed{4}",
    curriculum_retry_completion=(
        "<|meta|>\n"
        "confidence: 0.77\n"
        "The retrieved example suggests direct isolation.\n"
        "<|/meta|>\n"
        "Subtract 7 from both sides, so x = 5. \\boxed{5}"
    ),
    branch_completions=[
        "I still guess x=4. \\boxed{4}",
        "<|meta|>\nconfidence: 0.78\nI can solve it directly now.\n<|/meta|>\nSubtract 7 from both sides, so x = 5. \\boxed{5}",
        "<|meta|>\nconfidence: 0.64\nLet me verify by substitution.\n<|/meta|>\nWe get x=5 and 5+7=12, so \\boxed{5}",
    ],
)
summary = summarize_rq3_results([case])

check("root answer should be marked incorrect", case.root_judgment["is_correct"] is False)
check("trigger should fire", case.trigger_fired is True)
check("plain retry control should be available", case.plain_retry.available is True)
check("plain retry control should remain incorrect", case.plain_retry.judgment is not None and case.plain_retry.judgment["is_correct"] is False)
check("curriculum retry should be eligible", case.curriculum_retry.eligible is True)
check("curriculum retry should be applied", case.curriculum_retry.applied is True)
check("curriculum retry should improve over root", case.curriculum_retry.improved_over_root is True)
check(
    "curriculum retry should record next-meta transition",
    case.curriculum_retry.meta_transition is not None and case.curriculum_retry.meta_transition["confidence_recovered"] is True,
)
check(
    "curriculum retry should expose trigger reasons",
    "study_need" in case.root_analysis["trigger_reasons"] and "failure_diagnosis" in case.root_analysis["trigger_reasons"],
)
check("branching lane should be applied", case.selective_branching.applied is True)
check("branching lane should stay labeled side evidence", case.selective_branching.policy_label == "confidence_bucket_side_evidence")
check("best branch should beat plain retry", case.selective_branching.improved_over_plain_retry is True)
check("winner should prefer retrieval before branching when both fix root", case.winner == "retrieval_retry")
check(
    "best branch should include score breakdown",
    bool(case.selective_branching.branches[1]["score_breakdown"]) and "total" in case.selective_branching.branches[1]["score_breakdown"],
)
check("summary should track curriculum improvement", summary["curriculum_retry"]["improvement_rate_over_root"] == 1.0)
check("summary should track curriculum vs plain retry", summary["curriculum_retry"]["beats_plain_retry_rate"] == 1.0)
check("summary should track next-meta trigger clearance", summary["curriculum_retry"]["next_meta"]["trigger_clear_rate"] == 1.0)
check("summary should expose next-meta recovery alias", summary["curriculum_retry"]["next_meta"]["recovery_rate"] == 1.0)
check("summary should track branch improvement over root", summary["selective_branching"]["improvement_rate_over_root"] == 1.0)
check("summary should expose trigger rates", summary["triggering"]["trigger_rate"] == 1.0)

memory_records = build_dynamic_library_candidates([case])
check("dynamic library builder should keep successful repaired trace", len(memory_records) == 1)
check("dynamic library candidate should preserve source lane", memory_records[0].metadata["from_lane"] == "retrieval_retry")

memory_records_from_dict = build_dynamic_library_from_trace_dicts([case.to_dict()])
check("dynamic library dict builder should keep successful repaired trace", len(memory_records_from_dict) == 1)
check("dynamic library dict builder should tag source role", memory_records_from_dict[0].metadata["source_role"] == "dynamic_success")

print(f"\n=== SUMMARY: {passed} passed, {failed} failed ===")


def test_pytest_bridge():
    assert failed == 0


if __name__ == "__main__":
    if failed:
        sys.exit(1)
