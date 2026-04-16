"""RQ3 orchestration for diagnosis-triggered retry and selective branching.

This module keeps the two downstream uses of meta state separate:
1. curriculum / retrieval-based retry uses diagnosis as the trigger
2. confidence-bucket branching uses confidence as a branching prior

The goal is to log a comparable root -> intervention -> outcome trace for
each lane without overstating the branching helper as full MCTS or value
learning.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import re
from typing import Any

from src.curriculum.control_rag import (
    ExampleRecord,
    TfidfExampleRetriever,
    analyze_completion_for_rag,
    build_incontext_user_prompt,
    build_retrieval_query_bundle,
    run_redirect_rag_pass,
)
from src.curriculum.mcts_lite import run_mcts_lite_pass


BOXED_PATTERN = re.compile(r"\\boxed\{([^{}]+)\}")


def extract_boxed_answer(text: str) -> str:
    matches = BOXED_PATTERN.findall(text or "")
    if not matches:
        return ""
    return matches[-1].strip()


def normalize_answer(text: str) -> str:
    normalized = (text or "").strip().lower()
    normalized = normalized.replace("$", "")
    normalized = re.sub(r"\s+", "", normalized)
    normalized = normalized.rstrip(".")
    return normalized


def judge_completion(completion: str, gold_answer: str) -> dict[str, Any]:
    boxed_answer = extract_boxed_answer(completion)
    predicted = boxed_answer or completion.strip()
    normalized_prediction = normalize_answer(predicted)
    normalized_gold = normalize_answer(gold_answer)
    return {
        "boxed_answer": boxed_answer,
        "normalized_prediction": normalized_prediction,
        "normalized_gold": normalized_gold,
        "is_correct": bool(normalized_prediction) and normalized_prediction == normalized_gold,
    }


@dataclass
class CurriculumRetryTrace:
    lane_label: str = "curriculum_retry"
    evidence_class: str = "side_evidence"
    eligible: bool = False
    applied: bool = False
    query: str = ""
    prompt: str = ""
    retrieved: list[dict[str, Any]] = field(default_factory=list)
    retry_completion: str = ""
    retry_analysis: dict[str, Any] | None = None
    retry_judgment: dict[str, Any] | None = None
    meta_transition: dict[str, Any] | None = None
    improved_over_root: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BranchingTrace:
    lane_label: str = "confidence_bucket_branching"
    evidence_class: str = "side_evidence"
    eligible: bool = False
    applied: bool = False
    policy_label: str = "confidence_bucket_side_evidence"
    confidence_bucket: str = "unknown"
    branch_budget: int = 0
    best_branch_index: int | None = None
    best_branch_label: str = ""
    best_branch_value: float | None = None
    best_branch_completion: str = ""
    best_branch_judgment: dict[str, Any] | None = None
    plain_retry_judgment: dict[str, Any] | None = None
    improved_over_root: bool = False
    improved_over_plain_retry: bool = False
    branches: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlainRetryTrace:
    lane_label: str = "plain_retry_control"
    evidence_class: str = "side_evidence"
    available: bool = False
    completion: str = ""
    judgment: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RQ3CaseTrace:
    question: str
    gold_answer: str
    root_completion: str
    root_analysis: dict[str, Any]
    root_judgment: dict[str, Any]
    trigger_fired: bool
    plain_retry: PlainRetryTrace
    curriculum_retry: CurriculumRetryTrace
    selective_branching: BranchingTrace
    winner: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["plain_retry"] = self.plain_retry.to_dict()
        payload["curriculum_retry"] = self.curriculum_retry.to_dict()
        payload["selective_branching"] = self.selective_branching.to_dict()
        return payload


def _token_overlap_ratio(left: str, right: str) -> float:
    left_tokens = set(re.findall(r"[a-zA-Z0-9_]+", (left or "").lower()))
    right_tokens = set(re.findall(r"[a-zA-Z0-9_]+", (right or "").lower()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def evaluate_meta_transition(
    *,
    root_analysis: dict[str, Any],
    retry_completion: str,
    retry_analysis: dict[str, Any] | None,
    retry_judgment: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if retry_analysis is None:
        return None
    root_conf = root_analysis.get("min_confidence")
    retry_conf = retry_analysis.get("min_confidence")
    confidence_gain = None
    if root_conf is not None and retry_conf is not None:
        confidence_gain = retry_conf - root_conf
    root_need = root_analysis.get("study_need", "")
    retry_text = "\n".join([
        retry_analysis.get("diagnosis_text", ""),
        retry_completion or "",
    ])
    return {
        "root_min_confidence": root_conf,
        "retry_min_confidence": retry_conf,
        "confidence_gain": confidence_gain,
        "confidence_recovered": bool(
            root_conf is not None and retry_conf is not None and retry_conf >= root_conf + 0.15
        ),
        "trigger_cleared": bool(root_analysis.get("should_retrieve") and not retry_analysis.get("should_retrieve")),
        "low_confidence_cleared": bool(
            root_analysis.get("has_low_confidence") and not retry_analysis.get("has_low_confidence")
        ),
        "study_need_followthrough": _token_overlap_ratio(root_need, retry_text) if root_need else 0.0,
        "still_needs_retrieval": bool(retry_analysis.get("should_retrieve")),
        "retry_correct": bool(retry_judgment and retry_judgment.get("is_correct")),
    }


def build_dynamic_library_candidates(
    results: list[RQ3CaseTrace],
    *,
    min_confidence_gain: float = 0.10,
    source_label: str = "rq3_dynamic_memory",
) -> list[ExampleRecord]:
    """Filter successful repaired traces into a dynamic example library."""
    candidates: list[ExampleRecord] = []
    for case in results:
        if case.root_judgment["is_correct"]:
            continue

        lane = None
        completion = ""
        meta_transition = None
        if (
            case.curriculum_retry.retry_judgment is not None
            and case.curriculum_retry.retry_judgment["is_correct"]
        ):
            lane = "retrieval_retry"
            completion = case.curriculum_retry.retry_completion
            meta_transition = case.curriculum_retry.meta_transition or {}
        elif (
            case.selective_branching.best_branch_judgment is not None
            and case.selective_branching.best_branch_judgment["is_correct"]
        ):
            lane = "mcts_lite"
            completion = case.selective_branching.best_branch_completion
            meta_transition = None
        else:
            continue

        confidence_gain = meta_transition.get("confidence_gain") if meta_transition else None
        if lane == "retrieval_retry":
            if meta_transition and not (
                meta_transition.get("trigger_cleared")
                or meta_transition.get("retry_correct")
                or (
                    confidence_gain is not None
                    and confidence_gain >= min_confidence_gain
                )
            ):
                continue

        candidates.append(
            ExampleRecord(
                question=case.question,
                solution=completion,
                answer=case.gold_answer,
                source=source_label,
                metadata={
                    "topic": "",
                    "difficulty": "",
                    "benchmark": "",
                    "source": source_label,
                    "from_lane": lane,
                    "root_was_correct": False,
                    "winner": case.winner,
                    "study_need": case.root_analysis.get("study_need", ""),
                    "trigger_reasons": case.root_analysis.get("trigger_reasons", []),
                    "root_min_confidence": case.root_analysis.get("min_confidence"),
                    "meta_transition": meta_transition,
                },
            )
        )
    return candidates


def build_dynamic_library_from_trace_dicts(
    rows: list[dict[str, Any]],
    *,
    min_confidence_gain: float = 0.10,
    source_label: str = "dynamic_success_library",
) -> list[ExampleRecord]:
    """Materialize a dynamic library from saved JSON traces."""
    candidates: list[ExampleRecord] = []
    for row in rows:
        root_judgment = row.get("root_judgment", {}) or {}
        if root_judgment.get("is_correct"):
            continue

        question = str(row.get("question", "")).strip()
        gold_answer = str(row.get("gold_answer", "")).strip()
        root_analysis = row.get("root_analysis", {}) or {}
        winner = str(row.get("winner", "")).strip()
        if not question or not gold_answer:
            continue

        lane = None
        completion = ""
        meta_transition = None
        curriculum_retry = row.get("curriculum_retry", {}) or {}
        selective_branching = row.get("selective_branching", {}) or {}

        retry_judgment = curriculum_retry.get("retry_judgment", {}) or {}
        if retry_judgment.get("is_correct"):
            lane = "retrieval_retry"
            completion = str(curriculum_retry.get("retry_completion", "")).strip()
            meta_transition = curriculum_retry.get("meta_transition", {}) or {}
            confidence_gain = meta_transition.get("confidence_gain")
            if not (
                meta_transition.get("trigger_cleared")
                or meta_transition.get("retry_correct")
                or (
                    confidence_gain is not None
                    and confidence_gain >= min_confidence_gain
                )
            ):
                continue
        else:
            best_branch_judgment = selective_branching.get("best_branch_judgment", {}) or {}
            if best_branch_judgment.get("is_correct"):
                lane = "mcts_lite"
                completion = str(selective_branching.get("best_branch_completion", "")).strip()
                meta_transition = None
            else:
                continue

        if not completion:
            continue

        candidates.append(
            ExampleRecord(
                question=question,
                solution=completion,
                answer=gold_answer,
                source=source_label,
                metadata={
                    "topic": "",
                    "difficulty": "",
                    "benchmark": "",
                    "source": source_label,
                    "source_role": "dynamic_success",
                    "from_lane": lane,
                    "root_was_correct": False,
                    "winner": winner or lane,
                    "study_need": root_analysis.get("study_need", ""),
                    "trigger_reasons": root_analysis.get("trigger_reasons", []),
                    "root_min_confidence": root_analysis.get("min_confidence"),
                    "meta_transition": meta_transition,
                },
            )
        )
    return candidates


def build_plain_retry_trace(
    *,
    gold_answer: str,
    plain_retry_completion: str | None,
) -> PlainRetryTrace:
    if plain_retry_completion is None:
        return PlainRetryTrace()
    return PlainRetryTrace(
        available=True,
        completion=plain_retry_completion,
        judgment=judge_completion(plain_retry_completion, gold_answer),
    )


def choose_winner(
    *,
    root_judgment: dict[str, Any],
    plain_retry: PlainRetryTrace,
    curriculum_retry: CurriculumRetryTrace,
    selective_branching: BranchingTrace,
) -> str:
    """Pick the lowest-cost successful lane.

    Ordering is intentional:
    root -> plain_retry -> retrieval_retry -> selective_branching
    """
    if root_judgment["is_correct"]:
        return "root"
    if plain_retry.judgment is not None and plain_retry.judgment["is_correct"]:
        return "plain_retry"
    if curriculum_retry.retry_judgment is not None and curriculum_retry.retry_judgment["is_correct"]:
        return "retrieval_retry"
    if selective_branching.best_branch_judgment is not None and selective_branching.best_branch_judgment["is_correct"]:
        return "mcts_lite"
    return "none"


def run_curriculum_retry_lane(
    *,
    question: str,
    gold_answer: str,
    root_completion: str,
    retriever: TfidfExampleRetriever | None = None,
    curriculum_retry_completion: str | None = None,
    model=None,
    tokenizer=None,
    top_k: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> CurriculumRetryTrace:
    root_analysis = analyze_completion_for_rag(root_completion)
    trace = CurriculumRetryTrace(eligible=root_analysis.get("should_retrieve", False))
    if not trace.eligible or retriever is None:
        return trace

    query_bundle = build_retrieval_query_bundle(question, root_analysis)
    trace.query = query_bundle.to_text()
    retrieved = retriever.search(query_bundle, top_k=top_k)
    trace.retrieved = [
        {
            "question": item["record"].question,
            "answer": item["record"].answer,
            "source": item["record"].source,
            "score": item["score"],
            "score_breakdown": item.get("score_breakdown", {}),
        }
        for item in retrieved
    ]
    if not retrieved:
        return trace

    trace.prompt = build_incontext_user_prompt(question, root_analysis, retrieved)
    if curriculum_retry_completion is not None:
        trace.applied = True
        trace.retry_completion = curriculum_retry_completion
    elif model is not None and tokenizer is not None:
        result = run_redirect_rag_pass(
            model,
            tokenizer,
            question,
            root_completion,
            retriever,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        if not result["rag_used"]:
            return trace
        trace.applied = True
        trace.prompt = result["rag_prompt"]
        trace.retry_completion = result["rag_completion"]
        trace.retrieved = result["retrieved"]
    else:
        return trace

    trace.retry_analysis = analyze_completion_for_rag(trace.retry_completion)
    trace.retry_judgment = judge_completion(trace.retry_completion, gold_answer)
    trace.meta_transition = evaluate_meta_transition(
        root_analysis=root_analysis,
        retry_completion=trace.retry_completion,
        retry_analysis=trace.retry_analysis,
        retry_judgment=trace.retry_judgment,
    )
    return trace


def run_branching_lane(
    *,
    question: str,
    gold_answer: str,
    root_completion: str,
    retriever: TfidfExampleRetriever | None = None,
    branch_completions: list[str] | None = None,
    model=None,
    tokenizer=None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> BranchingTrace:
    root_analysis = analyze_completion_for_rag(root_completion)
    trace = BranchingTrace(eligible=root_analysis.get("meta_count", 0) > 0)
    if not trace.eligible:
        return trace
    if branch_completions is None and (model is None or tokenizer is None):
        return trace

    result = run_mcts_lite_pass(
        question=question,
        root_completion=root_completion,
        retriever=retriever,
        model=model,
        tokenizer=tokenizer,
        branch_completions=branch_completions,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    trace.applied = True
    trace.policy_label = result.policy_label
    trace.confidence_bucket = result.confidence_bucket
    trace.branch_budget = result.branch_budget
    trace.best_branch_index = result.best_branch_index
    trace.branches = []

    plain_retry_judgment = None
    for branch in result.branches:
        branch_judgment = judge_completion(branch.completion, gold_answer)
        if branch.label == "plain_retry":
            plain_retry_judgment = branch_judgment
        trace.branches.append(
            {
                "label": branch.label,
                "source": branch.source,
                "retrieved_questions": branch.retrieved_questions,
                "value": branch.value,
                "score_breakdown": branch.score_breakdown,
                "analysis": branch.analysis,
                "judgment": branch_judgment,
                "completion": branch.completion,
            }
        )

    trace.plain_retry_judgment = plain_retry_judgment
    if result.best_branch is not None:
        trace.best_branch_label = result.best_branch.label
        trace.best_branch_value = result.best_branch.value
        trace.best_branch_completion = result.best_branch.completion
        trace.best_branch_judgment = judge_completion(result.best_branch.completion, gold_answer)
    return trace


def evaluate_rq3_case(
    *,
    question: str,
    gold_answer: str,
    root_completion: str,
    retriever: TfidfExampleRetriever | None = None,
    plain_retry_completion: str | None = None,
    curriculum_retry_completion: str | None = None,
    branch_completions: list[str] | None = None,
    model=None,
    tokenizer=None,
    top_k: int = 1,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> RQ3CaseTrace:
    root_analysis = analyze_completion_for_rag(root_completion)
    root_judgment = judge_completion(root_completion, gold_answer)
    trigger_fired = bool(root_analysis.get("should_retrieve", False))
    if plain_retry_completion is None and branch_completions:
        plain_retry_completion = branch_completions[0]
    plain_retry = build_plain_retry_trace(
        gold_answer=gold_answer,
        plain_retry_completion=plain_retry_completion,
    )
    curriculum_trace = run_curriculum_retry_lane(
        question=question,
        gold_answer=gold_answer,
        root_completion=root_completion,
        retriever=retriever,
        curriculum_retry_completion=curriculum_retry_completion,
        model=model,
        tokenizer=tokenizer,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if curriculum_trace.retry_judgment is not None:
        curriculum_trace.improved_over_root = (
            curriculum_trace.retry_judgment["is_correct"] and not root_judgment["is_correct"]
        )

    branching_trace = run_branching_lane(
        question=question,
        gold_answer=gold_answer,
        root_completion=root_completion,
        retriever=retriever,
        branch_completions=branch_completions,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    if branching_trace.best_branch_judgment is not None:
        branching_trace.improved_over_root = (
            branching_trace.best_branch_judgment["is_correct"] and not root_judgment["is_correct"]
        )
    if branching_trace.best_branch_judgment is not None and plain_retry.judgment is not None:
        branching_trace.improved_over_plain_retry = (
            branching_trace.best_branch_judgment["is_correct"]
            and not plain_retry.judgment["is_correct"]
        )
    winner = choose_winner(
        root_judgment=root_judgment,
        plain_retry=plain_retry,
        curriculum_retry=curriculum_trace,
        selective_branching=branching_trace,
    )

    return RQ3CaseTrace(
        question=question,
        gold_answer=gold_answer,
        root_completion=root_completion,
        root_analysis=root_analysis,
        root_judgment=root_judgment,
        trigger_fired=trigger_fired,
        plain_retry=plain_retry,
        curriculum_retry=curriculum_trace,
        selective_branching=branching_trace,
        winner=winner,
    )


def summarize_rq3_results(results: list[RQ3CaseTrace]) -> dict[str, Any]:
    total = len(results)
    if total == 0:
        return {
            "num_cases": 0,
            "root_accuracy": 0.0,
            "curriculum_retry": {},
            "selective_branching": {},
        }

    root_correct = sum(case.root_judgment["is_correct"] for case in results)
    triggered = [case for case in results if case.trigger_fired]
    trigger_on_wrong_root = [case for case in triggered if not case.root_judgment["is_correct"]]
    false_trigger_on_correct_root = [case for case in triggered if case.root_judgment["is_correct"]]

    curriculum_eligible = [case for case in results if case.curriculum_retry.eligible]
    curriculum_applied = [case for case in results if case.curriculum_retry.applied]
    curriculum_improved = [case for case in curriculum_applied if case.curriculum_retry.improved_over_root]
    curriculum_beats_plain = [
        case
        for case in curriculum_applied
        if case.curriculum_retry.retry_judgment
        and case.curriculum_retry.retry_judgment["is_correct"]
        and (case.plain_retry.judgment is None or not case.plain_retry.judgment["is_correct"])
    ]
    curriculum_meta_transitions = [
        case.curriculum_retry.meta_transition
        for case in curriculum_applied
        if case.curriculum_retry.meta_transition is not None
    ]

    branching_applied = [case for case in results if case.selective_branching.applied]
    branching_improved = [case for case in branching_applied if case.selective_branching.improved_over_root]
    branching_improved_over_plain = [
        case for case in branching_applied if case.selective_branching.improved_over_plain_retry
    ]
    branching_expanded = [case for case in branching_applied if case.selective_branching.branch_budget > 1]
    branching_wasted = [
        case for case in branching_expanded if not case.selective_branching.improved_over_root
    ]

    bucket_counts: dict[str, int] = {}
    for case in branching_applied:
        bucket = case.selective_branching.confidence_bucket
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    return {
        "num_cases": total,
        "root_accuracy": ratio(root_correct, total),
        "triggering": {
            "trigger_rate": ratio(len(triggered), total),
            "trigger_on_wrong_root_rate": ratio(len(trigger_on_wrong_root), total),
            "false_trigger_on_correct_root_rate": ratio(len(false_trigger_on_correct_root), total),
        },
        "curriculum_retry": {
            "eligible_rate": ratio(len(curriculum_eligible), total),
            "applied_rate": ratio(len(curriculum_applied), total),
            "retry_accuracy": ratio(
                sum(
                    1
                    for case in curriculum_applied
                    if case.curriculum_retry.retry_judgment
                    and case.curriculum_retry.retry_judgment["is_correct"]
                ),
                len(curriculum_applied),
            ),
            "improvement_rate_over_root": ratio(len(curriculum_improved), len(curriculum_applied)),
            "beats_plain_retry_rate": ratio(len(curriculum_beats_plain), len(curriculum_applied)),
            "next_meta": {
                "confidence_recovery_rate": ratio(
                    sum(1 for item in curriculum_meta_transitions if item["confidence_recovered"]),
                    len(curriculum_meta_transitions),
                ),
                "trigger_clear_rate": ratio(
                    sum(1 for item in curriculum_meta_transitions if item["trigger_cleared"]),
                    len(curriculum_meta_transitions),
                ),
                "mean_study_need_followthrough": (
                    sum(item["study_need_followthrough"] for item in curriculum_meta_transitions)
                    / len(curriculum_meta_transitions)
                    if curriculum_meta_transitions else 0.0
                ),
                "correct_with_cleared_trigger_rate": ratio(
                    sum(
                        1
                        for item in curriculum_meta_transitions
                        if item["retry_correct"] and item["trigger_cleared"]
                    ),
                    len(curriculum_meta_transitions),
                ),
                "recovery_rate": ratio(
                    sum(
                        1
                        for item in curriculum_meta_transitions
                        if item["retry_correct"] and item["trigger_cleared"]
                    ),
                    len(curriculum_meta_transitions),
                ),
            },
        },
        "selective_branching": {
            "applied_rate": ratio(len(branching_applied), total),
            "bucket_counts": bucket_counts,
            "best_branch_accuracy": ratio(
                sum(
                    1
                    for case in branching_applied
                    if case.selective_branching.best_branch_judgment
                    and case.selective_branching.best_branch_judgment["is_correct"]
                ),
                len(branching_applied),
            ),
            "improvement_rate_over_root": ratio(len(branching_improved), len(branching_applied)),
            "beats_plain_retry_rate": ratio(len(branching_improved_over_plain), len(branching_applied)),
            "beats_retrieval_retry_rate": ratio(
                sum(
                    1
                    for case in branching_applied
                    if case.selective_branching.best_branch_judgment
                    and case.selective_branching.best_branch_judgment["is_correct"]
                    and (
                        case.curriculum_retry.retry_judgment is None
                        or not case.curriculum_retry.retry_judgment["is_correct"]
                    )
                ),
                len(branching_applied),
            ),
            "wasted_expansion_rate": ratio(len(branching_wasted), len(branching_expanded)),
        },
        "winner_distribution": {
            label: sum(1 for case in results if case.winner == label)
            for label in ["root", "plain_retry", "retrieval_retry", "mcts_lite", "none"]
        },
    }


def summarize_rq3_records(results: list[RQ3CaseTrace]) -> dict[str, Any]:
    """Backward-compatible alias for callers using the older helper name."""
    return summarize_rq3_results(results)
