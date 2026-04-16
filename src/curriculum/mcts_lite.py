"""Confidence-bucket search for RQ3 side-evidence.

This module is intentionally lightweight:
1. It reuses the existing redirect-analysis signals.
2. It treats confidence as a branching prior, not as a standalone value target.
3. It keeps search outside the mainline RL contract.

The search policy is selective rather than full MCTS:
- low confidence  -> larger branch budget
- mid confidence  -> moderate branch budget
- high confidence -> minimal/no branching

Retrieved exemplars can seed alternative branches, and branch scoring can use
either a custom value function or a simple heuristic over the generated text.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from src.curriculum.control_rag import (
    TfidfExampleRetriever,
    analyze_completion_for_rag,
    build_incontext_user_prompt,
    build_retrieval_query_bundle,
    generate_from_messages,
)


ValueFn = Callable[[str], float]


@dataclass
class SearchBranch:
    """One candidate branch in the confidence-bucket search."""

    label: str
    prompt: str
    source: str
    retrieved_questions: list[str] = field(default_factory=list)
    completion: str = ""
    value: float | None = None
    score_breakdown: dict[str, float] | None = None
    analysis: dict[str, Any] | None = None


@dataclass
class SearchResult:
    """Structured search output for later auditing."""

    question: str
    root_completion: str
    root_analysis: dict[str, Any]
    policy_label: str
    confidence_bucket: str
    branch_budget: int
    branches: list[SearchBranch]
    best_branch_index: int | None

    @property
    def best_branch(self) -> SearchBranch | None:
        if self.best_branch_index is None:
            return None
        return self.branches[self.best_branch_index]


def confidence_bucket(confidence: float | None) -> str:
    """Map confidence to {low, mid, high, unknown}."""
    if confidence is None:
        return "unknown"
    if confidence <= 0.40:
        return "low"
    if confidence <= 0.70:
        return "mid"
    return "high"


def branch_budget_for_bucket(bucket: str) -> int:
    """Selective expansion policy used by the side-evidence search."""
    if bucket == "low":
        return 3
    if bucket == "mid":
        return 2
    if bucket == "high":
        return 1
    return 1


def heuristic_value_details(completion: str) -> dict[str, Any]:
    """Return an auditable heuristic score breakdown for one branch.

    This is intentionally not a learned value head. It is a transparent
    heuristic that scores whether a branch looks more repair-oriented.
    """
    analysis = analyze_completion_for_rag(completion)
    score = 0.0
    breakdown: dict[str, float] = {
        "low_confidence_awareness": 0.0,
        "failure_diagnosis": 0.0,
        "failure_decomposition": 0.0,
        "next_strategy": 0.0,
        "retrieval_ready": 0.0,
        "meta_presence": 0.0,
        "confidence_prior": 0.0,
        "mid_confidence_bonus": 0.0,
    }
    if analysis["has_low_confidence"]:
        breakdown["low_confidence_awareness"] = 0.2
        score += 0.2
    if analysis["has_diagnosis"]:
        breakdown["failure_diagnosis"] = 0.2
        score += 0.2
    if analysis["has_decomposition"]:
        breakdown["failure_decomposition"] = 0.2
        score += 0.2
    if analysis["has_next_strategy"] or analysis["has_switch"]:
        breakdown["next_strategy"] = 0.2
        score += 0.2
    if analysis["should_retrieve"]:
        breakdown["retrieval_ready"] = 0.1
        score += 0.1
    min_conf = analysis.get("min_confidence")
    if analysis.get("meta_count", 0) > 0:
        breakdown["meta_presence"] = 0.05
        score += 0.05
    if min_conf is not None:
        # Confidence acts only as a weak prior here, not as a learned value target.
        breakdown["confidence_prior"] = 0.10 * min_conf
        score += breakdown["confidence_prior"]
        if 0.55 <= min_conf <= 0.85:
            breakdown["mid_confidence_bonus"] = 0.05
            score += 0.05
    breakdown["total"] = score
    return {
        "analysis": analysis,
        "score_breakdown": breakdown,
        "value": score,
    }


def heuristic_value(completion: str) -> float:
    """Compatibility wrapper for the transparent branch heuristic."""
    return float(heuristic_value_details(completion)["value"])


def build_branch_prompts(
    question: str,
    root_analysis: dict[str, Any],
    retriever: TfidfExampleRetriever | None,
) -> tuple[str, int, list[SearchBranch]]:
    """Construct candidate prompts from the root completion analysis."""
    bucket = confidence_bucket(root_analysis.get("min_confidence"))
    budget = branch_budget_for_bucket(bucket)

    branches: list[SearchBranch] = [
        SearchBranch(
            label="plain_retry",
            prompt=question,
            source="retry",
        )
    ]

    if retriever is None or not root_analysis.get("should_retrieve"):
        return bucket, budget, branches[:budget]

    query = build_retrieval_query_bundle(question, root_analysis)
    hits = retriever.search(query, top_k=max(0, budget - 1))
    for idx, hit in enumerate(hits, start=1):
        prompt = build_incontext_user_prompt(question, root_analysis, [hit])
        branches.append(
            SearchBranch(
                label=f"retrieval_{idx}",
                prompt=prompt,
                source="retrieval",
                retrieved_questions=[hit["record"].question],
            )
        )
    return bucket, budget, branches[:budget]


def _complete_branch(
    branch: SearchBranch,
    *,
    model,
    tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> SearchBranch:
    completion, _, _, _ = generate_from_messages(
        model,
        tokenizer,
        [{"role": "user", "content": branch.prompt}],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    branch.completion = completion
    branch.analysis = analyze_completion_for_rag(completion)
    return branch


def run_mcts_lite_pass(
    *,
    question: str,
    root_completion: str,
    retriever: TfidfExampleRetriever | None = None,
    model=None,
    tokenizer=None,
    value_fn: ValueFn | None = None,
    branch_completions: list[str] | None = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
) -> SearchResult:
    """Run selective confidence-bucket branching from a root completion.

    Either `model`+`tokenizer` or `branch_completions` must be provided.
    `branch_completions` is useful for deterministic tests.
    """
    root_analysis = analyze_completion_for_rag(root_completion)
    bucket, budget, branches = build_branch_prompts(question, root_analysis, retriever)

    if branch_completions is not None and len(branch_completions) < len(branches):
        raise ValueError("branch_completions must cover all constructed branches")
    if branch_completions is None and (model is None or tokenizer is None):
        raise ValueError("Provide either model+tokenizer or branch_completions")

    scorer = value_fn or heuristic_value
    completed: list[SearchBranch] = []
    for idx, branch in enumerate(branches):
        if branch_completions is not None:
            branch.completion = branch_completions[idx]
            branch.analysis = analyze_completion_for_rag(branch.completion)
        else:
            branch = _complete_branch(
                branch,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )
        if scorer is heuristic_value:
            score_info = heuristic_value_details(branch.completion)
            branch.analysis = score_info["analysis"]
            branch.score_breakdown = score_info["score_breakdown"]
            branch.value = float(score_info["value"])
        else:
            branch.value = float(scorer(branch.completion))
        completed.append(branch)

    best_idx = None
    if completed:
        best_idx = max(range(len(completed)), key=lambda i: completed[i].value if completed[i].value is not None else float("-inf"))

    return SearchResult(
        question=question,
        root_completion=root_completion,
        root_analysis=root_analysis,
        policy_label="confidence_bucket_side_evidence",
        confidence_bucket=bucket,
        branch_budget=budget,
        branches=completed,
        best_branch_index=best_idx,
    )
