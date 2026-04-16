"""Trace normalization for self-distillation and SDPO-style regeneration."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd


BOXED_RE = re.compile(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}')


@dataclass
class NormalizedTeacherTrace:
    question: str
    teacher_completion: str
    gold_answer: str = ""
    source: str = ""
    benchmark: str = ""
    origin: str = ""
    scenario: str = ""
    difficulty: str = ""
    root_completion: str = ""
    diagnosis_text: str = ""
    study_need: str = ""
    intervention_summary: str = ""
    confidence_gain: float | None = None
    trigger_cleared: bool | None = None
    teacher_feedback_kind: str = ""
    teacher_feedback_context: dict[str, Any] | None = None
    candidate_count: int = 0
    selected_candidate_id: str = ""
    selector_mode: str = ""
    selection_score_total: float | None = None
    selection_meta_commit_quality: float | None = None
    selection_margin: float | None = None
    selection_score_breakdown: dict[str, Any] | None = None


def read_table(path: str | Path) -> list[dict[str, Any]]:
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(input_path).to_dict(orient="records")
    if suffix == ".jsonl":
        return [
            json.loads(line)
            for line in input_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "results" in payload and isinstance(payload["results"], list):
                return payload["results"]
            if "cases" in payload and isinstance(payload["cases"], list):
                return payload["cases"]
        if isinstance(payload, list):
            return payload
    raise ValueError(f"Unsupported input artifact: {input_path}")


def load_messages(raw_messages: Any) -> list[dict[str, str]]:
    if isinstance(raw_messages, list):
        return raw_messages
    if isinstance(raw_messages, str):
        return json.loads(raw_messages)
    raise TypeError(f"Unsupported messages payload type: {type(raw_messages)!r}")


def first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def extract_last_boxed(text: str) -> str:
    matches = BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else ""


def parse_study_need(text: str) -> str:
    match = re.search(r"study_need:\s*(.+)", text or "", re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _normalize_from_messages(row: dict[str, Any]) -> NormalizedTeacherTrace | None:
    messages = load_messages(row["messages"])
    user_text = ""
    assistant_text = ""
    for message in messages:
        role = str(message.get("role", ""))
        content = str(message.get("content", ""))
        if role == "user" and not user_text:
            user_text = content.strip()
        elif role == "assistant":
            assistant_text = content.strip()
    if not user_text or not assistant_text:
        return None
    return NormalizedTeacherTrace(
        question=user_text,
        teacher_completion=assistant_text,
        gold_answer=first_text(row, ["full_gold_answer", "gold_answer", "answer"]) or extract_last_boxed(assistant_text),
        source=first_text(row, ["source"]) or "messages",
        benchmark=first_text(row, ["benchmark"]),
        origin="messages",
        scenario=first_text(row, ["scenario"]),
        difficulty=first_text(row, ["difficulty"]),
        study_need=first_text(row, ["study_need"]) or parse_study_need(assistant_text),
    )


def _normalize_from_eval(row: dict[str, Any]) -> NormalizedTeacherTrace | None:
    question = first_text(row, ["full_question", "question", "problem"])
    completion = first_text(row, ["completion", "response", "solution"])
    if not question or not completion:
        return None
    return NormalizedTeacherTrace(
        question=question,
        teacher_completion=completion,
        gold_answer=first_text(row, ["full_gold_answer", "gold_answer", "answer"]) or extract_last_boxed(completion),
        source=first_text(row, ["source", "benchmark"]) or "eval",
        benchmark=first_text(row, ["benchmark"]),
        origin="eval",
        scenario=first_text(row, ["scenario"]),
        difficulty=first_text(row, ["difficulty"]),
        study_need=first_text(row, ["study_need"]) or parse_study_need(completion),
    )


def _normalize_from_rq3_case(row: dict[str, Any]) -> NormalizedTeacherTrace | None:
    question = first_text(row, ["question"])
    gold_answer = first_text(row, ["gold_answer", "answer"])
    if not question:
        return None

    if first_text(row, ["selected_completion"]):
        selector = row.get("selector") or {}
        selected_feedback_context = row.get("selected_feedback_context") or {}
        selected_feedback_kind = first_text(row, ["selected_feedback_kind"])
        selected_meta_transition = row.get("selected_meta_transition") or {}
        root_analysis = row.get("root_analysis") or {}
        repair_candidates = row.get("repair_candidates") or []
        return NormalizedTeacherTrace(
            question=question,
            teacher_completion=first_text(row, ["selected_completion"]),
            gold_answer=gold_answer or extract_last_boxed(first_text(row, ["selected_completion"])),
            source=first_text(row, ["source"]) or "online_self_distill",
            benchmark=first_text(row, ["benchmark"]),
            origin=first_text(row, ["generation_mode"]) or "online_self_distill",
            scenario=first_text(row, ["scenario"]) or "repair_selection",
            difficulty=first_text(row, ["difficulty"]),
            root_completion=first_text(row, ["root_completion"]),
            diagnosis_text=str(root_analysis.get("diagnosis_text", "")).strip(),
            study_need=str(root_analysis.get("study_need", "")).strip(),
            intervention_summary=first_text(row, ["selected_prompt_kind"]) or "selected_completion",
            confidence_gain=(
                float(selected_meta_transition["confidence_gain"])
                if selected_meta_transition.get("confidence_gain") is not None
                else None
            ),
            trigger_cleared=(
                bool(selected_meta_transition["trigger_cleared"])
                if selected_meta_transition.get("trigger_cleared") is not None
                else None
            ),
            teacher_feedback_kind=selected_feedback_kind,
            teacher_feedback_context=selected_feedback_context or None,
            candidate_count=len(repair_candidates),
            selected_candidate_id=str(selector.get("selected_candidate_id", "")).strip(),
            selector_mode=str(selector.get("selector_mode", "")).strip(),
            selection_score_total=(
                float(selector["selected_score"])
                if selector.get("selected_score") is not None
                else None
            ),
            selection_meta_commit_quality=(
                float((selector.get("selected_breakdown") or {}).get("meta_commit_quality"))
                if (selector.get("selected_breakdown") or {}).get("meta_commit_quality") is not None
                else None
            ),
            selection_margin=(
                float(selector["score_margin"])
                if selector.get("score_margin") is not None
                else None
            ),
            selection_score_breakdown=selector.get("selected_breakdown") or {},
        )

    curriculum = row.get("curriculum_retry") or {}
    branching = row.get("selective_branching") or {}
    teacher_completion = ""
    intervention_summary = ""
    confidence_gain = None
    trigger_cleared = None
    teacher_feedback_kind = ""
    teacher_feedback_context: dict[str, Any] | None = None

    retry_judgment = curriculum.get("retry_judgment") or {}
    if curriculum.get("retry_completion") and retry_judgment.get("is_correct"):
        teacher_completion = str(curriculum["retry_completion"]).strip()
        intervention_summary = "retrieval_retry"
        meta_transition = curriculum.get("meta_transition") or {}
        confidence_gain = meta_transition.get("confidence_gain")
        trigger_cleared = meta_transition.get("trigger_cleared")
        retrieved = curriculum.get("retrieved") or []
        evidence_items = []
        for item in retrieved:
            if not isinstance(item, dict):
                continue
            evidence_items.append({
                "question": str(item.get("question", "")).strip(),
                "source": str(item.get("source", "")).strip(),
                "score": item.get("score"),
                "score_breakdown": item.get("score_breakdown", {}),
            })
        if evidence_items:
            teacher_feedback_kind = "teacher_only_rag"
            teacher_feedback_context = {
                "lane": "retrieval_retry",
                "evidence_items": evidence_items,
            }
    elif branching.get("best_branch_completion") and (branching.get("best_branch_judgment") or {}).get("is_correct"):
        teacher_completion = str(branching["best_branch_completion"]).strip()
        intervention_summary = "mcts_lite"
        branch_items = []
        best_branch_index = branching.get("best_branch_index")
        branches = branching.get("branches") or []
        if isinstance(best_branch_index, int) and 0 <= best_branch_index < len(branches):
            branch = branches[best_branch_index] or {}
            retrieved_questions = branch.get("retrieved_questions") or []
            branch_items = [{"question": str(q).strip()} for q in retrieved_questions if str(q).strip()]
        if branch_items:
            teacher_feedback_kind = "branch_side_evidence"
            teacher_feedback_context = {
                "lane": "mcts_lite",
                "evidence_items": branch_items,
            }
    else:
        # Root fallback: only use root completion if correct AND not easy
        # Plan item 7: "D2a/D2b dataset should not include easy problems where root was originally correct in bulk"
        root_judgment = row.get("root_judgment") or {}
        difficulty = (first_text(row, ["difficulty"]) or "").lower()
        is_easy = difficulty in ("easy", "trivial", "")
        if first_text(row, ["root_completion"]) and root_judgment.get("is_correct") and not is_easy:
            teacher_completion = first_text(row, ["root_completion"])
            intervention_summary = "root"

    if not teacher_completion:
        return None

    root_analysis = row.get("root_analysis") or {}
    return NormalizedTeacherTrace(
        question=question,
        teacher_completion=teacher_completion,
        gold_answer=gold_answer or extract_last_boxed(teacher_completion),
        source=first_text(row, ["source"]) or "rq3_case",
        benchmark=first_text(row, ["benchmark"]),
        origin="rq3_case",
        scenario="redirect",
        difficulty=first_text(row, ["difficulty"]),
        root_completion=first_text(row, ["root_completion"]),
        diagnosis_text=str(root_analysis.get("diagnosis_text", "")).strip(),
        study_need=str(root_analysis.get("study_need", "")).strip(),
        intervention_summary=intervention_summary,
        confidence_gain=float(confidence_gain) if confidence_gain is not None else None,
        trigger_cleared=bool(trigger_cleared) if trigger_cleared is not None else None,
        teacher_feedback_kind=teacher_feedback_kind,
        teacher_feedback_context=teacher_feedback_context,
        candidate_count=0,
    )


def normalize_teacher_row(row: dict[str, Any]) -> NormalizedTeacherTrace | None:
    if "messages" in row:
        normalized = _normalize_from_messages(row)
        if normalized is not None:
            return normalized
    if "curriculum_retry" in row or "selective_branching" in row or "selected_completion" in row:
        normalized = _normalize_from_rq3_case(row)
        if normalized is not None:
            return normalized
    return _normalize_from_eval(row)


__all__ = [
    "NormalizedTeacherTrace",
    "extract_last_boxed",
    "first_text",
    "load_messages",
    "normalize_teacher_row",
    "parse_study_need",
    "read_table",
]
