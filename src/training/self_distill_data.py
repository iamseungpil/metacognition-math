"""Utilities for building self-distillation datasets.

The design intentionally separates:
1. normalization into a stable teacher-trace IR
2. lane-specific transforms for naive vs epistemic-preserving distill
3. projection back into the repo's standard `messages` parquet contract

This keeps the first implementation compatible with the existing offline
`src/training/sft.py` path while preserving enough structure to later feed
teacher-conditioned distillation or RL-style training.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
from typing import Any, Iterable

import pandas as pd

from src.metacot.prompt import META_END, META_START, parse_meta_blocks


META_BLOCK_RE = re.compile(
    rf"{re.escape(META_START)}(.*?){re.escape(META_END)}",
    re.DOTALL | re.IGNORECASE,
)
BOXED_RE = re.compile(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}')
CONF_RE = re.compile(r"confidence\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
STUDY_NEED_RE = re.compile(r"study_need:\s*(.+)", re.IGNORECASE)
EPISTEMIC_RE = re.compile(
    r"\b(i think|maybe|perhaps|probably|possibly|not sure|uncertain|it seems|"
    r"might be|could be|feels like|I guess)\b",
    re.IGNORECASE,
)
DIAGNOSIS_RE = re.compile(
    r"\b(the issue is|the problem is|route is weak|what is missing|"
    r"overcommitted|forcing|does not control|mismatch|contradiction)\b",
    re.IGNORECASE,
)

SELF_DISTILL_COLUMNS = [
    "messages",
    "question",
    "gold_answer",
    "source",
    "benchmark",
    "self_distill_mode",
    "teacher_origin",
    "scenario",
    "difficulty",
    "root_completion",
    "teacher_completion_raw",
    "teacher_completion_built",
    "diagnosis_text",
    "study_need",
    "intervention_summary",
    "confidence_gain",
    "trigger_cleared",
    "teacher_feedback_kind",
    "teacher_feedback_available",
    "teacher_feedback_context_json",
    "teacher_num_meta_blocks",
    "teacher_avg_confidence",
    "teacher_has_diagnosis",
    "teacher_has_study_need",
    "teacher_has_epistemic",
    "teacher_completion_length_chars",
    "teacher_completion_length_tokens_approx",
]

SUPPORTED_SELF_DISTILL_MODES = ("naive", "epistemic", "feedback_conditioned")


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


def _read_table(path: str | Path) -> list[dict[str, Any]]:
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


def _load_messages(raw_messages: Any) -> list[dict[str, str]]:
    if isinstance(raw_messages, list):
        return raw_messages
    if isinstance(raw_messages, str):
        return json.loads(raw_messages)
    raise TypeError(f"Unsupported messages payload type: {type(raw_messages)!r}")


def _first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _extract_last_boxed(text: str) -> str:
    matches = BOXED_RE.findall(text or "")
    return matches[-1].strip() if matches else ""


def _mean_confidence(text: str) -> float | None:
    parsed = parse_meta_blocks(text or "")
    confs = parsed.get("confidences", [])
    if not confs:
        return None
    return sum(confs) / len(confs)


def _parse_study_need(text: str) -> str:
    match = STUDY_NEED_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _strip_meta_blocks(text: str) -> str:
    cleaned = META_BLOCK_RE.sub("", text or "")
    cleaned = cleaned.replace(META_START, "").replace(META_END, "")
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _suppress_epistemic_language(text: str) -> str:
    lines: list[str] = []
    for line in (text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if EPISTEMIC_RE.search(stripped):
            stripped = EPISTEMIC_RE.sub("", stripped)
            stripped = re.sub(r"\s{2,}", " ", stripped).strip(" ,.;:-")
        if stripped:
            lines.append(stripped)
    return "\n".join(lines).strip()


def _normalize_from_messages(row: dict[str, Any]) -> NormalizedTeacherTrace | None:
    messages = _load_messages(row["messages"])
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
        gold_answer=_first_text(row, ["full_gold_answer", "gold_answer", "answer"]) or _extract_last_boxed(assistant_text),
        source=_first_text(row, ["source"]) or "messages",
        benchmark=_first_text(row, ["benchmark"]),
        origin="messages",
        scenario=_first_text(row, ["scenario"]),
        difficulty=_first_text(row, ["difficulty"]),
        study_need=_first_text(row, ["study_need"]) or _parse_study_need(assistant_text),
    )


def _normalize_from_eval(row: dict[str, Any]) -> NormalizedTeacherTrace | None:
    question = _first_text(row, ["full_question", "question", "problem"])
    completion = _first_text(row, ["completion", "response", "solution"])
    if not question or not completion:
        return None
    return NormalizedTeacherTrace(
        question=question,
        teacher_completion=completion,
        gold_answer=_first_text(row, ["full_gold_answer", "gold_answer", "answer"]) or _extract_last_boxed(completion),
        source=_first_text(row, ["source", "benchmark"]) or "eval",
        benchmark=_first_text(row, ["benchmark"]),
        origin="eval",
        scenario=_first_text(row, ["scenario"]),
        difficulty=_first_text(row, ["difficulty"]),
        study_need=_first_text(row, ["study_need"]) or _parse_study_need(completion),
    )


def _normalize_from_rq3_case(row: dict[str, Any]) -> NormalizedTeacherTrace | None:
    question = _first_text(row, ["question"])
    gold_answer = _first_text(row, ["gold_answer", "answer"])
    if not question:
        return None

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
        root_judgment = row.get("root_judgment") or {}
        if _first_text(row, ["root_completion"]) and root_judgment.get("is_correct"):
            teacher_completion = _first_text(row, ["root_completion"])
            intervention_summary = "root"

    if not teacher_completion:
        return None

    root_analysis = row.get("root_analysis") or {}
    return NormalizedTeacherTrace(
        question=question,
        teacher_completion=teacher_completion,
        gold_answer=gold_answer or _extract_last_boxed(teacher_completion),
        source=_first_text(row, ["source"]) or "rq3_case",
        benchmark=_first_text(row, ["benchmark"]),
        origin="rq3_case",
        scenario="redirect",
        difficulty=_first_text(row, ["difficulty"]),
        root_completion=_first_text(row, ["root_completion"]),
        diagnosis_text=str(root_analysis.get("diagnosis_text", "")).strip(),
        study_need=str(root_analysis.get("study_need", "")).strip(),
        intervention_summary=intervention_summary,
        confidence_gain=float(confidence_gain) if confidence_gain is not None else None,
        trigger_cleared=bool(trigger_cleared) if trigger_cleared is not None else None,
        teacher_feedback_kind=teacher_feedback_kind,
        teacher_feedback_context=teacher_feedback_context,
    )


def normalize_teacher_row(row: dict[str, Any]) -> NormalizedTeacherTrace | None:
    if "messages" in row:
        normalized = _normalize_from_messages(row)
        if normalized is not None:
            return normalized
    if "curriculum_retry" in row or "selective_branching" in row:
        normalized = _normalize_from_rq3_case(row)
        if normalized is not None:
            return normalized
    return _normalize_from_eval(row)


def build_naive_teacher_completion(
    trace: NormalizedTeacherTrace,
    *,
    remove_meta: bool = True,
    suppress_epistemic: bool = True,
) -> str:
    text = trace.teacher_completion
    if remove_meta:
        text = _strip_meta_blocks(text)
    if suppress_epistemic:
        text = _suppress_epistemic_language(text)
    return text.strip()


def _render_epistemic_wrapper(trace: NormalizedTeacherTrace) -> str:
    diagnosis = trace.diagnosis_text or "The earlier route did not control the key constraint."
    study_need = trace.study_need.strip()
    action_hint = "I should recover with a corrected route and verify that the failure signal is cleared."
    lines = [
        META_START,
        "confidence: 0.34",
        diagnosis,
    ]
    if study_need:
        lines.append(f"study_need: {study_need}")
    if trace.intervention_summary:
        lines.append(f"I will use the {trace.intervention_summary.replace('_', ' ')} evidence to redirect cleanly.")
    lines.append(action_hint)
    lines.append(META_END)
    return "\n".join(lines)


def _render_recovery_summary(trace: NormalizedTeacherTrace) -> str:
    conf = 0.78
    if trace.confidence_gain is not None:
        conf = max(0.55, min(0.95, 0.45 + float(trace.confidence_gain)))
    lines = [
        META_START,
        f"confidence: {conf:.2f}",
        "The new route resolves the earlier issue and is now supported by the actual reasoning.",
    ]
    if trace.trigger_cleared is not None:
        if trace.trigger_cleared:
            lines.append("The earlier trigger looks cleared rather than merely ignored.")
        else:
            lines.append("Some caution remains, so the recovery should still be checked against the final answer.")
    lines.append(META_END)
    return "\n".join(lines)


def build_epistemic_teacher_completion(trace: NormalizedTeacherTrace) -> str:
    text = trace.teacher_completion.strip()
    parsed = parse_meta_blocks(text)
    if parsed.get("num_blocks", 0) > 0:
        return text
    prefix = _render_epistemic_wrapper(trace)
    suffix = _render_recovery_summary(trace)
    return f"{prefix}\n\n{text}\n\n{suffix}".strip()


def _format_feedback_evidence(trace: NormalizedTeacherTrace) -> str:
    context = trace.teacher_feedback_context or {}
    items = context.get("evidence_items") or []
    if not items:
        return ""
    lines = []
    lane = str(context.get("lane", "")).strip()
    if lane:
        lines.append(f"Teacher recovery evidence lane: {lane}")
    for idx, item in enumerate(items[:3], start=1):
        if not isinstance(item, dict):
            continue
        header_bits = [f"Evidence {idx}"]
        source = str(item.get("source", "")).strip()
        if source:
            header_bits.append(f"source={source}")
        score = item.get("score")
        if isinstance(score, (int, float)):
            header_bits.append(f"score={float(score):.3f}")
        lines.append(" | ".join(header_bits))
        question = str(item.get("question", "")).strip()
        if question:
            lines.append(f"Related solved problem: {question}")
        score_breakdown = item.get("score_breakdown") or {}
        if isinstance(score_breakdown, dict) and score_breakdown:
            lines.append(
                "Score breakdown: "
                + json.dumps(score_breakdown, ensure_ascii=False, sort_keys=True)
            )
    return "\n".join(lines).strip()


def build_feedback_conditioned_messages(trace: NormalizedTeacherTrace) -> list[dict[str, str]]:
    if not trace.teacher_feedback_context:
        raise ValueError("feedback_conditioned mode requires teacher feedback context")

    teacher_completion = build_epistemic_teacher_completion(trace)
    user_lines = [
        "You are learning to recover from a failed route using privileged teacher-side feedback.",
        "Use the provided evidence as strategic guidance rather than copying surface forms.",
        "Preserve explicit diagnosis, study_need, and justified recovery.",
        "",
        "Original problem:",
        trace.question.strip(),
    ]
    if trace.root_completion.strip():
        user_lines.extend([
            "",
            "Failed root attempt that should be repaired:",
            trace.root_completion.strip(),
        ])
    if trace.diagnosis_text.strip():
        user_lines.extend([
            "",
            "Failure diagnosis:",
            trace.diagnosis_text.strip(),
        ])
    if trace.study_need.strip():
        user_lines.append(f"study_need: {trace.study_need.strip()}")
    if trace.intervention_summary.strip():
        user_lines.append(f"teacher_intervention: {trace.intervention_summary.strip()}")
    feedback_block = _format_feedback_evidence(trace)
    if feedback_block:
        user_lines.extend([
            "",
            "Privileged teacher-side recovery evidence:",
            feedback_block,
        ])
    if trace.confidence_gain is not None or trace.trigger_cleared is not None:
        transition_bits = []
        if trace.confidence_gain is not None:
            transition_bits.append(f"confidence_gain={float(trace.confidence_gain):.3f}")
        if trace.trigger_cleared is not None:
            transition_bits.append(f"trigger_cleared={bool(trace.trigger_cleared)}")
        user_lines.extend([
            "",
            "Observed recovery outcome:",
            ", ".join(transition_bits),
        ])
    return [
        {"role": "user", "content": "\n".join(line for line in user_lines if line is not None).strip()},
        {"role": "assistant", "content": teacher_completion},
    ]


def record_teacher_metrics(completion: str) -> dict[str, Any]:
    parsed = parse_meta_blocks(completion)
    stripped = completion.strip()
    return {
        "teacher_num_meta_blocks": parsed.get("num_blocks", 0),
        "teacher_avg_confidence": _mean_confidence(stripped),
        "teacher_has_diagnosis": bool(DIAGNOSIS_RE.search(stripped)),
        "teacher_has_study_need": bool(_parse_study_need(stripped)),
        "teacher_has_epistemic": bool(EPISTEMIC_RE.search(stripped)),
        "teacher_completion_length_chars": len(stripped),
        "teacher_completion_length_tokens_approx": max(1, len(stripped) // 4),
    }


def build_teacher_feedback_payload(trace: NormalizedTeacherTrace) -> dict[str, Any]:
    payload = {
        "feedback_kind": trace.teacher_feedback_kind,
        "study_need": trace.study_need,
        "diagnosis_text": trace.diagnosis_text,
        "intervention_summary": trace.intervention_summary,
        "confidence_gain": trace.confidence_gain,
        "trigger_cleared": trace.trigger_cleared,
        "teacher_feedback_context": trace.teacher_feedback_context or {},
    }
    return payload


def build_self_distill_dataframe(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    source_tag: str | None = None,
    require_boxed: bool = True,
) -> pd.DataFrame:
    if mode not in SUPPORTED_SELF_DISTILL_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    built_rows: list[dict[str, Any]] = []
    for raw_row in rows:
        trace = normalize_teacher_row(raw_row)
        if trace is None:
            continue
        if mode == "naive":
            completion = build_naive_teacher_completion(trace)
            messages = [
                {"role": "user", "content": trace.question},
                {"role": "assistant", "content": completion},
            ]
        elif mode == "epistemic":
            completion = build_epistemic_teacher_completion(trace)
            messages = [
                {"role": "user", "content": trace.question},
                {"role": "assistant", "content": completion},
            ]
        else:
            if not trace.teacher_feedback_context:
                continue
            messages = build_feedback_conditioned_messages(trace)
            completion = str(messages[-1]["content"]).strip()

        if not completion:
            continue
        if require_boxed and "\\boxed" not in completion:
            continue

        metrics = record_teacher_metrics(completion)
        built_rows.append({
            "messages": json.dumps(messages, ensure_ascii=False),
            "question": trace.question,
            "gold_answer": trace.gold_answer,
            "source": source_tag or trace.source or trace.origin or f"self_distill_{mode}",
            "benchmark": trace.benchmark,
            "self_distill_mode": mode,
            "teacher_origin": trace.origin,
            "scenario": trace.scenario,
            "difficulty": trace.difficulty,
            "root_completion": trace.root_completion,
            "teacher_completion_raw": trace.teacher_completion,
            "teacher_completion_built": completion,
            "diagnosis_text": trace.diagnosis_text,
            "study_need": trace.study_need,
            "intervention_summary": trace.intervention_summary,
            "confidence_gain": trace.confidence_gain,
            "trigger_cleared": trace.trigger_cleared,
            "teacher_feedback_kind": trace.teacher_feedback_kind,
            "teacher_feedback_available": bool(trace.teacher_feedback_context),
            "teacher_feedback_context_json": json.dumps(build_teacher_feedback_payload(trace), ensure_ascii=False),
            **metrics,
        })

    if not built_rows:
        return pd.DataFrame(columns=SELF_DISTILL_COLUMNS)
    return pd.DataFrame(built_rows, columns=SELF_DISTILL_COLUMNS)


def build_self_distill_dataset(
    input_path: str | Path,
    *,
    mode: str,
    source_tag: str | None = None,
    require_boxed: bool = True,
) -> pd.DataFrame:
    rows = _read_table(input_path)
    return build_self_distill_dataframe(
        rows,
        mode=mode,
        source_tag=source_tag,
        require_boxed=require_boxed,
    )


def summarize_self_distill_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {
            "rows": 0,
            "meta_emission_rate": 0.0,
            "avg_num_meta_blocks": 0.0,
            "avg_confidence": None,
            "study_need_rate": 0.0,
            "diagnosis_rate": 0.0,
            "epistemic_rate": 0.0,
            "feedback_available_rate": 0.0,
            "avg_completion_length_chars": 0.0,
        }
    conf = df["teacher_avg_confidence"].dropna()
    return {
        "rows": int(len(df)),
        "meta_emission_rate": float((df["teacher_num_meta_blocks"] > 0).mean()),
        "avg_num_meta_blocks": float(df["teacher_num_meta_blocks"].mean()),
        "avg_confidence": float(conf.mean()) if len(conf) else None,
        "study_need_rate": float(df["teacher_has_study_need"].mean()),
        "diagnosis_rate": float(df["teacher_has_diagnosis"].mean()),
        "epistemic_rate": float(df["teacher_has_epistemic"].mean()),
        "feedback_available_rate": float(df["teacher_feedback_available"].mean()) if "teacher_feedback_available" in df.columns else 0.0,
        "avg_completion_length_chars": float(df["teacher_completion_length_chars"].mean()),
    }


def normalize_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Backward-compatible alias for smoke scripts and tests."""
    return summarize_self_distill_dataframe(df)


__all__ = [
    "NormalizedTeacherTrace",
    "build_epistemic_teacher_completion",
    "build_feedback_conditioned_messages",
    "build_naive_teacher_completion",
    "build_self_distill_dataframe",
    "build_self_distill_dataset",
    "normalize_summary",
    "normalize_teacher_row",
    "record_teacher_metrics",
    "summarize_self_distill_dataframe",
    "SUPPORTED_SELF_DISTILL_MODES",
]
