"""Dataset builders for self-distillation and SDPO-style teacher regeneration."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.metacot.prompt import META_END, META_START, parse_meta_blocks
from src.training.self_distill.trace import (
    NormalizedTeacherTrace,
    normalize_teacher_row,
    parse_study_need,
    read_table,
)


META_BLOCK_RE = re.compile(
    rf"{re.escape(META_START)}(.*?){re.escape(META_END)}",
    re.DOTALL | re.IGNORECASE,
)
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

MODE_NAIVE = "naive"
MODE_EPISTEMIC = "epistemic"
MODE_SDPO_REGEN = "sdpo_regen"
MODE_FEEDBACK_CONDITIONED = "feedback_conditioned"  # backward-compatible alias

SUPPORTED_SELF_DISTILL_MODES = (
    MODE_NAIVE,
    MODE_EPISTEMIC,
    MODE_SDPO_REGEN,
    MODE_FEEDBACK_CONDITIONED,
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
    "teacher_prompt_kind",
    "teacher_prompt_text",
    "teacher_failed_attempt",
    "teacher_feedback_text",
    "diagnosis_text",
    "study_need",
    "intervention_summary",
    "confidence_gain",
    "trigger_cleared",
    "teacher_feedback_kind",
    "teacher_feedback_available",
    "teacher_feedback_context_json",
    "candidate_count",
    "selected_candidate_id",
    "selector_mode",
    "selection_score_total",
    "selection_meta_commit_quality",
    "selection_margin",
    "selection_score_breakdown_json",
    "synthetic_meta_injected",
    "teacher_num_meta_blocks",
    "teacher_avg_confidence",
    "teacher_has_diagnosis",
    "teacher_has_study_need",
    "teacher_has_epistemic",
    "teacher_completion_length_chars",
    "teacher_completion_length_tokens_approx",
]


def canonical_mode(mode: str) -> str:
    if mode == MODE_FEEDBACK_CONDITIONED:
        return MODE_SDPO_REGEN
    return mode


def _mean_confidence(text: str) -> float | None:
    parsed = parse_meta_blocks(text or "")
    confs = parsed.get("confidences", [])
    if not confs:
        return None
    return sum(confs) / len(confs)


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
            # Save trailing punctuation before stripping
            trailing_punct = stripped[-1] if stripped and stripped[-1] in ".?!;:" else ""
            stripped = EPISTEMIC_RE.sub("", stripped)
            stripped = re.sub(r"\s{2,}", " ", stripped).strip()
            # Remove leading artifacts (orphaned commas, etc.) but preserve trailing punct
            stripped = re.sub(r"^[,;:\-\s]+", "", stripped)
            if stripped and trailing_punct and stripped[-1] not in ".?!;:":
                stripped = stripped.rstrip(" ,") + trailing_punct
        if stripped:
            lines.append(stripped)
    return "\n".join(lines).strip()


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
    lines = [META_START, "confidence: 0.34", diagnosis]
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


def build_epistemic_teacher_completion(
    trace: NormalizedTeacherTrace,
    *,
    allow_synthetic_meta: bool = True,
) -> str:
    text = trace.teacher_completion.strip()
    parsed = parse_meta_blocks(text)
    if parsed.get("num_blocks", 0) > 0:
        return text
    if not allow_synthetic_meta:
        return ""
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


def build_teacher_feedback_payload(trace: NormalizedTeacherTrace) -> dict[str, Any]:
    return {
        "feedback_kind": trace.teacher_feedback_kind,
        "study_need": trace.study_need,
        "diagnosis_text": trace.diagnosis_text,
        "intervention_summary": trace.intervention_summary,
        "confidence_gain": trace.confidence_gain,
        "trigger_cleared": trace.trigger_cleared,
        "teacher_feedback_context": trace.teacher_feedback_context or {},
    }


def build_sdpo_regen_user_prompt(trace: NormalizedTeacherTrace) -> str:
    """Approximate the SDPO reprompt path with evidence-conditioned regeneration."""
    if not trace.teacher_feedback_context:
        raise ValueError("sdpo_regen mode requires teacher feedback context")

    failed_attempt = trace.root_completion.strip() or "No failed attempt was logged."
    evidence_block = _format_feedback_evidence(trace)
    diagnosis = trace.diagnosis_text.strip() or "The previous route was unreliable."
    study_need = trace.study_need.strip() or "None stated."
    outcome_bits = []
    if trace.confidence_gain is not None:
        outcome_bits.append(f"confidence_gain={float(trace.confidence_gain):.3f}")
    if trace.trigger_cleared is not None:
        outcome_bits.append(f"trigger_cleared={bool(trace.trigger_cleared)}")
    outcome_line = ", ".join(outcome_bits) if outcome_bits else "No explicit recovery outcome metadata."

    sections = [
        trace.question.strip(),
        "",
        "The following is an unsuccessful earlier attempt:",
        failed_attempt,
        "",
        "The following is teacher-side recovery feedback and evidence:",
        f"Failure diagnosis:\n{diagnosis}",
        f"study_need: {study_need}",
    ]
    if trace.intervention_summary.strip():
        sections.append(f"teacher_intervention: {trace.intervention_summary.strip()}")
    if evidence_block:
        sections.extend(["", evidence_block])
    sections.extend([
        "",
        "Observed recovery metadata:",
        outcome_line,
        "",
        "Correctly solve the original question. Use the evidence to recover, not to copy blindly.",
    ])
    return "\n".join(sections).strip()


def build_sdpo_regen_messages(trace: NormalizedTeacherTrace) -> list[dict[str, str]]:
    teacher_completion = build_epistemic_teacher_completion(trace)
    return [
        {"role": "user", "content": build_sdpo_regen_user_prompt(trace)},
        {"role": "assistant", "content": teacher_completion},
    ]


def build_feedback_conditioned_messages(trace: NormalizedTeacherTrace) -> list[dict[str, str]]:
    """Backward-compatible alias for old naming."""
    return build_sdpo_regen_messages(trace)


def record_teacher_metrics(completion: str, *, synthetic_meta_injected: bool = False) -> dict[str, Any]:
    parsed = parse_meta_blocks(completion)
    stripped = completion.strip()
    return {
        "synthetic_meta_injected": bool(synthetic_meta_injected),
        "teacher_num_meta_blocks": parsed.get("num_blocks", 0),
        "teacher_avg_confidence": _mean_confidence(stripped),
        "teacher_has_diagnosis": bool(DIAGNOSIS_RE.search(stripped)),
        "teacher_has_study_need": bool(parse_study_need(stripped)),
        "teacher_has_epistemic": bool(EPISTEMIC_RE.search(stripped)),
        "teacher_completion_length_chars": len(stripped),
        "teacher_completion_length_tokens_approx": max(1, len(stripped) // 4),
    }


def _build_messages_for_mode(
    trace: NormalizedTeacherTrace,
    mode: str,
    *,
    claim_bearing: bool = False,
) -> tuple[list[dict[str, str]], str, str, str, bool]:
    mode = canonical_mode(mode)
    if claim_bearing and mode == MODE_SDPO_REGEN:
        raise ValueError("sdpo_regen is side-evidence only and cannot be marked claim-bearing")
    if mode == MODE_NAIVE:
        completion = build_naive_teacher_completion(trace)
        messages = [
            {"role": "user", "content": trace.question},
            {"role": "assistant", "content": completion},
        ]
        return messages, completion, "plain_question", trace.question, False
    if mode == MODE_EPISTEMIC:
        allow_synthetic_meta = not claim_bearing
        completion = build_epistemic_teacher_completion(trace, allow_synthetic_meta=allow_synthetic_meta)
        synthetic_meta_injected = bool(completion and not parse_meta_blocks(trace.teacher_completion).get("num_blocks", 0))
        if claim_bearing and synthetic_meta_injected:
            return [], "", "", "", False
        messages = [
            {"role": "user", "content": trace.question},
            {"role": "assistant", "content": completion},
        ]
        return messages, completion, "plain_question", trace.question, synthetic_meta_injected
    if not trace.teacher_feedback_context:
        raise ValueError("sdpo_regen mode requires teacher feedback context")
    messages = build_sdpo_regen_messages(trace)
    completion = str(messages[-1]["content"]).strip()
    return messages, completion, MODE_SDPO_REGEN, str(messages[0]["content"]), False


def build_self_distill_dataframe(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    source_tag: str | None = None,
    require_boxed: bool = True,
    claim_bearing: bool = False,
) -> pd.DataFrame:
    if mode not in SUPPORTED_SELF_DISTILL_MODES:
        raise ValueError(f"Unsupported mode: {mode}")

    built_rows: list[dict[str, Any]] = []
    for raw_row in rows:
        selected_judgment = raw_row.get("selected_judgment") or {}
        if raw_row.get("selected_completion") and selected_judgment and not bool(selected_judgment.get("is_correct")):
            continue
        trace = normalize_teacher_row(raw_row)
        if trace is None:
            continue
        try:
            messages, completion, prompt_kind, prompt_text, synthetic_meta_injected = _build_messages_for_mode(
                trace,
                mode,
                claim_bearing=claim_bearing,
            )
        except ValueError:
            continue

        if not completion:
            continue
        if require_boxed and "\\boxed" not in completion:
            continue

        metrics = record_teacher_metrics(completion, synthetic_meta_injected=synthetic_meta_injected)
        feedback_text = _format_feedback_evidence(trace)
        built_rows.append({
            "messages": json.dumps(messages, ensure_ascii=False),
            "question": trace.question,
            "gold_answer": trace.gold_answer,
            "source": source_tag or trace.source or trace.origin or f"self_distill_{canonical_mode(mode)}",
            "benchmark": trace.benchmark,
            "self_distill_mode": canonical_mode(mode),
            "teacher_origin": trace.origin,
            "scenario": trace.scenario,
            "difficulty": trace.difficulty,
            "root_completion": trace.root_completion,
            "teacher_completion_raw": trace.teacher_completion,
            "teacher_completion_built": completion,
            "teacher_prompt_kind": prompt_kind,
            "teacher_prompt_text": prompt_text,
            "teacher_failed_attempt": trace.root_completion,
            "teacher_feedback_text": feedback_text,
            "diagnosis_text": trace.diagnosis_text,
            "study_need": trace.study_need,
            "intervention_summary": trace.intervention_summary,
            "confidence_gain": trace.confidence_gain,
            "trigger_cleared": trace.trigger_cleared,
            "teacher_feedback_kind": trace.teacher_feedback_kind,
            "teacher_feedback_available": bool(trace.teacher_feedback_context),
            "teacher_feedback_context_json": json.dumps(build_teacher_feedback_payload(trace), ensure_ascii=False),
            "candidate_count": int(trace.candidate_count),
            "selected_candidate_id": trace.selected_candidate_id,
            "selector_mode": trace.selector_mode,
            "selection_score_total": trace.selection_score_total,
            "selection_meta_commit_quality": trace.selection_meta_commit_quality,
            "selection_margin": trace.selection_margin,
            "selection_score_breakdown_json": json.dumps(trace.selection_score_breakdown or {}, ensure_ascii=False),
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
    claim_bearing: bool = False,
) -> pd.DataFrame:
    return build_self_distill_dataframe(
        read_table(input_path),
        mode=mode,
        source_tag=source_tag,
        require_boxed=require_boxed,
        claim_bearing=claim_bearing,
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
            "synthetic_meta_injected_rate": 0.0,
            "avg_candidate_count": 0.0,
            "avg_completion_length_chars": 0.0,
            "sdpo_prompt_rate": 0.0,
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
        "synthetic_meta_injected_rate": float(df["synthetic_meta_injected"].mean()) if "synthetic_meta_injected" in df.columns else 0.0,
        "avg_candidate_count": float(df["candidate_count"].mean()) if "candidate_count" in df.columns else 0.0,
        "avg_completion_length_chars": float(df["teacher_completion_length_chars"].mean()),
        "sdpo_prompt_rate": float((df["teacher_prompt_kind"] == MODE_SDPO_REGEN).mean()) if "teacher_prompt_kind" in df.columns else 0.0,
    }


def normalize_summary(df: pd.DataFrame) -> dict[str, Any]:
    return summarize_self_distill_dataframe(df)


__all__ = [
    "MODE_EPISTEMIC",
    "MODE_FEEDBACK_CONDITIONED",
    "MODE_NAIVE",
    "MODE_SDPO_REGEN",
    "SELF_DISTILL_COLUMNS",
    "SUPPORTED_SELF_DISTILL_MODES",
    "build_epistemic_teacher_completion",
    "build_feedback_conditioned_messages",
    "build_naive_teacher_completion",
    "build_sdpo_regen_messages",
    "build_sdpo_regen_user_prompt",
    "build_self_distill_dataframe",
    "build_self_distill_dataset",
    "build_teacher_feedback_payload",
    "canonical_mode",
    "normalize_summary",
    "record_teacher_metrics",
    "summarize_self_distill_dataframe",
]
