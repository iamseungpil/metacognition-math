"""Control-span-weighted KL helpers for self-distill.

This module keeps the dense-teacher path narrow and claim-bearing:
1. KL is optional and only activates when teacher top-k targets exist.
2. KL is focused on control-critical spans, not the uniform full trace.
3. The default mask prioritizes wrapped meta, study_need / diagnosis cues,
   and the post-meta recovery segment.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

from src.metacot.prompt import META_END, META_START


_META_BLOCK_RE = re.compile(
    rf"{re.escape(META_START)}(.*?){re.escape(META_END)}",
    re.DOTALL | re.IGNORECASE,
)
_VERIFY_RE = re.compile(
    r"\b(verify|verified|verification|double-check|check again|re-check|"
    r"substitut\w*\s+back|plug\w*\s+(?:back|in)|cross-?check|sanity check)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class TeacherTopKPayload:
    token_ids: list[list[int]]
    logprobs: list[list[float]]
    target_logprobs: list[float]
    assistant_token_ids: list[int]


def _parse_json_list(raw: Any, *, name: str) -> list[Any]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        loaded = json.loads(raw)
        if isinstance(loaded, list):
            return loaded
    raise TypeError(f"Unsupported {name} payload type: {type(raw)!r}")


def load_teacher_topk_payload(row: dict[str, Any]) -> TeacherTopKPayload | None:
    if "teacher_topk_token_ids_json" not in row or "teacher_topk_logprobs_json" not in row:
        return None
    token_ids = _parse_json_list(row.get("teacher_topk_token_ids_json"), name="teacher_topk_token_ids_json")
    logprobs = _parse_json_list(row.get("teacher_topk_logprobs_json"), name="teacher_topk_logprobs_json")
    target_logprobs = _parse_json_list(row.get("teacher_target_logprobs_json"), name="teacher_target_logprobs_json")
    assistant_token_ids = _parse_json_list(row.get("assistant_token_ids_json"), name="assistant_token_ids_json")
    return TeacherTopKPayload(
        token_ids=[[int(x) for x in position] for position in token_ids],
        logprobs=[[float(x) for x in position] for position in logprobs],
        target_logprobs=[float(x) for x in target_logprobs],
        assistant_token_ids=[int(x) for x in assistant_token_ids],
    )


def _find_spans(text: str, needle: str) -> list[tuple[int, int]]:
    needle = (needle or "").strip()
    if len(needle) < 6:
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        spans.append((idx, idx + len(needle)))
        start = idx + len(needle)
    return spans


def _assistant_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    try:
        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = encoded["input_ids"]
        offsets = encoded["offset_mapping"]
        return [int(x) for x in input_ids], [(int(s), int(e)) for s, e in offsets]
    except Exception:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded["input_ids"]
        offsets: list[tuple[int, int]] = []
        cursor = 0
        for token_id in input_ids:
            piece = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            piece = piece.replace(" ", "")
            if not piece:
                offsets.append((cursor, cursor))
                continue
            start = text.find(piece, cursor)
            if start < 0:
                start = cursor
            end = min(len(text), start + len(piece))
            offsets.append((start, end))
            cursor = end
        return [int(x) for x in input_ids], offsets


def build_control_span_weights(
    *,
    tokenizer,
    assistant_text: str,
    expected_length: int,
    mask_mode: str = "control_spans",
    diagnosis_text: str = "",
    study_need: str = "",
    meta_weight: float = 1.0,
    diagnosis_weight: float = 1.25,
    study_need_weight: float = 1.4,
    recovery_weight: float = 0.7,
    verify_weight: float = 1.1,
) -> list[float]:
    """Build per-assistant-token weights for dense teacher KL.

    The mask intentionally ignores prompt tokens and most derivation tokens.
    This keeps the KL objective focused on controller preservation and
    recovery-critical spans.
    """
    if expected_length <= 0:
        return []

    base_ids, offsets = _assistant_offsets(tokenizer, assistant_text)
    weights = [0.0] * len(base_ids)

    def add_span_weight(char_start: int, char_end: int, value: float) -> None:
        if char_end <= char_start:
            return
        for token_idx, (start, end) in enumerate(offsets):
            if end <= char_start or start >= char_end:
                continue
            weights[token_idx] = max(weights[token_idx], float(value))

    if mask_mode not in {"control_spans", "meta_only"}:
        raise ValueError(f"Unsupported mask_mode: {mask_mode}")

    meta_matches = list(_META_BLOCK_RE.finditer(assistant_text))
    for match in meta_matches:
        add_span_weight(match.start(), match.end(), meta_weight)

    if mask_mode == "meta_only":
        if len(weights) < expected_length:
            weights.extend([0.0] * (expected_length - len(weights)))
        return weights[:expected_length]

    for start, end in _find_spans(assistant_text, diagnosis_text):
        add_span_weight(start, end, diagnosis_weight)
    for start, end in _find_spans(assistant_text, study_need):
        add_span_weight(start, end, study_need_weight)

    if meta_matches:
        last_end = meta_matches[-1].end()
        boxed_idx = assistant_text.rfind("\\boxed")
        recovery_end = boxed_idx if boxed_idx > last_end else len(assistant_text)
        add_span_weight(last_end, recovery_end, recovery_weight)
        recovery_text = assistant_text[last_end:recovery_end]
        for verify_match in _VERIFY_RE.finditer(recovery_text):
            add_span_weight(
                last_end + verify_match.start(),
                last_end + verify_match.end(),
                verify_weight,
            )

    if len(weights) < expected_length:
        weights.extend([0.0] * (expected_length - len(weights)))
    return weights[:expected_length]


def trim_teacher_payload(
    payload: TeacherTopKPayload,
    *,
    target_length: int,
) -> TeacherTopKPayload:
    n = min(
        int(target_length),
        len(payload.token_ids),
        len(payload.logprobs),
        len(payload.target_logprobs),
        len(payload.assistant_token_ids),
    )
    return TeacherTopKPayload(
        token_ids=payload.token_ids[:n],
        logprobs=payload.logprobs[:n],
        target_logprobs=payload.target_logprobs[:n],
        assistant_token_ids=payload.assistant_token_ids[:n],
    )


__all__ = [
    "TeacherTopKPayload",
    "build_control_span_weights",
    "load_teacher_topk_payload",
    "trim_teacher_payload",
]
