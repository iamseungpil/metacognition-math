"""Shared quality signals for self-distill selection, dense KL, and RL smoke rewards.

The goal is to keep "good meta" semantics consistent across:
1. question-only teacher selection
2. teacher-KL weighting
3. RL / RLSD-lite smoke reward shaping

These signals are analysis-driven rather than style-driven:
- reward structured wrapped meta
- reward concise meta -> reasoning -> commit behavior
- penalize no-commit, repeated-meta loops, and decoherence-like text drift
"""
from __future__ import annotations

import math
import re
from typing import Any

from src.metacot.prompt import META_END, META_START, parse_meta_blocks


WRAPPED_META_RE = re.compile(
    rf"{re.escape(META_START)}(.*?){re.escape(META_END)}",
    re.DOTALL | re.IGNORECASE,
)
DIAGNOSIS_SIGNAL_RE = re.compile(
    r"\b(route is weak|earlier route|diagnosis|missing|contradiction|unsupported|"
    r"does not control|failed because|the issue is|the problem is)\b",
    re.IGNORECASE,
)
STUDY_NEED_RE = re.compile(r"study_need\s*:\s*(.+)", re.IGNORECASE)
BOXED_RE = re.compile(r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}')
EPISTEMIC_OUTSIDE_RE = re.compile(r"\b(maybe|perhaps|probably|not sure|uncertain|i think)\b", re.IGNORECASE)
REPEATED_TAIL_RE = re.compile(r"(.{12,}?)\1{2,}", re.DOTALL)


def assistant_offsets(tokenizer, text: str) -> tuple[list[int], list[tuple[int, int]]]:
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


def _delimiter_balance_penalty(text: str) -> float:
    penalties = 0.0
    for left, right in [("{", "}"), ("(", ")"), ("[", "]")]:
        balance = abs(text.count(left) - text.count(right))
        penalties += min(1.0, balance / 6.0)
    return min(1.0, penalties / 3.0)


def _tail_repetition_penalty(text: str) -> float:
    tail = text[-400:]
    if not tail.strip():
        return 0.0
    if REPEATED_TAIL_RE.search(tail):
        return 1.0
    lines = [line.strip() for line in tail.splitlines() if line.strip()]
    if len(lines) >= 3 and len(set(lines[-3:])) == 1:
        return 1.0
    return 0.0


def score_meta_commit_quality(completion: str) -> dict[str, float]:
    text = str(completion or "").strip()
    wrapped_blocks = list(WRAPPED_META_RE.finditer(text))
    parsed_wrapped = parse_meta_blocks(text, allow_free_text_fallback=False)
    diagnosis_present = bool(DIAGNOSIS_SIGNAL_RE.search(text))
    study_need_present = bool(STUDY_NEED_RE.search(text))
    boxed_matches = list(BOXED_RE.finditer(text))
    boxed_idx = text.rfind("\\boxed")

    wrapped_meta_present = 1.0 if wrapped_blocks else 0.0
    boxed_present = 1.0 if boxed_matches else 0.0
    boxed_after_meta = 0.0
    post_meta_budget_efficiency = 0.0
    repeated_meta_penalty = 0.0
    overlong_post_meta_penalty = 0.0
    epistemic_outside_meta_penalty = 0.0
    multiple_boxed_penalty = 0.0
    post_boxed_text_penalty = 0.0
    diagnosis_score = 1.0 if diagnosis_present else 0.0
    study_need_score = 1.0 if study_need_present else 0.0
    boxed_count = len(boxed_matches)
    delimiter_balance_penalty = _delimiter_balance_penalty(text)
    tail_repetition_penalty = _tail_repetition_penalty(text)
    single_meta_bonus = 0.0

    if wrapped_blocks:
        single_meta_bonus = 1.0 if len(wrapped_blocks) == 1 else 0.0
        last_meta_end = wrapped_blocks[-1].end()
        if boxed_idx > last_meta_end:
            boxed_after_meta = 1.0
            post_meta_chars = max(1, boxed_idx - last_meta_end)
            post_meta_budget_efficiency = max(0.0, 1.0 - min(1.0, post_meta_chars / 1200.0))
            if post_meta_chars > 2400:
                overlong_post_meta_penalty = min(1.0, (post_meta_chars - 2400) / 2400.0)
        elif boxed_present:
            overlong_post_meta_penalty = 0.5
        if len(wrapped_blocks) > 2:
            repeated_meta_penalty = min(1.0, (len(wrapped_blocks) - 2) / 3.0)

        outside_text = text
        for match in reversed(wrapped_blocks):
            outside_text = outside_text[:match.start()] + " " + outside_text[match.end():]
        if EPISTEMIC_OUTSIDE_RE.search(outside_text):
            epistemic_outside_meta_penalty = 0.5
    else:
        if re.search(r"\bconfidence\s*:\s*\d", text, re.IGNORECASE):
            epistemic_outside_meta_penalty = 1.0

    if boxed_count > 1:
        multiple_boxed_penalty = min(1.0, (boxed_count - 1) / 2.0)
    if boxed_matches:
        tail_after_boxed = text[boxed_matches[-1].end():].strip()
        if len(tail_after_boxed) > 40:
            post_boxed_text_penalty = min(1.0, (len(tail_after_boxed) - 40) / 160.0)

    no_boxed_penalty = 0.0 if boxed_present else 1.0
    confidence_present = 1.0 if parsed_wrapped.get("confidences") else 0.0
    decoherence_penalty = max(delimiter_balance_penalty, tail_repetition_penalty)
    total = (
        0.18 * wrapped_meta_present
        + 0.08 * single_meta_bonus
        + 0.12 * confidence_present
        + 0.10 * diagnosis_score
        + 0.10 * study_need_score
        + 0.20 * boxed_after_meta
        + 0.18 * post_meta_budget_efficiency
        - 0.12 * repeated_meta_penalty
        - 0.22 * no_boxed_penalty
        - 0.14 * overlong_post_meta_penalty
        - 0.10 * epistemic_outside_meta_penalty
        - 0.10 * multiple_boxed_penalty
        - 0.12 * post_boxed_text_penalty
        - 0.16 * delimiter_balance_penalty
        - 0.16 * tail_repetition_penalty
    )
    return {
        "total": float(total),
        "wrapped_meta_present": wrapped_meta_present,
        "single_meta_bonus": single_meta_bonus,
        "confidence_present": confidence_present,
        "diagnosis_present": diagnosis_score,
        "study_need_present": study_need_score,
        "boxed_after_meta": boxed_after_meta,
        "post_meta_budget_efficiency": float(post_meta_budget_efficiency),
        "repeated_meta_penalty": float(repeated_meta_penalty),
        "no_boxed_penalty": float(no_boxed_penalty),
        "overlong_post_meta_penalty": float(overlong_post_meta_penalty),
        "epistemic_outside_meta_penalty": float(epistemic_outside_meta_penalty),
        "multiple_boxed_penalty": float(multiple_boxed_penalty),
        "post_boxed_text_penalty": float(post_boxed_text_penalty),
        "delimiter_balance_penalty": float(delimiter_balance_penalty),
        "tail_repetition_penalty": float(tail_repetition_penalty),
        "decoherence_penalty": float(decoherence_penalty),
    }


def summarize_entropy_profile(
    *,
    tokenizer,
    assistant_text: str,
    entropy_values: list[float],
) -> dict[str, float | None]:
    if not entropy_values:
        return {
            "teacher_entropy_mean": None,
            "teacher_entropy_std": None,
            "teacher_meta_entropy_mean": None,
            "teacher_pre_meta_entropy_mean": None,
            "teacher_post_meta_entropy_mean": None,
            "teacher_tail_entropy_mean": None,
            "teacher_entropy_delta_post_vs_pre": None,
        }

    _, offsets = assistant_offsets(tokenizer, assistant_text)
    usable = min(len(offsets), len(entropy_values))
    offsets = offsets[:usable]
    entropy_values = [float(x) for x in entropy_values[:usable]]

    wrapped_blocks = list(WRAPPED_META_RE.finditer(assistant_text))
    if wrapped_blocks:
        meta_start = wrapped_blocks[0].start()
        meta_end = wrapped_blocks[-1].end()
    else:
        meta_start = -1
        meta_end = -1
    boxed_idx = assistant_text.rfind("\\boxed")

    def _slice_mean(mask_fn):
        vals = [entropy_values[i] for i, (s, e) in enumerate(offsets) if mask_fn(s, e)]
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    def _mean(vals: list[float]) -> float | None:
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    overall_mean = _mean(entropy_values)
    overall_std = None
    if entropy_values:
        mu = overall_mean or 0.0
        overall_std = float(math.sqrt(sum((x - mu) ** 2 for x in entropy_values) / len(entropy_values)))

    meta_mean = _slice_mean(lambda s, e: meta_start >= 0 and not (e <= meta_start or s >= meta_end))
    pre_mean = _slice_mean(lambda s, e: meta_start >= 0 and e <= meta_start)
    post_mean = _slice_mean(lambda s, e: meta_end >= 0 and s >= meta_end and (boxed_idx < 0 or e <= boxed_idx))
    tail_mean = _slice_mean(lambda s, e: max(0, len(assistant_text) - 96) <= s)

    delta = None
    if post_mean is not None and pre_mean is not None:
        delta = float(post_mean - pre_mean)

    return {
        "teacher_entropy_mean": overall_mean,
        "teacher_entropy_std": overall_std,
        "teacher_meta_entropy_mean": meta_mean,
        "teacher_pre_meta_entropy_mean": pre_mean,
        "teacher_post_meta_entropy_mean": post_mean,
        "teacher_tail_entropy_mean": tail_mean,
        "teacher_entropy_delta_post_vs_pre": delta,
    }


def compute_teacher_kl_row_scale(
    row: dict[str, Any],
    *,
    entropy_penalty_strength: float = 0.35,
) -> float:
    scale = 1.0

    meta_quality = row.get("selection_meta_commit_quality")
    if meta_quality is None:
        meta_quality = row.get("teacher_meta_commit_quality")
    if meta_quality is not None:
        try:
            scale *= max(0.35, min(1.50, 0.85 + 0.55 * float(meta_quality)))
        except Exception:
            pass

    for key, factor in [
        ("teacher_no_boxed_penalty", 0.65),
        ("teacher_post_boxed_text_penalty", 0.80),
        ("teacher_repeated_meta_penalty", 0.80),
        ("teacher_decoherence_penalty", 0.60),
    ]:
        value = row.get(key)
        if value is not None:
            try:
                scale *= 1.0 - (1.0 - factor) * max(0.0, min(1.0, float(value)))
            except Exception:
                pass

    delta = row.get("teacher_entropy_delta_post_vs_pre")
    if delta is not None:
        try:
            scale *= max(0.60, 1.0 - entropy_penalty_strength * max(0.0, float(delta)))
        except Exception:
            pass

    return max(0.10, min(1.75, float(scale)))


def apply_entropy_weighting(
    *,
    position_weights: list[float],
    entropy_values: list[float] | None,
    beta: float = 0.0,
    floor: float = 0.65,
    ceil: float = 1.10,
) -> list[float]:
    if not entropy_values or beta <= 0.0:
        return position_weights
    usable = min(len(position_weights), len(entropy_values))
    weighted = list(position_weights)
    for idx in range(usable):
        weight = float(weighted[idx])
        if weight <= 0.0:
            continue
        entropy = max(0.0, float(entropy_values[idx]))
        factor = math.exp(-beta * entropy)
        factor = max(floor, min(ceil, factor))
        weighted[idx] = weight * factor
    return weighted


__all__ = [
    "assistant_offsets",
    "apply_entropy_weighting",
    "compute_teacher_kl_row_scale",
    "score_meta_commit_quality",
    "summarize_entropy_profile",
]
