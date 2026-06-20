"""Tests for student-state-conditioned teacher prompt builders.

These prompts ask the TRAPI teacher to generate distillation demos CONDITIONED
on the student's measured state (wrong prefix + measured confidence). The teacher
must state confidence ~= the STUDENT's measured value (never its own, never
inflated), then take the right action (redirect / verify) and finish correctly.

GPU-free / network-free: we only inspect the prompt strings the builders return.
"""

import pytest

from src.metacot.prompt_redirect_verify import (
    META_START,
    META_END,
    TEACHER_DISTILL_SYSTEM_PROMPT,
    CONTROL_CONTINUATION_SYSTEM_PROMPT,
    build_redirect_demo_prompt,
    build_verify_demo_prompt,
    build_control_continuation_prompt,
)

PROBLEM = "If 3x + 7 = 22, what is x?"
WRONG_PREFIX = "I think x = 6 because 3 times 6 is 18 and 18 plus 7 is 25."
STUDENT_ATTEMPT = "Subtracting 7 gives 3x = 15, so x = 5."


# --------------------------------------------------------------------------- #
# System prompt: forbids decorative meta + forbids inflating confidence.
# --------------------------------------------------------------------------- #
def test_system_prompt_forbids_decorative_meta():
    sp = TEACHER_DISTILL_SYSTEM_PROMPT.lower()
    assert "decorative" in sp or "fake" in sp
    # the meta block must change/check behavior, not be filler
    assert "filler" in sp or "decorative" in sp


def test_system_prompt_forbids_inflating_confidence():
    sp = TEACHER_DISTILL_SYSTEM_PROMPT.lower()
    # teacher must not raise confidence above the given (student) value
    assert "inflate" in sp or "above" in sp
    assert "student" in sp


def test_system_prompt_mentions_meta_format_and_boxed():
    assert META_START in TEACHER_DISTILL_SYSTEM_PROMPT
    assert META_END in TEACHER_DISTILL_SYSTEM_PROMPT
    assert "\\boxed" in TEACHER_DISTILL_SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Redirect demo prompt.
# --------------------------------------------------------------------------- #
def test_redirect_prompt_embeds_confidence():
    p = build_redirect_demo_prompt(PROBLEM, WRONG_PREFIX, conf=0.23)
    text = _as_text(p)
    # the student's measured value, formatted as 0.23, must appear
    assert "0.23" in text


def test_redirect_prompt_embeds_wrong_prefix():
    p = build_redirect_demo_prompt(PROBLEM, WRONG_PREFIX, conf=0.23)
    text = _as_text(p)
    assert WRONG_PREFIX in text
    assert PROBLEM in text


def test_redirect_prompt_demands_switch_method():
    p = build_redirect_demo_prompt(PROBLEM, WRONG_PREFIX, conf=0.23)
    text = _as_text(p).lower()
    # must instruct a genuine method switch
    assert "switch" in text
    assert "different method" in text or "different strategy" in text
    # the decision is now a TEXT field inside the meta block (no special token)
    assert "decision: redirect" in _as_text(p)


def test_redirect_prompt_begins_at_meta_and_forbids_repeating_prefix():
    # New design: the wrong prefix is ALREADY written and precedes the teacher's
    # continuation, so the teacher must NOT restate it and must begin at <|meta|>
    # (prevents the prefix-duplication seen in the first real sample).
    p = build_redirect_demo_prompt(PROBLEM, WRONG_PREFIX, conf=0.23)
    text = _as_text(p).lower()
    assert "begin" in text and "<|meta|>" in text
    assert "do not repeat" in text or "do not restate" in text


def test_redirect_prompt_says_confidence_is_student_not_teacher():
    p = build_redirect_demo_prompt(PROBLEM, WRONG_PREFIX, conf=0.23)
    text = _as_text(p).lower()
    assert "student" in text
    # must not let the teacher use its own (higher) confidence
    assert "not your own" in text or "not the teacher" in text or "do not inflate" in text


# --------------------------------------------------------------------------- #
# Control continuation prompt (the no-redirect CONTROL arm of the causal filter).
# --------------------------------------------------------------------------- #
def test_control_prompt_embeds_problem_and_prefix():
    p = build_control_continuation_prompt(PROBLEM, WRONG_PREFIX)
    text = _as_text(p)
    assert PROBLEM in text
    assert WRONG_PREFIX in text


def test_control_prompt_forbids_meta_and_switch():
    """The control arm must CONTINUE the same flawed approach: no meta block, no
    method switch. Otherwise the control could also recover and the causal filter
    'redirect_ok and not control_ok' becomes vacuous."""
    text = _as_text(build_control_continuation_prompt(PROBLEM, WRONG_PREFIX)).lower()
    assert "no meta" in text or "do not" in text
    # it must tell the teacher NOT to switch / NOT to open a meta block
    assert "switch" in text  # referenced only to forbid it
    assert "same" in text  # carry the same (flawed) approach


def test_control_prompt_uses_control_system_prompt_not_always_correct():
    """The control arm must NOT use the 'always end correct' distill system prompt
    (that is exactly what made the control non-falsifiable)."""
    msgs = build_control_continuation_prompt(PROBLEM, WRONG_PREFIX)
    assert msgs[0]["content"] == CONTROL_CONTINUATION_SYSTEM_PROMPT
    assert msgs[0]["content"] != TEACHER_DISTILL_SYSTEM_PROMPT
    # the control system prompt must NOT instruct "always reach the correct result"
    assert "always reach the correct" not in CONTROL_CONTINUATION_SYSTEM_PROMPT.lower()


def test_control_prompt_returns_system_and_user():
    msgs = build_control_continuation_prompt(PROBLEM, WRONG_PREFIX)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_control_prompt_empty_prefix_rejected():
    with pytest.raises(ValueError):
        build_control_continuation_prompt(PROBLEM, "   ")


# --------------------------------------------------------------------------- #
# Verify demo prompt.
# --------------------------------------------------------------------------- #
def test_verify_prompt_embeds_confidence():
    p = build_verify_demo_prompt(PROBLEM, STUDENT_ATTEMPT, conf=0.88)
    text = _as_text(p)
    assert "0.88" in text


def test_verify_prompt_embeds_attempt_and_problem():
    p = build_verify_demo_prompt(PROBLEM, STUDENT_ATTEMPT, conf=0.88)
    text = _as_text(p)
    assert STUDENT_ATTEMPT in text
    assert PROBLEM in text


def test_verify_prompt_demands_independent_check():
    p = build_verify_demo_prompt(PROBLEM, STUDENT_ATTEMPT, conf=0.88)
    text = _as_text(p).lower()
    assert "independent" in text
    assert "substitut" in text or "recomput" in text


def test_verify_prompt_does_not_demand_switch():
    # verify confirms/corrects; it should not force a method switch up front
    p = build_verify_demo_prompt(PROBLEM, STUDENT_ATTEMPT, conf=0.88)
    text = _as_text(p).lower()
    assert "you must switch" not in text


def test_verify_prompt_demands_decision_verify_field():
    # the decision is a TEXT field inside the meta block (no special token)
    p = build_verify_demo_prompt(PROBLEM, STUDENT_ATTEMPT, conf=0.88)
    assert "decision: verify" in _as_text(p)


# --------------------------------------------------------------------------- #
# Confidence formatting edge cases.
# --------------------------------------------------------------------------- #
def test_confidence_formatting_two_decimals():
    p = build_redirect_demo_prompt(PROBLEM, WRONG_PREFIX, conf=0.5)
    assert "0.50" in _as_text(p)
    p2 = build_verify_demo_prompt(PROBLEM, STUDENT_ATTEMPT, conf=0.9)
    assert "0.90" in _as_text(p2)


def test_confidence_out_of_range_rejected():
    with pytest.raises(ValueError):
        build_redirect_demo_prompt(PROBLEM, WRONG_PREFIX, conf=1.5)
    with pytest.raises(ValueError):
        build_verify_demo_prompt(PROBLEM, STUDENT_ATTEMPT, conf=-0.1)


def test_empty_prefix_rejected_for_redirect():
    with pytest.raises(ValueError):
        build_redirect_demo_prompt(PROBLEM, "   ", conf=0.3)


# --------------------------------------------------------------------------- #
# Message structure: builders return chat-style messages (system+user).
# --------------------------------------------------------------------------- #
def test_builders_return_messages_with_system_and_user():
    for builder, second in (
        (build_redirect_demo_prompt, WRONG_PREFIX),
        (build_verify_demo_prompt, STUDENT_ATTEMPT),
    ):
        msgs = builder(PROBLEM, second, conf=0.5)
        assert isinstance(msgs, list)
        roles = [m["role"] for m in msgs]
        assert roles == ["system", "user"]
        assert msgs[0]["content"] == TEACHER_DISTILL_SYSTEM_PROMPT


def _as_text(messages) -> str:
    """Flatten chat messages to one string for substring assertions."""
    if isinstance(messages, str):
        return messages
    return "\n".join(m["content"] for m in messages)
