"""_grade_answer must verify the SHORT extracted final answer (never the full essay)
and must NEVER hand a pathological expression to math_verify/sympy — that (with the
disabled thread-timeout) is what grew the build to ~117GB RSS.

GPU-free + network-free. The pathological cases assert FAST return (a wall-clock
budget) so a regression that re-introduces full-essay sympy parsing fails loudly.
"""
from __future__ import annotations

import time

from scripts.build_confidence_redirect_verify_sft import (
    _grade_answer,
    _extract_final_answer,
)


def test_extracts_last_boxed_balanced():
    assert _extract_final_answer(r"work... \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert _extract_final_answer("blah The answer is 42.") == "42"
    assert _extract_final_answer("no answer here") == ""


def test_grades_correct_and_wrong_short_answers():
    assert _grade_answer(r"...so \boxed{42}.", "42") is True
    assert _grade_answer(r"...so \boxed{41}.", "42") is False
    # symbolic equality on the SHORT answer still works (1/2 == 0.5).
    assert _grade_answer(r"\boxed{\frac{1}{2}}", "0.5") is True


def test_pathological_full_essay_is_safe_and_fast():
    """A control sample that 'continues a flawed approach' and emits a giant nested
    exponent in the BODY must not be sympy-evaluated: the grader parses only the
    (clean) boxed final answer, and the whole call returns in well under a second."""
    essay = (
        "Continuing: consider 9^9^9^9 then expand 123456789^987654321 ... "
        r"after much algebra \boxed{7}."
    )
    t0 = time.monotonic()
    out = _grade_answer(essay, "7")
    assert time.monotonic() - t0 < 1.0, "must not sympy-parse the pathological body"
    assert out is True  # the clean boxed 7 == gold 7


def test_pathological_boxed_answer_skips_sympy():
    """Even if the FINAL answer itself is pathological (rare), skip sympy -> plain
    equality, fast, no blow-up."""
    t0 = time.monotonic()
    out = _grade_answer(r"\boxed{9^{9^{9}}}", "9^{9^{9}}")
    assert time.monotonic() - t0 < 1.0
    assert out is True  # exact string equality, never evaluated
