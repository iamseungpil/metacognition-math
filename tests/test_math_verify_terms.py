#!/usr/bin/env python
r"""verify_terms — «말한 검산 / 수행한 검산» 탐지기의 계약 고정.

왜: 이 항들은 나중에 **보상**이 된다. 보상이 되는 순간 탐지기의 느슨함은 곧 해킹 통로이므로,
  경계 사례(말만 한 것 / 계산줄만 있고 판정이 없는 것 / 중괄호 균형)를 여기서 못 박는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.verify_terms import (  # noqa: E402
    all_boxed, first_candidate_pos, performed_check, performed_template,
    recovered_value_appears_in_problem, self_correction, template_sections, verdict_grade,
)

# ── 합성 텍스트 ────────────────────────────────────────────────────────────────────
RECOMPUTE_OK = r"""First method gives \boxed{12}.
## Method 2
a = 3 + 4
b = a * 2 = 14
c = b - 2 = 12
Both methods give the same answer, 12.
\boxed{12}"""

RECOMPUTE_SAID_ONLY = r"""I will now solve it by a second method and compare the two answers.
The second method also gives the same answer.
\boxed{12}"""

BACKWARD_OK = r"""So the speed is \boxed{60}.
Now I work backwards: assuming my answer is correct,
distance = 60 * 2 = 120, which matches the stated value of 120. ✓
\boxed{60}"""

MAGNITUDE_OK = r"""I expect the answer should be roughly 100, and positive.
sum = 42 + 55 = 97
That is consistent with my estimate, so it makes sense.
\boxed{97}"""

MAGNITUDE_POSTHOC_ONLY = r"""sum = \boxed{97}
That value is reasonable.
"""


def test_all_boxed_brace_balanced():
    t = r"x = \boxed{\frac{1}{2}} then y = \boxed {7}"
    assert all_boxed(t) == [r"\frac{1}{2}", "7"]


def test_all_boxed_unclosed_is_dropped():
    assert all_boxed(r"\boxed{12} and \boxed{oops") == ["12"]


def test_first_candidate_pos_picks_earliest():
    t = "the answer is 5, so " + r"\boxed{5}"
    assert first_candidate_pos(t) == 0
    t2 = r"\boxed{5}. Thus the final answer is 5."
    assert first_candidate_pos(t2) == 0
    assert first_candidate_pos("no answer at all here") is None


def test_recompute_performed():
    r = performed_check(RECOMPUTE_OK, "recompute")
    assert r["performed"] == 1
    assert r["n_eq_lines_after_cand"] >= 3
    assert r["verdict"] == "same"
    assert isinstance(r["verdict_pos"], int)


def test_recompute_said_but_not_performed():
    """★핵심 사례 — 'second method' 를 말하지만 '=' 줄이 하나도 없다."""
    r = performed_check(RECOMPUTE_SAID_ONLY, "recompute")
    assert r["performed"] == 0
    assert r["n_eq_lines_after_cand"] == 0


def test_recompute_eq_lines_without_comparison_is_not_performed():
    t = "the answer is 12\na = 1 = 1\nb = 2 = 2\nc = 3 = 3\n"
    assert performed_check(t, "recompute")["performed"] == 0


def test_recompute_verdict_different_wins():
    t = RECOMPUTE_OK + "\nActually the two answers do not match, so I recheck."
    assert performed_check(t, "recompute")["verdict"] == "different"


def test_backward_performed_and_lexicon_alone_fails():
    assert performed_check(BACKWARD_OK, "backward")["performed"] == 1
    assert performed_check("Let me work backwards to check.", "backward")["performed"] == 0


def test_magnitude_requires_expectation_before_first_boxed():
    assert performed_check(MAGNITUDE_OK, "magnitude")["performed"] == 1
    assert performed_check(MAGNITUDE_POSTHOC_ONLY, "magnitude")["performed"] == 0


def test_unknown_act_raises():
    try:
        performed_check("x", "units")
    except KeyError:
        return
    raise AssertionError("모르는 act 는 fail-loud 여야 한다")


def test_self_correction_equivalence_not_string():
    assert self_correction(r"\boxed{0.5} ... \boxed{\frac{1}{2}}")["changed"] == 0
    assert self_correction(r"\boxed{12} ... \boxed{13}")["changed"] == 1
    assert self_correction(r"\boxed{12}") == {"changed": 0, "n_boxed": 1}


def test_verdict_grade_categories():
    assert verdict_grade(RECOMPUTE_OK, "recompute", "12")["category"] == "confirm_right"
    assert verdict_grade(RECOMPUTE_OK, "recompute", "99")["category"] == "confirm_wrong"
    revised = RECOMPUTE_OK.replace(r"\boxed{12}" + "\n", r"\boxed{12}" + "\n", 1)
    revised = revised + "\nOn reflection the two answers do not match.\n" + r"\boxed{13}"
    g = verdict_grade(revised, "recompute", "13")
    assert g["changed"] == 1 and g["category"] == "revise_right"
    g2 = verdict_grade(revised, "recompute", "12")
    assert g2["category"] == "revise_wrong"


def test_verdict_grade_none_when_not_performed():
    g = verdict_grade(RECOMPUTE_SAID_ONLY, "recompute", "12")
    assert g["performed"] == 0 and g["category"] == "none" and g["final_correct"] == 1


# ══ 템플릿 인식 탐지기 ═════════════════════════════════════════════════════════════
# 왜 이 묶음이 있는가(0915): V1 은 템플릿을 시켜 놓고 자유서술 탐지기로 수행을 쟀다 —
#   backward 수행률 .19 는 «수행 안 함»이 아니라 «탐지기가 못 읽음»이었다(템플릿 인식 .79).
#   그 사고를 여기서 못 박는다.
TMPL_BACKWARD_OK = r"""The train travels at \boxed{60} km/h.

## Backward check
Assuming the answer is correct, distance = 60 * 2 = 120.
Recovered: 120 vs stated: 120 ✅

## Compare
The two values are the same.

## Final
\boxed{60}"""

TMPL_RECOMPUTE_OK = r"""First pass gives \boxed{12}.

## Method 2
a = 3 + 4
b = a * 2 = 14
c = b - 2 = 12
Method 2 result: 12

## Compare
Both are the same.

## Final
\boxed{12}"""

TMPL_MAGNITUDE_OK = r"""## Expectation
Positive, roughly 100.
sum = 42 + 55 = 97

## Solution
sum = 42 + 55

## Compare
The answer matches the expectation.

## Final
\boxed{97}"""


def test_template_sections_reads_headings():
    secs = template_sections(TMPL_BACKWARD_OK)
    assert set(secs) >= {"backward check", "compare", "final"}
    assert "Recovered: 120" in secs["backward check"]


def test_performed_template_positive_for_each_act():
    for text, act in ((TMPL_RECOMPUTE_OK, "recompute"), (TMPL_BACKWARD_OK, "backward"),
                      (TMPL_MAGNITUDE_OK, "magnitude")):
        r = performed_template(text, act)
        assert r["performed"] == 1, act
        assert r["sections"] == {"main": 1, "slot": 1, "compare": 1}
        assert r["verdict"] == "same"
    assert performed_template(TMPL_BACKWARD_OK, "backward")["recovered_matches_stated"] is True


def test_performed_template_missing_slot_is_not_performed():
    t = TMPL_BACKWARD_OK.replace("Recovered: 120 vs stated: 120 ✅", "It all checks out.")
    r = performed_template(t, "backward")
    assert r["performed"] == 0 and r["sections"]["main"] == 1 and r["sections"]["slot"] == 0


def test_performed_template_compliant_but_no_computation_is_not_performed():
    """★형식(sections)은 다 맞췄지만 본 섹션에 계산줄이 없으면 performed=0 이어야 한다 —
    템플릿 준수(compliance)와 실제 수행(performed)을 가르는 핵심 사례."""
    t = TMPL_RECOMPUTE_OK.replace(
        "a = 3 + 4\nb = a * 2 = 14\nc = b - 2 = 12\nMethod 2 result: 12",
        "Method 2 result: 12")
    r = performed_template(t, "recompute")
    assert r["sections"] == {"main": 1, "slot": 1, "compare": 1}
    assert r["n_eq_lines_main"] == 0
    assert r["performed"] == 0


def test_performed_template_verbatim_placeholder_is_not_filled():
    """★템플릿 문자열을 그대로 베낀 것은 «채운 것»이 아니다 — 보상이 되면 이게 해킹 통로다."""
    t = TMPL_RECOMPUTE_OK.replace("Method 2 result: 12", "Method 2 result: <value>")
    assert performed_template(t, "recompute")["performed"] == 0
    t2 = TMPL_BACKWARD_OK.replace("Recovered: 120 vs stated: 120 ✅",
                                  "Recovered: <value> vs stated: <value>")
    assert performed_template(t2, "backward")["performed"] == 0


def test_verdict_read_only_inside_compare_section():
    """Compare 섹션 **밖**의 'matches' 는 판정이 아니다."""
    t = TMPL_RECOMPUTE_OK.replace("## Compare\nBoth are the same.\n\n", "")
    t = t.replace("Method 2 result: 12", "Method 2 result: 12\nThis matches the first answer.")
    r = performed_template(t, "recompute")
    assert r["verdict"] is None and r["sections"]["compare"] == 0 and r["performed"] == 0


def test_template_verdict_different_beats_same_substring():
    t = TMPL_RECOMPUTE_OK.replace("Both are the same.", "The two results are different.")
    assert performed_template(t, "recompute")["verdict"] == "different"
    t2 = TMPL_RECOMPUTE_OK.replace("Both are the same.", "They are inconsistent.")
    assert performed_template(t2, "recompute")["verdict"] == "different"


def test_magnitude_expectation_must_precede_first_boxed():
    t = TMPL_MAGNITUDE_OK.replace("## Expectation\nPositive, roughly 100.\n\n", "")
    t = t + "\n\n## Expectation\nPositive, roughly 100."
    assert performed_template(t, "magnitude")["performed"] == 0


def test_free_text_outputs_score_zero_under_template_detector():
    """★자유서술(대조군에서 자연발생한 검산)은 템플릿 탐지기에서 0 이어야 한다 — 두 잣대의
    분모가 섞이면 «대조군도 수행했다»는 거짓 그림이 생긴다."""
    for text, act in ((RECOMPUTE_OK, "recompute"), (BACKWARD_OK, "backward"),
                      (MAGNITUDE_OK, "magnitude")):
        assert performed_template(text, act)["performed"] == 0


def test_recovered_value_appears_in_problem():
    prob = "A train covers 120 km in 2 hours. What is its speed?"
    assert recovered_value_appears_in_problem(prob, "120") is True
    assert recovered_value_appears_in_problem(prob, "$120$ ✅") is True
    assert recovered_value_appears_in_problem(prob, "60") is False
    assert recovered_value_appears_in_problem(prob, "") is False


def test_performed_template_unknown_act_raises():
    try:
        performed_template("x", "units")
    except KeyError:
        return
    raise AssertionError("모르는 act 는 fail-loud 여야 한다")
