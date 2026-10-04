"""mc.grade — 고정 회귀 케이스(tests/test_math_meta.py 의 0916 목록을 그대로 옮김)."""
from __future__ import annotations

import pytest

from mc import grade as G

# 0916 수리가 고정한 6건 + 옛 목록의 오답 대조.
_PINNED = [
    ("\\boxed{t^7}", "t^7", True),
    ("\\boxed{4x + 18}", "4x + 18", True),
    ("\\boxed{\\csc 10}", "\\csc 10", True),
    ("\\boxed{\\frac83}", "\\frac{8}{3}", True),
    ("\\boxed{\\$347}", "347", True),
    ("\\boxed{100000}", "100,\\!000", True),
    # 실측 거짓음성 — 이제 True
    ("The answer is \\boxed{(100, 101)}.", "(100,101)", True),
    ("So \\boxed{\\frac{37}{50}}", "\\dfrac{37}{50}", True),
    ("\\boxed{y = 10x - 4}", "y=10x-4", True),
    ("\\boxed{\\dfrac{\\sqrt{6}}{2}}", "\\frac{\\sqrt6}2", True),
    # 실제 오답 — 계속 False
    ("\\boxed{153}", "306", False),
    ("\\boxed{24}", "243", False),
    ("\\boxed{C}", "0.20", False),
    ("\\boxed{(-5,7)}", "(-5,1)", False),
    ("\\boxed{15}", "30^\\circ", False),
    ("\\boxed{0.19}", "0.75", False),
]


@pytest.mark.parametrize("pred,gold,want", _PINNED)
def test_pinned(pred, gold, want):
    assert G.grade_math(pred, gold) is want


def test_selftest_passes():
    G.selftest()


def test_never_raises():
    for pred, gold in [(None, None), ("", ""), ("\\boxed{", "1"), ("\\frac{", "\\frac{")]:
        assert G.grade_math(pred, gold) in (True, False)


def test_bare_comma_list_gold_keeps_order():
    assert G.grade_math(r"\boxed{(2,1)}", "1,2") is False
    assert G.grade_math(r"\boxed{(1,2)}", "1,2") is False
    assert G.grade_math(r"\boxed{1,2}", "1,2") is True
    assert G.grade_math(r"\boxed{2,1}", "1,2") is True


def test_boxed_extraction():
    t = r"first \boxed{3} then \boxed {4}"
    assert G.boxed_answer(t) == "4"
    assert G.first_boxed(t) == "3"
    assert G.boxed_answer("no box") is None
    assert len(G.boxed_spans(t)) == 2
    assert G.boxed_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"   # 중괄호 균형


def test_answers_equivalent():
    assert G.answers_equivalent("7", "7.0")
    assert G.answers_equivalent("0.5", r"\frac{1}{2}")
    assert not G.answers_equivalent("3", "5")
    assert not G.answers_equivalent("", "3")


def test_answers_equivalent_loose():
    assert G.answers_equivalent_loose(r"0 \text{ and } -3", r"\{-3, 0\}")
    assert G.answers_equivalent_loose("2 and 3", r"\{3, 2\}")
    assert not G.answers_equivalent_loose("3", "5")
    assert not G.answers_equivalent_loose("1, 2, 3", "1, 2")      # 항 개수가 다르면 False


def test_choice_marker_stripped_from_pred():
    r"""객관식 표식이 붙은 박스(`\text{(E)}\ 7/2`)가 gold `7/2` 와 같게 채점돼야 한다."""
    assert G.grade_math(r"\boxed{\text{(E)}\ \dfrac{7}{2}}", r"\dfrac{7}{2}") is True
    assert G.grade_math(r"\boxed{\text{(C) } \frac{1}{6}}", r"\frac{1}{6}") is True
    assert G.grade_math(r"\boxed{\textbf{(B)}\ 12}", "12") is True
    assert G.grade_math(r"\boxed{(D) 5}", "5") is True
    assert G.grade_math(r"\boxed{\text{(E)}}", "E") is True          # gold 가 맨 글자
    assert G.grade_math(r"\boxed{\text{(C) } \frac{1}{2}}", r"\frac{1}{6}") is False
    assert G.grade_math(r"\boxed{C}", "0.20") is False               # 맨 글자는 표식이 아니다


# ── 객관식 선택지 대응(0924) — 학습 풀 실측 표기 ─────────────────────────────────
_AMC = ("The ratio of the area of the circle to the area of the triangle is\n$\\textbf{(A) }\\frac{\\pi r}{h+2r}"
        "\\qquad \\textbf{(B) }\\frac{\\pi r}{h+r}\\qquad \\textbf{(C) }\\frac{\\pi}{2h+r}\\qquad "
        "\\textbf{(D) }\\frac{\\pi r^2}{r^2+h^2}\\qquad \\textbf{(E) }\\text{none of these}$\n")
_PLAIN = "How many blocks differ in exactly 2 ways?\n(A) 29 (B) 39 (C) 48 (D) 56 (E) 62\n"
_TAB = ("What was her score on the sixth test? $\\textbf{(A)} 92 \\qquad\\textbf{(B)} 94 \\qquad\textbf{(C)} 96 "
        "\\qquad\\textbf{(D)} 98 \\qquad\\textbf{(E)} 100$")        # `\t` 가 탭으로 깨진 실측 행
_ASY = "[asy]\ndot(A);\ndot(B);\ndot(C);\ndot(D);\ndot(E);\n[/asy] Find the area."


def test_parse_options_real_formats():
    o = G.parse_options(_AMC)
    assert o == {"A": "\\frac{\\pi r}{h+2r}", "B": "\\frac{\\pi r}{h+r}", "C": "\\frac{\\pi}{2h+r}",
                 "D": "\\frac{\\pi r^2}{r^2+h^2}", "E": "\\text{none of these}"}
    assert G.parse_options(_PLAIN) == {"A": "29", "B": "39", "C": "48", "D": "56", "E": "62"}
    assert G.parse_options(_TAB) == {"A": "92", "B": "94", "C": "96", "D": "98", "E": "100"}
    assert G.parse_options(_ASY) is None                      # asy 의 dot(A) 는 선택지가 아니다
    assert G.parse_options("Let (A) be a set. Find |A|.") is None
    assert G.parse_options("") is None


@pytest.mark.parametrize("ans,gold,prob,want", [
    ("\\text{B}", "\\frac{\\pi r}{h+r}", _AMC, True),         # 글자 → 값(학습 풀 망침 66/82 의 꼴)
    ("\\textbf{(B)}\\ \\frac{\\pi r}{h+r}", "\\frac{\\pi r}{h+r}", _AMC, True),
    ("(B)", "\\frac{\\pi r}{h+r}", _AMC, True),
    ("\\text{A}", "\\frac{\\pi r}{h+r}", _AMC, False),        # 다른 글자는 여전히 오답
    ("\\text{A}", "29", _PLAIN, True),
    ("39", "\\text{(A)}", _PLAIN, False),                     # 역방향: 값 대 글자 gold
    ("29", "\\text{(A)}", _PLAIN, True),
    ("B", "\\text{(B)}", _PLAIN, True),                       # 글자 대 글자
    ("\\text{E}", "100", _TAB, True),
])
def test_grade_answer_with_options(ans, gold, prob, want):
    assert G.grade_answer(ans, gold, G.parse_options(prob)) is want


def test_grade_answer_without_options_is_unchanged():
    assert G.grade_answer("\\text{B}", "\\frac{\\pi r}{h+r}") is False      # 옛 채점(V5·PFX) 그대로
    assert G.grade_answer("29", "29", None) is True
    assert G.choice_value("\\text{B}", None) == "\\text{B}"
    assert G.choice_value("7", G.parse_options(_PLAIN)) == "7"         # 글자 아닌 답은 그대로
