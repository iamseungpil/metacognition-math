"""Tests for src/eval/rml_verdict.py — the R-B' causal-measurement core.

The estimand is R - B' (redirect CONTENT given a matched second attempt). These
tests pin down row usability (parse / off-policy degeneracy / B' still
redirecting), the saved/broke McNemar counts, the placebo-margin requirement,
the monotone-saturating bias-sweep gate, and the composed final verdict.

regex_only=True (or a stub judge) is used so tests stay GPU-free. Spec
2026-06-18 REV-6 §5.
"""
from src.eval.rml_verdict import (
    row_usable,
    saved_broke,
    beats_placebo,
    is_monotone_saturating,
    rml_verdict,
)


# ---- helpers -------------------------------------------------------------

CLEAN = (
    "We carefully compute the value step by step, multiplying the two given "
    "numbers and then adding the constant offset from the problem statement to "
    "obtain a result we can verify. The final answer is \\boxed{42}."
)
CLEAN_WRONG = (
    "We carefully compute the value step by step, multiplying the two given "
    "numbers and then adding the constant offset from the problem statement but "
    "making a small arithmetic slip somewhere. The final answer is \\boxed{7}."
)
REDIRECT_PROSE = (
    "We carefully compute the value step by step, multiplying the numbers and "
    "adding the offset. Hmm, wait, that is wrong. Let me reconsider and try a "
    "different approach entirely. The final answer is \\boxed{42}."
)


def _arm(correct, text=CLEAN):
    return {"correct": correct, "text": text}


def _row(r_correct, b_correct, p_correct=True,
         r_text=CLEAN, b_text=CLEAN):
    return {
        "R": _arm(r_correct, r_text),
        "Bprime": _arm(b_correct, b_text),
        "P": {"correct": p_correct},
    }


def _saved_row(p_correct=False):
    # R right, B' wrong -> redirect SAVED it; placebo also wrong by default
    return _row(True, False, p_correct=p_correct,
                r_text=CLEAN, b_text=CLEAN_WRONG)


# ---- row_usable ----------------------------------------------------------

def test_row_usable_clean_pair():
    assert row_usable(_row(True, False), regex_only=True) is True


def test_row_usable_drops_unparseable_arm():
    row = _row(True, False, b_text="no answer here just words and prose")
    # b_text has digits? none -> not parsed -> drop
    assert row_usable({"R": _arm(True, CLEAN),
                       "Bprime": _arm(False, "qwerty asdf no number")},
                      regex_only=True) is False


def test_row_usable_drops_degenerate_bprime_when_r_clean():
    # B' is degenerate (repetition loop) while R is clean -> off-policy artifact, drop
    loop = ("answer " * 30) + "\\boxed{1}"
    row = {"R": _arm(True, CLEAN), "Bprime": _arm(False, loop)}
    assert row_usable(row, regex_only=True) is False


def test_row_usable_drops_bprime_that_still_redirects():
    # B' routed around the ban in prose -> drop (would fake separability)
    row = {"R": _arm(True, CLEAN), "Bprime": _arm(False, REDIRECT_PROSE)}
    assert row_usable(row, regex_only=True) is False


def test_row_usable_stub_judge():
    # judge says B' redirects -> drop
    judge = lambda t: "different approach" in t
    row = {"R": _arm(True, CLEAN), "Bprime": _arm(False, REDIRECT_PROSE)}
    assert row_usable(row, llm_judge=judge) is False
    # judge says no redirect -> kept (parse + degeneracy ok)
    row2 = {"R": _arm(True, CLEAN), "Bprime": _arm(False, CLEAN_WRONG)}
    assert row_usable(row2, llm_judge=lambda t: False) is True


# ---- saved_broke ---------------------------------------------------------

def test_saved_broke_counts():
    rows = (
        [_row(True, False) for _ in range(12)]   # saved (b)
        + [_row(False, True) for _ in range(3)]  # broke (c)
        + [_row(True, True) for _ in range(5)]   # concordant
        + [_row(False, False) for _ in range(4)]  # concordant
    )
    b, c = saved_broke(rows, regex_only=True)
    assert b == 12
    assert c == 3


def test_saved_broke_ignores_unusable_rows():
    rows = [_row(True, False) for _ in range(10)]
    # add a B'-still-redirects row that would have been a (b) save if counted
    rows.append({"R": _arm(True, CLEAN), "Bprime": _arm(False, REDIRECT_PROSE)})
    b, c = saved_broke(rows, regex_only=True)
    assert b == 10
    assert c == 0


# ---- beats_placebo -------------------------------------------------------

def test_beats_placebo_true():
    # acc_R=1.0, acc_Bprime=0.0, acc_P=0.0 -> (1-0) > (1-0)? equal -> NOT a win
    # Make placebo do BETTER than B' so redirect advantage over B' exceeds over P.
    # acc_R - acc_Bprime must exceed acc_R - acc_P  <=>  acc_P > acc_Bprime
    rows = [_saved_row(p_correct=True) for _ in range(10)]
    # acc_R=1, acc_Bprime=0, acc_P=1 -> (1-0)=1 > (1-1)=0  TRUE
    assert beats_placebo(rows, regex_only=True) is True


def test_beats_placebo_tie_is_false():
    # placebo no better than B' -> redirect's edge over B' is not above its edge over P
    rows = [_saved_row(p_correct=False) for _ in range(10)]
    # acc_R=1, acc_Bprime=0, acc_P=0 -> (1-0)=1 > (1-0)=1 ? no (not strictly) -> False
    assert beats_placebo(rows, regex_only=True) is False


def test_beats_placebo_uses_usable_rows_only():
    rows = [_saved_row(p_correct=True) for _ in range(10)]
    rows.append({"R": _arm(True, CLEAN),
                 "Bprime": _arm(True, REDIRECT_PROSE),  # dropped
                 "P": {"correct": False}})
    assert beats_placebo(rows, regex_only=True) is True


# ---- is_monotone_saturating ---------------------------------------------

def test_monotone_saturating_true():
    # increments: 0.4, 0.1, 0.05 ; last (0.05) <= 0.25*first (0.1) -> saturating
    eff = [(2, 0.0), (5, 0.4), (20, 0.5), (1e9, 0.55)]
    assert is_monotone_saturating(eff) is True


def test_non_monotone_is_false():
    eff = [(2, 0.0), (5, 0.4), (20, 0.3), (1e9, 0.6)]
    assert is_monotone_saturating(eff) is False


def test_non_saturating_blowup_is_false():
    # keeps growing: increments 0.2, 0.2, 0.2 -> last not <= 0.25*first -> artifact
    eff = [(2, 0.0), (5, 0.2), (20, 0.4), (1e9, 0.6)]
    assert is_monotone_saturating(eff) is False


# ---- rml_verdict ---------------------------------------------------------

def _mono():
    return [(2, 0.0), (5, 0.4), (20, 0.5), (1e9, 0.55)]


def test_verdict_clean_win_significant():
    rows = (
        [_saved_row(p_correct=True) for _ in range(20)]
        + [_row(False, True, p_correct=True) for _ in range(2)]
    )
    out = rml_verdict(rows, effects_by_bias=_mono(), regex_only=True)
    assert out["status"] == "SIGNIFICANT"
    assert out["b"] == 20
    assert out["c"] == 2
    assert out["n_usable"] == 22
    assert out["beats_placebo"] is True
    assert out["monotone_saturating"] is True


def test_verdict_bprime_redirects_rows_dropped():
    # all B' arms still redirect -> dropped -> insufficient
    rows = [{"R": _arm(True, CLEAN),
             "Bprime": _arm(False, REDIRECT_PROSE),
             "P": {"correct": False}} for _ in range(20)]
    out = rml_verdict(rows, regex_only=True)
    assert out["n_usable"] == 0
    assert out["status"] == "INSUFFICIENT"


def test_verdict_placebo_tie_not_significant():
    # strong saved>broke + monotone, but placebo not beaten -> INSUFFICIENT (not SIG)
    rows = [_saved_row(p_correct=False) for _ in range(20)]
    out = rml_verdict(rows, effects_by_bias=_mono(), regex_only=True)
    assert out["beats_placebo"] is False
    assert out["status"] == "INSUFFICIENT"


def test_verdict_underpowered_few_rows():
    # beats placebo, monotone, but n_usable >= min_discordant yet b+c too small
    rows = (
        [_saved_row(p_correct=True) for _ in range(8)]
        + [_row(True, True, p_correct=True) for _ in range(6)]  # concordant pad
    )
    # n_usable = 14 (>= min_discordant 10) but discordant b+c = 8 < 10 -> UNDERPOWERED
    out = rml_verdict(rows, effects_by_bias=_mono(), regex_only=True)
    assert out["n_usable"] == 14
    assert out["status"] == "UNDERPOWERED"


def test_verdict_insufficient_when_too_few_usable():
    rows = [_saved_row(p_correct=True) for _ in range(5)]  # 5 < min_discordant 10
    out = rml_verdict(rows, effects_by_bias=_mono(), regex_only=True)
    assert out["n_usable"] == 5
    assert out["status"] == "INSUFFICIENT"


def test_verdict_non_monotone_sweep_insufficient():
    rows = [_saved_row(p_correct=True) for _ in range(20)]
    bad = [(2, 0.0), (5, 0.4), (20, 0.3), (1e9, 0.6)]  # non-monotone
    out = rml_verdict(rows, effects_by_bias=bad, regex_only=True)
    assert out["monotone_saturating"] is False
    assert out["status"] == "INSUFFICIENT"


def test_verdict_no_sweep_monotone_is_none():
    rows = (
        [_saved_row(p_correct=True) for _ in range(20)]
        + [_row(False, True, p_correct=True) for _ in range(2)]
    )
    out = rml_verdict(rows, effects_by_bias=None, regex_only=True)
    assert out["monotone_saturating"] is None
    assert out["status"] == "SIGNIFICANT"
