"""Unit tests for the redirect-priming CONTINUOUS counterfactual R_meta helper
(src.training.dcpo_region.redirect_cf_rmeta + rmeta_pos/rmeta_neg), spec
2026-06-18 §3. These cover the NEW additive path only; the legacy R_meta = c_with
- c_without behavior is exercised by tests/test_dcpo_rewards.py and is unchanged.

PURE PYTHON. regex_only=True (the cheap pre-filter path) is used throughout so no
LLM judge is needed (spec §5.3 allows it for the pre-filter / tests).
"""
from src.training.dcpo_region import redirect_cf_rmeta, rmeta_pos, rmeta_neg

# A clean, correct, non-redirecting suppressed continuation (>= min_len tokens,
# high unique ratio, has a final boxed answer, no redirect cue).
CLEAN_CORRECT = (
    "We carefully add the two given quantities together and then simplify the "
    "resulting expression to obtain the final boxed value \\boxed{4}"
)
# Same length/shape but redirects in prose ("let me reconsider" / "instead").
REDIRECTING = (
    "Hmm, let me reconsider this problem from scratch using a different approach "
    "instead, because my first attempt clearly went wrong \\boxed{4}"
)
# Garbled / degenerate: a long single-token run -> degeneracy_flags repetition.
GARBLED = "no no no no no no no no no no no no no no no no \\boxed{4}"


def test_continuous_c_without_averaging():
    # 2 clean-correct + 2 clean-wrong -> c_without = 0.5; c_with=1 -> base=0.5.
    r = redirect_cf_rmeta(
        c_with=1,
        c_without_draws=[1, 1, 0, 0],
        emit_switch=False,
        cf_texts=[CLEAN_CORRECT, CLEAN_CORRECT,
                  "The simplification gives a clearly different final value \\boxed{9}",
                  "Another careful computation yields a distinct final answer \\boxed{7}"],
        in_hard_band=True,
    )
    assert abs(r - 0.5) < 1e-9


def test_garbled_draw_excluded():
    # correct=1 but degenerate text -> NOT counted as a success. Single draw,
    # so c_without_continuous = 0 -> base = c_with - 0 = 1.
    r = redirect_cf_rmeta(
        c_with=1,
        c_without_draws=[1],
        emit_switch=False,
        cf_texts=[GARBLED],
        in_hard_band=True,
    )
    assert abs(r - 1.0) < 1e-9


def test_redirecting_draw_excluded():
    # correct=1 and clean, but STILL redirects in prose -> not a meta-free
    # continuation -> excluded -> c_without = 0 -> base = 1.
    r = redirect_cf_rmeta(
        c_with=1,
        c_without_draws=[1],
        emit_switch=False,
        cf_texts=[REDIRECTING],
        in_hard_band=True,
    )
    assert abs(r - 1.0) < 1e-9


def test_negative_term_fires_when_emit_and_suppressed_correct():
    # emit_switch True AND suppressed arm correct (c_without=1 >= 0.5) -> the
    # redirect was UNNECESSARY -> subtract lam. base = 1 - 1 = 0; r = 0 - 0.25.
    r = redirect_cf_rmeta(
        c_with=1,
        c_without_draws=[1],
        emit_switch=True,
        cf_texts=[CLEAN_CORRECT],
        in_hard_band=True,
        lam=0.25,
    )
    assert abs(r - (-0.25)) < 1e-9


def test_negative_term_not_fired_without_emit():
    # Same as above but emit_switch False -> no penalty; base = 0.
    r = redirect_cf_rmeta(
        c_with=1,
        c_without_draws=[1],
        emit_switch=False,
        cf_texts=[CLEAN_CORRECT],
        in_hard_band=True,
    )
    assert abs(r - 0.0) < 1e-9


def test_negative_term_not_fired_when_suppressed_below_threshold():
    # emit_switch True but c_without_continuous = 0 (< 0.5) -> redirect was
    # warranted -> no penalty. base = 1 - 0 = 1.
    r = redirect_cf_rmeta(
        c_with=1,
        c_without_draws=[0],
        emit_switch=True,
        cf_texts=["The recomputation gives a different final value \\boxed{9}"],
        in_hard_band=True,
    )
    assert abs(r - 1.0) < 1e-9


def test_positive_credited_only_in_hard_band():
    args = dict(
        c_with=1,
        c_without_draws=[0],
        emit_switch=False,
        cf_texts=["A clean independent computation lands on a different value \\boxed{9}"],
    )
    # In band: positive base = 1 - 0 = 1 is credited.
    r_in = redirect_cf_rmeta(in_hard_band=True, **args)
    assert abs(r_in - 1.0) < 1e-9
    # Out of band: positive base is clamped to 0 (no reward on easy rows).
    r_out = redirect_cf_rmeta(in_hard_band=False, **args)
    assert abs(r_out - 0.0) < 1e-9


def test_penalty_allowed_outside_hard_band():
    # Negative base must still pass through when not in_hard_band (penalty-only).
    # c_with=0, suppressed arm correct -> base = 0 - 1 = -1.
    r = redirect_cf_rmeta(
        c_with=0,
        c_without_draws=[1],
        emit_switch=False,
        cf_texts=[CLEAN_CORRECT],
        in_hard_band=False,
    )
    assert abs(r - (-1.0)) < 1e-9


def test_no_draws_gives_c_without_zero():
    r = redirect_cf_rmeta(
        c_with=1,
        c_without_draws=[],
        emit_switch=False,
        cf_texts=[],
        in_hard_band=True,
    )
    assert abs(r - 1.0) < 1e-9


def test_rmeta_pos_neg_thresholds_at_quarter():
    # POSITIVE at +0.25 boundary (inclusive), not below.
    assert rmeta_pos(0.25) is True
    assert rmeta_pos(0.2499) is False
    assert rmeta_pos(1.0) is True
    # NEGATIVE at -0.25 boundary (inclusive), not above.
    assert rmeta_neg(-0.25) is True
    assert rmeta_neg(-0.2499) is False
    assert rmeta_neg(-1.0) is True
    # Mid-zone is neither.
    assert rmeta_pos(0.0) is False
    assert rmeta_neg(0.0) is False
