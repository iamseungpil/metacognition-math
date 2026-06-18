"""Tests for the redirect-priming continuous CF reward (torch-free module).
Covers intent-check w4udybnbv fixes C1-C5 + the continuous-regime thresholds."""
from src.training.redirect_cf import redirect_cf_rmeta, rmeta_pos, rmeta_neg

NOJUDGE = lambda s: False   # stub: "no prose redirect in this draw"
# clean continuation: >=20 tokens, has digits (parses), no redirect cue, not repetitive
CLEAN = ("Janet collects 16 eggs each morning from her ducks, then she eats 3 for breakfast "
         "and bakes 4 into muffins, so 9 eggs remain and at 2 dollars each that gives 18 total.")


def test_C1_non_emit_row_cannot_earn_positive():
    # straight-solved row (no switch emitted), suppressed draws fail -> must NOT be +1
    r = redirect_cf_rmeta(c_with=1, c_without_draws=[0, 0], emit_switch=False,
                          cf_texts=[CLEAN, CLEAN], in_hard_band=True, llm_judge=NOJUDGE)
    assert r <= 0.0


def test_C1_positive_credit_when_emit_and_in_band():
    r = redirect_cf_rmeta(c_with=1, c_without_draws=[0, 0], emit_switch=True,
                          cf_texts=[CLEAN, CLEAN], in_hard_band=True, llm_judge=NOJUDGE)
    assert r > 0.0 and rmeta_pos(r)


def test_C2_all_invalid_draws_abstain():
    # every suppressed draw is degenerate -> CF never established -> abstain (0.0), not +1
    r = redirect_cf_rmeta(c_with=1, c_without_draws=[1, 1], emit_switch=True,
                          cf_texts=["the the the the the the the the", "hi"],
                          in_hard_band=True, llm_judge=NOJUDGE)
    assert r == 0.0


def test_C3_hollow_with_arm_abstains():
    r = redirect_cf_rmeta(c_with=1, c_without_draws=[0], emit_switch=True, cf_texts=[CLEAN],
                          in_hard_band=True, with_text="the the the the the the the the", llm_judge=NOJUDGE)
    assert r == 0.0


def test_C4_live_path_requires_judge():
    import pytest
    with pytest.raises(ValueError):
        # default regex_only=False + no judge -> detect_redirect fails closed
        redirect_cf_rmeta(c_with=1, c_without_draws=[0], emit_switch=True,
                          cf_texts=[CLEAN], in_hard_band=True)


def test_C5_unnecessary_redirect_penalty_survives_out_of_band():
    # easy row, redirect emitted, suppressed arm also correct -> -lam even out of band
    r = redirect_cf_rmeta(c_with=1, c_without_draws=[1], emit_switch=True, cf_texts=[CLEAN],
                          in_hard_band=False, lam=0.25, llm_judge=NOJUDGE)
    assert r == -0.25


def test_negative_term_in_band():
    r = redirect_cf_rmeta(c_with=1, c_without_draws=[1], emit_switch=True, cf_texts=[CLEAN],
                          in_hard_band=True, lam=0.25, llm_judge=NOJUDGE)
    assert r == -0.25  # base 0 - lam


def test_thresholds():
    assert rmeta_pos(0.25) and not rmeta_pos(0.24)
    assert rmeta_neg(-0.25) and not rmeta_neg(-0.24)
