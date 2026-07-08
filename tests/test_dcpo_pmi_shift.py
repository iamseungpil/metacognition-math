r"""CPU unit tests for the PMI-SHIFT-ACROSS-META sign-reversal reward core
(src/training/dcpo_pmi_shift.py).

Spec (2026-06-25-asymmetric-counterfactual-meta-rl-design.md, PMI-SHIFT variant):
  shift = PMI_close − PMI_open
  R_shift = scale·clip(shift) PLUS asymmetric sign-reversal bonus/penalty:
    decoy→gold (open<0, close>0): +reversal_save
    gold→decoy (open>0, close<0): −reversal_derail  (derail >= save)
  NaN/empty -> 0 (fail-closed).
"""
import numpy as np

from src.training.dcpo_pmi_shift import (
    pmi_shift_reward,
    compute_pmi_shift_reward,
)


# ── sign-reversal cases ──────────────────────────────────────────────────────
def test_save_reversal_is_positive():
    # decoy->gold: open leaning decoy (-1.0), close leaning gold (+1.0)
    r = pmi_shift_reward(-1.0, 1.0, scale=1.0, reversal_save=1.0,
                         reversal_derail=2.0, clip=2.0)
    # continuous shift = 2.0 -> clip 2.0 * scale 1.0 = 2.0; + save 1.0 = 3.0
    assert r > 0.0
    assert r == 3.0


def test_derail_reversal_is_strongly_negative():
    # gold->decoy: open leaning gold (+1.0), close leaning decoy (-1.0)
    r = pmi_shift_reward(1.0, -1.0, scale=1.0, reversal_save=1.0,
                         reversal_derail=2.0, clip=2.0)
    # continuous shift = -2.0 -> clip -2.0; - derail 2.0 = -4.0
    assert r < 0.0
    assert r == -4.0


def test_asymmetry_derail_magnitude_ge_save():
    # symmetric belief moves of equal magnitude: derail must hurt >= save helps
    save = pmi_shift_reward(-1.0, 1.0, scale=1.0, reversal_save=1.0,
                            reversal_derail=2.0, clip=2.0)
    derail = pmi_shift_reward(1.0, -1.0, scale=1.0, reversal_save=1.0,
                              reversal_derail=2.0, clip=2.0)
    assert abs(derail) >= abs(save)
    # and strictly bigger with the default asymmetric knobs
    assert abs(derail) > abs(save)


def test_no_shift_near_zero():
    # belief unchanged (open==close, no reversal) -> ~0
    r = pmi_shift_reward(0.5, 0.5, scale=1.0, reversal_save=1.0,
                         reversal_derail=2.0, clip=2.0)
    assert r == 0.0


def test_zero_crossing_exactly_at_zero_not_a_reversal():
    # close lands exactly at 0 -> NOT a save/derail (flat belief)
    r_save = pmi_shift_reward(-1.0, 0.0, scale=1.0, reversal_save=5.0,
                              reversal_derail=9.0, clip=2.0)
    # only the continuous term (shift=+1.0), no +5.0 bonus
    assert r_save == 1.0
    r_der = pmi_shift_reward(1.0, 0.0, scale=1.0, reversal_save=5.0,
                             reversal_derail=9.0, clip=2.0)
    assert r_der == -1.0


def test_same_sign_no_reversal_only_continuous():
    # open and close both positive (gold-leaning both) -> no reversal, just shift
    r = pmi_shift_reward(0.2, 1.2, scale=1.0, reversal_save=5.0,
                         reversal_derail=9.0, clip=2.0)
    assert r == 1.0  # shift 1.0, no bonus


# ── eps threshold removes the zero-crossing discontinuity ────────────────────
def test_reversal_min_magnitude_suppresses_marginal_crossing():
    # A marginal crossing 0.0->0.1 (open exactly 0 -> not a reversal already), and a
    # near-zero crossing -0.05->0.05 within eps must NOT earn the save bonus.
    r = pmi_shift_reward(-0.05, 0.05, scale=1.0, reversal_save=5.0,
                         reversal_derail=9.0, clip=2.0, reversal_min_magnitude=0.1)
    # only the continuous shift (0.1), NO +5.0 save bonus (both within eps band)
    assert abs(r - 0.1) < 1e-9


def test_reversal_min_magnitude_allows_genuine_flip():
    # A genuine flip clears eps on both sides -> save bonus applies.
    r = pmi_shift_reward(-1.0, 1.0, scale=1.0, reversal_save=5.0,
                         reversal_derail=9.0, clip=2.0, reversal_min_magnitude=0.1)
    # continuous shift clip 2.0 + save 5.0 = 7.0
    assert r == 7.0


def test_reversal_min_magnitude_default_zero_is_backcompat():
    # eps=0 (default) reproduces the original step-function behavior exactly.
    r = pmi_shift_reward(-1.0, 1.0, scale=1.0, reversal_save=1.0,
                         reversal_derail=2.0, clip=2.0)
    assert r == 3.0


def test_reversal_min_magnitude_one_side_within_eps_no_bonus():
    # open clears eps but close does not (0.05 < eps 0.1) -> NOT a reversal.
    r = pmi_shift_reward(-1.0, 0.05, scale=1.0, reversal_save=5.0,
                         reversal_derail=9.0, clip=2.0, reversal_min_magnitude=0.1)
    # continuous shift 1.05, no save bonus
    assert abs(r - 1.05) < 1e-9


# ── scale taken as magnitude (negative config cannot invert reward) ──────────
def test_scale_negative_is_treated_as_magnitude():
    pos = pmi_shift_reward(0.2, 0.7, scale=4.0, reversal_save=1.0,
                           reversal_derail=2.0, clip=2.0)
    neg = pmi_shift_reward(0.2, 0.7, scale=-4.0, reversal_save=1.0,
                           reversal_derail=2.0, clip=2.0)
    # negative scale must NOT flip the sign of the continuous term
    assert abs(neg - pos) < 1e-9
    assert neg > 0.0


# ── clip ─────────────────────────────────────────────────────────────────────
def test_clip_bounds_continuous_term():
    # huge shift clipped to +clip
    r = pmi_shift_reward(-10.0, 10.0, scale=1.0, reversal_save=1.0,
                         reversal_derail=2.0, clip=2.0)
    # shift 20 -> clip 2.0; + save 1.0 = 3.0
    assert r == 3.0
    # negative huge shift clipped to -clip (gold->decoy derail)
    r2 = pmi_shift_reward(10.0, -10.0, scale=1.0, reversal_save=1.0,
                          reversal_derail=2.0, clip=2.0)
    assert r2 == -4.0  # -2.0 clip - 2.0 derail


def test_scale_multiplies_continuous_only():
    r = pmi_shift_reward(0.2, 0.7, scale=4.0, reversal_save=1.0,
                         reversal_derail=2.0, clip=2.0)
    # shift 0.5 * scale 4 = 2.0, no reversal
    assert abs(r - 2.0) < 1e-6


# ── NaN / fail-closed ────────────────────────────────────────────────────────
def test_nan_open_fails_closed():
    assert pmi_shift_reward(float("nan"), 1.0) == 0.0


def test_nan_close_fails_closed():
    assert pmi_shift_reward(1.0, float("inf")) == 0.0


# ── batch wrapper ────────────────────────────────────────────────────────────
def test_compute_pmi_shift_reward_batch_and_counts():
    rows = [
        {"pmi_open": -1.0, "pmi_close": 1.0},   # save
        {"pmi_open": 1.0, "pmi_close": -1.0},   # derail
        {"pmi_open": 0.5, "pmi_close": 0.5},    # neutral
        {"pmi_open": None, "pmi_close": 1.0},   # fail-closed
        {"pmi_open": 1.0, "pmi_close": float("nan")},  # fail-closed
    ]
    r, diag = compute_pmi_shift_reward(
        rows, scale=1.0, reversal_save=1.0, reversal_derail=2.0, clip=2.0)
    assert r.dtype == np.float32
    assert r[0] == 3.0
    assert r[1] == -4.0
    assert r[2] == 0.0
    assert r[3] == 0.0 and diag["failures"][3] is True
    assert r[4] == 0.0 and diag["failures"][4] is True
    assert diag["n_save"] == 1
    assert diag["n_derail"] == 1
    assert np.isnan(diag["raw_shift"][3])
    assert np.isfinite(diag["raw_shift"][0])


def test_compute_pmi_shift_reward_eps_gates_counts_and_reward():
    # marginal crossing within eps must NOT count as a save NOR pay a save bonus.
    rows = [
        {"pmi_open": -0.05, "pmi_close": 0.05},  # within eps -> neutral
        {"pmi_open": -1.0, "pmi_close": 1.0},    # genuine flip -> save
    ]
    r, diag = compute_pmi_shift_reward(
        rows, scale=1.0, reversal_save=5.0, reversal_derail=9.0, clip=2.0,
        reversal_min_magnitude=0.1)
    assert diag["n_save"] == 1   # only the genuine flip counts
    assert diag["n_derail"] == 0
    # row0 gets ONLY the continuous shift (0.1), no +5.0 bonus
    assert abs(float(r[0]) - 0.1) < 1e-6


# ── signal-test pure helpers (src/eval/pmi_shift_signal.py) ──────────────────
from src.eval.pmi_shift_signal import (
    parse_open_close,
    rank_auc,
    reversal_label,
    corrupt_meta_text,
    gold_is_default,
    _pmi_position,
    _summarize,
)


def test_parse_open_close_splits_at_tags():
    body_open, body_close = parse_open_close("abc<|meta|>check<|/meta|>tail")
    assert body_open == "abc"
    assert body_close == "abc<|meta|>check<|/meta|>"


def test_parse_open_close_none_without_close():
    assert parse_open_close("abc<|meta|>unclosed tail") is None
    assert parse_open_close("no meta here") is None


def test_reversal_label_cases():
    assert reversal_label(-1.0, 1.0) == 1     # decoy->gold SAVE
    assert reversal_label(1.0, -1.0) == -1    # gold->decoy DERAIL
    assert reversal_label(0.5, 0.5) == 0
    assert reversal_label(-1.0, 0.0) == 0     # lands at 0, not a reversal


def test_rank_auc_perfect_separation():
    auc = rank_auc([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0])
    assert auc == 1.0


def test_pmi_position_sums_divergent_only():
    # gold_lp - decoy_lp = [1, 1, 1]; mask picks positions 0 and 2 -> sum 2.0
    val = _pmi_position([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [True, False, True])
    assert val == 2.0


def test_pmi_position_length_mismatch_fails_closed():
    # decoy shorter than gold: previously zero-PADDED (inflating contrast); now NaN.
    import numpy as _np
    val = _pmi_position([1.0, 1.0, 1.0, 1.0], [0.0, 0.0], [True, True, True, True])
    assert _np.isnan(val)


def test_pmi_position_zero_divergent_is_nan():
    # all-False mask (gold==decoy on the span) -> no divergent positions -> NaN.
    import numpy as _np
    val = _pmi_position([1.0, 1.0], [0.5, 0.5], [False, False])
    assert _np.isnan(val)


# ── placebo / safe-default helpers ───────────────────────────────────────────
def test_corrupt_meta_preserves_tags_and_length_changes_content():
    block = "<|meta|>let me verify this carefully here<|/meta|>"
    out = corrupt_meta_text(block, seed=1)
    assert out.startswith("<|meta|>")
    assert out.endswith("<|/meta|>")
    # same token multiset, content (order) destroyed for a seed that permutes
    inner_in = block[len("<|meta|>"):-len("<|/meta|>")].split()
    inner_out = out[len("<|meta|>"):-len("<|/meta|>")].split()
    assert sorted(inner_in) == sorted(inner_out)


def test_corrupt_meta_single_token_unchanged():
    block = "<|meta|>verify<|/meta|>"
    assert corrupt_meta_text(block, seed=1) == block


def test_gold_is_default_flag():
    assert gold_is_default(0.5) == 1   # model already leans gold at OPEN
    assert gold_is_default(-0.5) == 0
    assert gold_is_default(0.0) == 0


def test_summarize_confound_stratification():
    # own!=gold rows where shift STILL points to gold (positive) = genuine update.
    results = [
        {"pmi_open": -1.0, "pmi_close": 1.0, "shift": 2.0, "shift_placebo": 0.1,
         "r_shift": 3.0, "reversal": 1, "correct": 1, "own_eq_gold": 0,
         "gold_is_default": 0},
        {"pmi_open": -1.0, "pmi_close": 0.5, "shift": 1.5, "shift_placebo": 0.0,
         "r_shift": 1.5, "reversal": 1, "correct": 0, "own_eq_gold": 0,
         "gold_is_default": 0},
        {"pmi_open": 1.0, "pmi_close": 2.0, "shift": 1.0, "shift_placebo": 0.9,
         "r_shift": 1.0, "reversal": 0, "correct": 1, "own_eq_gold": 1,
         "gold_is_default": 1},
    ]
    rep = _summarize(results)
    assert rep["n_scored"] == 3
    assert rep["confound"]["n_own_ne_gold"] == 2
    assert rep["confound"]["n_own_eq_gold"] == 1
    # own!=gold mean close-PMI is positive (toward gold) -> genuine, not confound.
    assert rep["confound"]["mean_pmi_close_own_ne_gold"] > 0.0
    assert rep["confound"]["save_rate_own_ne_gold"] == 1.0
    # placebo gap large (real 1.75 vs placebo 0.05) -> content-driven (genuine).
    assert rep["confound"]["placebo"]["placebo_gap_own_ne_gold"] > 1.0


def test_summarize_reports_missing_own_eq_gold():
    # own_eq_gold None (extraction/checker unavailable) must be counted, not silent.
    results = [
        {"pmi_open": -1.0, "pmi_close": 1.0, "shift": 2.0, "shift_placebo": 0.1,
         "r_shift": 3.0, "reversal": 1, "correct": 1, "own_eq_gold": None,
         "gold_is_default": 0},
        {"pmi_open": -1.0, "pmi_close": 0.5, "shift": 1.5, "shift_placebo": 0.0,
         "r_shift": 1.5, "reversal": 1, "correct": 0, "own_eq_gold": 0,
         "gold_is_default": 0},
    ]
    rep = _summarize(results)
    assert rep["confound"]["n_own_eq_gold_missing"] == 1
    assert rep["confound"]["n_own_ne_gold"] == 1


def test_summarize_dropped_counts_passthrough():
    results = [
        {"pmi_open": -1.0, "pmi_close": 1.0, "shift": 2.0, "shift_placebo": 0.1,
         "r_shift": 3.0, "reversal": 1, "correct": 1, "own_eq_gold": 0,
         "gold_is_default": 0},
    ]
    rep = _summarize(results, diag_counts={"n_len_mismatch": 3, "n_zero_divergent": 2})
    assert rep["dropped"]["n_len_mismatch"] == 3
    assert rep["dropped"]["n_zero_divergent"] == 2


def test_summarize_confound_check_is_fooled_by_presence_confound():
    """DEMONSTRATION (review finding #1/#6): the own≠gold stratification ALONE does
    NOT catch a meta-presence-as-confidence confound. Here shift>0 on own≠gold (so
    the bare confound check passes / looks 'genuine'), but the PLACEBO gap is ~0 —
    the shift survives content destruction, i.e. it is presence-driven, NOT content.
    The bare own≠gold metrics are fooled; only the placebo check exposes it."""
    results = [
        # own≠gold, shift>0 (bare check says 'genuine'), but placebo shift == real.
        {"pmi_open": -1.0, "pmi_close": 1.0, "shift": 2.0, "shift_placebo": 2.0,
         "r_shift": 3.0, "reversal": 1, "correct": 1, "own_eq_gold": 0,
         "gold_is_default": 0},
        {"pmi_open": -1.0, "pmi_close": 0.5, "shift": 1.5, "shift_placebo": 1.5,
         "r_shift": 1.5, "reversal": 1, "correct": 0, "own_eq_gold": 0,
         "gold_is_default": 0},
    ]
    rep = _summarize(results)
    c = rep["confound"]
    # Bare own≠gold check is FOOLED: positive shift / positive close-PMI look genuine.
    assert c["mean_pmi_close_own_ne_gold"] > 0.0
    assert c["mean_shift_own_ne_gold"] > 0.0
    # ...but the placebo gap ~0 EXPOSES the presence-as-confidence confound.
    assert abs(c["placebo"]["placebo_gap_own_ne_gold"]) < 1e-9


def test_summarize_safe_default_substratification():
    # own≠gold split into gold-is-default vs not: shift>0 ONLY in default subset =
    # confounded; here non-default also has shift>0 -> more likely genuine.
    results = [
        {"pmi_open": 1.0, "pmi_close": 2.0, "shift": 1.0, "shift_placebo": 0.0,
         "r_shift": 1.0, "reversal": 0, "correct": 1, "own_eq_gold": 0,
         "gold_is_default": 1},
        {"pmi_open": -1.0, "pmi_close": 1.0, "shift": 2.0, "shift_placebo": 0.0,
         "r_shift": 3.0, "reversal": 1, "correct": 1, "own_eq_gold": 0,
         "gold_is_default": 0},
    ]
    rep = _summarize(results)
    sd = rep["confound"]["safe_default"]
    assert sd["n_own_ne_gold_is_default"] == 1
    assert sd["n_own_ne_gold_not_default"] == 1
    assert sd["mean_shift_own_ne_gold_not_default"] == 2.0
