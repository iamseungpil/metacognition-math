"""Unit tests for the INTENT metrics (does meta work as intended).

Covers well-calibrated vs over-confident, appropriate vs misfiring actions,
causal-redirect rate, accuracy delta, and meta survival over RL steps.
"""
import math

from src.eval.redirect_verify_metrics import (
    confidence_calibration,
    action_appropriateness,
    redirect_causal_rate,
    accuracy_delta,
    meta_survival,
    intent_report,
)


# ----------------------------------------------------------------------------
# (1) confidence_calibration
# ----------------------------------------------------------------------------
def test_calibration_perfect_is_zero_ece():
    # confidence exactly equals empirical correctness within each bin:
    # the 0.0-conf bin is all wrong, the 1.0-conf bin is all right.
    confs = [0.0, 0.0, 1.0, 1.0]
    correct = [False, False, True, True]
    r = confidence_calibration(confs, correct, n_bins=10)
    assert r["ece"] == 0.0
    assert r["n"] == 4


def test_calibration_overconfident_has_gap_and_signed_positive():
    # always says 0.9 confident but only right half the time -> overconfident
    confs = [0.9, 0.9, 0.9, 0.9]
    correct = [True, False, True, False]
    r = confidence_calibration(confs, correct, n_bins=10)
    assert r["ece"] > 0.3
    # signed_gap = mean_conf - mean_acc > 0 => over-confident
    assert r["signed_gap"] > 0
    assert math.isclose(r["mean_conf"], 0.9)
    assert math.isclose(r["mean_acc"], 0.5)


def test_calibration_underconfident_signed_negative():
    confs = [0.2, 0.2, 0.2, 0.2]
    correct = [True, True, True, False]
    r = confidence_calibration(confs, correct, n_bins=10)
    assert r["signed_gap"] < 0  # under-confident: acc exceeds conf


def test_calibration_empty_returns_none_safely():
    r = confidence_calibration([], [], n_bins=10)
    assert r["n"] == 0
    assert r["ece"] is None


def test_calibration_length_mismatch_raises():
    try:
        confidence_calibration([0.5], [True, False])
    except ValueError:
        return
    raise AssertionError("expected ValueError on length mismatch")


def test_calibration_clamps_out_of_range_conf():
    # confidences outside [0,1] should be clamped, not crash binning
    r = confidence_calibration([1.4, -0.3], [True, False], n_bins=10)
    assert r["ece"] is not None


# ----------------------------------------------------------------------------
# (2) action_appropriateness
# ----------------------------------------------------------------------------
def _rec(action, conf, recoverable_wrong, flipped=None, confirmed=None):
    return {
        "action": action,                       # 'redirect' | 'verify' | 'none'
        "confidence": conf,
        "recoverable_wrong": recoverable_wrong,  # wrong-but-fixable (gold-derived)
        "flipped_to_right": flipped,             # for redirect: wrong->right?
        "verify_confirmed": confirmed,           # for verify: confirmed/corrected?
    }


def test_appropriateness_all_appropriate():
    recs = [
        # redirect on a recoverable-wrong, low-conf case -> appropriate
        _rec("redirect", 0.2, recoverable_wrong=True, flipped=True),
        # verify on a high-conf case -> appropriate
        _rec("verify", 0.9, recoverable_wrong=False, confirmed=True),
    ]
    r = action_appropriateness(recs, high_conf=0.7, low_conf=0.5)
    assert r["redirect_appropriate_rate"] == 1.0
    assert r["verify_appropriate_rate"] == 1.0
    assert r["redirect_misfire_rate"] == 0.0
    assert r["verify_misfire_rate"] == 0.0


def test_appropriateness_redirect_misfire_on_high_conf_correct():
    # redirect emitted on a NOT-recoverable-wrong, high-conf case -> misfire
    recs = [
        _rec("redirect", 0.95, recoverable_wrong=False, flipped=False),
        _rec("redirect", 0.1, recoverable_wrong=True, flipped=True),
    ]
    r = action_appropriateness(recs, high_conf=0.7, low_conf=0.5)
    assert math.isclose(r["redirect_appropriate_rate"], 0.5)
    assert math.isclose(r["redirect_misfire_rate"], 0.5)
    assert r["n_redirect"] == 2


def test_appropriateness_verify_misfire_on_low_conf():
    # verify emitted on a low-conf case is a misfire (should redirect instead)
    recs = [
        _rec("verify", 0.2, recoverable_wrong=True, confirmed=False),
        _rec("verify", 0.9, recoverable_wrong=False, confirmed=True),
    ]
    r = action_appropriateness(recs, high_conf=0.7, low_conf=0.5)
    assert math.isclose(r["verify_appropriate_rate"], 0.5)
    assert math.isclose(r["verify_misfire_rate"], 0.5)


def test_appropriateness_no_actions_safe():
    recs = [_rec("none", 0.5, recoverable_wrong=False)]
    r = action_appropriateness(recs)
    assert r["n_redirect"] == 0
    assert r["redirect_appropriate_rate"] is None
    assert r["verify_appropriate_rate"] is None


# ----------------------------------------------------------------------------
# (3) redirect_causal_rate
# ----------------------------------------------------------------------------
def test_redirect_causal_rate_counts_only_flips():
    recs = [
        _rec("redirect", 0.2, True, flipped=True),
        _rec("redirect", 0.2, True, flipped=False),
        _rec("redirect", 0.2, True, flipped=True),
        _rec("verify", 0.9, False, confirmed=True),  # ignored: not a redirect
        _rec("none", 0.5, False),                     # ignored
    ]
    r = redirect_causal_rate(recs)
    assert r["n_redirect"] == 3
    assert r["n_flipped"] == 2
    assert math.isclose(r["causal_rate"], 2 / 3)


def test_redirect_causal_rate_no_redirects_none():
    r = redirect_causal_rate([_rec("none", 0.5, False)])
    assert r["n_redirect"] == 0
    assert r["causal_rate"] is None


# ----------------------------------------------------------------------------
# (4) accuracy_delta
# ----------------------------------------------------------------------------
def test_accuracy_delta_signs():
    r = accuracy_delta(meta_on_acc=0.80, meta_off_acc=0.75, baseline=0.786)
    assert math.isclose(r["delta_on_off"], 0.05)
    assert math.isclose(r["delta_vs_baseline"], 0.80 - 0.786)
    assert r["beats_baseline"] is True
    assert r["meta_helps"] is True


def test_accuracy_delta_forming_collapse_flagged():
    # the v1 collapse: meta on (0.651) < baseline (0.786)
    r = accuracy_delta(meta_on_acc=0.651, meta_off_acc=0.70, baseline=0.786)
    assert r["beats_baseline"] is False
    assert r["meta_helps"] is False


def test_accuracy_delta_baseline_optional():
    r = accuracy_delta(meta_on_acc=0.8, meta_off_acc=0.7, baseline=None)
    assert r["delta_vs_baseline"] is None
    assert r["beats_baseline"] is None
    assert r["meta_helps"] is True


# ----------------------------------------------------------------------------
# (5) meta_survival
# ----------------------------------------------------------------------------
def test_meta_survival_stable_no_collapse():
    # wellformed rate stays high across RL steps
    steps = [
        {"step": 0, "wellformed_rate": 0.95},
        {"step": 50, "wellformed_rate": 0.93},
        {"step": 100, "wellformed_rate": 0.94},
    ]
    r = meta_survival(steps, collapse_threshold=0.5, drop_frac=0.5)
    assert r["survived"] is True
    assert r["collapsed"] is False
    assert math.isclose(r["final_rate"], 0.94)
    assert math.isclose(r["min_rate"], 0.93)


def test_meta_survival_forming_collapse_detected():
    # emit rate crashes toward zero (v3l-style mode collapse)
    steps = [
        {"step": 0, "wellformed_rate": 0.90},
        {"step": 30, "wellformed_rate": 0.40},
        {"step": 60, "wellformed_rate": 0.02},
    ]
    r = meta_survival(steps, collapse_threshold=0.5, drop_frac=0.5)
    assert r["survived"] is False
    assert r["collapsed"] is True
    assert r["final_rate"] < 0.5


def test_meta_survival_relative_drop_collapse():
    # final rate above absolute floor but dropped > drop_frac from the peak
    steps = [
        {"step": 0, "wellformed_rate": 0.99},
        {"step": 50, "wellformed_rate": 0.40},
    ]
    r = meta_survival(steps, collapse_threshold=0.1, drop_frac=0.5)
    # 0.40 > 0.1 absolute floor, but 0.40 < 0.5*0.99 -> relative collapse
    assert r["collapsed"] is True
    assert r["survived"] is False


def test_meta_survival_empty_safe():
    r = meta_survival([])
    assert r["survived"] is None
    assert r["final_rate"] is None


# ----------------------------------------------------------------------------
# intent_report (structured composite)
# ----------------------------------------------------------------------------
def test_intent_report_structure():
    recs = [
        _rec("redirect", 0.2, True, flipped=True),
        _rec("verify", 0.9, False, confirmed=True),
    ]
    rep = intent_report(
        emitted_confs=[0.2, 0.9],
        correct_flags=[False, True],
        records=recs,
        meta_on_acc=0.80,
        meta_off_acc=0.75,
        baseline=0.786,
        survival_steps=[{"step": 0, "wellformed_rate": 0.9},
                        {"step": 50, "wellformed_rate": 0.9}],
    )
    for key in ("calibration", "appropriateness", "redirect_causal",
                "accuracy", "survival"):
        assert key in rep
    assert rep["accuracy"]["beats_baseline"] is True
    assert rep["redirect_causal"]["n_flipped"] == 1
