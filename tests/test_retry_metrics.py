"""retry_metrics 회귀 시험 — within-problem 선택성 vs «난이도만 반영하는 redirect»를 가른다.

두 시나리오를 모의로 구성한다:
  A) redirect 가 문제 «안에서» 틀림을 따라간다(선택성 진짜) → mixed 선택성 양수, AUC>0.5.
  B) redirect 가 문제 «난이도»만 따라간다(같은 문제 안에서는 wrong/right 구분 없이 결정) →
     within-problem AUC ≈ 0.5 (난이도가 통제됐으므로).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.training import retry_metrics as RM  # noqa: E402


def _row(group_id, first_correct, decision, final_correct=None):
    return {"group_id": group_id, "first_correct": int(first_correct), "decision": decision,
            "final_correct": int(final_correct if final_correct is not None else first_correct)}


# ── Scenario A: redirect tracks wrongness within each mixed problem ──────────
def _scenario_wrongness_tracking():
    rows = []
    # problem "hard1": 8 samples, 4 correct(right) 4 wrong. redirect exactly on the wrong ones,
    # and redirect rescues them (final_correct=1) to show a genuine lift too.
    for i in range(8):
        wrong = i < 4
        rows.append(_row("hard1", first_correct=not wrong, decision=("redirect" if wrong else "verify"),
                         final_correct=1 if wrong else 1))
    # problem "hard2": another mixed problem, same pattern.
    for i in range(8):
        wrong = i < 3
        rows.append(_row("hard2", first_correct=not wrong, decision=("redirect" if wrong else "verify"),
                         final_correct=1 if wrong else 1))
    # a non-mixed (all-correct) problem, should be excluded from mixed metrics.
    for i in range(8):
        rows.append(_row("easy1", first_correct=1, decision="verify", final_correct=1))
    return rows


def test_mixed_selectivity_positive_when_redirect_tracks_wrongness():
    rows = _scenario_wrongness_tracking()
    sel = RM.mixed_problem_selectivity(rows)
    assert sel["n_mixed_problems"] == 2
    assert sel["redirect_rate_given_wrong_mixed"] == 1.0
    assert sel["redirect_rate_given_right_mixed"] == 0.0
    assert sel["selectivity_mixed"] == 1.0


def test_per_problem_auc_high_when_redirect_tracks_wrongness():
    rows = _scenario_wrongness_tracking()
    res = RM.per_problem_auc(rows)
    assert res["n_problems_auc"] == 2
    assert res["mean_within_problem_auc"] == 1.0
    assert res["frac_problems_auc_gt_half"] == 1.0


def test_difficulty_bucket_lift_shows_gain_on_mixed_bucket():
    rows = _scenario_wrongness_tracking()
    res = RM.difficulty_bucket_lift(rows)
    # hard1: first_pass_rate=0.5 -> bucket p0_50; hard2: 5/8=.625 -> bucket p50_100
    assert res["retry_lift_p0_50"] > 0        # wrong rows rescued -> final_acc > first_acc
    assert res["retry_lift_p50_100"] > 0
    assert res["retry_lift_p100"] == 0.0      # easy1: already all correct, no room to lift


def test_judgment_acc_mixed_perfect_when_selective():
    rows = _scenario_wrongness_tracking()
    acc = RM.judgment_acc_mixed(rows)
    assert acc == 1.0


# ── Scenario B: redirect tracks only problem difficulty, not within-problem wrongness ─
def _scenario_difficulty_only():
    rows = []
    # "hard1": mixed (4/8 correct). redirect fires on a FIXED HALF regardless of correctness
    # (i.e. decision is uncorrelated with first_correct within the problem).
    pattern = [1, 0, 1, 0, 1, 0, 1, 0]     # first_correct per sample
    redirect_half = [1, 1, 1, 1, 0, 0, 0, 0]  # decision uncorrelated with pattern
    for fc, rd in zip(pattern, redirect_half):
        rows.append(_row("hard1", first_correct=fc, decision=("redirect" if rd else "verify")))
    # "hard2": same construction, different fixed half.
    pattern2 = [0, 1, 0, 1, 0, 1, 0, 1]
    redirect_half2 = [1, 1, 1, 1, 0, 0, 0, 0]
    for fc, rd in zip(pattern2, redirect_half2):
        rows.append(_row("hard2", first_correct=fc, decision=("redirect" if rd else "verify")))
    return rows


def test_within_problem_auc_near_half_when_redirect_tracks_only_difficulty():
    rows = _scenario_difficulty_only()
    res = RM.per_problem_auc(rows)
    assert res["n_problems_auc"] == 2
    assert abs(res["mean_within_problem_auc"] - 0.5) < 1e-9


# ── pass_rate override (external, e.g. build_math_parquet group_pass_rate) ───
def test_pass_rate_override_used_instead_of_batch_local():
    # Batch has only 1 sample per group (can't tell mixed-ness locally) but external pass_rate says mixed.
    rows = [
        {"group_id": "u1", "first_correct": 0, "decision": "redirect", "final_correct": 1},
        {"group_id": "u2", "first_correct": 1, "decision": "verify", "final_correct": 1},
    ]
    pass_rate = {"u1": 0.4, "u2": 0.6}
    sel = RM.mixed_problem_selectivity(rows, group_key="group_id", pass_rate=pass_rate)
    assert sel["n_mixed_problems"] == 2
    assert sel["redirect_rate_given_wrong_mixed"] == 1.0
    assert sel["redirect_rate_given_right_mixed"] == 0.0


def test_pass_rate_none_values_excluded_from_mixed():
    rows = [
        {"group_id": "u1", "first_correct": 0, "decision": "redirect", "final_correct": 1},
        {"group_id": "u2", "first_correct": 1, "decision": "verify", "final_correct": 1},
    ]
    pass_rate = {"u1": None, "u2": 0.5}
    sel = RM.mixed_problem_selectivity(rows, group_key="group_id", pass_rate=pass_rate)
    assert sel["n_mixed_problems"] == 1


def test_all_metrics_returns_all_keys():
    rows = _scenario_wrongness_tracking()
    m = RM.all_metrics(rows)
    for k in ("selectivity_mixed", "mean_within_problem_auc", "frac_problems_auc_gt_half",
             "retry_lift_p0", "retry_lift_p0_50", "retry_lift_p50_100", "retry_lift_p100",
             "judgment_acc_mixed"):
        assert k in m, k


def test_empty_rows_are_nan_not_crash():
    m = RM.all_metrics([])
    assert m["selectivity_mixed"] != m["selectivity_mixed"]  # NaN
    assert m["mean_within_problem_auc"] != m["mean_within_problem_auc"]
