"""`scripts/local/math_trial2_eval.py` 의 집계·판정 테스트 (설계 §6 관문 ②). GPU 없음."""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))

import pytest  # noqa: E402

import math_trial2_eval as mte  # noqa: E402


def _wrong(n, p_self, p_none, **kw):
    return [{"roll_id": f"r{i}", "p_self": p_self, "p_none": p_none,
             "trunc_self": 0.0, "trunc_none": 0.0,
             "note_generic_self": 0.0, **kw} for i in range(n)]


def test_summary_keys_and_rescue_means():
    recs = _wrong(30, 0.75, 0.60)
    s = mte.summarize_two_trial(recs, k=8, n_boot=200)
    assert s["n_wrong"] == 30 and s["n_paired"] == 30
    assert s["rescue_self"]["mean"] == pytest.approx(0.75)
    assert s["rescue_none"]["mean"] == pytest.approx(0.60)
    # `.617` 기준선 대비 = 평균 − 기준선 (같은 부트스트랩 자)
    assert s["vs_reference_self"]["mean"] == pytest.approx(0.75 - mte.FACT_ONLY_REFERENCE)
    assert s["paired_self_minus_none"]["mean"] == pytest.approx(0.15)
    assert s["reference_fact_only"] == mte.FACT_ONLY_REFERENCE


def test_paired_difference_uses_only_rows_with_both_modes():
    """짝이 아닌 행을 섞으면 (self − none) 이 «모집단 차이»를 잰다 — 배제되어야 한다."""
    recs = _wrong(10, 0.8, 0.6) + [{"roll_id": "x", "p_self": 0.0}]
    s = mte.summarize_two_trial(recs, k=8, n_boot=200)
    assert s["n_wrong"] == 11 and s["n_paired"] == 10
    assert s["paired_self_minus_none"]["mean"] == pytest.approx(0.2)
    assert s["rescue_self"]["n"] == 11 and s["rescue_none"]["n"] == 10


def test_false_alarm_from_correct_rows():
    corr = [{"flip_self": 0.25, "flip_none": 0.125} for _ in range(8)]
    s = mte.summarize_two_trial(_wrong(8, 0.7, 0.6), corr, k=8, n_boot=200)
    assert s["n_correct"] == 8
    assert s["false_alarm_self"]["mean"] == pytest.approx(0.25)
    assert s["false_alarm_none"]["mean"] == pytest.approx(0.125)


def test_gate2_pass_requires_both_reference_and_paired_gain():
    strong = mte.summarize_two_trial(_wrong(60, 0.80, 0.62), k=8, n_boot=400)
    ok, misses = mte.two_trial_pass(strong)
    assert ok and not misses
    # 기준선은 넘지만 none 팔과 차이가 없다 → 반성문이 기여했다고 말할 수 없다
    flat = mte.summarize_two_trial(_wrong(60, 0.80, 0.80), k=8, n_boot=400)
    ok, misses = mte.two_trial_pass(flat)
    assert not ok and any("paired" in m for m in misses)
    # 차이는 있지만 `.617` 에 못 미친다
    low = mte.summarize_two_trial(_wrong(60, 0.40, 0.20), k=8, n_boot=400)
    ok, misses = mte.two_trial_pass(low)
    assert not ok and any("rescue_self" in m for m in misses)


def test_empty_input_does_not_crash():
    s = mte.summarize_two_trial([], [], k=8, n_boot=50)
    assert s["n_wrong"] == 0 and s["n_paired"] == 0
    assert not math.isfinite(s["rescue_self"]["mean"])
    ok, misses = mte.two_trial_pass(s)
    assert not ok and len(misses) == 2


def test_cli_rejects_two_trial_without_rollouts():
    with pytest.raises(SystemExit):
        mte.main(["--stage", "two_trial", "--model_path", "m", "--out_dir", "o"])
