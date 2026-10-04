"""H2 «밀도 1 리셋 2턴» 항 `trial2_score` 테스트 (docs/PLAN_h2_twoturn_0921.md §1·§2·§5)."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import trial2 as t2  # noqa: E402
from src.training.math_meta import MATH_ARM_SPECS  # noqa: E402

_ALPHA_ENV = ("SCORE_ALPHA_POS", "SCORE_ALPHA_NEG")


@pytest.fixture
def _clean_alphas(monkeypatch):
    """알파 손잡이를 지운 상태 = 기본값(1.0 / 2.0)."""
    for k in _ALPHA_ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


# ── 손잡이 ──────────────────────────────────────────────────────────────────
def test_alpha_defaults_and_negative_rejection(_clean_alphas):
    assert t2.score_alpha_pos() == 1.0
    assert t2.score_alpha_neg() == 2.0          # α⁻ = 2α⁺ (설계 §1)
    _clean_alphas.setenv("SCORE_ALPHA_POS", "0.5")
    _clean_alphas.setenv("SCORE_ALPHA_NEG", "3")
    assert (t2.score_alpha_pos(), t2.score_alpha_neg()) == (0.5, 3.0)
    _clean_alphas.setenv("SCORE_ALPHA_POS", "-0.1")
    with pytest.raises(ValueError):
        t2.score_alpha_pos()
    _clean_alphas.setenv("SCORE_ALPHA_POS", "1.0")
    _clean_alphas.setenv("SCORE_ALPHA_NEG", "-1")
    with pytest.raises(ValueError):
        t2.score_alpha_neg()


# ── 진리표 ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("r1,r2,want", [
    (0.0, 0.0, 0.0),        # 못 고침 — 벌도 상도 없다
    (0.0, 1.0, 2.0),        # 구제: R2 + α⁺ = 1 + 1
    (1.0, 0.0, -2.0),       # 파괴: 0 − α⁻ = −2  ← 이 팔의 레버
    (1.0, 1.0, 1.0),        # 지킴
])
def test_a2_truth_table_with_default_alphas(_clean_alphas, r1, r2, want):
    assert t2.row_reward(t2.SCORE_TERM, "a2", 1, r1, r2, 0.6) == pytest.approx(want)


def test_a1_and_note_stages(_clean_alphas):
    for r1 in (0.0, 1.0):
        for r2 in (0.0, 1.0):
            assert t2.row_reward(t2.SCORE_TERM, "a1", 1, r1, r2, 0.6) == r1
            assert t2.row_reward(t2.SCORE_TERM, "note", 1, r1, r2, 0.6) == 0.0


def test_inactive_rows_are_zero(_clean_alphas):
    for st in t2.STAGES:
        assert t2.row_reward(t2.SCORE_TERM, st, 0, 1.0, 0.0, 0.6) == 0.0


def test_custom_alphas_move_only_the_delta(_clean_alphas):
    _clean_alphas.setenv("SCORE_ALPHA_POS", "0.5")
    _clean_alphas.setenv("SCORE_ALPHA_NEG", "4.0")
    assert t2.row_reward(t2.SCORE_TERM, "a2", 1, 0.0, 1.0, 0.6) == pytest.approx(1.5)
    assert t2.row_reward(t2.SCORE_TERM, "a2", 1, 1.0, 0.0, 0.6) == pytest.approx(-4.0)
    assert t2.row_reward(t2.SCORE_TERM, "a2", 1, 1.0, 1.0, 0.6) == 1.0


# ── 기존 팔 불변 ────────────────────────────────────────────────────────────
def test_credit_and_outcome_unchanged(_clean_alphas):
    """SCORE 항을 들여도 CREDIT/OUTCOME 의 행 보상은 한 값도 안 바뀐다."""
    g = 0.6
    want = {
        (t2.CREDIT_TERM, "a1"): lambda r1, r2: r1 + g * r2,
        (t2.CREDIT_TERM, "note"): lambda r1, r2: g * r2,
        (t2.CREDIT_TERM, "a2"): lambda r1, r2: r2,
        (t2.OUTCOME_TERM, "a1"): lambda r1, r2: r1,
        (t2.OUTCOME_TERM, "note"): lambda r1, r2: 0.0,
        (t2.OUTCOME_TERM, "a2"): lambda r1, r2: r2,
    }
    for (term, st), f in want.items():
        for r1 in (0.0, 1.0):
            for r2 in (0.0, 1.0):
                assert t2.row_reward(term, st, 1, r1, r2, g) == pytest.approx(f(r1, r2)), (term, st)
                assert t2.row_reward(term, st, 0, r1, r2, g) == 0.0


# ── 팔 해석 · 재시도 계열 소속 ──────────────────────────────────────────────
def test_arm_spec_and_retry_family():
    s = MATH_ARM_SPECS["M_TRIAL2_SCORE"]
    assert (s["variant"], s["meta_term"], s["require_meta"]) == ("math_opt", t2.SCORE_TERM, False)
    assert s["note"].strip()
    # 2-시도(재시도) 계열 소속 = 항 이름 집합. verl_sdc 의 S3 게이트가 이걸로 슬롯 3K·
    # select_retry_rows·recredit 훅을 켠다 — M_TRIAL2_CREDIT 과 같은 취급이어야 한다.
    assert t2.SCORE_TERM in t2.TRIAL2_TERMS
    assert t2.TRIAL2_TERMS == {t2.CREDIT_TERM, t2.OUTCOME_TERM, t2.SCORE_TERM}
    assert t2.rollout_n(8) == 24


def test_launcher_defaults_6144_and_alpha_env():
    """런처가 M_TRIAL2_SCORE 에 RESP_LEN/A2_RESP_LEN 6144·ROLLOUT_N 24 와 α 를 싣는가."""
    import os
    import subprocess
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    for k in ("RESP_LEN", "A2_RESP_LEN", "ROLLOUT_N", *_ALPHA_ENV):
        env.pop(k, None)
    out = subprocess.run(["bash", str(root / "scripts/local/run_math_arm.sh"),
                          "M_TRIAL2_SCORE", "1", "1", "--dry-run"],
                         capture_output=True, text=True, env=env, cwd=str(root)).stdout
    line = next(ln for ln in out.splitlines() if "LINEAGE=" in ln and "ARM=" in ln)
    for frag in ("ARM=M_TRIAL2_SCORE", "RESP_LEN=6144", "A2_RESP_LEN=6144",
                 "ROLLOUT_N=24", "SCORE_ALPHA_POS=1.0", "SCORE_ALPHA_NEG=2.0"):
        assert frag in line, (frag, line)


# ── 텔레메트리: 구제율·파괴율 ───────────────────────────────────────────────
def _fab_all(r1_seq, r2_seq):
    """RETRY_GATE=all 배치 위조 — 한 문제 K 행, 전 행이 시도 2 를 받는다."""
    k = len(r1_seq)
    uids, stages, active, tkeys, own = [], [], [], [], []
    for st in t2.STAGES:
        for j in range(k):
            uids.append("u0")
            stages.append(st)
            active.append(1)
            tkeys.append(f"u0#t{j}")
            own.append(float(r1_seq[j]) if st == "a1" else
                       (0.0 if st == "note" else float(r2_seq[j])))
    return uids, stages, active, tkeys, own


def test_p_fix_and_break_rate(_clean_alphas):
    # R1: 1 1 1 1 0 0 0 0  → 정답 4 · 오답 4
    # R2: 1 1 0 0 1 0 0 0  → 파괴 2/4 = .5 · 구제 1/4 = .25
    uids, stages, active, tkeys, own = _fab_all((1, 1, 1, 1, 0, 0, 0, 0),
                                                (1, 1, 0, 0, 1, 0, 0, 0))
    rew, keys, tel = t2.recredit(t2.SCORE_TERM, own, stages, active, uids, tkeys, 0.6)
    assert tel["n_r1_right"] == 4.0 and tel["n_r1_wrong"] == 4.0
    assert tel["break_rate"] == pytest.approx(0.5)
    assert tel["p_fix"] == pytest.approx(0.25)
    assert tel["r1_mean"] == pytest.approx(0.5)
    assert tel["r2_mean"] == pytest.approx(3 / 8)
    assert tel["two_trial_acc"] == pytest.approx(5 / 8)
    # 시도-2 행 보상이 진리표 그대로인가
    a2 = [rew[i] for i in range(len(stages)) if stages[i] == "a2"]
    assert a2 == pytest.approx([1.0, 1.0, -2.0, -2.0, 2.0, 0.0, 0.0, 0.0])


def test_rates_ignore_trials_without_attempt2(_clean_alphas):
    """재시도를 안 받은 trial 은 분모에서 빠진다 — 안 굴린 시도를 파괴로 세면 안 된다."""
    uids, stages, active, tkeys, own = _fab_all((1, 0), (0, 1))
    active = [1 if st == "a1" else (1 if tkeys[i].endswith("t1") else 0)
              for i, st in enumerate(stages)]
    own = [own[i] if active[i] else 0.0 for i in range(len(own))]
    _, _, tel = t2.recredit(t2.SCORE_TERM, own, stages, active, uids, tkeys, 0.6)
    assert tel["n_r1_right"] == 0.0 and tel["break_rate"] == 0.0     # 분모 0 → 0
    assert tel["n_r1_wrong"] == 1.0 and tel["p_fix"] == pytest.approx(1.0)
