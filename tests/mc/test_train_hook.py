"""mc.train_hook — 손잡이 검증 · PFX 중단 규칙 · 미구현 항의 즉사."""
from __future__ import annotations

import pytest

from mc import train_hook as H


def test_arm_specs_shape():
    # 0925: 유일한 팔은 SPONT_PFX — SPONT_SCORE/SPONT_V5/SPONT_PMI2 와 그보다 앞선 주입 2턴/H3/SHIFT 팔은
    # 전부 경로째 삭제됐다.
    assert set(H.MC_ARM_SPECS) == {"SPONT_PFX"}
    assert H.arm_spec("SPONT_PFX")["pfx"] is True
    for attr in ("SCORE_TERM", "PMI_TERM", "DISTILL_TERM", "distill_token_rewards",
                 "teacher_path", "w_distill", "notice_mode", "w_turn", "turn_span",
                 "entropy_tensor", "entropy_mask_q", "kl_first_coef", "within_row_outcome",
                 "alphas", "score_credit", "spont_credit", "zone_weights", "siblings",
                 "row_class", "opt_grade", "v5_mode", "transition_rates", "hack_telemetry",
                 "pmi2_score", "pmi2_knobs", "pmi2_credit", "u_credit", "u_class", "pmi2_telemetry"):
        assert not hasattr(H, attr), attr
    with pytest.raises(ValueError):
        H.arm_spec("NOPE")


def test_label_knob(monkeypatch):
    monkeypatch.setenv("LABEL", "majority")
    assert H.label_mode() == "majority"
    monkeypatch.setenv("LABEL", "nope")
    with pytest.raises(ValueError):
        H.label_mode()


def test_write_aborted(tmp_path, monkeypatch):
    monkeypatch.setenv("MC_CKPT_DIR", str(tmp_path / "ck"))
    H.write_aborted("because")
    assert (tmp_path / "ck" / "ABORTED.txt").read_text().strip() == "because"


def test_stop_reason_is_pfx_guard_only():
    """0925: PFX 만 남았다 — `mode` 가 "pfx" 가 아니면 규칙이 없다(옛 사행·파괴율·정체 규칙은 삭제)."""
    assert H.stop_reason([{"r1_acc": 0.1}] * 30) is None
    assert H.stop_reason([{"r1_acc": 0.1}] * 30, "group") is None
    assert H.stop_reason([{"pfx_break_right": v} for v in (0.01, 0.01, 0.01, 0.5, 0.5, 0.5)], "pfx")


def test_run_sh_rejects_non_pfx_arms():
    """SPONT_PFX 가 아닌 어떤 이름도 런처에서 즉사해야 한다(옛 팔이 이름만 남아 도는 일을 막는다)."""
    import os
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for arm in ("H2A_SCORE", "H2B_PMI", "H3_DISTILL", "SPONT_PMI", "SPONT_DIR", "SPONT_DIR_ANCHOR",
                "SPONT_SCORE", "SPONT_V5", "SPONT_PMI2"):
        r = subprocess.run(["bash", "mc/run.sh", arm, "2", "3", "--dry-run"],
                          cwd=root, capture_output=True, text=True)
        assert r.returncode != 0 and "SPONT_PFX" in r.stderr, arm


def _dry_run(ckpt_dir, **extra):
    import os
    import subprocess
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = {k: v for k, v in os.environ.items() if k != "CKPT_DIR"}
    env["CKPT_DIR"] = str(ckpt_dir)
    env.update(extra)
    return subprocess.run(["bash", "mc/run.sh", "SPONT_PFX", "2", "3", "--dry-run"],
                         cwd=root, env=env, capture_output=True, text=True).stdout


def test_run_sh_train_done_marker_skips_training(tmp_path):
    """운용 사고 대비: 학습 50/50 이 끝난 뒤 병합·평가가 실패해도 재시도가 **학습을 다시
    띄우지 않아야** 한다(옛 run_math_arm.sh 가 GPU 를 30분 놀린 자리). 마커가 그 계약이다."""
    ck = tmp_path / "ck"
    out = _dry_run(ck)
    assert "학습을 돌리지 않는다" in out and "TRAIN_DONE 있음" not in out
    (ck).mkdir(parents=True, exist_ok=True)
    (ck / "TRAIN_DONE").write_text("2026-09-21T00:00:00Z\n")
    out2 = _dry_run(ck)
    assert "TRAIN_DONE 있음" in out2 and "학습을 건너뛰고 병합·평가만" in out2
    # 마커가 있어도 병합·평가 단계는 그대로 계획된다
    assert "post-train = merge ->" in out2 and "mc.eval {single}" in out2


def test_run_sh_post_train_eval_is_single_12k(tmp_path):
    """단일 패스 팔의 사후 평가는 single 12,288 하나다(2턴 학습 경로가 없으므로)."""
    out = _dry_run(tmp_path / "ck")
    assert "mc.eval {single}" in out and "twoturn" not in out
    assert "k=8 seed=11" in out
    assert "merged_mc_SPONT_PFX_gold_s2_r4096_step<N>" in out
