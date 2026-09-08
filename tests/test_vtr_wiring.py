"""OPT_VTR/OPT_VTRW(§12, 2026-09-08) 배선 회귀 테스트.

세 갈래:
  1. `countdown_rewards.vtr_batch_gate` — 순수 함수, tau 임계값·K=4 twin 평균 계산·
     온라인 불가 시 오프라인(`opd_gate`) 폴백을 확인한다. GPU/모델 불필요.
  2. `countdown_rewards.r_when` — 순수 함수, redirect/continue × family_dead 4칸 +
     미발화 0칸을 확인한다.
  3. `scripts/local/run_arm.sh --dry-run` — OPT_VTR/OPT_VTRW 가 새 data_hint
     (`mixed_vtr`)로 라우팅되고 셔플 오버라이드를 받는지, `test_run_arm_resp_len.py`
     와 같은 스타일로 확인한다. GPU 불필요.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from src.training import countdown_rewards as cdr

ROOT = Path(__file__).resolve().parents[1]
RUN_ARM = ROOT / "scripts" / "local" / "run_arm.sh"


# ══════════════════════════════════════════════════════════════════════════
# 1. vtr_batch_gate
# ══════════════════════════════════════════════════════════════════════════

def _row(role, key, *, emitted=0, corr=0.0, opd_gate=0):
    return {"vtr_role": role, "vtr_key": key, "emitted": emitted, "corr": corr,
            "opd_gate": opd_gate}


def test_vtr_gate_online_pass_when_delta_at_or_above_tau():
    rows = [
        _row("main", "s1", emitted=0, corr=0.0),   # 무발화 main
        _row("main", "s1", emitted=1, corr=1.0),   # 발화 main(비교 대상 아님)
        _row("twin", "s1", corr=1.0), _row("twin", "s1", corr=1.0),
        _row("twin", "s1", corr=1.0), _row("twin", "s1", corr=0.0),
    ]
    gate = cdr.vtr_batch_gate(rows, tau=0.10)
    # mean(hint)=0.75, mean(nometa)=0.0, delta=0.75 >= 0.10
    assert gate["s1"] == (1, "online")


def test_vtr_gate_online_fail_when_delta_below_tau():
    rows = [
        _row("main", "s1", emitted=0, corr=1.0),
        _row("twin", "s1", corr=1.0), _row("twin", "s1", corr=1.0),
        _row("twin", "s1", corr=1.0), _row("twin", "s1", corr=1.0),
    ]
    gate = cdr.vtr_batch_gate(rows, tau=0.10)
    # mean(hint)=1.0, mean(nometa)=1.0, delta=0.0 < 0.10
    assert gate["s1"] == (0, "online")


def test_vtr_gate_tau_boundary_is_inclusive():
    rows = [
        _row("main", "s1", emitted=0, corr=0.0),
        _row("twin", "s1", corr=0.10), _row("twin", "s1", corr=0.10),
    ]
    gate = cdr.vtr_batch_gate(rows, tau=0.10)
    assert gate["s1"] == (1, "online")


def test_vtr_gate_k4_mean_uses_all_twins_not_first_one():
    # K=4 twin, mean = (1+1+1+0)/4 = 0.75 — 단일 twin 만 봤다면(1.0) 다른 값이 나온다.
    rows = [
        _row("main", "s1", emitted=0, corr=0.0),
        _row("twin", "s1", corr=1.0), _row("twin", "s1", corr=1.0),
        _row("twin", "s1", corr=1.0), _row("twin", "s1", corr=0.0),
    ]
    gate = cdr.vtr_batch_gate(rows, tau=0.80)
    assert gate["s1"] == (0, "online")   # 0.75 < 0.80
    gate2 = cdr.vtr_batch_gate(rows, tau=0.70)
    assert gate2["s1"] == (1, "online")  # 0.75 >= 0.70


def test_vtr_gate_falls_back_to_offline_when_no_nometa_sample():
    # 이 배치가 뽑은 main 롤아웃이 전원 발화 — 온라인 비교 불가, opd_gate=1 로 폴백.
    rows = [
        _row("main", "s1", emitted=1, corr=1.0, opd_gate=1),
        _row("main", "s1", emitted=1, corr=0.0, opd_gate=1),
        _row("twin", "s1", corr=1.0), _row("twin", "s1", corr=1.0),
    ]
    gate = cdr.vtr_batch_gate(rows, tau=0.10)
    assert gate["s1"] == (1, "fallback")


def test_vtr_gate_falls_back_to_offline_when_no_twin_sample():
    rows = [
        _row("main", "s1", emitted=0, corr=0.0, opd_gate=0),
    ]
    gate = cdr.vtr_batch_gate(rows, tau=0.10)
    assert gate["s1"] == (0, "fallback")


def test_vtr_gate_empty_key_rows_are_not_listed():
    rows = [_row("none", "", emitted=0, corr=0.0)]
    gate = cdr.vtr_batch_gate(rows, tau=0.10)
    assert gate == {}


def test_vtr_gate_env_override(monkeypatch):
    monkeypatch.setenv("VTR_TAU", "0.5")
    assert cdr.resolved_vtr_tau() == pytest.approx(0.5)
    rows = [
        _row("main", "s1", emitted=0, corr=0.0),
        _row("twin", "s1", corr=0.3),
    ]
    gate = cdr.vtr_batch_gate(rows)   # tau=None -> resolved_vtr_tau() 읽음
    assert gate["s1"] == (0, "online")   # 0.3 < 0.5


# ══════════════════════════════════════════════════════════════════════════
# 2. r_when
# ══════════════════════════════════════════════════════════════════════════

def test_when_redirect_dead_is_positive():
    r = cdr.r_when({"emitted": 1, "family_dead": 1, "dec_redirect": 1})
    assert r == pytest.approx(cdr.resolved_vtr_when_w())


def test_when_continue_alive_is_positive():
    r = cdr.r_when({"emitted": 1, "family_dead": 0, "dec_redirect": 0})
    assert r == pytest.approx(cdr.resolved_vtr_when_w())


def test_when_redirect_alive_is_negative():
    r = cdr.r_when({"emitted": 1, "family_dead": 0, "dec_redirect": 1})
    assert r == pytest.approx(-cdr.resolved_vtr_when_w())


def test_when_continue_dead_is_negative():
    r = cdr.r_when({"emitted": 1, "family_dead": 1, "dec_redirect": 0})
    assert r == pytest.approx(-cdr.resolved_vtr_when_w())


def test_when_no_meta_is_zero_never_forces_emission():
    r = cdr.r_when({"emitted": 0, "family_dead": 1, "dec_redirect": 1})
    assert r == 0.0


def test_when_undeterminable_family_is_zero():
    r = cdr.r_when({"emitted": 1, "family_dead": None, "dec_redirect": 1})
    assert r == 0.0


def test_when_env_override(monkeypatch):
    monkeypatch.setenv("VTR_WHEN_W", "0.4")
    assert cdr.resolved_vtr_when_w() == pytest.approx(0.4)
    r = cdr.r_when({"emitted": 1, "family_dead": 1, "dec_redirect": 1})
    assert r == pytest.approx(0.4)


def test_opt_vtr_and_vtrw_arm_specs_registered():
    assert "OPT_VTR" in cdr.ARM_SPECS
    assert "OPT_VTRW" in cdr.ARM_SPECS
    assert cdr.OPD_TERM_C in cdr.ARM_SPECS["OPT_VTR"]["terms"]
    assert cdr.WHEN_TERM in cdr.ARM_SPECS["OPT_VTRW"]["terms"]
    assert cdr.WHEN_TERM not in cdr.ARM_SPECS["OPT_VTR"]["terms"]
    assert cdr.ARM_SPECS["OPT_VTR"]["data_hint"] == "mixed_vtr"
    assert cdr.ARM_SPECS["OPT_VTRW"]["data_hint"] == "mixed_vtr"


def test_arm_signature_distinguishes_vtr_from_opdg():
    # OPT_OPDG 와 OPT_VTR 은 항 구성이 같다(opd_meta_c) — vtr_tau 조각이 없으면
    # 두 팔의 서명이 같아져 로그가 거짓말을 한다.
    assert cdr.arm_signature("OPT_VTR") != cdr.arm_signature("OPT_OPDG")
    assert "vtr_tau=" in cdr.arm_signature("OPT_VTR")
    assert "when_w=" in cdr.arm_signature("OPT_VTRW")
    assert "when_w=" not in cdr.arm_signature("OPT_VTR")


# ══════════════════════════════════════════════════════════════════════════
# 3. run_arm.sh --dry-run
# ══════════════════════════════════════════════════════════════════════════

def _dry_run(env_extra: dict, args: list[str]) -> str:
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", **env_extra)
    r = subprocess.run(
        ["bash", str(RUN_ARM), *args, "--dry-run"],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"run_arm.sh --dry-run failed: rc={r.returncode}\n{r.stdout}\n{r.stderr}"
    return r.stdout + r.stderr


def test_opt_vtr_routes_to_vtr_opt_parquet():
    out = _dry_run(
        {"SLIM": "1", "SITES_DIR": "sites_v4", "MIXED_DATA": "mixed_train_v4"},
        ["OPT_VTR", "1", "100", "opt"],
    )
    assert "data.train_files=" in out
    assert "sites_v4/mixed_train_v4_vtr_opt.parquet" in out
    assert "DATA_HINT=mixed_vtr" in out
    assert "data.shuffle=false" in out


def test_opt_vtrw_routes_to_vtr_opt_parquet():
    out = _dry_run(
        {"SLIM": "1", "SITES_DIR": "sites_v4", "MIXED_DATA": "mixed_train_v4"},
        ["OPT_VTRW", "1", "100", "opt"],
    )
    assert "sites_v4/mixed_train_v4_vtr_opt.parquet" in out
    assert "DATA_HINT=mixed_vtr" in out
    assert "data.shuffle=false" in out


def test_opt_vtr_default_mixed_data_name():
    out = _dry_run(
        {"SLIM": "1", "SITES_DIR": "sites_v4"},
        ["OPT_VTR", "1", "100", "opt"],
    )
    assert "sites_v4/mixed_train_v4_vtr_opt.parquet" in out


def test_opt_vtr_uses_opt_prompt_variant():
    out = _dry_run(
        {"SLIM": "1", "SITES_DIR": "sites_v4", "MIXED_DATA": "mixed_train_v4"},
        ["OPT_VTR", "1", "100", "p3"],   # VARIANT_ARG 는 무시돼야 한다
    )
    assert "DATA_VARIANT=opt" in out


def test_twin_advantages_are_zeroed():
    """§12 수정 ②: vtr_role=="twin" 행의 advantages 는 0, 나머지는 그대로."""
    import torch
    from src.training.verl_sdc import _countdown_mask_twin_advantages

    class _D:  # verl DataProto 흉내(batch dict + non_tensor_batch dict)
        pass
    d = _D()
    d.batch = {"advantages": torch.ones(4, 3)}
    d.non_tensor_batch = {"extra_info": [{"vtr_role": "main"}, {"vtr_role": "twin"}, {}, {"vtr_role": "twin"}]}
    out = _countdown_mask_twin_advantages(d)
    assert out.batch["advantages"].sum().item() == 6.0
    assert out.batch["advantages"][1].abs().sum().item() == 0.0
    assert out.batch["advantages"][3].abs().sum().item() == 0.0


def test_twin_mask_is_noop_without_column():
    import torch
    from src.training.verl_sdc import _countdown_mask_twin_advantages

    class _D:
        pass
    d = _D()
    d.batch = {"advantages": torch.ones(2, 3)}
    d.non_tensor_batch = {"extra_info": [{}, {}]}
    assert _countdown_mask_twin_advantages(d).batch["advantages"].sum().item() == 6.0


def test_build_opd_arms_full_span_scores_rows_without_meta():
    """§12-b: full_span_rows 에 든 행은 메타가 없어도 프리픽스 직후 구간이 증류 대상이 된다."""
    from src.training.verl_sdc import _build_opd_arms

    class _Tok:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [ord(c) % 100 for c in text]}
        def apply_chat_template(self, msgs, tokenize=False, **kw):
            return "\n".join(m["content"] for m in msgs)
    prefix = "Numbers: [20, 10, 7, 10]\n20+10 = 30 → too low\n10*7 = 70 → too low\n"
    msgs = [[{"role": "system", "content": "s"}, {"role": "user", "content": "Numbers: [20, 10, 7, 10] Target: 27"}]]
    resp = ["Let me try 20-7 = 13 → too low, then 13+10 = 23."]  # 메타 없음
    _, _, attempts, per_row, diag = _build_opd_arms(_Tok(), msgs, resp, [prefix], [[20, 10, 7, 10]], [27],
                                                  full_span_rows=[0], full_span_tok=8)
    assert diag["full_span"] == 1 and per_row[0]["opd_status"] == "pending" and per_row[0]["opd_n_tok"] == 8
    _, _, _, per_row2, diag2 = _build_opd_arms(_Tok(), msgs, resp, [prefix], [[20, 10, 7, 10]], [27])
    assert diag2["no_meta"] == 1 and per_row2[0]["opd_status"] == "no_meta"
