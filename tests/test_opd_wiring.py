r"""OPD(힌트 교사) 항의 «선언된 레버, 배선 0» 회귀 테스트. `tests/test_osd_wiring.py`
와 같은 규약 — verl_sdc.py 를 실제로 import 하므로 torch/verl 이 있는 환경에서만
돈다(`source scripts/local/env.sh`).
"""
import inspect
import math

import pytest

from src.training import countdown_rewards as cr


def _row(**kw):
    r = {"r_corr": 1, "format_ok": 1, "emitted": 1, "opd_kl": 0.02}
    r.update(kw)
    return r


def test_opd_term_name_is_single_sourced():
    """★핵심 회귀. `_compute_countdown_arm_stash` 의 fail-loud 가드가 OPD_TERM 을
    읽는데 그 이름이 팔의 terms 에 없으면 가드가 영영 안 돈다(0825 "meta_osd" vs
    "osd" 사고의 재발 방지, OPD 판)."""
    assert {cr.OPD_TERM} & set(cr.ARM_SPECS["OPT_OPD"]["terms"]), (
        f"OPD_TERM={cr.OPD_TERM!r} 이 ARM_SPECS['OPT_OPD']['terms']"
        f"={cr.ARM_SPECS['OPT_OPD']['terms']} 에 없다 — fail-loud 가드가 죽는다.")
    assert cr.OPD_TERM in cr.TERMS
    assert cr.OPD_TERM in cr.META_TERMS


def test_opt_opd_arm_is_not_bit_identical_to_opt():
    """OPT_OPD 팔이 OPT 팔과 같은 총보상을 내면 처치가 배선되지 않은 것이다."""
    row = _row(opd_kl=0.02)
    opt, _ = cr.arm_reward("OPT", row, step=30)
    opd, comp = cr.arm_reward("OPT_OPD", row, step=30)
    assert comp.get(cr.OPD_TERM, 0.0) != 0.0, f"opd_meta 성분이 0 이다: {comp}"
    assert abs(opt - opd) > 1e-9, f"OPT_OPD({opd}) 와 OPT({opt}) 총보상이 동일 — 배선 0."


def test_opd_kl_none_is_fail_loud_when_missing():
    """`opd_kl` 이 행에 아예 없으면(스코어러가 안 돌았다는 배선 사고) KeyError."""
    with pytest.raises(KeyError):
        cr.arm_reward("OPT_OPD", {"r_corr": 1, "format_ok": 1, "emitted": 1}, step=30)


def test_opd_kl_none_value_is_silent_zero():
    """`opd_kl=None`(잴 스팬이 없었다 — 정상 행)은 KeyError 가 아니라 조용히 0.0."""
    _tot, comp = cr.arm_reward("OPT_OPD", _row(opd_kl=None), step=30)
    assert comp[cr.OPD_TERM] == 0.0


def test_opd_kl_nan_is_fail_closed_zero():
    """NaN = «쟀는데 비유한»(OSD 의 delta_cert=NaN 과 같은 규약). 0 으로 닫혀
    포이즌 행이 형제의 센터링을 망치지 않는다. `+inf`/`-inf` 는 NaN 이 아니라
    **유한값처럼 클립된다**(clip(·,0,C) 가 이미 경계를 정의한다) — +inf 는 벌
    최대치(-1.0)로, -inf 는 벌 없음(0.0)으로."""
    assert cr.r_opd_meta(float("nan")) == 0.0
    assert cr.r_opd_meta(math.inf) == pytest.approx(-1.0)
    assert cr.r_opd_meta(-math.inf) == 0.0


def test_c_is_read_at_call_time_not_import_time():
    """관문이 실측 p95 를 박으면 서명과 보상이 **함께** 움직여야 한다(OSD_C 회귀와 같은 계기)."""
    old_c, old_prov = cr.OPD_C, cr.OPD_C_PROVISIONAL
    try:
        cr.OPD_C = 0.10
        assert cr.r_opd_meta(0.05) == pytest.approx(-0.5)
        cr.OPD_C = 0.20
        assert cr.r_opd_meta(0.05) == pytest.approx(-0.25), (
            "OPD_C 를 바꿨는데 r_opd_meta 가 안 따라온다 — 기본 인자로 캡처됐다.")
        assert "opd_c=0.2" in cr.arm_signature("OPT_OPD")
    finally:
        cr.OPD_C, cr.OPD_C_PROVISIONAL = old_c, old_prov


def test_opd_is_one_sided_penalty_only():
    """상은 없다 — 항상 <= 0."""
    for kl in (-5.0, -0.01, 0.0, 0.01, cr.OPD_C, cr.OPD_C * 10):
        assert cr.r_opd_meta(kl) <= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 소스텍스트 가드 — `_compute_countdown_arm_stash` 가 `_compute_countdown_opd` 를
# opd_meta 항이 켜진 팔에서만 부르는가(계산 일치). OSD/INV 회귀 규약과 같은 이유로
# `inspect.getsource` 로 실제 배선 순서를 검사한다 — «선언은 있고 배선은 없음»은
# arm_reward 단위 테스트만으로는 못 잡는다(그건 스코어러가 실제로 호출됐다고 가정한다).
# ══════════════════════════════════════════════════════════════════════════════

def test_stash_calls_compute_countdown_opd_gated_on_opd_term():
    from src.training import verl_sdc as vs

    src = inspect.getsource(vs._compute_countdown_arm_stash)
    assert "_opd_terms = {_cdr.OPD_TERM} & set(" in src, (
        "opd_meta 항 게이트(_opd_terms)가 없다 — 스코어러가 팔과 무관하게 항상/전혀 "
        "안 돌 수 있다.")
    assert "_compute_countdown_opd(" in src
    # ★핵심: `_compute_countdown_opd(` 호출이 `if not _opd_on:` 의 **else** 분기
    # (즉 `_opd_on` 이 참일 때)에서만 나타나야 한다 — OSD 의 `_osd_on`/`_compute_
    # countdown_osd` 와 같은 구조. 텍스트 순서로 확인: 게이트 변수 선언 뒤, 그리고
    # "if not _opd_on:" 블록이 스코어러를 부르지 않는다는 것도 함께 확인한다.
    gate_idx = src.index("_opd_on = bool(_opd_terms)")
    call_idx = src.index("_compute_countdown_opd(")
    off_branch_idx = src.index("if not _opd_on:")
    assert gate_idx < off_branch_idx < call_idx, (
        "호출 순서가 «게이트 선언 → off 분기 → 실제 호출»이 아니다 — "
        "opd_meta 가 꺼진 팔에서도 GPU forward 가 돌 수 있다.")
    # off 분기 자체는 스코어러를 부르지 않는다(그 사이 텍스트에 호출이 없어야 한다).
    off_to_call = src[off_branch_idx:call_idx]
    assert "_compute_countdown_opd(" not in off_to_call


def test_stash_reraises_on_ref_failure_when_term_is_on():
    from src.training import verl_sdc as vs

    src = inspect.getsource(vs._compute_countdown_arm_stash)
    assert "if _opd_terms:" in src and "raise" in src.split("if _opd_terms:")[1][:80], (
        "opd_meta 항이 켜진 팔에서 ref 스코어링 실패를 삼키면 그 팔이 무음 0 으로 "
        "OPT 팔과 같아진다 — fail-loud 가 없다.")


def test_hint_builder_never_contains_witness_across_random_instances():
    """건축 규율(설계 §1.1/§5) — 임의 인스턴스에서도 힌트가 witness 를 담지 않는다."""
    import random

    from src.training import countdown_opd as opd
    from src.training import countdown_sites as cds

    rng = random.Random(0)
    n_checked = 0
    for _ in range(200):
        nums = [rng.randint(1, 13) for _ in range(4)]
        target = rng.randint(1, 200)
        try:
            counts, witness = cds.enumerate_solutions(nums, target)
        except Exception:
            continue
        if sum(counts.values()) == 0 or not witness:
            continue
        prefix = f"{nums[0]}+{nums[1]}=99\n"
        try:
            hint = opd.build_hint(nums, target, prefix)
        except RuntimeError:
            continue
        n_checked += 1
        assert witness not in hint
        assert "\\boxed" not in hint
    assert n_checked > 20, "표본이 너무 적다 — 무작위 시드/범위를 다시 보라."
