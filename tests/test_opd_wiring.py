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
    assert "_opd_terms = {_cdr.OPD_TERM, _cdr.OPD_TERM_C} & set(" in src, (
        "opd_meta/opd_meta_c 항 게이트(_opd_terms)가 없다 — 스코어러가 팔과 무관하게 "
        "항상/전혀 안 돌 수 있다.")
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


def test_opd_kl_sign_penalizes_student_tokens_that_hint_finds_unlikely():
    """E-132: 학생이 뽑은 토큰 위에서 (lp_student − lp_teacher) 평균이어야 한다.
    학생이 확신한 토큰을 힌트 교사가 낯설어하면(lp_s > lp_t) 양의 KL → 벌.
    힌트 교사가 더 좋아하면(lp_t > lp_s) 음 → 클립 0(벌 없음). 첫 구현은 반대였다."""
    from types import SimpleNamespace as NS
    from src.training.verl_sdc import _read_opd_from_ref_logprobs
    from src.training.countdown_rewards import r_opd_meta
    # 행 0: teacher(hint) 팔, 행 1: student(plain) 팔 — 학생이 더 확신(−0.5 vs −2.0)
    ref_lp = [[-2.0, -2.0, -2.0, 0.0], [-0.5, -0.5, -0.5, 0.0]]
    kl = _read_opd_from_ref_logprobs(ref_lp, [NS(w_len=3)])
    assert kl[0] > 0 and abs(kl[0] - 1.5) < 1e-9
    assert r_opd_meta(kl[0], c=0.075) == -1.0
    # 반대: 교사가 학생 토큰을 더 좋아함 → 음 → 벌 없음
    kl2 = _read_opd_from_ref_logprobs([[-0.5, -0.5, -0.5], [-2.0, -2.0, -2.0]], [NS(w_len=3)])
    assert kl2[0] < 0 and r_opd_meta(kl2[0], c=0.075) == 0.0


# ══════════════════════════════════════════════════════════ 13. OPT_OPDC (그룹 중심화)

def test_opd_c_term_name_is_single_sourced():
    """OPD_TERM 과 같은 회귀 — OPD_TERM_C 가 OPT_OPDC 의 terms 와 META_TERMS 에 없으면
    fail-loud 가드와 텔레메트리 집계기가 영영 안 돈다."""
    assert {cr.OPD_TERM_C} & set(cr.ARM_SPECS["OPT_OPDC"]["terms"]), (
        f"OPD_TERM_C={cr.OPD_TERM_C!r} 이 ARM_SPECS['OPT_OPDC']['terms']"
        f"={cr.ARM_SPECS['OPT_OPDC']['terms']} 에 없다 — fail-loud 가드가 죽는다.")
    assert cr.OPD_TERM_C in cr.TERMS
    assert cr.OPD_TERM_C in cr.META_TERMS


def test_opt_opdc_arm_is_not_bit_identical_to_opt_or_opt_opd():
    """OPT_OPDC 가 OPT/OPT_OPD 와 같은 총보상을 내면 처치가 배선되지 않은 것이다."""
    row = _row(opd_kl=0.02)
    row["opd_kl_c"] = 0.4   # 그룹 중심화 결과(verl_sdc 가 그룹 단위로 미리 채우는 값)
    opt, _ = cr.arm_reward("OPT", row, step=30)
    opd, _ = cr.arm_reward("OPT_OPD", row, step=30)
    opdc, comp = cr.arm_reward("OPT_OPDC", row, step=30)
    assert comp.get(cr.OPD_TERM_C, 0.0) != 0.0, f"opd_meta_c 성분이 0 이다: {comp}"
    assert abs(opt - opdc) > 1e-9, "OPT_OPDC 와 OPT 총보상이 동일 — 배선 0."
    assert abs(opd - opdc) > 1e-9, "OPT_OPDC 와 OPT_OPD 총보상이 동일 — 항이 안 갈린다."


def test_opd_c_missing_material_dies_loud():
    """`opd_kl_c` 가 행에 아예 없으면(그룹 중심화가 배선 안 됐다는 사고) KeyError."""
    with pytest.raises(KeyError):
        cr.arm_reward("OPT_OPDC", {"r_corr": 1, "format_ok": 1, "emitted": 1}, step=30)


def test_opd_c_none_placeholder_is_silent_zero():
    _tot, comp = cr.arm_reward("OPT_OPDC", _row(opd_kl_c=None), step=30)
    assert comp[cr.OPD_TERM_C] == 0.0


def test_opd_center_rows_single_scored_row_is_all_zero():
    """비교 상대가 없는(채점 대상 1개) 그룹은 전원 0 — 발화했어도 상벌이 없다."""
    out = cr.opd_center_rows([0.05, None, None], c=0.075)
    assert out == [0.0, 0.0, 0.0]


def test_opd_center_rows_group_mean_is_zero_and_clips():
    """3 행 그룹, NaN 하나 제외 — 평균은 남은 둘로만 잡고, 합은 0(기대값 중립),
    큰 편차는 ±1 에서 클립된다."""
    c = 0.075
    out = cr.opd_center_rows([0.0, float("nan"), c * 10], c=c)
    assert out[1] == 0.0                                  # NaN 은 채점·평균 모두 제외
    assert out[0] == pytest.approx(1.0)                    # (mean-0)/c 가 1 을 넘어 클립
    assert out[2] == pytest.approx(-1.0)                   # (mean-10c)/c 가 -1 미만이라 클립
    scored = [out[0], out[2]]
    # 실제 클립 전 값들의 합은 0(대칭 평균 정의) — 클립이 안 걸리는 완만한 예로 재확인.
    out2 = cr.opd_center_rows([0.01, 0.03], c=c)
    assert sum(out2) == pytest.approx(0.0)
    assert out2[0] > 0 and out2[1] < 0    # 평균보다 낮은 kl(0.01)이 +를 받는다


def test_opd_center_rows_none_and_group_lt2_stay_zero():
    assert cr.opd_center_rows([], c=0.075) == []
    assert cr.opd_center_rows([None] * 5, c=0.075) == [0.0] * 5


def test_opdg_arm_and_gate_are_wired_in_the_stash():
    """OPT_OPDG(§11) 소스텍스트 가드: ①게이트 마스크가 `opd_center_rows` 로 들어가고,
    ②배치 게이트 행 수가 `[COUNTDOWN][WIRED]` 에 찍힌다(선언만 있고 배선 0 방지)."""
    from src.training import verl_sdc as vs

    assert cr.OPD_TERM_C in cr.ARM_SPECS["OPT_OPDG"]["terms"]
    assert cr.ARM_SPECS["OPT_OPDG"].get("opd_gate") is True
    src = inspect.getsource(vs._compute_countdown_arm_stash)
    assert 'opd_gate' in src, "verl_sdc 가 extra_info.opd_gate 를 안 읽는다."
    assert 'gate=_gate' in src, "게이트 마스크가 opd_center_rows 로 안 들어간다."
    assert 'opdg_gated_rows=' in src, "[COUNTDOWN][WIRED] 에 게이트 행 수가 없다."


def test_stash_gate_covers_both_opd_terms_and_centers_before_reward_assembly():
    """소스텍스트 가드: ①게이트가 opd_meta·opd_meta_c 둘 다 커버, ②그룹 중심화
    (`opd_center_rows` 호출)가 per-row `arm_reward` 조립 **전**에 있다."""
    from src.training import verl_sdc as vs

    src = inspect.getsource(vs._compute_countdown_arm_stash)
    assert "{_cdr.OPD_TERM, _cdr.OPD_TERM_C}" in src, (
        "OPD 게이트가 opd_meta_c 를 커버하지 않는다 — OPT_OPDC 에서 opd_kl 이 안 채워질 "
        "수 있다.")
    assert "opd_center_rows(" in src, "그룹 중심화 호출이 없다."
    center_idx = src.index("opd_center_rows(")
    reward_loop_idx = src.index('_cdr.arm_reward(arm, r, step=step, phat=phat_of[uid[i]])')
    assert center_idx < reward_loop_idx, (
        "그룹 중심화가 per-row arm_reward 조립보다 뒤에 있다 — opd_kl_c 가 그 행의 "
        "arm_reward 호출 시점에 아직 없을 수 있다.")


# ══════════════════════════════════════════════════════════ 14. OPT_CF (반사실 쌍둥이)
# §8, docs/RESULTS_cd7.md "같은 자리 인과 검사" — main(메타 허가)/twin(메타 없음)
# 쌍을 배치 전체에서 cf_key(=site_id)로 짝짓는 배선(`_compute_countdown_arm_stash`
# 의 `_cf_on`/`cf_center_rows` 블록)이 실제로 걸려 있는지, OSD/OPD 회귀와 같은
# `inspect.getsource` 소스텍스트 가드로 확인한다.

def test_cf_term_name_is_single_sourced():
    assert {cr.CF_TERM} & set(cr.ARM_SPECS["OPT_CF"]["terms"]), (
        f"CF_TERM={cr.CF_TERM!r} 이 ARM_SPECS['OPT_CF']['terms']"
        f"={cr.ARM_SPECS['OPT_CF']['terms']} 에 없다 — fail-loud 가드가 죽는다.")
    assert cr.CF_TERM in cr.TERMS
    assert cr.CF_TERM in cr.META_TERMS


def test_opt_cf_arm_is_not_bit_identical_to_opt():
    row = {"r_corr": 1, "format_ok": 1, "emitted": 1, "cf_meta_raw": 0.5}
    opt, _ = cr.arm_reward("OPT", row, step=30)
    cf, comp = cr.arm_reward("OPT_CF", row, step=30)
    assert comp.get(cr.CF_TERM, 0.0) != 0.0, f"cf_meta 성분이 0 이다: {comp}"
    assert abs(opt - cf) > 1e-9, f"OPT_CF({cf}) 와 OPT({opt}) 총보상이 동일 — 배선 0."


def test_cf_meta_raw_missing_is_fail_loud():
    with pytest.raises(KeyError):
        cr.arm_reward("OPT_CF", {"r_corr": 1, "format_ok": 1, "emitted": 1}, step=30)


def test_cf_meta_raw_none_is_silent_zero():
    row = {"r_corr": 1, "format_ok": 1, "emitted": 1, "cf_meta_raw": None}
    _tot, comp = cr.arm_reward("OPT_CF", row, step=30)
    assert comp[cr.CF_TERM] == 0.0


def test_stash_computes_cf_center_rows_gated_on_cf_term():
    """소스텍스트 가드: ①게이트(`_cf_on`)가 CF_TERM 을 커버, ②`cf_center_rows` 호출이
    꺼진 팔에서는 안 나오고(else 분기가 cf_meta_raw=0.0 으로만 채운다), ③배치 전체
    센터링이 per-row `arm_reward` 조립 **전**에 끝난다(그래야 그 행의 arm_reward 호출
    시점에 `cf_meta_raw` 가 이미 있다)."""
    from src.training import verl_sdc as vs

    src = inspect.getsource(vs._compute_countdown_arm_stash)
    assert "_cf_on = _cdr.CF_TERM in _cdr.ARM_SPECS[arm][\"terms\"]" in src, (
        "cf_meta 게이트(_cf_on)가 없다 — cf_center_rows 가 팔과 무관하게 항상/전혀 "
        "안 돌 수 있다.")
    assert "cf_center_rows(" in src, "배치 전체 센터링 호출이 없다."
    center_idx = src.index("cf_center_rows(")
    reward_loop_idx = src.index('_cdr.arm_reward(arm, r, step=step, phat=phat_of[uid[i]])')
    assert center_idx < reward_loop_idx, (
        "cf_center_rows 호출이 per-row arm_reward 조립보다 뒤에 있다 — cf_meta_raw 가 "
        "그 행의 arm_reward 호출 시점에 아직 없을 수 있다.")


def test_wired_log_line_reports_cf_scored_and_pairs_found():
    """★배선의 유일한 증거 — [COUNTDOWN][WIRED] 가 cf_main_scored/cf_pairs_found 를
    찍는다(런처 가드가 grep 하는 그 줄). 조건부 계산이 실제로 이 줄까지 이어지는지
    소스텍스트로 확인한다(값 자체는 통합 테스트 없이는 못 재현한다)."""
    from src.training import verl_sdc as vs

    src = inspect.getsource(vs._compute_countdown_arm_stash)
    assert "cf_main_scored={cf_main_scored}" in src
    assert "cf_pairs_found={cf_pairs_found}/{cf_pairs_total}" in src


# ══════════════════════════════════════════════════════════ 15. OPT_CFG (그룹 내부판)
# §10 — OPT_CF 의 twin 프롬프트 교란(학습 전 정책 plain 0.426 vs opt 0.352, 7.4pp)을
# 없앤 판. 같은 GRPO 그룹(uid) 안에서 emit(E)/no-emit(NE) 을 갈라 짝짓는 배선
# (`_compute_countdown_arm_stash` 의 `_cfg_on`/`cf_group_rows` 블록)이 실제로 걸려
# 있는지, OPT_CF 회귀와 같은 `inspect.getsource` 소스텍스트 가드로 확인한다.

def test_cf_group_term_name_is_single_sourced():
    assert {cr.CF_GROUP_TERM} & set(cr.ARM_SPECS["OPT_CFG"]["terms"]), (
        f"CF_GROUP_TERM={cr.CF_GROUP_TERM!r} 이 ARM_SPECS['OPT_CFG']['terms']"
        f"={cr.ARM_SPECS['OPT_CFG']['terms']} 에 없다 — fail-loud 가드가 죽는다.")
    assert cr.CF_GROUP_TERM in cr.TERMS
    assert cr.CF_GROUP_TERM in cr.META_TERMS


def test_opt_cfg_arm_is_not_bit_identical_to_opt():
    row = {"r_corr": 1, "format_ok": 1, "emitted": 1, "cf_group_raw": 0.5}
    opt, _ = cr.arm_reward("OPT", row, step=30)
    cfg, comp = cr.arm_reward("OPT_CFG", row, step=30)
    assert comp.get(cr.CF_GROUP_TERM, 0.0) != 0.0, f"cf_group 성분이 0 이다: {comp}"
    assert abs(opt - cfg) > 1e-9, f"OPT_CFG({cfg}) 와 OPT({opt}) 총보상이 동일 — 배선 0."


def test_cf_group_raw_missing_is_fail_loud():
    with pytest.raises(KeyError):
        cr.arm_reward("OPT_CFG", {"r_corr": 1, "format_ok": 1, "emitted": 1}, step=30)


def test_cf_group_raw_none_is_silent_zero():
    row = {"r_corr": 1, "format_ok": 1, "emitted": 1, "cf_group_raw": None}
    _tot, comp = cr.arm_reward("OPT_CFG", row, step=30)
    assert comp[cr.CF_GROUP_TERM] == 0.0


def test_stash_computes_cf_group_rows_gated_on_cf_group_term():
    """소스텍스트 가드: ①게이트(`_cfg_on`)가 CF_GROUP_TERM 을 커버, ②`cf_group_rows`
    호출이 꺼진 팔에서는 안 나오고(else 분기가 cf_group_raw=0.0 으로만 채운다),
    ③배치 전체 계산이 per-row `arm_reward` 조립 **전**에 끝난다(그래야 그 행의
    arm_reward 호출 시점에 `cf_group_raw` 가 이미 있다)."""
    from src.training import verl_sdc as vs

    src = inspect.getsource(vs._compute_countdown_arm_stash)
    assert "_cfg_on = _cdr.CF_GROUP_TERM in _cdr.ARM_SPECS[arm][\"terms\"]" in src, (
        "cf_group 게이트(_cfg_on)가 없다 — cf_group_rows 가 팔과 무관하게 항상/전혀 "
        "안 돌 수 있다.")
    assert "cf_group_rows(" in src, "그룹 내부 계산 호출이 없다."
    center_idx = src.index("cf_group_rows(")
    reward_loop_idx = src.index('_cdr.arm_reward(arm, r, step=step, phat=phat_of[uid[i]])')
    assert center_idx < reward_loop_idx, (
        "cf_group_rows 호출이 per-row arm_reward 조립보다 뒤에 있다 — cf_group_raw 가 "
        "그 행의 arm_reward 호출 시점에 아직 없을 수 있다.")


def test_wired_log_line_reports_cfg_scored_and_groups_ok():
    """★배선의 유일한 증거 — [COUNTDOWN][WIRED] 가 cfg_scored/cfg_groups_ok 를
    찍는다(런처 가드가 grep 하는 그 줄). 조건부 계산이 실제로 이 줄까지 이어지는지
    소스텍스트로 확인한다(값 자체는 통합 테스트 없이는 못 재현한다)."""
    from src.training import verl_sdc as vs

    src = inspect.getsource(vs._compute_countdown_arm_stash)
    assert "cfg_scored={cfg_scored}" in src
    assert "cfg_groups_ok={cfg_groups_ok}/{len(groups)}" in src
