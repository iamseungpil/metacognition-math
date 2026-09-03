"""OSD 항의 «선언된 레버, 배선 0» 회귀 테스트.

0825 적대검증에서 실제로 발생한 결함들을 박아 둔다. 이 파일이 없어서 776 개 테스트가
전부 통과하면서도 OSD 팔이 A 팔과 **비트 동일한 보상**을 내는 상태가 살아남았다.
"""
import math
import pytest

from src.training import countdown_rewards as cr


def _row(**kw):
    r = {"r_corr": 1, "format_ok": 1, "emitted": 1, "delta_cert": 0.05}
    r.update(kw)
    return r


def test_osd_term_name_is_single_sourced():
    """★핵심 회귀. 스코어러의 fail-loud 가드가 OSD_TERM 을 읽는데 그 이름이 팔의
    terms 에 없으면 가드가 영영 안 돈다 — 0825 에 정확히 그랬다("meta_osd" vs "osd")."""
    assert {cr.OSD_TERM} & set(cr.ARM_SPECS["OSD"]["terms"]), (
        f"OSD_TERM={cr.OSD_TERM!r} 이 ARM_SPECS['OSD']['terms']"
        f"={cr.ARM_SPECS['OSD']['terms']} 에 없다 — fail-loud 가드가 죽는다.")
    assert cr.OSD_TERM in cr.TERMS
    assert cr.OSD_TERM in cr.META_TERMS


def test_osd_arm_is_not_bit_identical_to_control():
    """OSD 팔이 A 팔과 같은 총보상을 내면 처치가 배선되지 않은 것이다."""
    row = _row(delta_cert=0.05)
    a, _ = cr.arm_reward("A", row, step=30)
    o, comp = cr.arm_reward("OSD", row, step=30)
    assert comp.get("osd", 0.0) != 0.0, f"osd 성분이 0 이다: {comp}"
    assert abs(a - o) > 1e-9, f"OSD({o}) 와 A({a}) 의 총보상이 동일하다 — 배선 0."


def test_delta_cert_none_is_fail_loud():
    """None = «스코어러가 안 돌았다». 조용한 0 은 이 팔을 A 팔로 위장시킨다.

    ⚠이 테스트만으로는 부족하다 — 0826 에 이것이 통과하는 상태로 학습이 step 1 에서
    죽었다. «구조적으로 W 를 못 만드는 정상 행» 까지 None 을 받아 즉사했기 때문이다.
    아래 `test_undefined_w_rows_are_zero_not_none` 이 그 짝이다. 둘은 함께 있어야 한다.
    """
    with pytest.raises(ValueError, match="delta_cert"):
        cr.arm_reward("OSD", _row(delta_cert=None), step=30)


def test_undefined_w_rows_are_zero_not_none():
    """★0826 회귀. W 가 구조적으로 정의 불가한 행은 «정상 행»이고 R_osd=0 이다.

    no_boxed / meta_at_end / boxed_before_meta 는 512행 중 3~30행으로 **매 스텝** 나온다.
    이들이 None 을 받으면 fail-loud 가 매 스텝 학습을 죽인다(실제로 죽였다).
    """
    from src.training import verl_sdc as vs
    for status in ("no_boxed", "meta_at_end", "boxed_before_meta"):
        row = vs._osd_empty_row(status)
        assert row["delta_cert"] == 0.0, (
            f"status={status} 는 «쟀는데 W 가 없다»이므로 0.0 이어야 한다. "
            f"None 이면 매 스텝 학습이 죽는다.")
        # 그리고 실제로 보상 계산이 통과해야 한다
        r = _row(delta_cert=row["delta_cert"])
        tot, comp = cr.arm_reward("OSD", r, step=30)
        assert comp["osd"] == 0.0

    for status in ("off", "ref_error", "span_error"):
        row = vs._osd_empty_row(status)
        assert row["delta_cert"] is None, (
            f"status={status} 는 «스코어러가 안 돌았다»이므로 None 이어야 한다.")


def test_auc_monitor_key_is_read_by_aggregator():
    """★0826 회귀. 감시가 넣는 키가 META_TERMS 에 없으면 AUC 가 항상 0.500 이 된다.

    완벽 판별 신호를 넣어도 0.5 가 나오면 회로차단기가 무조건 자폭하고, 그 순간
    이 팔은 대조군과 비트 동일해진다 — 안전장치를 경유한 «선언된 레버, 배선 0».
    """
    rows = [{"r_corr": 1}] * 30 + [{"r_corr": 0}] * 30
    comps = [{cr.OSD_TERM: +1.0}] * 30 + [{cr.OSD_TERM: -1.0}] * 30
    gids = ["g0"] * 60
    mq = cr.meta_outcome_discrimination(rows, comps, group_ids=gids)
    assert mq["auc"] > 0.99, (
        f"완벽 판별 신호인데 AUC={mq['auc']} — 집계기가 이 키를 못 읽는다. "
        f"META_TERMS={cr.META_TERMS}")


def test_delta_cert_nan_is_fail_closed_zero():
    """NaN = «쟀는데 비유한». 포이즌 행이 형제의 센터링을 망치지 않게 0 으로 닫는다."""
    assert cr.r_osd(float("nan"), 1) == 0.0
    assert cr.r_osd(math.inf, 0) == 0.0


def test_c_is_read_at_call_time_not_import_time():
    """관문이 실측 p90 을 박으면 서명과 보상이 **함께** 움직여야 한다."""
    old_c, old_prov = cr.OSD_C, cr.OSD_C_PROVISIONAL
    try:
        cr.OSD_C = 0.10
        assert cr.r_osd(0.05, 1) == pytest.approx(0.5)
        cr.OSD_C = 0.20
        assert cr.r_osd(0.05, 1) == pytest.approx(0.25), (
            "OSD_C 를 바꿨는데 r_osd 가 안 따라온다 — 기본 인자로 캡처됐다.")
        assert "osdc=0.2" in cr.arm_signature("OSD")
    finally:
        cr.OSD_C, cr.OSD_C_PROVISIONAL = old_c, old_prov


def test_four_quadrant_signs():
    """보상 대상은 «확신» 이 아니라 «확신과 결과의 정렬»(calibration) 이다."""
    assert cr.r_osd(0.5, 1) > 0   # 확신↑ 정답  → 상
    assert cr.r_osd(0.5, 0) < 0   # 확신↑ 오답  → 벌 (과신)
    assert cr.r_osd(-0.5, 1) < 0  # 확신↓ 정답  → 벌
    assert cr.r_osd(-0.5, 0) > 0  # 확신↓ 오답  → 상 (정직한 불확실성)


def test_leak_guard_has_exactly_one_definition():
    """공백-단어 기준 판본이 부활하면 사양(토큰 8-그램)보다 2~3배 헐거워진다."""
    assert not hasattr(cr, "osd_leak_guard"), (
        "countdown_rewards.osd_leak_guard 가 되살아났다. 누출 가드의 정의처는 "
        "verl_sdc._osd_leak_guard(토큰 기준) 하나여야 한다.")


def test_existing_eight_arms_untouched():
    """OSD 추가가 기존 팔의 정체 서명을 건드리면 안 된다."""
    for arm in "ABCDEFGH":
        sig = cr.arm_signature(arm)
        assert "osd" not in sig, f"{arm} 서명에 osd 가 새어들었다: {sig}"
