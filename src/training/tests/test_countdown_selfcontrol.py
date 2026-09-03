"""countdown_selfcontrol 단위 테스트 — GPU·torch·verl 없이 CPU 로 돈다.

무엇을 지키는가:
  · 프리픽스 특징(n_att_pre·pairs_pre·pos_frac·has_boxed_pre)이 <meta> 경계에서
    정확히 갈리는가.
  · checked 는 **참인** 등식만 인정한다(거짓 등식은 0).
  · novel 은 "이미 결합된 쌍"을 다시 새 계열로 안 센다.
  · r_verify 네 사분면(hi×dec×checked/wrong 조합)이 SC_DESIGN.md 식과 일치하는가.
  · arm_reward("SC", row) 가 손으로 계산한 기대 총합과 맞는가(explore/verify/early_cost
    각각 하나씩 켜지는 네 시나리오).
  · arm_signature 가 SC_K_STUCK·SC_CONF_HI 를 담아 로그에서 임계값을 확인할 수 있는가.
  · ARM_SPECS/META_TERMS 등 기존 불변식이 SC/SCg 추가 이후에도 유지되는가
    (test_countdown_rewards.py 의 ADDED_ARMS·warmed 갱신과 짝을 이룬다).

실행:  python -m pytest src/training/tests/test_countdown_selfcontrol.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root

from src.training import countdown_rewards as cr
from src.training import countdown_selfcontrol as sc


NUMS = [12, 5, 20, 3, 30, 2, 8, 4, 15, 15]


def _resp(before: str, meta: str, after: str) -> str:
    return f"{before}\n<meta>\n{meta}\n</meta>\n{after}"


# ══════════════════════════════════════════════════════════ 1. prefix_features

def test_prefix_features_no_meta_uses_whole_text():
    text = "12+5=17 some prose with no meta tag at all"
    pf = sc.prefix_features(text, NUMS)
    assert pf["has_meta"] == 0
    assert pf["pos_frac"] == 1.0          # prefix == 전체
    assert pf["n_att_pre"] == 1


def test_prefix_features_counts_equalities_before_meta_only():
    text = _resp("12+5=17 20-3=17 30*2=60", "confidence: 0.9\ndecision: redirect\nnext: 8*4",
                 "8*4=32 more stuff")
    pf = sc.prefix_features(text, NUMS)
    assert pf["has_meta"] == 1
    assert pf["n_att_pre"] == 3           # 메타 뒤 8*4=32 는 안 센다
    assert pf["pairs_pre"] == {(3, 20), (5, 12), (2, 30)}
    assert 0.0 < pf["pos_frac"] < 1.0


def test_prefix_features_has_boxed_pre():
    text = _resp("\\boxed{17}", "confidence: 0.5\ndecision: verify\nnext: 8*4", "8*4=32")
    pf = sc.prefix_features(text, NUMS)
    assert pf["has_boxed_pre"] == 1


def test_prefix_features_empty_text_is_safe():
    pf = sc.prefix_features("", NUMS)
    assert pf["pos_frac"] == 0.0
    assert pf["n_att_pre"] == 0
    assert pf["has_meta"] == 0


# ══════════════════════════════════════════════════════════ 2. meta_fields

def test_meta_fields_parses_next_and_ruled_out():
    text = _resp("12+5=17", "confidence: 0.9\ndecision: redirect\nruled_out: 12+5\nnext: 8*4",
                 "8*4=32")
    mf = sc.meta_fields(text)
    assert mf["confidence"] == 0.9
    assert mf["decision"] == "redirect"
    assert mf["next"] == (8, "*", 4)
    assert "12+5" in mf["ruled_out"]
    assert "next" not in mf["ruled_out"]   # 비탐욕 매치 — next 절을 삼키지 않는다


def test_meta_fields_no_meta_returns_none_fields():
    mf = sc.meta_fields("no meta block here")
    assert mf["confidence"] is None
    assert mf["decision"] is None
    assert mf["next"] is None
    assert mf["ruled_out"] == ""


def test_meta_fields_confidence_not_clamped():
    """countdown_rewards.parse_meta 규약과 동일 — 0/1 도 있는 그대로 읽는다."""
    text = _resp("", "confidence: 1.0\ndecision: verify\nnext: 8*4", "")
    assert sc.meta_fields(text)["confidence"] == 1.0
    text0 = _resp("", "confidence: 0.0\ndecision: verify\nnext: 8*4", "")
    assert sc.meta_fields(text0)["confidence"] == 0.0


# ══════════════════════════════════════════════════════════ 3. post_meta_checked

def test_checked_true_equation_after_meta():
    text = _resp("", "confidence: 0.9\ndecision: verify\nnext: 8*4",
                 "8*4=32 \\boxed{32}")
    assert sc.post_meta_checked(text, NUMS) == 1


def test_checked_false_equation_gives_zero():
    """거짓 등식은 «재계산했다»로 인정하지 않는다 — 실제로 evaluate 한다."""
    text = _resp("", "confidence: 0.9\ndecision: verify\nnext: 8*4",
                 "8*4=99 \\boxed{99}")
    assert sc.post_meta_checked(text, NUMS) == 0


def test_checked_no_equation_after_meta_is_zero():
    text = _resp("", "confidence: 0.9\ndecision: verify\nnext: 8*4", "\\boxed{32}")
    assert sc.post_meta_checked(text, NUMS) == 0


def test_checked_no_meta_is_zero():
    assert sc.post_meta_checked("8*4=32 \\boxed{32}", NUMS) == 0


def test_checked_ignores_equations_before_meta():
    """검산 구간은 </meta> 뒤·boxed 앞 뿐이다. 프리픽스의 참인 등식은 안 센다."""
    text = _resp("8*4=32", "confidence: 0.9\ndecision: verify\nnext: 30*2", "\\boxed{60}")
    assert sc.post_meta_checked(text, NUMS) == 0


def test_checked_exact_division_only():
    text_exact = _resp("", "confidence: 0.9\ndecision: verify\nnext: 8*4",
                       "20/4=5 \\boxed{5}")
    assert sc.post_meta_checked(text_exact, NUMS) == 1
    text_inexact = _resp("", "confidence: 0.9\ndecision: verify\nnext: 8*4",
                         "20/3=6 \\boxed{6}")     # 20/3 은 정수가 아니다 — 거짓
    assert sc.post_meta_checked(text_inexact, NUMS) == 0


# ══════════════════════════════════════════════════════════ 4. novel / followed (via sc_row)

def test_novel_true_for_fresh_pair_and_zero_if_already_combined():
    fresh = _resp("12+5=17 20-3=17", "confidence: 0.9\ndecision: redirect\nnext: 30*2",
                 "30*2=60 \\boxed{60}")
    row = sc.sc_row(fresh, NUMS, target=60, r_corr=1, K_S=4, CONF_HI=0.8)
    assert row["novel"] == 1

    stale = _resp("12+5=17 20-3=17 30*2=60", "confidence: 0.9\ndecision: redirect\nnext: 30*2",
                 "30*2=60 \\boxed{60}")
    row2 = sc.sc_row(stale, NUMS, target=60, r_corr=1, K_S=4, CONF_HI=0.8)
    assert row2["novel"] == 0             # 이미 프리픽스에서 결합됐던 쌍


def test_novel_requires_both_numbers_in_nums():
    """next 가 지목한 수가 원래 다중집합에 없으면 novel 이 될 수 없다."""
    text = _resp("12+5=17", "confidence: 0.9\ndecision: redirect\nnext: 99*77", "")
    row = sc.sc_row(text, NUMS, target=60, r_corr=1, K_S=4, CONF_HI=0.8)
    assert row["novel"] == 0


def test_followed_requires_immediate_next_attempt_to_match():
    followed_text = _resp("12+5=17 20-3=17", "confidence: 0.9\ndecision: redirect\nnext: 30*2",
                          "30*2=60 \\boxed{60}")
    row = sc.sc_row(followed_text, NUMS, target=60, r_corr=1, K_S=4, CONF_HI=0.8)
    assert row["followed"] == 1

    not_followed = _resp("12+5=17 20-3=17", "confidence: 0.9\ndecision: redirect\nnext: 30*2",
                         "8-4=4 \\boxed{4}")
    row2 = sc.sc_row(not_followed, NUMS, target=60, r_corr=1, K_S=4, CONF_HI=0.8)
    assert row2["followed"] == 0


# ══════════════════════════════════════════════════════════ 5. r_verify 사분면

def _vrow(hi, dec_verify, checked, y):
    return {"hi": hi, "dec_verify": dec_verify, "checked": checked, "y": y}


def test_r_verify_not_hi_is_zero():
    assert cr.r_verify(_vrow(0, 1, 1, 1)) == 0.0
    assert cr.r_verify(_vrow(0, 1, 0, -1)) == 0.0


def test_r_verify_hi_verify_checked_is_positive_one():
    assert cr.r_verify(_vrow(1, 1, 1, 1)) == 1.0


def test_r_verify_hi_unchecked_wrong_is_negative_one():
    assert cr.r_verify(_vrow(1, 1, 0, -1)) == -1.0
    assert cr.r_verify(_vrow(1, 0, 0, -1)) == -1.0     # decision 이 verify 가 아니어도 벌은 동일


def test_r_verify_hi_unchecked_correct_is_zero():
    """확신에 찼고 검산은 안 했지만 맞혔다 — 벌도 상도 없다(y>=0 이라 wrong=0)."""
    assert cr.r_verify(_vrow(1, 0, 0, 1)) == 0.0


def test_r_verify_hi_checked_but_not_verify_decision_is_zero():
    """checked=1 이라도 decision!=verify 면 dec_verify=0 이라 상은 없다."""
    assert cr.r_verify(_vrow(1, 0, 1, 1)) == 0.0


# ══════════════════════════════════════════════════════════ 6. arm_reward("SC", ...) 손계산

def _base_row(**kw):
    row = dict(r_corr=1, format_ok=1, emitted=1,
              stuck=0, dec_redirect=0, novel=0, followed=0,
              hi=0, dec_verify=0, checked=0, y=1, early=0)
    row.update(kw)
    return row


def test_arm_SC_explore_only():
    row = _base_row(stuck=1, dec_redirect=1, novel=1, followed=1)
    total, comps = cr.arm_reward("SC", row, step=999)
    assert comps["explore"] == 1.0
    assert comps["verify"] == 0.0
    assert comps["early_cost"] == 0.0
    assert abs(total - (1.0 + 0.35 + 0.02 + 1.0)) < 1e-9


def test_arm_SC_verify_positive():
    row = _base_row(hi=1, dec_verify=1, checked=1, y=1)
    total, comps = cr.arm_reward("SC", row, step=999)
    assert comps["explore"] == 0.0
    assert abs(comps["verify"] - 0.5) < 1e-9
    assert abs(total - (1.0 + 0.35 + 0.02 + 0.5)) < 1e-9


def test_arm_SC_verify_negative():
    row = _base_row(r_corr=0, hi=1, dec_verify=1, checked=0, y=-1)
    total, comps = cr.arm_reward("SC", row, step=999)
    assert abs(comps["verify"] - (-0.5)) < 1e-9
    assert abs(total - (0.0 + 0.35 + 0.02 - 0.5)) < 1e-9


def test_arm_SC_early_cost():
    row = _base_row(early=1)
    total, comps = cr.arm_reward("SC", row, step=999)
    assert abs(comps["early_cost"] - (-0.25)) < 1e-9
    assert abs(total - (1.0 + 0.35 + 0.02 - 0.25)) < 1e-9


def test_arm_SC_early_cost_not_warmed_up_at_step_zero():
    """early_cost 는 워밍업을 안 받는다 — step=0 에서도 전액 지급된다."""
    row = _base_row(early=1)
    _, comps = cr.arm_reward("SC", row, step=0)
    assert comps["early_cost"] == -0.25


def test_arm_SC_explore_and_verify_are_zeroed_at_step_zero():
    row = _base_row(stuck=1, dec_redirect=1, novel=1, followed=1,
                    hi=1, dec_verify=1, checked=1)
    _, comps = cr.arm_reward("SC", row, step=0)
    assert comps["explore"] == 0.0
    assert comps["verify"] == 0.0


def test_arm_SCg_multiplies_explore_by_plan_ok():
    row = _base_row(stuck=1, dec_redirect=1, novel=1, followed=1, plan_ok=1)
    total, comps = cr.arm_reward("SCg", row, step=999)
    assert comps["explore_g"] == 1.0

    row0 = _base_row(stuck=1, dec_redirect=1, novel=1, followed=1, plan_ok=0)
    total0, comps0 = cr.arm_reward("SCg", row0, step=999)
    assert comps0["explore_g"] == 0.0


def test_arm_SC_missing_material_dies_loud():
    """무효 레버 방지 — 켜진 항의 원재료가 없으면 KeyError."""
    import pytest
    incomplete = dict(r_corr=1, format_ok=1, emitted=1)
    with pytest.raises(KeyError):
        cr.arm_reward("SC", incomplete, step=999)


# ══════════════════════════════════════════════════════════ 7. arm_signature

def test_sc_signature_contains_k_and_conf_threshold():
    sig = cr.arm_signature("SC")
    assert f"sc_k={cr.SC_K_STUCK}" in sig
    assert f"sc_conf_hi={cr.SC_CONF_HI:g}" in sig
    assert "explore@" in sig and "verify@" in sig and "early_cost@" in sig


def test_scg_signature_differs_from_sc():
    assert cr.arm_signature("SC") != cr.arm_signature("SCg")


def test_sc_and_scg_signatures_are_distinct_from_every_other_arm():
    sigs = cr.all_arm_signatures()
    assert len(set(sigs.values())) == len(cr.ARM_SPECS)


# ══════════════════════════════════════════════════════════ 8. 스펙 불변식 (SC/SCg 반영)

def test_sc_and_scg_share_common_terms():
    for arm in ("SC", "SCg"):
        for t in ("corr", "format", "meta_floor"):
            assert t in cr.ARM_SPECS[arm]["terms"]
        assert cr.ARM_SPECS[arm]["meta_form"] == "new"


def test_sc_and_scg_are_registered_in_arm_specs_and_meta_terms():
    assert "SC" in cr.ARM_SPECS and "SCg" in cr.ARM_SPECS
    for t in ("explore", "explore_g", "verify", "early_cost"):
        assert t in cr.META_TERMS


def test_reads_every_meta_term_that_arm_reward_can_emit_including_sc():
    """test_countdown_rmeta_magnitude.py 의 일반 계약을 SC/SCg 포함해 다시 확인."""
    emitted = {t for spec in cr.ARM_SPECS.values() for t in spec["terms"]}
    meta_like = {t for t in emitted if t not in ("corr", "format", "meta_floor")}
    assert meta_like <= set(cr.META_TERMS)


def test_no_torch_or_verl_import():
    src = Path(sc.__file__).read_text()
    for bad in ("import torch", "from torch", "import verl", "from verl"):
        assert bad not in src


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
