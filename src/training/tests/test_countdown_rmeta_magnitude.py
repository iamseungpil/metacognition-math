"""rmeta_magnitude 단위 테스트 — R_meta **크기** 계기.

무엇을 지키는가: C-012 가 확정한 구별이다. b3p 붕괴의 근인은 "PMI-shift 가 작아서"가
아니라 그것이 **그룹 중심화되어 평균이 0**(순 견인력 없음)인 채로 `meta_floor` 가
꺼져 있었다는 것이다. 따라서 이 계기는 세 경우를 **서로 다르게** 말해야 한다:

  ① mean≈0 · std>0 · floor=0   → C-012 붕괴 조건 (재분배형 + 견인력 없음)
  ② abs_mean=0                 → 무효 레버 (항이 켜졌는데 지급이 없다)
  ③ mean>0 · std>0             → ok

셋을 한 문장으로 뭉뚱그리면 계기가 있으나 마나가 된다 — 그것이 이 파일의 이유다.

실행:  python -m pytest src/training/tests/test_countdown_rmeta_magnitude.py -q
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root

from src.training import countdown_rewards as cr


def _c(meta_key, vals, floor=0.02):
    return [{"corr": 1.0, "format": 0.35, "meta_floor": floor, meta_key: v} for v in vals]


# ── ① C-012 붕괴 조건 ─────────────────────────────────────────────────────────
def test_centered_meta_with_floor_off_is_flagged_as_c012():
    """평균 0 · 분산 있음 · floor 0 → 두 경고가 **둘 다** 떠야 한다."""
    mag = cr.rmeta_magnitude(_c("meta_pos", [+0.8, -0.8, +0.4, -0.4], floor=0.0))
    s = mag["terms"]["meta_pos"]
    assert abs(s["mean"]) < 1e-9          # 순 견인력 없음
    assert s["std"] > 0.1                 # 그러나 재분배는 살아 있다
    assert s["abs_mean"] > 0.1            # 지급 자체는 있다 (무효 레버가 아니다)
    joined = " ".join(mag["verdict"])
    assert "순 견인력" in joined
    assert "meta_floor: 0" in joined      # C-012 의 나머지 절반


def test_centered_meta_with_floor_on_does_not_claim_collapse_condition():
    """같은 평균 0 이라도 floor 가 켜져 있으면 붕괴 조건은 아니다."""
    mag = cr.rmeta_magnitude(_c("meta_pos", [+0.8, -0.8, +0.4, -0.4], floor=0.02))
    joined = " ".join(mag["verdict"])
    assert "순 견인력" in joined
    assert "meta_floor: 0" not in joined
    assert mag["meta_floor_abs_mean"] > 0.0


# ── ② 무효 레버 ───────────────────────────────────────────────────────────────
def test_all_zero_meta_term_is_dead_lever():
    """곱셈 팔이 sign=0 으로 전면 침묵한 경우. 널을 '안 통한다'로 읽으면 안 되는 자리."""
    mag = cr.rmeta_magnitude(_c("meta_mul", [0.0] * 8))
    s = mag["terms"]["meta_mul"]
    assert s["abs_mean"] == 0.0 and s["std"] == 0.0 and s["frac_zero"] == 1.0
    assert any("DEAD LEVER" in v for v in mag["verdict"])


def test_constant_nonzero_meta_term_is_dead_lever_too():
    """평균은 0 이 아닌데 분산이 0 — GRPO 센터링에서 사라지므로 역시 무효 레버다."""
    mag = cr.rmeta_magnitude(_c("meta_pos", [0.5] * 8))
    assert any("DEAD LEVER" in v and "분산 0" in v for v in mag["verdict"])


# ── ③ 정상 ────────────────────────────────────────────────────────────────────
def test_healthy_meta_term_reports_ok():
    mag = cr.rmeta_magnitude(_c("meta_mul", [0.9, -0.2, 0.5, 0.0, 0.7, -0.1, 0.3, 0.4]))
    assert mag["verdict"] == ["ok"]
    assert mag["terms"]["meta_mul"]["mean"] > 0.0


# ── 배선/계약 ─────────────────────────────────────────────────────────────────
def test_share_of_total_needs_matching_totals_length():
    comps = _c("meta_mul", [0.5] * 4)
    assert "meta_share_of_total" not in cr.rmeta_magnitude(comps)          # 길이 불일치 → 생략
    assert "meta_share_of_total" in cr.rmeta_magnitude(comps, totals=[1.0] * 4)


def test_reads_every_meta_term_that_arm_reward_can_emit():
    """계기가 보는 항 집합이 ARM_SPECS 가 실제로 켜는 메타 항을 전부 덮는가.

    새 팔이 새 메타 항을 들고 들어왔는데 META_TERMS 에 등록을 잊으면, 그 항은
    조용히 계기 밖으로 사라진다 — 이 테스트가 그것을 막는다.
    """
    emitted = {t for spec in cr.ARM_SPECS.values() for t in spec["terms"]}
    meta_like = {t for t in emitted if t not in ("corr", "format", "meta_floor")}
    assert meta_like <= set(cr.META_TERMS), f"계기 미등록 항: {meta_like - set(cr.META_TERMS)}"


def test_pure_function_does_not_mutate_input():
    comps = _c("meta_pos", [0.3, -0.3])
    before = [dict(c) for c in comps]
    cr.rmeta_magnitude(comps, totals=[1.0, 1.0])
    assert comps == before


def test_empty_input_is_safe():
    assert cr.rmeta_magnitude([])["n"] == 0


# ══════════════════════════════════════════════════════════════════════════════
# group_variance_decomposition — /σ 정규화 하에서의 실제 영향력
# ══════════════════════════════════════════════════════════════════════════════
def _grp(n_correct, metas=None, fmt=0.35):
    """8-롤아웃 그룹 하나. metas 가 None 이면 메타 항이 없는 팔(A)."""
    g = []
    for i in range(8):
        c = {"corr": 1.0 if i < n_correct else 0.0, "format": fmt, "meta_floor": 0.02}
        if metas is not None:
            c["meta_pos"] = metas[i]
        g.append(c)
    return g


def test_degenerate_group_variance_comes_entirely_from_meta():
    """corr 이 상수인 그룹(Countdown 의 64.8%)에서는 σ 를 메타가 만든다.

    이것이 `norm_adv_by_std_in_grpo=True` 의 위험이다 — 나누고 나면 그 그룹이
    «메타만으로 만든 단위분산 그라디언트» 를 정상 그룹과 같은 세기로 낸다.
    """
    metas = [0.9, -0.7, 0.4, -0.2, 0.6, -0.5, 0.1, -0.6]
    r = cr.group_variance_decomposition([_grp(0, metas) for _ in range(10)])
    assert r["std_corr"] < 1e-12               # 전원 오답 → corr 분산 0
    assert r["frac_corr_constant"] == 1.0
    assert r["meta_var_share"] > 0.99           # 분산이 사실상 전부 메타
    assert any("corr 상수" in v for v in r["verdict"])


def test_meta_dominance_is_flagged_even_when_corr_varies():
    metas = [3.0, -3.0, 2.0, -2.0, 1.0, -1.0, 0.5, -0.5]
    r = cr.group_variance_decomposition([_grp(4, metas) for _ in range(10)])
    assert r["std_corr"] > 0.0                  # corr 은 갈렸는데도
    assert r["meta_var_share"] > 0.5            # 메타가 분산을 지배
    assert any("지배" in v for v in r["verdict"])


def test_balanced_reward_reports_ok():
    metas = [0.10, -0.08, 0.05, -0.04, 0.06, -0.05, 0.02, -0.06]
    r = cr.group_variance_decomposition([_grp(4, metas) for _ in range(10)])
    assert r["meta_var_share"] < 0.5
    assert r["verdict"] == ["ok"]


def test_tiny_sigma_amplification_is_flagged():
    """σ 가 0 에 가까운 그룹 — ε=1e-6 로 나누면 사소한 차이가 폭증한다."""
    r = cr.group_variance_decomposition([_grp(0, [0.0] * 8) for _ in range(10)])
    assert r["std_total"] < 1e-12          # 부동소수점 잔차 허용
    assert r["frac_tiny_std"] == 1.0
    assert r["amp_p95"] > 1e5                   # 1/(0+1e-6)
    assert any("증폭" in v for v in r["verdict"])


def test_arm_without_meta_has_zero_meta_variance():
    r = cr.group_variance_decomposition([_grp(4) for _ in range(10)])
    assert r["std_meta"] < 1e-12 and r["meta_var_share"] < 1e-12


def test_empty_and_singleton_groups_are_safe():
    assert cr.group_variance_decomposition([])["n_groups"] == 0
    r = cr.group_variance_decomposition([[{"corr": 1.0}]])   # 1-롤아웃 그룹 → std 0
    assert r["n_groups"] == 1 and r["std_total"] < 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# meta_outcome_discrimination — «좋은 메타인지인가» 의 유일한 런타임 검정
# ══════════════════════════════════════════════════════════════════════════════
def _rc(solved, meta):
    return {"r_corr": solved}, {"corr": float(solved), "meta_pos": meta}


def _split(items):
    rows, comps = zip(*items)
    return list(rows), list(comps)


def test_auc_is_one_when_meta_perfectly_ranks_solved_above_unsolved():
    rows, comps = _split([_rc(1, 0.9), _rc(1, 0.8), _rc(0, 0.2), _rc(0, 0.1)])
    assert cr.meta_outcome_discrimination(rows, comps)["auc"] == 1.0


def test_auc_is_zero_when_meta_ranks_backwards():
    """★거꾸로 — R_meta 가 못 푼 쪽을 높게 매기는 경우. 반드시 따로 말해야 한다."""
    rows, comps = _split([_rc(1, 0.1), _rc(1, 0.2), _rc(0, 0.8), _rc(0, 0.9)])
    r = cr.meta_outcome_discrimination(rows, comps)
    assert r["auc"] == 0.0
    assert any("거꾸로" in v for v in r["verdict"])


def test_auc_near_half_is_flagged_as_no_signal():
    """AUC~0.5 의 널은 «공식이 나쁘다» 가 아니라 «신호가 없다» 다 — 그렇게 말해야 한다."""
    rows, comps = _split([_rc(1, 0.5), _rc(0, 0.5), _rc(1, 0.5), _rc(0, 0.5)])
    r = cr.meta_outcome_discrimination(rows, comps)
    assert abs(r["auc"] - 0.5) < 1e-9          # 전부 동점 → 0.5
    assert any("신호가 없다" in v for v in r["verdict"])


def test_auc_undefined_when_one_class_is_empty():
    """전원 오답 그룹(Countdown 의 63.7%)에서는 AUC 가 정의되지 않는다.
    0.5 로 채우면 «못 가림» 으로 오독되므로 NaN 이어야 한다."""
    rows, comps = _split([_rc(0, 0.3), _rc(0, 0.7)])
    r = cr.meta_outcome_discrimination(rows, comps)
    assert r["auc"] != r["auc"]                 # NaN
    assert any("정의 불가" in v for v in r["verdict"])


def test_length_mismatch_raises_instead_of_silently_truncating():
    rows, comps = _split([_rc(1, 0.5), _rc(0, 0.5)])
    try:
        cr.meta_outcome_discrimination(rows, comps[:1])
    except ValueError as e:
        assert "길이가 다르다" in str(e)
    else:
        raise AssertionError("길이 불일치를 조용히 넘겼다")


# ══════════════════════════════════════════════════════════════════════════════
# 감사에서 나온 두 결함의 회귀 방지
# ══════════════════════════════════════════════════════════════════════════════
def test_warmup_zero_is_not_reported_as_dead_lever():
    """★step<20 은 처치 항이 축소돼 있다. warmup=0 을 «DEAD LEVER» 로 부르면
    첫 20스텝이 전부 오경보가 된다."""
    comps = _c("meta_pos", [0.0] * 8)
    assert any("DEAD LEVER" in v for v in cr.rmeta_magnitude(comps)["verdict"])
    r = cr.rmeta_magnitude(comps, warmup=0.0)
    assert not any("DEAD LEVER" in v for v in r["verdict"])
    assert any("warmup=0" in v for v in r["verdict"])


def test_partial_warmup_is_annotated_not_silenced():
    r = cr.rmeta_magnitude(_c("meta_pos", [0.0] * 8), warmup=0.5)
    assert any("DEAD LEVER" in v and "warmup 중" in v for v in r["verdict"])


def test_arm_without_meta_term_is_not_called_a_bad_discriminator():
    """★A팔은 메타 항이 없어 AUC 가 0.5 로 나온다. 그것을 «못 가린다» 로 읽으면 안 된다."""
    rows = [{"r_corr": 1}, {"r_corr": 0}, {"r_corr": 1}, {"r_corr": 0}]
    comps = [{"corr": 1.0, "format": 0.35}, {"corr": 0.0, "format": 0.35},
             {"corr": 1.0, "format": 0.35}, {"corr": 0.0, "format": 0.35}]
    r = cr.meta_outcome_discrimination(rows, comps)
    assert any("메타 항이 없다" in v for v in r["verdict"])
    assert not any("못 가린다" in v for v in r["verdict"])


def test_total_auc_measures_goal_conformance():
    """★총보상 AUC = «푼 롤아웃이 위로 매겨지나». 사전등록 §1 을 수치로 옮긴 것."""
    rows = [{"r_corr": 1}, {"r_corr": 0}]
    aligned = [{"corr": 1.0, "meta_pos": 0.0}, {"corr": 0.0, "meta_pos": 0.5}]
    r = cr.meta_outcome_discrimination(rows, aligned)
    assert r["auc_total"] == 1.0 and r["inversion_rate"] == 0.0

    inverted = [{"corr": 1.0, "meta_pos": 0.0}, {"corr": 0.0, "meta_pos": 2.0}]
    r2 = cr.meta_outcome_discrimination(rows, inverted)
    assert r2["auc_total"] == 0.0 and r2["inversion_rate"] == 1.0
    assert any("목표 역전율" in v for v in r2["verdict"])


# ══════════════════════════════════════════════════════════════════════════════
# 그룹 내 제한 — 파일럿 0821 에서 잡은 오독의 회귀 방지
# ══════════════════════════════════════════════════════════════════════════════
def test_within_group_scope_removes_cross_group_difficulty_confound():
    """★E팔 오경보 재현. gate 는 정의상 «어려운 그룹»에서 높다.

    통짜로 재면 그것이 «메타가 못 푼 쪽을 높게 매긴다» 로 둔갑한다(16.7% 역전).
    GRPO 는 같은 그룹 안에서만 비교하므로 그룹 내로 제한하면 오경보가 사라진다.
    """
    # 그룹0: 어려움(전원 오답) gate 높음 / 그룹1: 쉬움(전원 정답) gate 낮음
    rows, comps, gids = [], [], []
    for g, (solved, gate) in enumerate([(0, 0.9), (1, 0.1)]):
        for _ in range(4):
            rows.append({"r_corr": solved})
            comps.append({"corr": float(solved), "gate": gate})
            gids.append(g)

    pooled = cr.meta_outcome_discrimination(rows, comps)
    assert pooled["scope"] == "pooled"
    assert pooled["auc"] == 0.0                      # 통짜: 완전히 거꾸로로 보인다

    within = cr.meta_outcome_discrimination(rows, comps, group_ids=gids)
    assert within["scope"] == "within_group"
    assert within["auc"] != within["auc"]            # 비교 가능한 쌍이 0 → NaN
    assert any("정의 불가" in v or "잴 수 없" in v for v in within["verdict"])


def test_within_group_auc_counts_only_same_group_pairs():
    rows = [{"r_corr": 1}, {"r_corr": 0}, {"r_corr": 1}, {"r_corr": 0}]
    comps = [{"meta_pos": 0.9}, {"meta_pos": 0.1},      # 그룹0: 정답이 위 → 1.0
             {"meta_pos": 0.1}, {"meta_pos": 0.9}]      # 그룹1: 정답이 아래 → 0.0
    r = cr.meta_outcome_discrimination(rows, comps, group_ids=[0, 0, 1, 1])
    assert abs(r["auc"] - 0.5) < 1e-9                  # (1.0 + 0.0) / 2


def test_group_ids_length_mismatch_raises():
    rows = [{"r_corr": 1}, {"r_corr": 0}]
    comps = [{"meta_pos": 0.5}, {"meta_pos": 0.1}]
    try:
        cr.meta_outcome_discrimination(rows, comps, group_ids=[0])
    except ValueError as e:
        assert "group_ids" in str(e)
    else:
        raise AssertionError("길이 불일치를 조용히 넘겼다")
