"""math_plan_gate 회귀 시험 (CPU, 모델 없음).

1. 계획 파싱 — 계획 3개 + Choice, 망가진 형식은 드롭.
2. 답 누출 가드.
3. 일곱 조건의 프롬프트 조립 — blind 바이트 동일, "Follow this approach" 접미, donor 는
   **서로 다른 세 문제**.
4. donor 셋 배정(assign_donor_triples) — 서로 다른 문제 셋, 부족하면 None.
5. 요약 산술 — argmax 비율·spread·짝지은 Δ·조건별 절단율.
6. 통과 규칙 진리표(PLAN-INFO / SELECTION) — max-of-3 편향이 donor 대조로 상쇄되는지가
   핵심(요건 6): own·donor 양쪽이 순수 잡음이면 p_best−p_blind 는 양이어도 PLAN-INFO 는
   FAIL 해야 한다(단일 donor 시절의 결함에 대한 회귀 시험).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_plan_gate as G  # noqa: E402
import math_ruler_pivot as P  # noqa: E402
from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: E402

TOK = P.MockTok()
PROBLEM = "What is the sum of the roots of x^2-5x+6?"
RAW = (
    "Plan 1: Use Vieta's formulas on the quadratic coefficients.\n"
    "Plan 2: Factor the quadratic and add the roots directly.\n"
    "Plan 3: Complete the square and read off the symmetric center.\n"
    "Choice: 2\n"
)


# ── 1. 파싱 ────────────────────────────────────────────────────────────────────
def test_parse_three_plans_and_choice():
    plans, choice = G.parse_plans(RAW)
    assert len(plans) == 3 and choice == 2
    assert plans[0].startswith("Use Vieta") and plans[1].startswith("Factor")


def test_parse_tolerates_markdown_and_repeats_first_wins():
    raw = ("**Plan 1**: alpha route\n- Plan 2. beta route\nPlan 3 - gamma route\n"
           "Plan 1: duplicate that must be ignored\n**Choice:** 3\n")
    plans, choice = G.parse_plans(raw)
    assert plans == ["alpha route", "beta route", "gamma route"] and choice == 3


def test_parse_drops_malformed():
    # 계획이 둘뿐
    assert G.parse_plans("Plan 1: a route\nPlan 2: b route\nChoice: 1") is None
    # Choice 없음
    assert G.parse_plans("Plan 1: a\nPlan 2: b\nPlan 3: c\n") is None
    # Choice 범위 밖 — 조용히 1로 떨어뜨리면 «선택 능력»이 기본값 성능이 된다
    assert G.parse_plans("Plan 1: a\nPlan 2: b\nPlan 3: c\nChoice: 4") is None
    assert G.parse_plans("Plan 1: a\nPlan 2: b\nPlan 3: c\nChoice: 0") is None
    # 서로 같은 계획(형식만 3개)
    assert G.parse_plans("Plan 1: same\nPlan 2: same\nPlan 3: same\nChoice: 1") is None
    assert G.parse_plans("") is None


# ── 2. 누출 가드 ───────────────────────────────────────────────────────────────
def test_leak_guard_flags_boxed_and_gold_value():
    plans = ["Use Vieta's formulas", "The answer is \\boxed{5}", "Factor it"]
    assert G.plans_leak(plans, "5")
    assert G.plans_leak(["Note that the total equals 5 at the end", "a", "b"], "5")
    # 금 답과 무관한 숫자는 통과(2차식의 «2»)
    assert not G.plans_leak(["Use Vieta on the degree 2 polynomial", "Factor", "Complete"], "5")


# ── 3. 프롬프트 조립 ───────────────────────────────────────────────────────────
def test_blind_prompt_is_byte_identical_to_generation_prompt():
    assert (G.solve_prompt(TOK, "math_opt", PROBLEM, None)
            == render_generation_prompt(TOK, "math_opt", PROBLEM))


def test_plan_prompt_differs_only_by_ask_suffix():
    blind = G.solve_prompt(TOK, "math_opt", PROBLEM, None)
    ask = G.plan_prompt(TOK, "math_opt", PROBLEM)
    assert ask != blind and G.PLAN_ASK.strip() in ask
    assert ask.replace(G.PLAN_ASK, "") == blind          # system 은 그대로


def test_follow_suffix_is_the_only_difference_for_the_seven_conditions():
    blind = G.solve_prompt(TOK, "math_opt", PROBLEM, None)
    plans, _ = G.parse_plans(RAW)
    rendered = [G.solve_prompt(TOK, "math_opt", PROBLEM, p) for p in plans]
    for p, q in zip(plans, rendered):
        assert q != blind
        assert G.FOLLOW_PREFIX + p in q
        assert q.replace(G.FOLLOW_PREFIX + p, "") == blind
    assert len(set(rendered)) == 3                       # 세 계획이 서로 다른 프롬프트
    donors = [G.solve_prompt(TOK, "math_opt", PROBLEM, f"Donor {i}'s approach")
              for i in range(3)]
    assert len(set(donors)) == 3                         # 세 donor 도 서로 다른 프롬프트
    for i, d in enumerate(donors):
        assert G.FOLLOW_PREFIX + f"Donor {i}'s approach" in d


def test_seven_conditions_tuple():
    assert G.CONDS == ("blind", "plan1", "plan2", "plan3", "donor1", "donor2", "donor3")
    assert len(G.CONDS) == 7


# ── 4. donor 셋 배정 ───────────────────────────────────────────────────────────
def test_assign_donor_triples_distinct_indices_and_group_disjoint():
    recs = [{"group_id": f"g{i}"} for i in range(10)]
    d = G.assign_donor_triples(recs, random.Random(0))
    assert len(d) == 10
    for i, triple in enumerate(d):
        assert triple is not None and len(triple) == G.N_DONOR
        assert len(set(triple)) == G.N_DONOR                  # 서로 다른 인덱스(반복 없음)
        for j in triple:
            assert recs[j]["group_id"] != recs[i]["group_id"]  # 자기 문제가 아니다
        groups = {recs[j]["group_id"] for j in triple}
        assert len(groups) == G.N_DONOR                        # donor 셋도 서로 다른 문제


def test_assign_donor_triples_succeeds_with_exactly_three_other_problems():
    # self(2행) + 서로 다른 문제 3개(각 2행) — donor 후보로 딱 맞는 최소 구성
    recs = [{"group_id": "self"}, {"group_id": "self"}]
    for g in ("a", "b", "c"):
        recs += [{"group_id": g}, {"group_id": g}]
    d = G.assign_donor_triples(recs, random.Random(1))
    for i, triple in enumerate(d):
        assert triple is not None
        groups = {recs[j]["group_id"] for j in triple}
        assert len(groups) == 3
        assert recs[i]["group_id"] not in groups


def test_assign_donor_triples_fails_when_only_two_other_problems():
    # self + 서로 다른 문제 2개뿐 — 어느 행에서 봐도 «자기 아닌 문제» 는 최대 2개
    recs = [{"group_id": "self"}, {"group_id": "self"}]
    for g in ("a", "b"):
        recs += [{"group_id": g}, {"group_id": g}]
    d = G.assign_donor_triples(recs, random.Random(2))
    assert all(t is None for t in d)


# ── 선별·준수 ──────────────────────────────────────────────────────────────────
def test_select_mixed_problems_keeps_one_row_per_mixed_group():
    rolls = [
        {"group_id": "g0", "problem": "p0", "gold": "1", "r_corr": 1},
        {"group_id": "g0", "problem": "p0", "gold": "1", "r_corr": 0},
        {"group_id": "g1", "problem": "p1", "gold": "2", "r_corr": 0},   # 전부 오답 → 제외
        {"group_id": "g1", "problem": "p1", "gold": "2", "r_corr": 0},
        {"group_id": "g2", "problem": "p2", "gold": "3", "r_corr": 1},   # 전부 정답 → 제외
        {"group_id": "g3", "problem": "p3", "gold": "4", "r_corr": 0},
        {"group_id": "g3", "problem": "p3", "gold": "4", "r_corr": 1},
    ]
    got = G.select_mixed_problems(rolls, max_problems=150)
    assert [r["group_id"] for r in got] == ["g0", "g3"]
    assert got[0]["p_group"] == 0.5 and got[0]["problem"] == "p0"
    assert len(G.select_mixed_problems(rolls, max_problems=1)) == 1


def test_adherence_jaccard_on_content_words():
    plan = "Use Vieta formulas on the quadratic coefficients"
    assert G.adherence(plan, "Vieta formulas quadratic coefficients follow immediately") > 0.5
    assert G.adherence(plan, "Completely unrelated narrative about trains leaving stations") < 0.2
    assert math.isnan(G.adherence("", "anything"))


# ── 5. 요약 산술 ───────────────────────────────────────────────────────────────
def _rec(gid, p_blind, ps, choice, ds, *, adh=0.5, trunc=None):
    trunc = trunc if trunc is not None else {c: 0.0 for c in G.CONDS}
    r = G.per_problem_record({
        "group_id": gid, "p_blind": p_blind, "p_plan": list(ps), "choice": choice,
        "p_donor": list(ds), "plan_words": 10.0, "adherence_blind": 0.1,
        "adherence_donormean": adh, "adherence_planmean": adh,
        "adherence_plan": [adh, adh, adh], "trunc_rate_by_cond": trunc,
    })
    r["adherence_choice"] = r["adherence_plan"][choice - 1]
    return r


def test_per_problem_derived_quantities():
    r = _rec("g0", 0.2, [0.1, 0.7, 0.4], 2, [0.2, 0.5, 0.3])
    assert r["p_choice"] == 0.7 and r["p_best"] == 0.7 and r["p_worst"] == 0.1
    assert abs(r["p_plan_mean"] - 0.4) < 1e-12
    assert abs(r["plan_spread"] - 0.6) < 1e-12
    assert r["choice_is_argmax"] == 1
    assert abs(r["p_donor_mean"] - 1.0 / 3) < 1e-12
    assert r["p_donor_best"] == 0.5 and abs(r["donor_spread"] - 0.3) < 1e-12
    # 최선이 아닌 것을 고른 경우
    r2 = _rec("g1", 0.2, [0.1, 0.7, 0.4], 3, [0.2, 0.5, 0.3])
    assert r2["p_choice"] == 0.4 and r2["choice_is_argmax"] == 0
    # 동점이면 «맞혔다»로 친다
    r3 = _rec("g2", 0.2, [0.5, 0.5, 0.1], 2, [0.2, 0.5, 0.3])
    assert r3["choice_is_argmax"] == 1


def test_summary_paired_deltas_spread_and_argmax_fraction():
    recs = [
        _rec("g0", 0.2, [0.1, 0.7, 0.4], 2, [0.1, 0.2, 0.3]),  # best .7 donor_best .3
        _rec("g1", 0.5, [0.5, 0.5, 0.5], 1, [0.4, 0.4, 0.4]),  # spread 0, donor_spread 0
        _rec("g2", 0.4, [0.2, 0.3, 0.9], 1, [0.1, 0.2, 0.2]),  # best .9 donor_best .2
        _rec("g3", 0.1, [0.2, 0.1, 0.1], 1, [0.1, 0.1, 0.1]),  # spread .1
    ]
    s = G.summarize(recs, k=8, seed=3, n_boot=200)
    assert s["n_problems"] == 4
    # (a) choice − blind = (.5, 0, −.2, .1)/4  (참고 진단)
    assert abs(s["paired_choice_minus_blind"]["mean"] - 0.1) < 1e-9
    # (b) choice − planmean
    want_b = ((0.7 - 0.4) + 0.0 + (0.2 - 1.4 / 3) + (0.2 - 0.4 / 3)) / 4
    assert abs(s["paired_choice_minus_planmean"]["mean"] - want_b) < 1e-9
    # (c) best − donor_best = (.7-.3, .5-.4, .9-.2, .2-.1)/4
    want_c = ((0.7 - 0.3) + (0.5 - 0.4) + (0.9 - 0.2) + (0.2 - 0.1)) / 4
    assert abs(s["paired_best_minus_donorbest"]["mean"] - want_c) < 1e-9
    # (d) planmean − donormean
    want_d = ((0.4 - 0.2) + (0.5 - 0.4) + (1.4 / 3 - 0.5 / 3) + (0.4 / 3 - 0.3 / 3)) / 4
    assert abs(s["paired_planmean_minus_donormean"]["mean"] - want_d) < 1e-9
    # (e) own spread ≥ .25 인 문제 둘, donor spread ≥ .25 인 문제는 없다
    assert abs(s["frac_spread_ge_thresh"] - 0.5) < 1e-9
    assert abs(s["frac_donor_spread_ge_thresh"] - 0.0) < 1e-9
    # (f) argmax 셋 / 넷
    assert abs(s["frac_choice_is_argmax"]["mean"] - 0.75) < 1e-9
    assert s["choice_hist"] == {"1": 3, "2": 1, "3": 0}
    # (g) 준수는 조건별로 찍힌다
    for k in ("adherence_blind", "adherence_choice", "adherence_planmean", "adherence_donormean"):
        assert math.isfinite(s[k]["mean"])
    assert s["unparsed_rate"] == 0.0


def test_summary_counts_unparsed_and_leaked_in_rates():
    recs = [_rec("g0", 0.2, [0.1, 0.7, 0.4], 2, [0.1, 0.2, 0.3])]
    s = G.summarize(recs, k=8, seed=3, n_boot=100, n_unparsed=2, n_leaked=1)
    assert abs(s["unparsed_rate"] - 2 / 4) < 1e-9 and abs(s["leak_rate"] - 1 / 4) < 1e-9


def test_trunc_rate_by_cond_arithmetic_over_seven_conditions():
    trunc_a = {"blind": 0.0, "plan1": 0.1, "plan2": 0.2, "plan3": 0.3,
               "donor1": 0.4, "donor2": 0.5, "donor3": 0.6}
    trunc_b = {"blind": 1.0, "plan1": 0.0, "plan2": 0.0, "plan3": 0.0,
               "donor1": 0.0, "donor2": 0.0, "donor3": 0.0}
    recs = [
        _rec("g0", 0.2, [0.1, 0.7, 0.4], 2, [0.1, 0.2, 0.3], trunc=trunc_a),
        _rec("g1", 0.5, [0.5, 0.5, 0.5], 1, [0.4, 0.4, 0.4], trunc=trunc_b),
    ]
    s = G.summarize(recs, k=8, seed=1, n_boot=50)
    assert set(s["trunc_rate_by_cond"]) == set(G.CONDS)
    assert abs(s["trunc_rate_by_cond"]["blind"] - 0.5) < 1e-9        # (0.0+1.0)/2
    assert abs(s["trunc_rate_by_cond"]["plan2"] - 0.1) < 1e-9        # (0.2+0.0)/2
    assert abs(s["trunc_rate_by_cond"]["donor3"] - 0.3) < 1e-9       # (0.6+0.0)/2


# ── 6. 통과 규칙 진리표 ────────────────────────────────────────────────────────
def _ci(mean, lo, hi):
    return {"mean": mean, "lo": lo, "hi": hi, "n": 10}


def _summ(c, d, b):
    return {"paired_best_minus_donorbest": c, "paired_planmean_minus_donormean": d,
            "paired_choice_minus_planmean": b}


def test_plan_info_pass_rule_truth_table():
    ok_c, ok_d = _ci(0.09, 0.03, 0.15), _ci(0.04, 0.01, 0.08)
    assert G.plan_info_pass(_summ(ok_c, ok_d, _ci(0, -1, 1)))
    # (c) 평균이 +.05 를 못 넘으면 실패(CI 가 0 을 제외해도)
    assert not G.plan_info_pass(_summ(_ci(0.04, 0.01, 0.07), ok_d, _ci(0, -1, 1)))
    # (c) CI 가 0 을 포함하면 실패
    assert not G.plan_info_pass(_summ(_ci(0.20, -0.01, 0.40), ok_d, _ci(0, -1, 1)))
    # (d) CI 가 0 을 포함하면 실패 — donor_mean 대조를 못 넘은 것
    assert not G.plan_info_pass(_summ(ok_c, _ci(0.04, -0.01, 0.09), _ci(0, -1, 1)))
    # (d) 평균이 음수면 실패
    assert not G.plan_info_pass(_summ(ok_c, _ci(-0.05, -0.09, -0.01), _ci(0, -1, 1)))
    # 결측(nan)은 실패
    assert not G.plan_info_pass(_summ(_ci(float("nan"), 0.1, 0.2), ok_d, _ci(0, -1, 1)))
    assert not G.plan_info_pass({})


def test_selection_pass_rule_truth_table():
    assert G.selection_pass(_summ(_ci(0, -1, 1), _ci(0, -1, 1), _ci(0.03, 0.01, 0.06)))
    assert not G.selection_pass(_summ(_ci(0, -1, 1), _ci(0, -1, 1), _ci(0.03, -0.01, 0.06)))
    assert not G.selection_pass(_summ(_ci(0, -1, 1), _ci(0, -1, 1), _ci(-0.03, -0.06, -0.01)))
    assert not G.selection_pass({})


def test_summary_writes_both_pass_flags():
    recs = [_rec(f"g{i}", 0.0, [0.9, 0.9, 0.9], 1, [0.0, 0.0, 0.0]) for i in range(12)]
    s = G.summarize(recs, k=8, seed=1, n_boot=300)
    assert s["pass_plan_info"] == 1        # best−donor_best=.9, planmean−donormean=.9
    assert s["pass_selection"] == 0        # choice == planmean → Δ=0
    assert "PLAN-INFO PASS" in G.to_markdown(s) and "SELECTION FAIL" in G.to_markdown(s)


def test_max_of_three_bias_defect_regression():
    """단일 donor 시절의 결함: p_best 는 3개 중 최댓값이라 무정보 상태에서도 p_blind 보다
    체계적으로 높다(Binomial(K,p)/8 최댓값 편향). paired_best_minus_blind 만 보면 이 편향을
    «계획이 유용했다»로 오독한다 — donor 도 **같은 3개 최댓값**(p_donor_best) 이어야
    짝지은 차에서 그 편향이 상쇄된다.

    여기서는 own 쪽과 donor 쪽 계획을 **같은 분포**(둘 다 잡음)로 구성한다 — own 은 매
    문제마다 [best=.625, mid=.5, worst=.375] 패턴(문제별로 어느 계획이 최선인지만 회전),
    donor 도 똑같은 패턴을 쓴다. p_best 의 평균은 p_blind=.5 보다 뚜렷이 높지만(정의상
    3개 최댓값이므로), p_best 와 p_donor_best 는 항등적으로 같다 — 계획에 진짜 정보가
    없다는 뜻이다."""
    pattern = [0.625, 0.5, 0.375]
    recs = []
    for i in range(9):
        rot = pattern[i % 3:] + pattern[:i % 3]           # 문제마다 순서만 돌린다(순수 잡음)
        recs.append(_rec(f"g{i}", 0.5, rot, 1, rot))       # own == donor 분포
    s = G.summarize(recs, k=8, seed=5, n_boot=500)
    # p_best − p_blind 는 크고 양(옛 규칙이라면 통과했을 신호)
    assert s["paired_best_minus_blind"]["mean"] > 0.09
    # 하지만 p_best 와 p_donor_best 는 항등적으로 같다 — 새 ceiling 통계량은 0
    assert abs(s["paired_best_minus_donorbest"]["mean"]) < 1e-9
    # PLAN-INFO 는 반드시 FAIL — 이것이 이 시험의 핵심 단언이다
    assert s["pass_plan_info"] == 0


def test_own_plans_genuinely_beat_donor_plans_passes():
    recs = [_rec(f"g{i}", 0.3, [0.6, 0.7, 0.65], 2, [0.2, 0.25, 0.3]) for i in range(9)]
    s = G.summarize(recs, k=8, seed=7, n_boot=500)
    assert s["paired_best_minus_donorbest"]["mean"] > G.PASS_DELTA_CEIL
    assert s["paired_planmean_minus_donormean"]["mean"] > 0
    assert s["pass_plan_info"] == 1
