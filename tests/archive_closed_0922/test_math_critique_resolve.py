"""math_critique_resolve_gate + math_critique_ig_ruler 회귀 시험 (CPU, 모델 없음).

1. 비평 누출 가드 — \\boxed / 답 숫자 / 답 문자열.
2. 네 조건의 프롬프트 조립 — blind 바이트 동일, note 합성, donor 가 **다른 문제**,
   (iv) 이어쓰기의 assistant 접합이 원 풀이와 바이트 동일.
3. 짝지은 Δ 산술과 통과 규칙.
4. IG 계산(모의 forward 의 log p 차)과 «가장 짧은 정답 형제» 선택.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_critique_ig_ruler as I  # noqa: E402
import math_critique_resolve_gate as C  # noqa: E402
import math_ruler_pivot as P  # noqa: E402
from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: E402

TOK = P.MockTok()
PROBLEM = "What is 2 plus 2?"
# ★끝 공백이 있는 풀이 — chat 템플릿이 assistant 본문의 끝 공백을 지우는 함정을 고정한다.
SOL = "Step 1: add them.\nStep 2: I get five.\nThus \\boxed{5}   \n"
CRIT = "The second step adds incorrectly; a correct approach must recheck the addition."


# ── 1. 누출 가드 ────────────────────────────────────────────────────────────────
def test_leak_guard_flags_boxed_and_answer_number_and_answer_string():
    assert C.critique_leaks("the step is wrong \\boxed{5}", "5")
    assert C.critique_leaks("you should have gotten 5 instead", "5")
    assert C.critique_leaks("the value \\frac{1}{2} is wrong", "\\frac{1}{2}")
    # 답과 무관한 숫자는 통과해야 한다(2 는 답 5 가 아니다).
    assert not C.critique_leaks("step 2 of the derivation is unjustified", "5")
    assert not C.critique_leaks(CRIT, "5")


def test_leak_guard_ignores_non_numeric_answer_without_match():
    assert not C.critique_leaks("the algebraic setup is wrong", "\\frac{1}{2}")


def test_critique_len_counts_words():
    assert C.critique_len("a b c") == 3 and C.critique_len("") == 0


# ── 2. 프롬프트 조립 ────────────────────────────────────────────────────────────
def test_blind_prompt_is_byte_identical_to_generation_prompt():
    assert (C.resolve_prompt(TOK, "math_opt", PROBLEM, None)
            == render_generation_prompt(TOK, "math_opt", PROBLEM))


def test_note_prompt_differs_only_by_note_suffix_in_user_turn():
    blind = C.resolve_prompt(TOK, "math_opt", PROBLEM, None)
    note = C.resolve_prompt(TOK, "math_opt", PROBLEM, CRIT)
    assert note != blind
    assert C.NOTE_PREFIX.strip() in note and CRIT in note
    # system 프롬프트는 그대로 — note 를 지우면 blind 와 같아진다.
    assert note.replace(C.NOTE_PREFIX + CRIT, "", 1) == blind


def test_donor_prompt_uses_other_problem_critique():
    donor = "A different problem's critique about geometry."
    q = C.resolve_prompt(TOK, "math_opt", PROBLEM, donor)
    assert donor in q and CRIT not in q


def test_critique_prompt_holds_solution_byte_identically():
    q = C.build_critique_prompt(TOK, "math_opt", PROBLEM, SOL)
    assert SOL in q, "원 풀이가 끝 공백까지 그대로 들어가야 한다"
    assert C.CRITIQUE_ASK in q and C._SENTINEL not in q


def test_incontext_prompt_is_generation_context_plus_solution_plus_seed():
    q = C.incontext_prompt(TOK, "math_opt", PROBLEM, SOL, CRIT)
    head = render_generation_prompt(TOK, "math_opt", PROBLEM)
    assert q == head + SOL + C.INCONTEXT_SEED.format(critique=CRIT)
    assert q.endswith("Second attempt:") and "decision: redirect" in q and CRIT in q


def test_assign_donors_never_picks_same_problem():
    recs = [{"group_id": f"g{i // 2}"} for i in range(8)]      # 문제당 두 행
    d = C.assign_donors(recs, random.Random(0))
    assert all(j is not None for j in d)
    assert all(recs[j]["group_id"] != recs[i]["group_id"] for i, j in enumerate(d))


def test_assign_donors_returns_none_when_single_problem():
    recs = [{"group_id": "g0"}, {"group_id": "g0"}]
    assert C.assign_donors(recs, random.Random(0)) == [None, None]


# ── 3. Δ 산술과 통과 규칙 ───────────────────────────────────────────────────────
def _recs(p_blind, p_crit, p_donor, p_inc=0.0, n=40):
    return [{"p_blind": p_blind, "p_crit": p_crit, "p_donor": p_donor, "p_incontext": p_inc,
             "anchor_crit": 0.1, "anchor_blind": 0.1, "critique_words": 30, "trunc_rate": 0.0}
            for _ in range(n)]


def test_paired_deltas_and_pass_rule():
    s = C.summarize(_recs(0.40, 0.55, 0.45), k=8, seed=0, n_boot=200)
    assert abs(s["paired_crit_minus_blind"]["mean"] - 0.15) < 1e-9
    assert abs(s["paired_crit_minus_donor"]["mean"] - 0.10) < 1e-9
    assert abs(s["paired_incontext_minus_blind"]["mean"] + 0.40) < 1e-9
    assert s["pass"] == 1


def test_pass_rule_fails_when_donor_matches_critique():
    # 비평이 blind 보다 크게 올라도 donor 와 같으면(내용 무관) FAIL.
    s = C.summarize(_recs(0.40, 0.55, 0.55), k=8, seed=0, n_boot=200)
    assert s["paired_crit_minus_donor"]["mean"] == 0.0 and s["pass"] == 0


def test_pass_rule_fails_below_threshold():
    s = C.summarize(_recs(0.40, 0.42, 0.40), k=8, seed=0, n_boot=200)
    assert s["paired_crit_minus_blind"]["mean"] <= C.PASS_DELTA_BLIND and s["pass"] == 0


def test_leak_rate_counts_rejected_critiques():
    s = C.summarize(_recs(0.4, 0.5, 0.45, n=8), k=8, seed=0, n_boot=100,
                    n_leaked=2, n_no_critique=0)
    assert abs(s["leak_rate"] - 0.2) < 1e-9


# ── 4. IG 자 ────────────────────────────────────────────────────────────────────
def _rolls():
    """g0: 오답 1 + 정답 2개(길이 다름), g1: 오답 1 + 정답 1개."""
    return [
        {"group_id": "g0", "problem": "p0", "gold": "4", "text": "wrong long solution g0",
         "r_corr": 0, "truncated": 0},
        {"group_id": "g0", "problem": "p0", "gold": "4", "text": "short right", "r_corr": 1,
         "truncated": 0},
        {"group_id": "g0", "problem": "p0", "gold": "4", "text": "a much longer right one",
         "r_corr": 1, "truncated": 0},
        {"group_id": "g1", "problem": "p1", "gold": "9", "text": "wrong g1", "r_corr": 0,
         "truncated": 0},
        {"group_id": "g1", "problem": "p1", "gold": "9", "text": "right g1 here", "r_corr": 1,
         "truncated": 0},
    ]


def test_shortest_correct_sibling_picks_shortest_and_skips_truncated():
    rolls = _rolls()
    sib = I.shortest_correct_sibling(rolls, "g0")
    assert sib["text"] == "short right" and sib["roll_id"] == "g0#1"
    rolls[1]["truncated"] = 1
    assert I.shortest_correct_sibling(rolls, "g0")["text"] == "a much longer right one"


def test_shortest_correct_sibling_none_when_no_correct():
    assert I.shortest_correct_sibling(_rolls(), "g2") is None


def _crits():
    return [{"roll_id": "g0#0", "critique": "critique zero", "leaked": 0},
            {"roll_id": "g1#3", "critique": "critique one", "leaked": 0},
            {"roll_id": "g0#0", "critique": "leaky", "leaked": 1}]


def test_build_rows_drops_leaked_and_assigns_cross_problem_donor():
    rows = I.build_rows(_rolls(), _crits(), random.Random(0))
    assert len(rows) == 2 and all(r["critique"] != "leaky" for r in rows)
    for r in rows:
        assert r["donor_critique"] and r["donor_critique"] != r["critique"]
    assert rows[0]["s_plus"] == "short right"


def test_ig_uses_note_minus_plain_mean_logprob():
    rows = I.build_rows(_rolls(), _crits(), random.Random(0))
    built = I.build_jobs(TOK, rows, "math_opt", max_len=10_000)
    assert len(built["jobs"]) == 8 * len(built["index"])

    # ★모의 forward: 각 job 의 lp 합을 «대상 토큰 수 × 정해 둔 평균» 으로 심는다.
    want = {"plus_plain": -2.0, "plus_note": -1.5, "plus_donor": -1.9,
            "plus_generic": -1.8, "plus_shuffled": -1.7, "plus_masked": -1.6,
            "minus_plain": -3.0, "minus_note": -3.4}
    kind_of = {}
    for it in built["index"]:
        for k, j in it["jobs"].items():
            kind_of[j] = (k, it["n_tok"][k])

    def forward(jobs, layers):
        out = []
        for j, _ in enumerate(jobs):
            k, n = kind_of[j]
            out.append({"hidden": {}, "entropy": [], "lp": [want[k] * n]})
        return out

    recs = I.ig_records(forward(built["jobs"], []), built)
    r = recs[0]
    assert abs(r["ig"] - 0.5) < 1e-9
    assert abs(r["ig_donor"] - 0.1) < 1e-9
    assert abs(r["ig_generic"] - 0.2) < 1e-9
    assert abs(r["ig_shuffled"] - 0.3) < 1e-9
    assert abs(r["ig_masked"] - 0.4) < 1e-9
    assert abs(r["ig_minus"] + 0.4) < 1e-9
    assert abs(r["ig_advantage"] - 0.4) < 1e-9      # IG − IG_donor
    assert abs(r["ig_vs_generic"] - 0.3) < 1e-9     # IG − IG_generic
    assert abs(r["ig_vs_shuffled"] - 0.2) < 1e-9    # IG − IG_shuffled
    assert abs(r["ig_vs_masked"] - 0.1) < 1e-9      # IG − IG_masked
    assert abs(r["ig_direction"] - 0.9) < 1e-9      # IG − IG_minus

    s = I.summarize(recs, {"g0#0": {"p_crit": 0.5, "p_blind": 0.2}}, seed=0, n_boot=100)
    assert abs(s["ig"]["mean"] - 0.5) < 1e-9 and s["n_joined_resolve"] == 1
    assert abs(s["ig_generic"]["mean"] - 0.2) < 1e-9
    assert abs(s["ig_shuffled"]["mean"] - 0.3) < 1e-9
    assert abs(s["ig_masked"]["mean"] - 0.4) < 1e-9
    assert abs(s["ig_vs_generic"]["mean"] - 0.3) < 1e-9
    assert abs(s["ig_vs_shuffled"]["mean"] - 0.2) < 1e-9
    # 모든 행이 IG > IG_minus 이므로 held-out AUC 는 완벽히 분리된다.
    assert abs(s["auc_ig_vs_minus"] - 1.0) < 1e-9
    assert abs(s["frac_ig_gt_minus"] - 1.0) < 1e-9
    # ig_vs_masked = 0.1 은 임계값(0.02) 보다 커서 "안정" 행이 하나도 없다 → 게이트 실패.
    assert s["frac_masked_delta_small"] == 0.0
    assert s["pass"] == 0
    # 두 행 모두 IG_advantage 가 항상 0.4 로 같으므로 분산은 0.
    assert abs(s["ig_delta_var"]) < 1e-9


# ── 5. 새 통제(generic/shuffled/masked) + held-out AUC + PASS 규칙 ─────────────────
def test_build_rows_generic_critique_identical_across_rows():
    rows = I.build_rows(_rolls(), _crits(), random.Random(0))
    assert len(rows) >= 1
    assert all(r["generic_critique"] == I.GENERIC_CRITIQUE for r in rows)
    assert len({r["generic_critique"] for r in rows}) == 1


def test_shuffle_critique_preserves_word_multiset():
    c = "alpha beta gamma delta epsilon"
    s = I.shuffle_critique(c, random.Random(0))
    assert sorted(s.split()) == sorted(c.split())
    assert s != c or len(set(c.split())) <= 1   # 시드 0 에서 우연히 항등이면 통과시키지 않는다
    assert I.shuffle_critique("", random.Random(0)) == ""


def test_mask_critique_removes_digits_and_boxed():
    c = "Step 2 uses 3.5 wrongly; \\boxed{42} is not justified, off by -7."
    m = I.mask_critique(c)
    assert not any(ch.isdigit() for ch in m)
    assert "\\boxed" not in m
    assert I.MASK_TOKEN in m
    # \boxed{...} 안 숫자는 박스 전체 치환으로 한 번만 가려진다(중복 치환 없음).
    assert m.count(I.MASK_TOKEN) == 4   # "2", "3.5", "\boxed{42}", "-7"


def test_critique_leak_flag_detects_numbers_and_boxed():
    assert I.critique_leak_flag("the value \\boxed{5} is wrong") == 1
    assert I.critique_leak_flag("step 3 is wrong") == 1
    assert I.critique_leak_flag("the algebra and case analysis are wrong") == 0
    assert I.critique_leak_flag("") == 0


def test_held_out_auc_ranks_plus_above_minus():
    sep = [{"ig": 1.0, "ig_minus": -1.0}, {"ig": 0.5, "ig_minus": -0.5}]
    assert abs(I.held_out_auc(sep) - 1.0) < 1e-9
    tied = [{"ig": 0.0, "ig_minus": 0.0}]
    assert abs(I.held_out_auc(tied) - 0.5) < 1e-9
    reversed_ = [{"ig": -1.0, "ig_minus": 1.0}]
    assert abs(I.held_out_auc(reversed_) - 0.0) < 1e-9


def test_gate_pass_truth_table():
    good = {"lo": 0.1, "hi": 0.3, "mean": 0.2}
    crosses_zero = {"lo": -0.1, "hi": 0.3, "mean": 0.1}
    negative = {"lo": -0.3, "hi": -0.1, "mean": -0.2}

    def s(ig_advantage=good, ig_vs_generic=good, ig_vs_shuffled=good, frac=0.9):
        return {"ig_advantage": ig_advantage, "ig_vs_generic": ig_vs_generic,
                "ig_vs_shuffled": ig_vs_shuffled, "frac_masked_delta_small": frac}

    assert I.gate_pass(s()) is True
    assert I.gate_pass(s(ig_advantage=crosses_zero)) is False
    assert I.gate_pass(s(ig_vs_generic=negative)) is False
    assert I.gate_pass(s(ig_vs_shuffled=crosses_zero)) is False
    assert I.gate_pass(s(frac=0.79)) is False
    assert I.gate_pass(s(frac=0.80)) is True
    assert I.gate_pass({}) is False


def test_build_jobs_drops_row_whose_any_condition_exceeds_max_len():
    rows = I.build_rows(_rolls(), _crits(), random.Random(0))
    built = I.build_jobs(TOK, rows, "math_opt", max_len=5)
    assert built["index"] == [] and built["n_dropped_long"] == len(rows)


def test_ig_job_spans_score_only_the_target_solution():
    rows = I.build_rows(_rolls(), _crits(), random.Random(0))
    built = I.build_jobs(TOK, rows, "math_opt", max_len=10_000)
    it = built["index"][0]
    job = built["jobs"][it["jobs"]["plus_plain"]]
    (s0, s1) = job.lp_spans[0]
    assert s1 - s0 == it["n_tok"]["plus_plain"] == len(P._enc(TOK, "short right"))
    assert s1 == len(job.ids)
