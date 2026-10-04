"""math_disagree_gate 회귀 시험 (CPU, 모의 LLM).

1. 후보 스케치 조립 — 메타 제거, 마지막 \\boxed 앞 400자, 후보 번호.
2. 다수결 동점 규칙(먼저 나온 것).
3. Commit 파싱 — 정상·마크다운·범위 밖·없음 → **다수결 답으로 폴백**(형식 미달이 다수결을
   넘을 수 없다는 규약).
4. minority-correct / all-agree 부분집합 판정.
5. 조건별 프롬프트 — pick 에 진단 지시가 없고, pick_blind 에 스케치가 없고, solve_fresh 는
   render_generation_prompt 와 **바이트 동일**.
6. 짝지은 Δ 산술 + 통과 규칙 진리표 — diagnose 가 minority-correct 를 건지고 pick 은 못
   건지는 모집단에서 SELECT·CONTENT 둘 다 PASS, 무효(null) 모집단에서 둘 다 FAIL,
   nan 이면 FAIL.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_disagree_gate as G  # noqa: E402
import math_ruler_pivot as P  # noqa: E402
from src.metacot.math_meta_prompt import render_generation_prompt  # noqa: E402

TOK = P.MockTok()
PROBLEM = "What is the sum of the roots of x^2-5x+6?"
CANDS = [{"answer": "5", "sketch": "Vieta gives -b/a = 5."},
         {"answer": "6", "sketch": "Product of roots is 6, so the sum is 6."},
         {"answer": "5", "sketch": "Roots are 2 and 3, so 2+3."},
         {"answer": "1", "sketch": "Difference of roots is 1."}]


# ── 1. 후보 스케치 ─────────────────────────────────────────────────────────────
def test_strip_meta_removes_every_block():
    t = "head <meta>\nconfidence: 0.3\ndecision: verify\n</meta> tail <meta>x</meta> end"
    assert G.strip_meta(t) == "head  tail  end"
    assert G.strip_meta("no meta here") == "no meta here"


def test_sketch_is_last_chars_before_final_boxed_with_meta_stripped():
    body = "A" * 500 + " <meta>\ndecision: verify\n</meta> " + "B" * 30
    text = body + " So \\boxed{42}. Done."
    s = G.candidate_sketch(text, chars=400)
    assert len(s) <= 400
    assert "meta" not in s and "decision" not in s
    assert "\\boxed" not in s            # 마지막 박스 **앞**까지만
    assert s.endswith("So")
    assert s.startswith("A")


def test_sketch_without_boxed_falls_back_to_tail():
    assert G.candidate_sketch("abcdefgh", chars=3) == "fgh"


def test_build_candidates_prefers_final_answer_field():
    rows = [{"text": "x \\boxed{7}", "final_answer": "7"},
            {"text": "y <meta>m</meta> z \\boxed{9}", "final_answer": ""}]
    cs = G.build_candidates(rows, chars=400)
    assert [c["answer"] for c in cs] == ["7", "9"]
    assert cs[1]["sketch"] == "y  z"


def test_render_candidates_numbers_from_one_and_blind_drops_sketches():
    full = G.render_candidates(CANDS)
    assert "[Candidate 1]" in full and "[Candidate 4]" in full and "[Candidate 0]" not in full
    assert "Vieta" in full
    blind = G.render_candidates(CANDS, blind=True)
    assert "[Candidate 1]" in blind and "Vieta" not in blind and "Product of roots" not in blind


# ── 2. 다수결 ──────────────────────────────────────────────────────────────────
def test_majority_vote_picks_largest_cluster():
    assert G.majority_vote(["5", "6", "5", "1"]) == "5"


def test_majority_vote_tie_goes_to_first():
    assert G.majority_vote(["6", "5", "5", "6"]) == "6"
    assert G.majority_vote(["1", "2", "3", "4"]) == "1"


def test_majority_vote_uses_math_equivalence_not_string_equality():
    assert G.majority_vote(["0.5", "\\frac{1}{2}", "3"]) in ("0.5", "\\frac{1}{2}")
    assert len(G.answer_clusters(["7", "7.0", "3"])) == 2


# ── 3. Commit 파싱·폴백 ───────────────────────────────────────────────────────
def test_parse_commit_variants():
    assert G.parse_commit("blah\nCommit: 3\n\\boxed{5}", 4) == 3
    assert G.parse_commit("**Commit:** 2", 4) == 2
    assert G.parse_commit("Commit Candidate 4", 4) == 4
    # 마지막 매치가 최종 결정이다
    assert G.parse_commit("Commit: 1 ... on reflection Commit: 2", 4) == 2


def test_parse_commit_malformed_is_none():
    assert G.parse_commit("no decision at all", 4) is None
    assert G.parse_commit("Commit: 9", 4) is None
    assert G.parse_commit("Commit: 0", 4) is None
    assert G.parse_commit("", 4) is None


def test_score_generation_uses_boxed_answer():
    sc = G.score_generation("Candidate 2 divides wrongly.\nCommit: 3\n\\boxed{5}", CANDS, "6")
    assert sc == {"answer": "5", "commit": 3, "fallback": 0, "malformed": 0}


def test_score_generation_falls_back_to_candidate_answer_without_boxed():
    sc = G.score_generation("Commit: 2", CANDS, "5")
    assert sc["answer"] == "6" and sc["commit"] == 2 and sc["fallback"] == 1


def test_malformed_generation_is_scored_as_the_majority_answer():
    # ★규약: 형식 미달은 버리지도 0 으로 치지도 않는다 — 다수결과 같은 점수를 받는다.
    sc = G.score_generation("I cannot decide.", CANDS, "5")
    assert sc == {"answer": "5", "commit": None, "fallback": 0, "malformed": 1}


def test_diag_stats_counts_words_and_two_candidate_citations():
    d = G.diag_stats("Candidate 1 and Candidate 3 disagree on the sign.")
    assert d["cites2"] == 1 and d["diag_words"] >= 7
    assert G.diag_stats("Candidate 2 is right.")["cites2"] == 0


# ── 4. 부분집합 판정 ───────────────────────────────────────────────────────────
def test_minority_correct_subset_detection():
    # 다수결 "5"(2표)가 틀렸고 후보 중 "6" 이 맞다 → minority-correct
    f = G.subset_flags(CANDS, "6")
    assert f["maj_answer"] == "5" and f["maj_corr"] == 0
    assert f["minority_correct"] == 1 and f["all_agree"] == 0
    # 다수결이 맞으면 minority-correct 가 아니다
    f2 = G.subset_flags(CANDS, "5")
    assert f2["maj_corr"] == 1 and f2["minority_correct"] == 0
    # 아무도 못 맞히면 어떤 선택기도 못 건진다 → minority-correct 아님
    f3 = G.subset_flags(CANDS, "99")
    assert f3["maj_corr"] == 0 and f3["minority_correct"] == 0


def test_all_agree_subset_detection():
    same = [{"answer": "7", "sketch": "a"}, {"answer": "7.0", "sketch": "b"}]
    f = G.subset_flags(same, "7")
    assert f["all_agree"] == 1 and f["minority_correct"] == 0


# ── 5. 조건별 프롬프트 ─────────────────────────────────────────────────────────
def test_solve_fresh_prompt_is_byte_identical_to_generation_prompt():
    assert (G.cond_prompt(TOK, "math_opt", PROBLEM, CANDS, "solve_fresh")
            == render_generation_prompt(TOK, "math_opt", PROBLEM))


def test_diagnose_prompt_has_diagnosis_instruction_and_sketches():
    q = G.cond_prompt(TOK, "math_opt", PROBLEM, CANDS, "diagnose")
    assert "DIAGNOSIS" in q and "Commit:" in q and "Vieta" in q
    assert q.startswith(render_generation_prompt(TOK, "math_opt", PROBLEM)[:20])


def test_diagnose_prompt_forbids_resolving_from_scratch():
    # ★s1(0914): 재풀이가 max_tokens=600 예산을 태워 malformed_rate .738 을 만들었다 —
    #   재풀이 금지를 명시 지시로 못박는다(gens.jsonl 표본 10개로 원인 확인).
    q = G.cond_prompt(TOK, "math_opt", PROBLEM, CANDS, "diagnose")
    assert "Do NOT solve the problem from scratch" in q
    assert "Compare the candidates" in q
    assert "on its own line immediately after the diagnosis" in q


def test_pick_prompt_has_no_diagnosis_instruction():
    q = G.cond_prompt(TOK, "math_opt", PROBLEM, CANDS, "pick")
    assert "DIAGNOSIS" not in q and "Do not explain" in q
    assert "Answer in one line" in q
    assert "Vieta" in q                 # 후보 스케치는 그대로 보인다


def test_pick_blind_prompt_has_no_sketches_but_same_instruction():
    q = G.cond_prompt(TOK, "math_opt", PROBLEM, CANDS, "pick_blind")
    assert "Vieta" not in q and "Product of roots" not in q
    assert "Do not explain" in q and "DIAGNOSIS" not in q
    assert "Answer in one line" in q
    assert "\\boxed{6}" in q            # 최종답만 보여 준다


def test_conditions_differ_only_by_the_user_suffix():
    base = render_generation_prompt(TOK, "math_opt", PROBLEM)
    for c in G.GEN_CONDS:
        q = G.cond_prompt(TOK, "math_opt", PROBLEM, CANDS, c)
        assert q != base and len(q) > len(base)


def test_cond_tuples():
    assert G.CONDS == ("majority", "diagnose", "pick", "pick_blind", "solve_fresh")
    assert G.GEN_CONDS == ("diagnose", "pick", "pick_blind")
    assert G.SUBSETS == ("all", "disagree", "minority_correct", "all_agree")
    assert G.GATE_SUBSET == "disagree"


# ── 6. 요약 산술·통과 규칙 ─────────────────────────────────────────────────────
def _rec(i, *, maj, diag, pick, blind=None, fresh=None, minority=0, agree=0):
    return {"group_id": f"g{i}", "problem": PROBLEM, "gold": "6",
            "cand_answers": ["5", "6", "5", "1"], "maj_answer": "5", "maj_corr": maj,
            "all_agree": agree, "minority_correct": minority, "disagree": int(not agree),
            "p_majority": float(maj), "p_diagnose": diag, "p_pick": pick,
            "p_pick_blind": pick if blind is None else blind,
            "p_solve_fresh": float(maj) if fresh is None else fresh,
            "commit_hist": [2], "commit_is_minority_rate": float(minority),
            "commit_minority_flags": [minority],
            "diag_words": 40.0, "cites2_rate": 1.0,
            "malformed_rate": 0.0, "fallback_rate": 0.0, "trunc_rate": 0.0}


def _rescue_population(n_min=30, n_maj=70):
    """diagnose 는 minority-correct 를 전부 건지고 pick 은 하나도 못 건진다."""
    recs = [_rec(i, maj=0, diag=1.0, pick=0.0, minority=1) for i in range(n_min)]
    recs += [_rec(100 + i, maj=1, diag=1.0, pick=1.0) for i in range(n_maj)]
    return recs


def test_paired_delta_arithmetic_on_rescue_population():
    s = G.summarize(_rescue_population(), k_cand=4, n=8, seed=11, n_boot=400)
    b = s["by_subset"]["all"]
    assert b["n"] == 100
    assert math.isclose(b["p_majority"]["mean"], 0.70)
    assert math.isclose(b["p_diagnose"]["mean"], 1.00)
    assert math.isclose(b["paired_diagnose_minus_majority"]["mean"], 0.30)
    assert math.isclose(b["paired_diagnose_minus_pick"]["mean"], 0.30)
    assert math.isclose(b["paired_pick_minus_majority"]["mean"], 0.00)
    assert math.isclose(s["frac_minority_correct"], 0.30)


def test_rescue_population_passes_both_gates():
    s = G.summarize(_rescue_population(), k_cand=4, n=8, seed=11, n_boot=400)
    assert s["pass_select"] == 1 and s["pass_content"] == 1
    assert G.select_pass(s) and G.content_pass(s)
    # MINORITY-RESCUE 는 그 부분집합의 diagnose 정확도다(다수결은 거기서 정의상 0)
    assert math.isclose(s["minority_rescue_diagnose"]["mean"], 1.0)
    assert math.isclose(s["minority_rescue_pick"]["mean"], 0.0)
    assert math.isclose(s["by_subset"]["minority_correct"]["p_majority"]["mean"], 0.0)


def test_null_population_fails_both_gates():
    recs = [_rec(i, maj=i % 2, diag=float(i % 2), pick=float(i % 2)) for i in range(100)]
    s = G.summarize(recs, k_cand=4, n=8, seed=11, n_boot=400)
    assert math.isclose(s["by_subset"]["all"]["paired_diagnose_minus_majority"]["mean"], 0.0)
    assert s["pass_select"] == 0 and s["pass_content"] == 0


def test_select_can_pass_while_content_fails_when_pick_matches_diagnose():
    # 진단이 다수결은 넘지만 pick 과 같다 → 「고르기」의 값이지 「진단 텍스트」의 값이 아니다
    recs = [_rec(i, maj=0, diag=1.0, pick=1.0, minority=1) for i in range(30)]
    recs += [_rec(100 + i, maj=1, diag=1.0, pick=1.0) for i in range(70)]
    s = G.summarize(recs, k_cand=4, n=8, seed=11, n_boot=400)
    assert s["pass_select"] == 1 and s["pass_content"] == 0


def test_small_positive_delta_below_threshold_fails_select():
    recs = [_rec(i, maj=0, diag=1.0, pick=0.0, minority=1) for i in range(2)]
    recs += [_rec(100 + i, maj=1, diag=1.0, pick=1.0) for i in range(98)]
    s = G.summarize(recs, k_cand=4, n=8, seed=11, n_boot=400)
    assert math.isclose(s["by_subset"]["all"]["paired_diagnose_minus_majority"]["mean"], 0.02)
    assert s["pass_select"] == 0        # 평균 0.02 < +0.03


def test_all_agree_subset_deltas_are_zero_by_construction():
    recs = [_rec(i, maj=1, diag=1.0, pick=1.0, agree=1) for i in range(20)]
    recs += [_rec(100 + i, maj=0, diag=1.0, pick=0.0, minority=1) for i in range(20)]
    s = G.summarize(recs, k_cand=4, n=8, seed=11, n_boot=400)
    b = s["by_subset"]["all_agree"]
    assert b["n"] == 20
    assert math.isclose(b["paired_diagnose_minus_majority"]["mean"], 0.0)
    assert math.isclose(s["frac_all_agree"], 0.5)


def test_nan_deltas_fail_both_gates():
    recs = [_rec(i, maj=1, diag=float("nan"), pick=float("nan")) for i in range(20)]
    s = G.summarize(recs, k_cand=4, n=8, seed=11, n_boot=400)
    assert s["by_subset"]["all"]["paired_diagnose_minus_majority"]["n"] == 0
    assert s["pass_select"] == 0 and s["pass_content"] == 0
    assert G.select_pass({}) is False and G.content_pass({}) is False


def test_markdown_renders_every_subset_and_both_verdicts():
    s = G.summarize(_rescue_population(), k_cand=4, n=8, seed=11, n_boot=200)
    md = G.to_markdown(s)
    for sub in G.SUBSETS:
        assert f"### subset: {sub}" in md
    assert "SELECT PASS" in md and "CONTENT PASS" in md and "MINORITY-RESCUE" in md
    assert "paired_diagnose_minus_solve_fresh" in md
    assert "(disagree)" in md            # ★판정문이 all 이 아니라 disagree 를 인용한다


def test_disagree_subset_is_complement_of_all_agree():
    # ★0914 개정: SELECT/CONTENT 는 all 이 아니라 disagree(all_agree 여집합)에서 잰다.
    recs = [_rec(i, maj=1, diag=1.0, pick=1.0, agree=1) for i in range(10)]   # all_agree
    recs += _rescue_population(n_min=30, n_maj=70)                            # 전부 disagree
    s = G.summarize(recs, k_cand=4, n=8, seed=11, n_boot=400)
    assert s["by_subset"]["disagree"]["n"] == 100
    assert s["by_subset"]["all"]["n"] == 110
    assert s["by_subset"]["all_agree"]["n"] == 10
    # disagree 부분집합만으로 판정하므로 all_agree 오염이 섞인 population 도 여전히 PASS
    assert s["pass_select"] == 1 and s["pass_content"] == 1
    assert G.select_pass(s) and G.content_pass(s)


def test_disagree_subset_dilution_would_have_masked_the_rescue_signal():
    # all_agree 문제를 대량으로 섞으면 "all" 부분집합의 Δ 는 희석되지만 disagree 는 그대로다.
    recs = [_rec(i, maj=1, diag=1.0, pick=1.0, agree=1) for i in range(500)]  # 오염
    recs += _rescue_population(n_min=30, n_maj=70)
    s = G.summarize(recs, k_cand=4, n=8, seed=11, n_boot=400)
    all_delta = s["by_subset"]["all"]["paired_diagnose_minus_majority"]["mean"]
    dis_delta = s["by_subset"]["disagree"]["paired_diagnose_minus_majority"]["mean"]
    assert dis_delta > all_delta
    assert math.isclose(dis_delta, 0.30)


# ── 커밋 자리 편향 진단 ────────────────────────────────────────────────────────
def test_commit_position_bias_and_matches_majority_rate():
    # commit_hist = {2: n}, 모두 다수결과 일치(minority=0) → bias=1.0, matches=1.0
    recs = [_rec(i, maj=1, diag=1.0, pick=1.0, minority=0) for i in range(10)]
    s = G.summarize(recs, k_cand=4, n=8, seed=11, n_boot=200)
    assert math.isclose(s["commit_position_bias"], 1.0)
    assert math.isclose(s["commit_matches_majority_rate"], 1.0)
    assert s["commit_hist"] == {"2": 10}


def test_commit_position_bias_low_when_spread_and_matches_rate_reflects_minority():
    recs = [_rec(i, maj=1, diag=1.0, pick=1.0, minority=1) for i in range(5)]
    s = G.summarize(recs, k_cand=4, n=8, seed=11, n_boot=200)
    # 전부 minority(다수결과 불일치) → matches_majority_rate = 0
    assert math.isclose(s["commit_matches_majority_rate"], 0.0)


# ── 조건별 malformed/trunc/reached_commit(pooled 아님) ─────────────────────────
def test_to_markdown_reports_cond_stats_and_void_warning_when_diagnose_is_malformed():
    s = G.summarize(_rescue_population(), k_cand=4, n=8, seed=11, n_boot=200)
    s["cond_stats"] = {
        "diagnose": {"malformed_rate": 0.738, "trunc_rate": 0.58, "reached_commit_rate": 0.262},
        "pick": {"malformed_rate": 0.05, "trunc_rate": 0.01, "reached_commit_rate": 0.95},
        "pick_blind": {"malformed_rate": 0.02, "trunc_rate": 0.0, "reached_commit_rate": 0.98},
    }
    md = G.to_markdown(s)
    assert "reached_commit_rate" in md
    assert "[VOID?]" in md and "diagnose" in md.split("[VOID?]")[1].split("\n")[0]


def test_to_markdown_no_void_warning_when_all_conds_below_threshold():
    s = G.summarize(_rescue_population(), k_cand=4, n=8, seed=11, n_boot=200)
    s["cond_stats"] = {c: {"malformed_rate": 0.1, "trunc_rate": 0.05, "reached_commit_rate": 0.9}
                       for c in G.GEN_CONDS}
    md = G.to_markdown(s)
    assert "[VOID?]" not in md


def test_default_max_tokens_is_2048():
    # ★s1(0914): 600 은 diagnose malformed_rate .738 을 만들었다(재풀이가 예산을 태움).
    #   CLI 실기동(vLLM) 없이 스크립트 소스의 argparse 기본값을 직접 확인한다.
    src = (REPO / "scripts" / "local" / "math_disagree_gate.py").read_text()
    assert '"--max_tokens", type=int, default=2048' in src
