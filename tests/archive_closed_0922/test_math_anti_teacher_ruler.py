"""math_anti_teacher_ruler 회귀 시험 (CPU, 모델 없음 — MockTok + mock forward).

1. 프롬프트 **바이트 동일** — 세 문맥이 math_activation_gate 의 빌더가 내는 것과 한 바이트도
   다르지 않다(그 게이트가 y 를 만든 문맥이 clean 이다).
2. donor 회전은 **절대 자기 자신이 아니다**.
3. AUC 함수 — 합성 데이터에서 알려진 값, 그리고 한 클래스뿐인 행의 처리.
4. 재순위 동률 규칙(최초 등장)과 maj@8 정본 규칙.
5. Δ 특징(boxed/tail/옛답 구간)과 국소화 분수.
6. 관문 진리표(NaN → FAIL 포함).
7. 배선: mock forward 로 end-to-end 요약이 돈다.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_activation_gate as A  # noqa: E402
import math_anti_teacher_ruler as R  # noqa: E402
import math_ruler_pivot as P  # noqa: E402

TOK = P.MockTok()
PROBLEM = "What is 2 plus 2?"
WRONG = "I add badly.\nThus \\boxed{5}\n"
DONOR_WRONG = "Different problem, different error.\nThus \\boxed{99}\n"


def _row(**kw):
    base = {"roll_id": "g1#0", "group_id": "g1", "problem": PROBLEM, "gold": "4",
            "wrong_text": WRONG, "wrong_ans": "5", "donor_wrong_text": DONOR_WRONG}
    base.update(kw)
    return base


# ── 1. 프롬프트 바이트 동일 ─────────────────────────────────────────────────────
def test_contexts_are_byte_identical_to_activation_gate_builders():
    ctx = R.build_contexts(TOK, "math_opt", _row())
    assert ctx["clean"] == A.blind_external_prompt(TOK, "math_opt", PROBLEM)
    assert ctx["anti"] == A.external_prompt(TOK, "math_opt", PROBLEM, WRONG)
    assert ctx["donor"] == A.external_prompt(TOK, "math_opt", PROBLEM, DONOR_WRONG)


def test_clean_context_has_no_wrong_text_and_anti_does():
    ctx = R.build_contexts(TOK, "math_opt", _row())
    assert WRONG not in ctx["clean"] and A.BLIND_EXTERNAL_NOTE.strip() in ctx["clean"]
    assert WRONG in ctx["anti"] and A.EXTERNAL_CUE in ctx["anti"]
    assert WRONG not in ctx["donor"] and DONOR_WRONG in ctx["donor"]


def test_anti_and_donor_differ_only_in_the_inserted_attempt():
    ctx = R.build_contexts(TOK, "math_opt", _row())
    assert ctx["anti"].replace(WRONG, DONOR_WRONG) == ctx["donor"]


# ── 2. donor 회전 ───────────────────────────────────────────────────────────────
def test_rotate_donors_never_equals_own_row():
    for n in (2, 3, 8, 150):
        d = R.rotate_donors(n)
        assert len(d) == n
        assert all(d[i] != i for i in range(n))
        assert sorted(d) == list(range(n))          # 완전 순열(모든 오답이 한 번씩 donor)


def test_rotate_donors_degenerate():
    assert R.rotate_donors(1) == [-1]
    assert R.rotate_donors(0) == []


def test_build_rows_drops_rows_without_donor_and_never_self_assigns():
    rolls = [{"group_id": f"g{i}", "problem": f"p{i}", "gold": "1",
              "text": f"sol {i} \\boxed{{{i}}}"} for i in range(3)]
    gens = [{"roll_id": f"g{i}#{i}", "population": "wrong", "cond": R.COND_FACT,
             "text": f"retry {i} {j}"} for i in range(3) for j in range(2)]
    rows = R.build_rows(gens, rolls)
    assert len(rows) == 3
    assert all(r["donor_roll_id"] != r["roll_id"] for r in rows)
    assert all(len(r["retry_texts"]) == 2 for r in rows)


# ── 3. AUC ──────────────────────────────────────────────────────────────────────
def _rows_for_auc(spec):
    """spec: [[(score, label), ...] per row]"""
    return [{"retries": [{"s": s, "y": y} for s, y in row]} for row in spec]


def test_pooled_auc_perfect_and_inverted():
    rows = _rows_for_auc([[(0.0, 0), (1.0, 1)], [(0.1, 0), (0.9, 1)]])
    assert R.pooled_auc(rows, "s", "y") == 1.0
    rows_inv = _rows_for_auc([[(1.0, 0), (0.0, 1)], [(0.9, 0), (0.1, 1)]])
    assert R.pooled_auc(rows_inv, "s", "y") == 0.0


def test_within_auc_ignores_single_class_rows():
    rows = _rows_for_auc([[(0.0, 0), (1.0, 1)],       # 두 클래스 — AUC 1
                          [(5.0, 1), (6.0, 1)],       # 한 클래스 — 건너뛴다
                          [(0.0, 1), (1.0, 0)]])      # 두 클래스 — AUC 0
    w = R.within_auc(rows, "s", "y")
    assert w["n_rows_both_classes"] == 2
    assert abs(w["auc"] - 0.5) < 1e-9


def test_within_auc_beats_pooled_when_row_offsets_dominate():
    """행마다 상수 오프셋이 라벨과 반대로 크면 pooled 는 무너지지만 within 은 1 이다 —
    GRPO 그룹 중심화가 지우지 못하는 «문제 안» 신호가 바로 이 상황이다."""
    rows = _rows_for_auc([[(0.0, 0), (1.0, 1)], [(100.0, 0), (101.0, 1)]])
    rows[1]["retries"][0]["y"] = 1                    # 큰 오프셋 행의 라벨을 뒤집는다
    rows[1]["retries"][1]["y"] = 0
    assert R.within_auc(rows, "s", "y")["auc"] == 0.5
    rows2 = _rows_for_auc([[(0.0, 0), (1.0, 1)], [(-100.0, 0), (-99.0, 1)]])
    assert R.within_auc(rows2, "s", "y")["auc"] == 1.0
    assert R.pooled_auc(rows2, "s", "y") < 1.0


def test_within_auc_handles_all_single_class():
    rows = _rows_for_auc([[(0.0, 1), (1.0, 1)]])
    w = R.within_auc(rows, "s", "y")
    assert w["n_rows_both_classes"] == 0 and math.isnan(w["auc"])


def test_rank_normalize_ties_and_bounds():
    assert R.rank_normalize([1.0, 2.0, 3.0]) == [0.0, 0.5, 1.0]
    assert R.rank_normalize([5.0, 5.0]) == [0.5, 0.5]
    out = R.rank_normalize([float("nan"), 1.0, 2.0])
    assert math.isnan(out[0]) and out[1] == 0.0 and out[2] == 1.0
    assert R.rank_normalize([]) == []


# ── 4. 재순위 · maj@8 ───────────────────────────────────────────────────────────
def test_argmin_tie_goes_to_first_appearance():
    assert R._argmin([1.0, 1.0, 2.0]) == 0
    assert R._argmin([float("nan"), 3.0, 3.0]) == 1
    assert R._argmin([float("nan"), float("nan")]) is None


def test_rerank_row_selectors():
    retries = [
        {"sum_delta": 5.0, "box_delta": 1.0, "r_corr": 0, "ans": "5"},
        {"sum_delta": -3.0, "box_delta": 9.0, "r_corr": 1, "ans": "4"},
        {"sum_delta": 0.0, "box_delta": 0.5, "r_corr": 0, "ans": "5"},
        {"sum_delta": 1.0, "box_delta": 2.0, "r_corr": 0, "ans": "5"},
    ]
    r = R.rerank_row(retries, "4")
    assert r["pick_sum_idx"] == 1 and r["pick_sum"] == 1        # ΣΔ 최소 = 정답
    assert r["pick_box_idx"] == 2 and r["pick_box"] == 0        # boxΔ 최소 = 오답
    assert r["maj8"] == 0                                       # 다수 "5" 는 오답
    assert abs(r["random"] - 0.25) < 1e-9
    assert r["oracle"] == 1


def test_majority_answer_math_equivalence_excludes_empty_and_first_appearance_tie():
    # 동치 표기는 한 군집.
    assert R.majority_answer(["\\frac{1}{2}", "\\dfrac{1}{2}", "3"]) in ("\\frac{1}{2}",)
    # 무응답은 표를 안 던진다 — 빈 표가 이기지 못한다.
    assert R.majority_answer(["", "", "7"]) == "7"
    assert R.majority_answer(["", None]) is None
    # 동률(1:1)은 **최초 등장**.
    assert R.majority_answer(["11", "12"]) == "11"
    assert R.majority_answer(["12", "11"]) == "12"


def test_majority_correct_boxes_before_grading():
    assert R.majority_correct(["\\dfrac{1}{3}", "\\dfrac{1}{3}", "2"], "\\frac{1}{3}") == 1
    assert R.majority_correct(["2", "2", "\\dfrac{1}{3}"], "\\frac{1}{3}") == 0
    assert R.boxed_correct("", "4") == 0


def test_paired_boot_is_a_paired_difference():
    ci = R.paired_boot([1, 1, 1, 0], [0, 0, 0, 0], seed=0, n_boot=200)
    assert abs(ci["mean"] - 0.75) < 1e-9 and ci["n"] == 4


# ── 5. Δ 특징 ───────────────────────────────────────────────────────────────────
def test_delta_features_box_tail_and_oldans_spans():
    text = "aa bb \\boxed{5} zz"
    ids, offs = R.enc_with_offsets(TOK, text)
    assert len(ids) == len(offs)
    box = R.final_boxed_span(text)
    assert box is not None and text[box[0]:box[1]].startswith("\\boxed{")
    delta = [1.0] * len(offs)
    f = R.delta_features(delta, offs, box, R.literal_spans(text, "5"), tail=2)
    assert f["n_tok"] == len(offs)
    assert abs(f["sum_delta"] - len(offs)) < 1e-9
    assert abs(f["mean_delta"] - 1.0) < 1e-9
    assert f["n_tok_box"] >= 1 and abs(f["box_delta"] - f["n_tok_box"]) < 1e-9
    assert abs(f["tail_delta"] - 2.0) < 1e-9
    assert f["n_tok_oldans"] >= 1
    assert 0.0 <= f["frac_abs_mass_last10pct"] <= 1.0
    assert 0.0 <= f["frac_abs_mass_boxed"] <= 1.0


def test_delta_features_without_boxed_span():
    text = "no box here"
    _, offs = R.enc_with_offsets(TOK, text)
    assert R.final_boxed_span(text) is None
    f = R.delta_features([1.0] * len(offs), offs, None, [])
    assert math.isnan(f["box_delta"]) and f["n_tok_box"] == 0


def test_literal_spans_non_overlapping():
    assert R.literal_spans("aaaa", "aa") == [(0, 2), (2, 4)]
    assert R.literal_spans("abc", "") == []
    assert R.literal_spans("abc", "z") == []


def test_spans_to_token_mask():
    offs = [(0, 2), (2, 4), (4, 6)]
    assert R.spans_to_token_mask(offs, [(2, 4)]) == [False, True, False]
    assert R.spans_to_token_mask(offs, []) == [False, False, False]


# ── 6. 라벨 ─────────────────────────────────────────────────────────────────────
def test_label_retry_three_labels():
    a = R.label_retry("so \\boxed{5}", "5", "4")
    assert a["reemit"] == 1 and a["wrong"] == 1 and a["changed_and_right"] == 0
    b = R.label_retry("so \\boxed{4}", "5", "4")
    assert b["reemit"] == 0 and b["wrong"] == 0 and b["changed_and_right"] == 1
    c = R.label_retry("so \\boxed{9}", "5", "4")
    assert c["reemit"] == 0 and c["wrong"] == 1 and c["changed_and_right"] == 0
    d = R.label_retry("no answer at all", "5", "4")
    assert d["no_answer"] == 1 and d["wrong"] == 1 and d["reemit"] == 0


# ── 7. 관문 진리표 ──────────────────────────────────────────────────────────────
def _summ(within, own_minus_donor, rr_mean, rr_lo, rr_hi, score="sum_delta"):
    key = "pick_sum_minus_maj8" if score == "sum_delta" else "pick_box_minus_maj8"
    return {"auc": {f"{score}->reemit": {"within": within, "own_minus_donor": own_minus_donor,
                                         "pooled": 0.5, "within_donor": 0.5,
                                         "within_n_rows": 10, "within_ranknorm_mw": 0.5}},
            "rerank": {key: {"mean": rr_mean, "lo": rr_lo, "hi": rr_hi, "n": 10}}}


def test_gate_truth_table():
    assert R.gate_pass(_summ(0.80, 0.15, 0.05, 0.01, 0.09)) is True
    assert R.gate_pass(_summ(0.74, 0.15, 0.05, 0.01, 0.09)) is False   # AUC 미달
    assert R.gate_pass(_summ(0.80, 0.05, 0.05, 0.01, 0.09)) is False   # donor 대조 미달
    assert R.gate_pass(_summ(0.80, 0.15, 0.05, -0.01, 0.09)) is False  # CI 가 0 포함
    assert R.gate_pass(_summ(0.80, 0.15, -0.05, -0.09, -0.01)) is False  # 음의 방향
    nan = float("nan")
    assert R.gate_pass(_summ(nan, 0.15, 0.05, 0.01, 0.09)) is False
    assert R.gate_pass(_summ(0.80, nan, 0.05, 0.01, 0.09)) is False
    assert R.gate_pass({}) is False
    # boxΔ 쪽만 통과해도 PASS 다.
    assert R.gate_pass(_summ(0.90, 0.20, 0.06, 0.02, 0.10, score="box_delta")) is True


def test_gate_requires_the_same_ruler_in_all_three_clauses():
    """ΣΔ 가 AUC 를 통과하고 boxΔ 가 재순위를 통과해도 **섞어서는** 안 된다."""
    s = _summ(0.90, 0.20, 0.0, -0.05, 0.05)                      # ΣΔ: 재순위 실패
    s["auc"]["box_delta->reemit"] = {"within": 0.50, "own_minus_donor": 0.0, "pooled": 0.5,
                                     "within_donor": 0.5, "within_n_rows": 10,
                                     "within_ranknorm_mw": 0.5}
    s["rerank"]["pick_box_minus_maj8"] = {"mean": 0.06, "lo": 0.02, "hi": 0.10, "n": 10}
    assert R.gate_pass(s) is False


# ── 8. 배선(end-to-end, mock forward) ───────────────────────────────────────────
def test_end_to_end_with_mock_forward():
    rolls = [{"group_id": f"g{i}", "problem": f"What is {i} plus 1?", "gold": str(i + 1),
              "text": f"bad reasoning \\boxed{{{i + 9}}}"} for i in range(4)]
    gens = []
    for i in range(4):
        for j in range(3):
            ans = (i + 1) if j == 0 else (i + 9)
            gens.append({"roll_id": f"g{i}#{i}", "group_id": f"g{i}", "population": "wrong",
                         "cond": R.COND_FACT, "text": f"retry {j} gives \\boxed{{{ans}}}"})
    rows = R.build_rows(gens, rolls)
    assert len(rows) == 4
    built = R.build_jobs(TOK, rows, "math_opt", max_len=4096)
    assert len(built["jobs"]) == 3 * sum(len(r["retries"]) for r in built["index"])
    forward = P.mock_forward_factory(seed=0)
    recs = R.retry_records(forward(built["jobs"], []), built)
    summ = R.summarize(recs, seed=1, n_boot=50)
    assert summ["n_rows"] == 4 and summ["n_retries"] == 12
    for sk in R.SCORES:
        for lk in R.LABELS:
            assert f"{sk}->{lk}" in summ["auc"]
    assert set(summ["rerank"]["acc"]) == {"pick_sum", "pick_box", "maj8", "random", "oracle"}
    assert summ["rerank"]["acc"]["oracle"] >= summ["rerank"]["acc"]["random"]
    assert isinstance(summ["pass_anti_teacher"], int)
    md = R.to_markdown(summ)
    assert "ANTI-TEACHER" in md and "재순위" in md


def test_build_jobs_drops_retries_that_do_not_fit_all_three_contexts():
    rolls = [{"group_id": "g0", "problem": "p " * 50, "gold": "1",
              "text": "w " * 50 + "\\boxed{9}"},
             {"group_id": "g1", "problem": "q " * 50, "gold": "2",
              "text": "v " * 50 + "\\boxed{8}"}]
    gens = [{"roll_id": f"g{i}#{i}", "population": "wrong", "cond": R.COND_FACT,
             "text": "y " * 400 + "\\boxed{3}"} for i in range(2)]
    rows = R.build_rows(gens, rolls)
    built = R.build_jobs(TOK, rows, "math_opt", max_len=64)
    assert built["jobs"] == [] and built["index"] == []
    assert built["n_dropped_long"] == 2


# ── 9. 토큰별 log p 확장(math_ruler_pivot) ──────────────────────────────────────
def test_job_tok_lp_spans_is_additive_and_mock_forward_returns_per_token():
    j = P.Job(list(range(10)), tok_lp_spans=[(4, 10)])
    assert j.tok_lp_spans == [(4, 10)] and j.lp_spans == []
    plain = P.Job(list(range(10)))
    assert plain.tok_lp_spans == []
    out = P.mock_forward_factory(seed=0)([j, plain], [])
    assert len(out[0]["tok_lp"]) == 1 and len(out[0]["tok_lp"][0]) == 6
    assert "tok_lp" not in out[1]


# ══ B1 «기억상실 교사» 모드(--cond wait) ═══════════════════════════════════════
def test_wait_student_context_is_byte_identical_to_activation_gate_wait_builder():
    ctx = R.build_contexts(TOK, "math_opt", _row(), R.COND_WAIT)
    assert ctx["stud"] == A.wait_prompt(TOK, "math_opt", PROBLEM, WRONG)
    assert ctx["stud"].endswith(A.WAIT_CUE) and WRONG in ctx["stud"]


def test_wait_teacher_context_is_blind_external_prompt_and_hides_the_attempt():
    ctx = R.build_contexts(TOK, "math_opt", _row(), R.COND_WAIT)
    assert ctx["teach"] == A.blind_external_prompt(TOK, "math_opt", PROBLEM)
    assert WRONG not in ctx["teach"] and A.BLIND_EXTERNAL_NOTE.strip() in ctx["teach"]


def test_wait_donor_student_context_never_carries_own_attempt():
    ctx = R.build_contexts(TOK, "math_opt", _row(), R.COND_WAIT)
    assert ctx["stud_donor"] == A.wait_prompt(TOK, "math_opt", PROBLEM, DONOR_WRONG)
    assert ctx["stud_donor"] != ctx["stud"]
    assert WRONG not in ctx["stud_donor"] and DONOR_WRONG in ctx["stud_donor"]
    assert ctx["stud"].replace(WRONG, DONOR_WRONG) == ctx["stud_donor"]


def test_wait_delta_is_teacher_minus_student_and_donor_swaps_the_student():
    """L = teach − stud (부호까지) · donor 는 **학생 쪽**을 바꾼다."""
    rows = R.build_rows(_gens_rolls()[0], _gens_rolls()[1], cond=R.COND_WAIT)
    built = R.build_jobs(TOK, rows, "math_opt", max_len=4096, cond=R.COND_WAIT)
    res = P.mock_forward_factory(seed=0)(built["jobs"], [])
    recs = R.retry_records(res, built)
    s = built["index"][0]["retries"][0]
    lp = {c: res[s["jobs"][c]]["tok_lp"][0] for c in ("teach", "stud", "stud_donor")}
    x = recs[0]["retries"][0]
    assert x["delta"] == [a - b for a, b in zip(lp["teach"], lp["stud"])]
    assert x["delta_donor"] == [a - b for a, b in zip(lp["teach"], lp["stud_donor"])]
    assert x["cond"] == R.COND_WAIT


def test_skip16_features_drop_exactly_the_first_16_tokens():
    delta = [1.0] * 16 + [3.0] * 10
    offs = [(i, i + 1) for i in range(len(delta))]
    f = R.delta_features_all(delta, offs, None, [], skip=16)
    assert f["n_tok"] == 26 and f[R.SKIP + "n_tok"] == 10
    assert abs(f["sum_delta"] - 46.0) < 1e-9
    assert abs(f[R.SKIP + "sum_delta"] - 30.0) < 1e-9
    assert abs(f[R.SKIP + "mean_delta"] - 3.0) < 1e-9
    short = R.delta_features_all([1.0] * 3, offs[:3], None, [], skip=16)
    assert math.isnan(short[R.SKIP + "sum_delta"])


def test_tail_or_box_mass_fraction_is_a_union_not_a_sum():
    """박스가 꼬리 안에 있으면 두 번 세지 않는다 — 합이 1 을 넘지 않는다."""
    text = "a b c d e f g h i \\boxed{5}"
    _, offs = R.enc_with_offsets(TOK, text)
    box = R.final_boxed_span(text)
    n = len(offs)
    f = R.delta_features([1.0] * n, offs, box, [])
    assert 0.0 <= f["frac_abs_mass_tail_or_box"] <= 1.0
    assert f["frac_abs_mass_tail_or_box"] >= max(f["frac_abs_mass_last10pct"],
                                                 f["frac_abs_mass_boxed"]) - 1e-9
    assert f["frac_abs_mass_tail_or_box"] <= (f["frac_abs_mass_last10pct"]
                                              + f["frac_abs_mass_boxed"] + 1e-9)


def test_wait_labels_inherit_the_old_answer_when_no_new_box_and_flag_stopped_early():
    a = R.label_retry("Yes. Correct.", "5", "4", inherit_answer=True)
    assert a["no_answer"] == 1 and a["inherited_answer"] == 1
    assert a["eff_ans"] == "5" and a["reemit"] == 1 and a["wrong"] == 1
    assert a["stopped_early"] == 1
    b = R.label_retry("Yes. Correct.", "5", "4")           # S1 규약은 그대로(상속 없음)
    assert b["inherited_answer"] == 0 and b["reemit"] == 0 and b["eff_ans"] == ""
    c = R.label_retry("x " * 700 + "\\boxed{4}", "5", "4", inherit_answer=True)
    assert c["stopped_early"] == 0 and c["changed_and_right"] == 1


def test_mean_by_decile_shape_and_monotone_ramp():
    d = R.mean_by_decile(list(range(100)))
    assert len(d) == 10 and all(d[i] < d[i + 1] for i in range(9))
    assert all(math.isnan(v) for v in R.mean_by_decile([]))


def _amnesic_summ(within_raw, donor_raw, conc, score="box_delta", conc_lo=None):
    """orient=-1 이므로 within_oriented = 1 − within_raw."""
    auc = {f"{s}->reemit": {"within": 0.5, "within_oriented": 0.5,
                            "within_donor": 0.5, "within_donor_oriented": 0.5,
                            "own_minus_donor": 0.0, "pooled": 0.5, "within_n_rows": 10,
                            "within_ranknorm_mw": 0.5}
           for s in R.GATE_SCORES_AMNESIC}
    auc[f"{score}->reemit"] = {
        "within": within_raw, "within_oriented": R._orient(within_raw, -1),
        "within_donor": donor_raw, "within_donor_oriented": R._orient(donor_raw, -1),
        "own_minus_donor": R._orient(within_raw, -1) - R._orient(donor_raw, -1),
        "pooled": 0.5, "within_n_rows": 10, "within_ranknorm_mw": 0.5}
    lo = conc * 0.6 if conc_lo is None else conc_lo
    return {"auc": auc,
            "localization": {"frac_abs_mass_tail_or_box": {"mean": 0.3, "lo": 0.2,
                                                           "hi": 0.4, "n": 10},
                             "conc_ratio": {"mean": conc, "lo": lo, "hi": conc * 1.4,
                                            "n": 10}}}


def test_amnesic_gate_needs_all_three_clauses_on_the_same_ruler():
    assert R.gate_pass_amnesic(_amnesic_summ(0.20, 0.45, 2.5)) is True   # AUC(−L)=.80, d=.25
    assert R.gate_pass_amnesic(_amnesic_summ(0.30, 0.45, 2.5)) is False  # AUC(−L)=.70 미달
    assert R.gate_pass_amnesic(_amnesic_summ(0.20, 0.25, 2.5)) is False  # own−donor=.05
    assert R.gate_pass_amnesic(_amnesic_summ(0.20, 0.45, 1.8)) is False  # 집중 비율 미달
    # 평균은 서는데 CI 하한이 1.0 을 넘지 못하면 FAIL(균일 분포를 배제 못 한다).
    assert R.gate_pass_amnesic(_amnesic_summ(0.20, 0.45, 2.5, conc_lo=0.9)) is False
    assert R.gate_pass_amnesic(_amnesic_summ(0.20, 0.45, 2.5, conc_lo=1.05)) is True
    assert R.gate_pass_amnesic(_amnesic_summ(0.80, 0.45, 2.5)) is False  # 방향이 반대
    assert R.gate_pass_amnesic({}) is False
    nan = float("nan")
    assert R.gate_pass_amnesic(_amnesic_summ(nan, 0.45, 2.5)) is False
    assert R.gate_pass_amnesic(_amnesic_summ(0.20, 0.45, nan)) is False
    # 자를 섞지 않는다: boxL 이 AUC 만, tailL 이 donor 만 통과 → FAIL.
    s = _amnesic_summ(0.20, 0.45, 2.5, score="box_delta")
    s["auc"]["box_delta->reemit"]["own_minus_donor"] = 0.0
    s["auc"]["tail_delta->reemit"].update({"within_oriented": 0.5, "own_minus_donor": 0.3})
    assert R.gate_pass_amnesic(s) is False


def _gens_rolls():
    rolls = [{"group_id": f"g{i}", "problem": f"What is {i} plus 1?", "gold": str(i + 1),
              "text": f"bad reasoning \\boxed{{{i + 9}}}"} for i in range(4)]
    gens = []
    for i in range(4):
        for j in range(3):
            ans = (i + 1) if j == 0 else (i + 9)
            gens.append({"roll_id": f"g{i}#{i}", "group_id": f"g{i}", "population": "wrong",
                         "cond": R.COND_WAIT, "text": f"retry {j} gives \\boxed{{{ans}}}"})
    return gens, rolls


def test_wait_mode_end_to_end_with_mock_forward():
    gens, rolls = _gens_rolls()
    rows = R.build_rows(gens, rolls, cond=R.COND_WAIT)
    assert len(rows) == 4
    built = R.build_jobs(TOK, rows, "math_opt", max_len=4096, cond=R.COND_WAIT)
    assert built["cond"] == R.COND_WAIT
    assert len(built["jobs"]) == 3 * sum(len(r["retries"]) for r in built["index"])
    recs = R.retry_records(P.mock_forward_factory(seed=0)(built["jobs"], []), built)
    summ = R.summarize(recs, seed=1, n_boot=50, cond=R.COND_WAIT)
    assert summ["cond"] == R.COND_WAIT and summ["orient"] == -1
    for sk in R.SCORES + R.SCORES_SKIP:
        for lk in R.LABELS:
            assert f"{sk}->{lk}" in summ["auc"]
    assert "stopped_early" in summ["rate"]
    assert len(summ["localization"]["mean_delta_by_decile"]) == 10
    assert "frac_abs_mass_tail_or_box" in summ["localization"]
    assert isinstance(summ["pass_amnesic_teacher"], int)
    summ["top_negative_tokens"] = R.top_negative_tokens(recs, None, min_count=1)[:20]
    assert len(summ["top_negative_tokens"]) <= 20
    md = R.to_markdown(summ)
    assert "AMNESIC-TEACHER" in md


def test_nan_boxl_on_reemit_rows_yields_nan_auc_with_n_valid_not_a_spurious_half():
    """새 박스가 없는 이어쓰기(=상속 재발화)에서 boxL 은 NaN 이다 — 그 자의 AUC 는 **NaN**
    이고 n_valid 가 보고돼야 한다(0.5 로 눙치거나 터지면 안 된다). tailL 은 멀쩡하다."""
    nan = float("nan")
    rows = [{"retries": [{"box_delta": nan, "tail_delta": -9.0, "reemit": 1},
                         {"box_delta": nan, "tail_delta": -8.0, "reemit": 1},
                         {"box_delta": -1.0, "tail_delta": -1.0, "reemit": 0}]}
            for _ in range(3)]
    w = R.within_auc(rows, "box_delta", "reemit")
    assert math.isnan(w["auc"]) and w["n_rows_both_classes"] == 0 and w["n_valid"] == 3
    assert math.isnan(w["auc_ranknorm_pooled"])
    assert math.isnan(R.pooled_auc(rows, "box_delta", "reemit"))
    t = R.within_auc(rows, "tail_delta", "reemit")
    assert t["n_valid"] == 9 and t["n_rows_both_classes"] == 3 and t["auc"] == 0.0
    # 요약과 마크다운이 그 상태에서도 돌고, 부족 경고가 찍힌다.
    recs = [{"gold": "4", "retries": [dict(x, ans="", eff_ans="5", r_corr=0, wrong=1,
                                           n_tok=20, sum_delta=x["tail_delta"],
                                           mean_delta=x["tail_delta"] / 20.0,
                                           conc_ratio=1.2, expected_mass_share=0.12,
                                           decile_delta=[0.0] * 10) for x in r["retries"]]}
            for r in rows]
    summ = R.summarize(recs, seed=0, n_boot=20, cond=R.COND_WAIT)
    a = summ["auc"]["box_delta->reemit"]
    assert math.isnan(a["within"]) and math.isnan(a["within_oriented"]) and a["n_valid"] == 3
    assert math.isnan(a["own_minus_donor"])
    md = R.to_markdown(summ)
    assert "boxL 은 이어쓰기" in md and "AMNESIC-TEACHER FAIL" in md


def test_unknown_cond_is_refused():
    import pytest
    with pytest.raises(SystemExit):
        R.mode_spec("external")


def test_s1_mode_is_unchanged_by_the_wait_addition():
    """기본 모드의 문맥·부호·출력 파일 이름이 그대로다(회귀 방지)."""
    assert R.mode_spec(R.COND_FACT)["orient"] == 1
    assert R.mode_spec(R.COND_FACT)["out_name"] == "per_retry.jsonl"
    assert R.mode_spec(R.COND_WAIT)["out_name"] == "per_continuation.jsonl"
    ctx = R.build_contexts(TOK, "math_opt", _row())
    assert set(ctx) == {"clean", "anti", "donor"}
