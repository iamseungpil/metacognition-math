"""cd9 M_AGREE(형제 동의 예측, 0914b) 회귀 시험.

agreement: 줄 파싱 / agree_true(배치 uid 그룹의 형제 동의율) / 모니터링 항 / 통제 진리표
(다수/소수 기준) / RAND 그룹 부호 반전(통제 ±1 부분만) / forced 미정의 / 텔레메트리 키·TEL 줄 /
abort 규칙(agree_line_rate·재시도류 공유 워밍업) / held-out 평가(모의 생성기) / parquet 강제
탐색 빌더(--variant math_agree 기본 forced_variant) / 런처 dry-run(RESP_LEN 6144, EVAL_SCRIPT).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from src.training import math_meta as M  # noqa: E402


def _meta_agree(decision=None, conf="0.4", agree=None, body="The substitution step might be off."):
    lines = ["<meta>", f"confidence: {conf}", body]
    if agree is not None:
        lines.append(f"agreement: {agree}")
    if decision:
        lines.append(f"decision: {decision}")
    lines.append("</meta>")
    return "\n".join(lines)


def _arow(first, decision, second=None, *, agree=None, marker=True):
    """first: 첫 \\boxed 내용, second: 두 번째 \\boxed 내용(None = 없음), agree: agreement: 줄 값."""
    t = f"work \\boxed{{{first}}}\n"
    if decision is not False:
        t += _meta_agree(decision, agree=agree) + "\n"
    if second is not None:
        t += ("Second attempt: " if marker else "") + f"other method \\boxed{{{second}}}"
    return t


# ── agreement: 줄 파싱 ─────────────────────────────────────────────────────────
def test_parse_agreement_line_and_value():
    raw = _arow("7", "verify", agree="0.75")   # split_attempts 의 raw 는 <meta>...</meta> 전체
    sp = M.split_attempts(raw)
    ag = M.parse_agreement(sp["meta"]["raw"])
    assert ag["agree_line"] == 1 and ag["agree_pred"] == pytest.approx(0.75)


def test_parse_agreement_missing_line():
    raw = _arow("7", "verify")   # agreement 줄 없음
    sp = M.split_attempts(raw)
    ag = M.parse_agreement(sp["meta"]["raw"])
    assert ag["agree_line"] == 0 and ag["agree_pred"] is None


def test_parse_agreement_out_of_range_value_is_none_but_line_present():
    raw = _arow("7", "verify", agree="1.5")   # 줄은 있지만 값이 [0,1] 밖
    sp = M.split_attempts(raw)
    ag = M.parse_agreement(sp["meta"]["raw"])
    assert ag["agree_line"] == 1 and ag["agree_pred"] is None


def test_parse_agree_row_extracts_pred_and_line_alongside_retry_fields():
    t = _arow("7", "verify", agree="0.8")
    r = M.parse_agree_row(t, "7", "P")
    assert r["agree_pred"] == pytest.approx(0.8) and r["agree_line"] == 1
    assert r["first_correct"] == 1 and r["final_correct"] == 1   # gold=7, verify, 답 변경 없음
    assert r["has_second_attempt"] == 0


# ── agree_true: 배치 uid 그룹에서 사후 계산 ──────────────────────────────────────
def test_agree_true_synthetic_group_with_math_equivalence():
    """7.0 ≡ 7(수학 동치) — answers_equivalent 를 그대로 재사용한다는 스펙 지시를 확인한다."""
    firsts = ["7", "7", "7.0", "7", "9", "9", "7", None]
    rows = [{"first_answer": f} for f in firsts]
    keys = ["g"] * 8
    at = M.compute_agree_true(rows, keys)
    # row0: own="7", 형제(자기 제외, first_answer 있는 것) = idx1..6 = ["7","7.0","7","9","9","7"]
    # 동치인 것: "7","7.0","7","7" = 4/6
    assert at[0] == pytest.approx(4 / 6)
    assert at[6] == pytest.approx(4 / 6)      # 같은 값("7")이면 같은 비율
    assert at[7] is None                       # 자기 first_answer 가 없으면 None


def test_agree_true_none_when_fewer_than_two_siblings_with_answer():
    rows = [{"first_answer": "7"}, {"first_answer": None}, {"first_answer": None}]
    at = M.compute_agree_true(rows, ["g", "g", "g"])
    assert at[0] is None                        # 형제 중 first_answer 있는 것이 0개 (<2)


def test_agree_true_groups_are_isolated_by_key():
    rows = [{"first_answer": "1"}, {"first_answer": "1"}, {"first_answer": "1"},
            {"first_answer": "2"}, {"first_answer": "2"}, {"first_answer": "3"}]
    keys = ["g1", "g1", "g1", "g2", "g2", "g2"]
    at = M.compute_agree_true(rows, keys)
    assert at[0] == pytest.approx(1.0)          # g1: 형제 둘 다 "1" 과 동치
    assert at[3] == pytest.approx(0.5)          # g2(own="2"): 형제=["2","3"] → 1/2


# ── 모니터링 항 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("pred,true,want,defined", [
    (0.8, 0.8, 1.0, 1),
    (0.8, 0.3, 1 - 2 * 0.5, 1),
    (0.0, 1.0, -1.0, 1),
    (None, 0.5, 0.0, 0),
    (0.5, None, 0.0, 0),
    (None, None, 0.0, 0),
])
def test_agree_monitoring_term(pred, true, want, defined):
    got, d = M.agree_monitoring_term(pred, true)
    assert d == defined and got == pytest.approx(want)


# ── 통제 진리표(다수/소수 기준) ─────────────────────────────────────────────────
@pytest.mark.parametrize("agree_true,decision,second,final,first,final_ok,extra,want,defined", [
    (0.2, "redirect", 1, "42", "7", 1, 0, 1.0, 1),        # 소수·재시도·다른 답·최종 정답 → +1.0
    (0.2, "redirect", 1, "9", "7", 0, 0, 0.5, 1),         # 소수·재시도·다른 오답 → +0.5
    (0.2, "redirect", 1, "7", "7", 0, 0, 0.0, 1),         # 소수·재시도·베낌 → 0
    (0.2, "redirect", 1, "7.0", "7", 0, 0, 0.0, 1),       # 소수·재시도·수학 동치 베낌 → 0
    (0.2, "redirect", 0, "7", "7", 0, 0, 0.0, 1),         # 소수·선언만 → 0
    (0.2, "verify", 0, "7", "7", 0, 0, -1.0, 1),          # 소수·verify → −1
    (0.8, "verify", 0, "7", "7", 1, 0, 1.0, 1),           # 다수·verify → +1
    (0.8, "redirect", 1, "7", "7", 1, 0, -1.0, 1),        # 다수·재시도(0 토큰) → −1
    (0.8, "redirect", 1, "9", "7", 0, 2000, -1.4, 1),     # 다수·재시도 2k 토큰 → −1 − 0.2·2
    (0.8, "redirect", 0, "7", "7", 1, 2000, -1.0, 1),     # 다수·선언만 → 길이 비용 없음
    (0.5, "verify", 0, "7", "7", 1, 0, 1.0, 1),           # 정확히 0.5 는 다수(majority = agree_true≥0.5)
    (None, "verify", 0, "7", "7", 1, 0, 0.0, 0),          # agree_true 없음 → 미정의
    (0.2, None, 0, "7", "7", 0, 0, 0.0, 0),               # decision 없음 → 미정의
])
def test_agree_control_truth_table(agree_true, decision, second, final, first, final_ok, extra, want, defined):
    base, cost, d = M.agree_control_parts(agree_true, decision, second, final, first,
                                          final_correct=final_ok, extra_tokens=extra, len_cost=0.2)
    assert d == defined and (base - cost) == pytest.approx(want)


def test_agree_control_copied_uses_math_equivalence_with_string_fallback(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("math_verify down")
    monkeypatch.setattr(M, "grade_math", boom)
    base, cost, d = M.agree_control_parts(0.2, "redirect", 1, "9", "7", final_correct=0,
                                          extra_tokens=0, len_cost=0.2)
    assert d == 1 and (base - cost) == pytest.approx(0.5)   # 폴백 문자열 비교로도 «변경» 판정


# ── compute_rows(M_AGREE): 답 스팬은 최종 답 GOLD 정오만 ────────────────────────
def test_compute_rows_agree_answer_span_is_final_gold_only(monkeypatch):
    monkeypatch.setenv("MATH_AGREE_MON_W", "0.0")
    monkeypatch.setenv("MATH_AGREE_CTL_W", "0.5")
    monkeypatch.delenv("MATH_RETRY_LEN_COST", raising=False)
    # 3 행, 전부 first_answer="7" → 서로 동치 → agree_true=1.0(다수) 전원.
    texts = [_arow("7", "redirect", "42", agree="0.5"),   # 다수·재시도(구제) → −1·0.5=−0.5
             _arow("7", "verify", agree="0.5"),           # 다수·verify → +1·0.5=+0.5
             _arow("7", "verify", agree="0.5")]
    rows = M.compute_rows(texts, ["42"] * 3, ["P"] * 3, "M_AGREE", uids=["u"] * 3,
                          tok_len_fn=lambda s: 0)   # 길이 비용을 0 으로 고정해 진리표 값만 본다
    assert [r["final_correct"] for r in rows] == [1, 0, 0]     # gold=42: 재시도만 구제
    assert [r["answer_total"] for r in rows] == [1.0, 0.0, 0.0]
    assert rows[0]["agree_true"] == pytest.approx(1.0)
    assert rows[0]["meta_val"] == pytest.approx(-0.5)
    assert rows[1]["meta_val"] == pytest.approx(0.5) and rows[2]["meta_val"] == pytest.approx(0.5)
    assert all(r["meta_defined"] == 1 for r in rows)


def test_compute_rows_agree_meta_val_combines_monitoring_and_control(monkeypatch):
    monkeypatch.setenv("MATH_AGREE_MON_W", "0.5")
    monkeypatch.setenv("MATH_AGREE_CTL_W", "0.5")
    # 3 행 전부 first="7"(동치) → agree_true=1.0. row0 은 agree_pred=1.0(정확한 예측) → 모니터링=+1.
    texts = [_arow("7", "verify", agree="1.0"), _arow("7", "verify", agree="0.0"),
             _arow("7", "verify", agree="1.0")]
    rows = M.compute_rows(texts, ["7"] * 3, ["P"] * 3, "M_AGREE", uids=["u"] * 3)
    # row0: mon=1−2|1.0−1.0|=1.0, ctl=+1.0(다수·verify) → meta_val=0.5·1.0+0.5·1.0=1.0
    assert rows[0]["meta_val"] == pytest.approx(1.0)
    # row1: mon=1−2|0.0−1.0|=−1.0, ctl=+1.0 → meta_val=0.5·(−1.0)+0.5·1.0=0.0
    assert rows[1]["meta_val"] == pytest.approx(0.0)


# ── RAND: 통제의 ±1 부분만 uid 그룹 단위로 부호 반전(모니터링은 불변) ────────────────
def test_agree_rand_flips_control_sign_per_group_not_per_row(monkeypatch):
    monkeypatch.setenv("MATH_AGREE_MON_W", "0.0")
    monkeypatch.setenv("MATH_AGREE_CTL_W", "1.0")
    A = _arow("7", "redirect", "42")   # 전부 first="7"(동치) → 다수 → base=−1(재시도)
    B = _arow("7", "verify")           # 다수 → base=+1(verify)
    texts = ([A, A, B, B] * 4)
    uids = sum(([g] * 4 for g in "abcd"), [])
    base_rows = M.compute_rows(texts, ["42"] * 16, ["P"] * 16, "M_AGREE", uids=uids,
                               tok_len_fn=lambda s: 0)
    for i in range(0, 16, 4):
        assert base_rows[i]["meta_val"] == pytest.approx(-1.0)
        assert base_rows[i + 1]["meta_val"] == pytest.approx(-1.0)
        assert base_rows[i + 2]["meta_val"] == pytest.approx(1.0)
        assert base_rows[i + 3]["meta_val"] == pytest.approx(1.0)
    found_flip = False
    for seed in range(20):
        rows = M.compute_rows(texts, ["42"] * 16, ["P"] * 16, "M_AGREE_RAND", uids=uids,
                              rng=random.Random(seed), tok_len_fn=lambda s: 0)
        for i in range(0, 16, 4):
            assert rows[i]["meta_val"] == rows[i + 1]["meta_val"]              # 같은 그룹=같은 부호
            assert rows[i + 2]["meta_val"] == rows[i + 3]["meta_val"]
            assert rows[i]["meta_val"] == pytest.approx(-rows[i + 2]["meta_val"])   # A/B 는 항상 반대
            assert abs(rows[i]["meta_val"]) == pytest.approx(1.0)
        found_flip |= any(rows[i]["meta_val"] == pytest.approx(1.0) for i in range(0, 16, 4))
        assert [r["answer_total"] for r in rows] == [r["answer_total"] for r in base_rows]
        assert [r["meta_defined"] for r in rows] == [r["meta_defined"] for r in base_rows]
    assert found_flip
    a = M.compute_rows(texts, ["42"] * 16, ["P"] * 16, "M_AGREE_RAND", uids=uids, rng=random.Random(3))
    b = M.compute_rows(texts, ["42"] * 16, ["P"] * 16, "M_AGREE_RAND", uids=uids, rng=random.Random(3))
    assert [r["meta_val"] for r in a] == [r["meta_val"] for r in b]


def test_agree_rand_flips_base_but_not_length_cost(monkeypatch):
    monkeypatch.setenv("MATH_AGREE_MON_W", "0.0")
    monkeypatch.setenv("MATH_AGREE_CTL_W", "1.0")
    monkeypatch.setenv("MATH_RETRY_LEN_COST", "0.2")
    verify_row = _arow("7", "verify")
    redirect_row = _arow("7", "redirect", "9")   # 다수·재시도 2k 토큰 → −1 − 0.4 = −1.4
    texts = [verify_row, redirect_row, verify_row]      # 그룹 크기 3(형제 조건 충족), 전부 first="7"
    base = M.compute_rows(texts, ["42"] * 3, ["P"] * 3, "M_AGREE", uids=["u"] * 3,
                          tok_len_fn=lambda s: 2000)
    assert [r["meta_val"] for r in base] == pytest.approx([1.0, -1.4, 1.0])
    seen = set()
    for seed in range(20):
        rows = M.compute_rows(texts, ["42"] * 3, ["P"] * 3, "M_AGREE_RAND", uids=["u"] * 3,
                              rng=random.Random(seed), tok_len_fn=lambda s: 2000)
        vals = tuple(round(r["meta_val"], 6) for r in rows)
        assert vals in ((1.0, -1.4, 1.0), (-1.0, 0.6, -1.0)), vals   # 길이 비용은 반전 밖: +1.4 는 없다
        seen.add(vals)
    assert len(seen) == 2


# ── forced 행: 판단(모니터링+통제) 미정의, 답 스팬은 정상 ─────────────────────────
def test_agree_forced_row_judgment_undefined_but_answer_total_normal(monkeypatch):
    monkeypatch.setenv("MATH_AGREE_MON_W", "0.5")
    monkeypatch.setenv("MATH_AGREE_CTL_W", "0.5")
    texts = [_arow("7", "redirect", "42", agree="0.5")] * 3   # 그룹 크기 3, 전부 first="7"
    rows_free = M.compute_rows(texts, ["42"] * 3, ["P"] * 3, "M_AGREE", uids=["u"] * 3)
    assert rows_free[0]["meta_defined"] == 1 and rows_free[0]["meta_val"] != 0.0

    rows_forced = M.compute_rows(texts, ["42"] * 3, ["P"] * 3, "M_AGREE", uids=["u"] * 3,
                                 forced_redirect=[1, 0, 0])
    assert rows_forced[0]["meta_defined"] == 0
    assert rows_forced[0]["meta_val"] == 0.0
    assert rows_forced[0]["judge"] == 0.0
    assert rows_forced[0]["answer_total"] == rows_free[0]["answer_total"] == 1.0   # gold=42, 재시도로 구제


def test_agree_forced_rows_also_undefined_in_rand_arm():
    texts = [_arow("7", "redirect", "42", agree="0.5")] * 3
    rows = M.compute_rows(texts, ["42"] * 3, ["P"] * 3, "M_AGREE_RAND", uids=["u"] * 3,
                          forced_redirect=[1, 0, 0])
    assert rows[0]["meta_defined"] == 0 and rows[0]["meta_val"] == 0.0
    assert rows[0]["answer_total"] == 1.0


# ── 텔레메트리 ─────────────────────────────────────────────────────────────────
def test_agree_telemetry_keys_and_tel_line():
    texts = [_arow("7", "redirect", "42", agree="0.9"),
             _arow("7", "verify", agree="0.1"),
             _arow("7", "verify", agree="0.5")]
    rows = M.compute_rows(texts, ["42"] * 3, ["P"] * 3, "M_AGREE", uids=["u"] * 3)
    rep = M.telemetry(rows, arm="M_AGREE", step=7)
    # 재시도 키(같은 구조)와 agree 전용 키가 모두 있어야 한다(스펙 "appended to retry keys").
    for k in ("first_acc", "final_acc", "redirect_rate", "judgment_acc", "second_attempt_rate",
              "trunc_rate", "agree_pred_mean", "agree_true_mean", "agree_mae", "agree_auc_wrong",
              "minority_rate", "redirect_rate_given_minority", "redirect_rate_given_majority",
              "agree_line_rate"):
        assert k in rep, k
    assert rep["agree_line_rate"] == 1.0   # 세 행 모두 agreement: 줄을 냈다
    line = M.format_tel(rep)
    for frag in ("first_acc=", "agree_pred=", "agree_true=", "agree_mae=", "agree_auc_wrong=",
                "minority_rate=", "redirect|minority=", "redirect|majority=", "agree_line_rate="):
        assert frag in line, line
    # 다른 팔(M_RETRY)의 TEL 줄엔 agree 키가 없다(기존 계약 불변)
    rows_r = M.compute_rows([_arow("7", "verify")], ["7"], ["P"], "M_RETRY", uids=["u"])
    rep_r = M.telemetry(rows_r, arm="M_RETRY", step=1)
    assert "agree_pred_mean" not in rep_r and "agree_pred=" not in M.format_tel(rep_r)


def test_agree_line_rate_reflects_missing_lines():
    texts = [_arow("7", "verify", agree="0.5"), _arow("7", "verify")]   # 둘째는 agreement: 줄 없음
    rows = M.compute_rows(texts, ["7"] * 2, ["P"] * 2, "M_AGREE", uids=["u"] * 2)
    rep = M.telemetry(rows, arm="M_AGREE", step=1)
    assert rep["agree_line_rate"] == 0.5


# ── 중단 규칙 ──────────────────────────────────────────────────────────────────
def _rep(**kw):
    base = {"emit_rate": 0.5, "boxed_in_meta": 0.0, "boilerplate_rate": 0.01, "n_emitted": 100,
            "multi_block_rate": 0.0, "acc": 0.6, "redirect_rate": 0.3, "trunc_rate": 0.0,
            "agree_line_rate": 0.9, "step": 10}
    base.update(kw)
    return base


def _aborts(rep, arm):
    return {h["metric"] for h in M.check_abort(rep, arm=arm) if h["status"] == "abort"}


def test_agree_line_rate_abort_rule():
    assert M.check_abort(_rep(), arm="M_AGREE") == []
    assert _aborts(_rep(agree_line_rate=0.5), "M_AGREE") == {"agree_line_rate"}
    assert _aborts(_rep(agree_line_rate=0.5), "M_AGREE_RAND") == {"agree_line_rate"}
    assert _aborts(_rep(agree_line_rate=0.5, step=5), "M_AGREE") == set()   # 워밍업 step≤5
    assert _aborts(_rep(agree_line_rate=0.5, step=6), "M_AGREE") == {"agree_line_rate"}
    # 다른 팔엔 agree_line_rate 규칙이 없다(팔 전용)
    hits = M.check_abort(_rep(agree_line_rate=0.0), arm="M_RETRY")
    assert not [h for h in hits if h["metric"] == "agree_line_rate"]


def test_agree_shares_retry_like_warmup_rules(monkeypatch):
    monkeypatch.delenv("MATH_ACC_FLOOR", raising=False)
    # emit_rate/redirect_rate: step≤5 워밍업(M_AGREE 도 재시도 구조라 공유)
    assert _aborts(_rep(emit_rate=0.1, redirect_rate=0.01, step=5), "M_AGREE") == set()
    assert _aborts(_rep(emit_rate=0.1, redirect_rate=0.01, step=6), "M_AGREE") == \
        {"emit_rate", "redirect_rate"}
    # trunc_rate: step≤3 워밍업
    assert _aborts(_rep(trunc_rate=0.9, step=3), "M_AGREE") == set()
    assert _aborts(_rep(trunc_rate=0.9, step=4), "M_AGREE") == {"trunc_rate"}


# ── 프롬프트 변형 ──────────────────────────────────────────────────────────────
def test_prompt_variant_math_agree():
    from src.metacot import math_meta_prompt as P
    s = P.MATH_PROMPT_VARIANTS["math_agree"]
    assert s == P.SOLVE_MATH_AGREE
    for frag in ("first answer in \\boxed", "EXACTLY ONE", "<meta>", "confidence:",
                 "agreement:", "decision: verify", "decision: redirect", "Second attempt:",
                 "GENUINELY DIFFERENT", "LAST \\boxed"):
        assert frag in s, frag
    assert "You MAY" not in s and "At least once while solving" not in s
    # math_retry 와 머리·꼬리 문장이 글자 그대로 같다(agreement: 줄 하나만 는다).
    assert s.startswith(P._MATH_RETRY_RULES)
    assert s.replace(
        "agreement: <a number between 0 and 1: the fraction of independent attempts at this "
        "problem that you expect to reach the same final answer as yours>\n", "") == P.SOLVE_MATH_RETRY
    forced = P.MATH_PROMPT_VARIANTS["math_agree_forced"]
    assert "decision: redirect` — write it regardless" in forced and "agreement:" in forced
    assert M.MATH_ARM_SPECS["M_AGREE"]["variant"] == "math_agree" and M.MATH_ARM_SPECS["M_AGREE"]["require_meta"]
    assert M.MATH_ARM_SPECS["M_AGREE_RAND"]["variant"] == "math_agree"


# ── held-out 평가(모의 생성기) ─────────────────────────────────────────────────
def test_math_agree_eval_summary_on_mock_generations():
    import math_retry_eval as E
    problems = [{"problem": "p0", "gold": "42"}, {"problem": "p1", "gold": "42"}]

    def gen(prompts):
        assert len(prompts) == 2
        assert prompts[0][0]["content"] == E.MATH_PROMPT_VARIANTS["math_agree"]
        return [
            [(_arow("7", "redirect", "42", agree="0.9"), 0, 300),   # 구제, 소수 예측 높음
             (_arow("42", "verify", agree="0.2"), 0, 300),
             (_arow("42", "verify", agree="0.8"), 0, 300),
             (_arow("7", "redirect", "9", agree="0.9"), 0, 300)],   # 탈선(정답 아님)
            [(_arow("42", "verify", agree="0.9"), 0, 300)] * 4,     # 전원 만장일치 정답
        ]
    rows, tel = E.evaluate(problems, gen, num_samples=4, variant="math_agree")
    assert len(rows) == 8 and tel["n_groups"] == 2
    for k in ("agree_pred_mean", "agree_true_mean", "agree_mae", "agree_auc_wrong",
             "agree_auc_wrong_mixed", "minority_rate", "redirect_rate_given_minority",
             "redirect_rate_given_majority", "agree_line_rate", "agree_calibration_bins"):
        assert k in tel, k
    assert len(tel["agree_calibration_bins"]) == 5
    assert 0.0 <= tel["agree_pred_mean"] <= 1.0
    assert tel["agree_line_rate"] == 1.0     # 모든 모의 행이 agreement: 줄을 낸다
    s = E.format_summary(tel)
    for k in ("agree_pred_mean", "agree_mae", "agree_auc_wrong_mixed", "calibration"):
        assert k in s, s
    with pytest.raises(ValueError, match="math_agree"):
        E.evaluate(problems, gen, num_samples=4, variant="not_a_variant")


# ── build_math_parquet.py: --variant math_agree 기본 forced_variant ─────────
def _load_build_module():
    import build_math_parquet as B
    return B


def test_resolve_forced_variant_defaults_and_override():
    B = _load_build_module()
    assert B.resolve_forced_variant(None, "math_retry") == "math_retry_forced"
    assert B.resolve_forced_variant(None, "math_agree") == "math_agree_forced"
    assert B.resolve_forced_variant(None, "math_opt") is None
    assert B.resolve_forced_variant("custom_forced", "math_agree") == "custom_forced"


def test_build_math_parquet_agree_variant_forced_marking():
    B = _load_build_module()
    rows = [{"problem": f"p{i}", "solution": f"\\boxed{{{i}}}", "level": "Level 5", "type": "algebra"}
            for i in range(20)]
    train, val, stats = B.split_records(rows, set(), val_n=4, seed=11, variant="math_agree")
    assert len(train) == 16 and len(val) == 4
    forced_variant = B.resolve_forced_variant(None, "math_agree")
    assert forced_variant == "math_agree_forced"
    train_f, n_forced = B.apply_forced_redirect(train, forced_frac=0.25, forced_variant=forced_variant,
                                                seed=11)
    assert n_forced == round(0.25 * len(train))
    n_marked = sum(1 for r in train_f if r["extra_info"]["forced_redirect"] == 1)
    assert n_marked == n_forced
    for r in train_f:
        if r["extra_info"]["forced_redirect"] == 1:
            assert r["extra_info"]["prompt_variant"] == "math_agree_forced"
        else:
            assert r["extra_info"]["prompt_variant"] == "math_agree"
    assert all(r["extra_info"]["forced_redirect"] == 0 for r in val)


# ── verl_sdc 배선: M_AGREE 도 M_RETRY 와 같은 재료(truncated/tok_len_fn/forced/group_pass_rate) ──
class _Tok:
    def encode(self, s, add_special_tokens=False):
        return [0] * len(s)


def _fake_trainer_data(texts, uids, arm, *, plen=4, resp_w=None, tokenizer=None, extra_info=None):
    import torch
    from types import SimpleNamespace as NS
    L = resp_w or (max(len(t) for t in texts) + 1)
    B = len(texts)
    am = torch.zeros(B, plen + L, dtype=torch.long)
    am[:, :plen] = 1
    for i, t in enumerate(texts):
        am[i, plen:plen + min(len(t), L)] = 1
    nt = {"problem": ["P"] * B, "gold": ["7"] * B, "uid": list(uids)}
    if extra_info is not None:
        nt["extra_info"] = extra_info
    data = NS(batch={"prompts": torch.zeros(B, plen, dtype=torch.long), "attention_mask": am,
                     "responses": torch.zeros(B, L, dtype=torch.long)},
              non_tensor_batch=nt)
    self = NS(config=NS(algorithm=NS(math_arm=arm), data=NS(max_prompt_length=plen)), tokenizer=tokenizer)
    return self, data


def test_stash_wires_agree_arm_with_trunc_and_token_counter(capsys):
    from src.training import verl_sdc as V
    texts = [_arow("7", "verify", agree="0.5")] * 3   # 그룹 크기 3, 전부 first="7"(gold="7")
    self, data = _fake_trainer_data(texts, ["u"] * 3, "M_AGREE", tokenizer=_Tok())
    totals = V._compute_math_arm_stash(self, data, texts, 3, 4, 1)
    assert totals == [1.0, 1.0, 1.0]
    rows = V._MATH_STASH["rows"]
    assert all(r["agree_true"] == pytest.approx(1.0) for r in rows)
    out = capsys.readouterr().out
    assert "retry_w=" in out and "first_acc=" in out   # M_AGREE 도 재시도 구조 로그를 공유


# ── 런처 dry-run ───────────────────────────────────────────────────────────────
def test_run_math_arm_agree_defaults_resp_len_6144_and_uses_retry_eval():
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items()
          if k not in ("RESP_LEN", "MATH_AGREE_MON_W", "MATH_AGREE_CTL_W", "MATH_RETRY_LEN_COST")}
    r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_AGREE", "1", "50", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    for want in ("RESP_LEN=6144", "data.max_response_length=6144", "LINEAGE=cd9_M_AGREE_s1_r6144",
                 "math_train_math_agree.parquet", "MATH_AGREE_MON_W=0.5", "MATH_AGREE_CTL_W=0.5",
                 "MATH_RETRY_LEN_COST=0.2", "EVAL_SCRIPT=math_retry_eval.py"):
        assert want in out, (want, out)
    r2 = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_AGREE_RAND", "2", "--dry-run"],
                        cwd=REPO, capture_output=True, text=True, env={**env, "RESP_LEN": "4096"})
    assert r2.returncode == 0 and "RESP_LEN=4096" in r2.stdout and "EVAL_SCRIPT=math_retry_eval.py" in r2.stdout


def test_ray_env_forwarding_has_agree_names():
    src = (REPO / "src/training/verl_sdc.py").read_text()
    for k in ("MATH_AGREE_MON_W", "MATH_AGREE_CTL_W"):
        assert f'"{k}"' in src, k
    sh = (REPO / "scripts/local/run_math_arm.sh").read_text()
    assert "MATH_AGREE_MON_W" in sh and "MATH_AGREE_CTL_W" in sh
