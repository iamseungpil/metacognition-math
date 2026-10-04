"""cd9 사전등록 수정 3 — M_RETRY(재시도 판단) 회귀 시험.

split_attempts 분해 / 판단 진리표(길이 비용 포함) / RAND 그룹 부호 반전 / 텔레메트리 키·TEL 줄 /
중단 규칙(redirect_rate 워밍업, trunc_rate, 팔 전용) / Stage-1 라벨 빌더 / held-out 평가(모의 생성기) /
프롬프트 변형 / verl_sdc 스태시 배선(truncated·tok_len_fn·Ray env 전달).
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from src.training import math_meta as M  # noqa: E402


def _meta(decision=None, conf="0.4", body="The substitution step might be off."):
    d = f"decision: {decision}\n" if decision else ""
    return f"<meta>\nconfidence: {conf}\n{body}\n{d}</meta>"


def _row(first, decision, second=None, *, marker=True, meta_before=False):
    """first: 첫 \\boxed 내용, second: 두 번째 \\boxed 내용(None = 없음)."""
    t = ("" if not meta_before else _meta("verify") + "\n") + f"work \\boxed{{{first}}}\n"
    if decision is not False:
        t += _meta(decision) + "\n"
    if second is not None:
        t += ("Second attempt: " if marker else "") + f"other method \\boxed{{{second}}}"
    return t


# ── split_attempts ─────────────────────────────────────────────────────────────
def test_first_boxed_balanced():
    assert M.first_boxed("a \\boxed{\\frac{1}{2}} b \\boxed{3}") == "\\frac{1}{2}"
    assert M.first_boxed("nested \\boxed{\\{a, b\\}} end") == "\\{a, b\\}"
    assert M.first_boxed("unclosed \\boxed{42") == ""
    assert M.first_boxed("nothing") == ""


def test_split_no_meta():
    sp = M.split_attempts("work \\boxed{7}")
    assert sp["first_answer"] == "7" and sp["final_answer"] == "7"
    assert sp["meta"]["emitted"] == 0 and sp["has_second_attempt"] == 0 and sp["n_blocks"] == 0


def test_split_meta_then_verify():
    t = _row("7", "verify")
    sp = M.split_attempts(t)
    assert sp["meta"]["emitted"] == 1 and sp["meta"]["decision"] == "verify"
    assert sp["has_second_attempt"] == 0 and sp["final_answer"] == "7"
    s, e = sp["meta"]["start"], sp["meta"]["end"]
    assert t[s:e].startswith("<meta>") and t[s:e].endswith("</meta>")   # 오프셋은 전체 텍스트 기준


def test_split_redirect_with_second_boxed():
    sp = M.split_attempts(_row("7", "redirect", "42"))
    assert sp["first_answer"] == "7" and sp["final_answer"] == "42"
    assert sp["meta"]["decision"] == "redirect" and sp["has_second_attempt"] == 1


def test_split_redirect_copied_answer_and_marker_only():
    sp = M.split_attempts(_row("7", "redirect", "7"))
    assert sp["first_answer"] == sp["final_answer"] == "7" and sp["has_second_attempt"] == 1
    # ★0914 수리: 표식만 있고 두 번째 \boxed 가 없으면(잘림 등) 미완 시도 — 이제는 0 으로 센다
    #   (종전엔 «시도했다»로 1 을 줬으나, marker AND marker 뒤 \boxed 를 둘 다 요구하도록 조였다).
    #   최종 답은 여전히 첫 답으로 떨어진다.
    t = _row("7", "redirect") + "Second attempt: let me try again but never finish"
    sp = M.split_attempts(t)
    assert sp["has_second_attempt"] == 0 and sp["final_answer"] == "7"
    # 마커도 boxed 도 없으면 시도 없음
    assert M.split_attempts(_row("7", "redirect"))["has_second_attempt"] == 0


def test_split_verify_restated_boxed_is_not_second_attempt():
    # ★0914 수리 핵심 사례: verify 뒤에 답을 그냥 재진술한 \boxed 가 있어도(표식 없음) 두 번째
    #   시도가 아니다 — smoke 텔레메트리 second_attempt=.80 / redirect=.002 모순의 원인이었다.
    t = _row("7", "verify") + "so the answer is \\boxed{7}"
    sp = M.split_attempts(t)
    assert sp["has_second_attempt"] == 0
    assert sp["final_answer"] == "7"        # final_answer 는 자격 있는 마지막 boxed 그대로


def test_split_marker_case_insensitive_and_colon_optional():
    t = _row("7", "redirect") + "SECOND ATTEMPT let me redo \\boxed{9}"
    sp = M.split_attempts(t)
    assert sp["has_second_attempt"] == 1 and sp["final_answer"] == "9"


def test_split_redirect_boxed_without_marker_is_not_second_attempt():
    # 표식 없이 블록 뒤에 \boxed 만 있으면(구버전 규칙이 1 로 셌던 경우) 이제는 0
    t = _row("7", "redirect") + "other method \\boxed{42}"
    sp = M.split_attempts(t)
    assert sp["has_second_attempt"] == 0
    assert sp["final_answer"] == "42"       # final_answer 는 자격(scan) 판정과 무관하게 마지막 boxed


def test_split_marker_present_no_boxed_after_is_not_second_attempt():
    t = _row("7", "redirect") + "Second attempt: I will retry but stop here"
    sp = M.split_attempts(t)
    assert sp["has_second_attempt"] == 0
    assert sp["final_answer"] == "7"


def test_split_boxed_inside_meta_flagged():
    t = "work \\boxed{7}\n" + _meta("verify", body="I think \\boxed{7} is right.")
    sp = M.split_attempts(t)
    assert sp["boxed_in_meta"] == 1
    # 메타 안 \boxed 는 «블록 뒤 \boxed» 가 아니므로 두 번째 시도로 세지 않는다
    assert sp["has_second_attempt"] == 0


def test_split_two_blocks_is_multi_block_and_meta_before_first_boxed_not_counted():
    two = _row("7", "redirect", "42") + "\n" + _meta("verify")
    sp = M.split_attempts(two)
    assert sp["n_blocks"] == 2
    rows = M.compute_rows([two], ["42"], ["P"], "M_RETRY", uids=["u"])
    assert rows[0]["multi_block"] == 1 and rows[0]["meta_defined"] == 0 and rows[0]["meta_val"] == 0.0
    # 첫 \boxed 앞의 블록은 «답을 판단한 것»이 아니다 — 발화로 치지 않는다(n_blocks 는 1)
    sp = M.split_attempts(_row("7", False, meta_before=True))
    assert sp["n_blocks"] == 1 and sp["meta"]["emitted"] == 0


def test_boxed_scanner_is_space_tolerant_everywhere():
    # 첫/마지막 박스 스캐너와 has_second_attempt/boxed_in_meta 가 같은 _BOXED_RE(\\boxed\\s*\\{)를 쓴다
    assert M.first_boxed("work \\boxed {7} end") == "7"
    assert M.last_boxed("a \\boxed{1} b \\boxed {\\frac{1}{2}}") == "\\frac{1}{2}"
    sp = M.split_attempts("work \\boxed {7}\n" + _meta("verify"))
    assert sp["first_answer"] == "7" and sp["final_answer"] == "7" and sp["meta"]["decision"] == "verify"


def test_boxed_inside_meta_excluded_from_first_and_final_answers():
    # 메타가 첫 \boxed 앞에 있고 그 안에만 \boxed 가 있다 → 첫 답 없음, 누출 플래그 1, 발화 0
    t = _meta("verify", body="I think \\boxed{42} is right.") + "\nno real answer"
    sp = M.split_attempts(t)
    assert sp["first_answer"] is None and sp["final_answer"] is None
    assert sp["boxed_in_meta"] == 1 and sp["meta"]["emitted"] == 0
    r = M.parse_retry_row(t, "42", "P")
    assert r["first_correct"] == 0 and r["final_correct"] == 0 and r["boxed_in_meta"] == 1 and r["emitted"] == 0
    # 메타(안에 \boxed) 뒤에 진짜 첫 답 → 그 답이 첫 답이고 메타는 «답 앞» 이라 발화가 아니다
    t2 = _meta("verify", body="maybe \\boxed{42}") + "\nwork \\boxed{7}"
    sp2 = M.split_attempts(t2)
    assert sp2["first_answer"] == "7" and sp2["final_answer"] == "7" and sp2["boxed_in_meta"] == 1
    assert sp2["meta"]["emitted"] == 0
    # 첫 답 → 메타(안에 \boxed{9}) → 재시도 \boxed{42}: 메타 안 박스는 최종 답 후보가 아니다
    t3 = "work \\boxed{7}\n" + _meta("redirect", body="try \\boxed{9}") + "\nSecond attempt: \\boxed{42}"
    sp3 = M.split_attempts(t3)
    assert sp3["first_answer"] == "7" and sp3["final_answer"] == "42"
    assert sp3["boxed_in_meta"] == 1 and sp3["has_second_attempt"] == 1
    # 첫 답 → 메타(안에 \boxed{9}) 로 끝: 최종 답은 첫 답(메타 안 박스 제외)
    t4 = "work \\boxed{7}\n" + _meta("verify", body="so \\boxed{9}")
    assert M.split_attempts(t4)["final_answer"] == "7"


# ── 판단 진리표 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("first_ok,decision,second,final,first,final_ok,extra,want,defined", [
    # ★수정 3b: 오답·재시도·변경(수학 동치 기준) → +0.5 + 0.5·최종정답
    (0, "redirect", 1, "42", "7", 1, 0, 1.0, 1),      # 오답·재시도·다른 답·최종 정답 → +1.0
    (0, "redirect", 1, "9", "7", 0, 0, 0.5, 1),       # 오답·재시도·다른 오답 → +0.5 (판단 크레딧만)
    (0, "redirect", 1, "7", "7", 0, 0, 0.0, 1),       # 오답·재시도·베낌 → 0
    (0, "redirect", 1, "7.0", "7", 0, 0, 0.0, 1),     # 오답·재시도·베낌(7 ≡ 7.0, 수학 동치) → 0
    (0, "redirect", 1, "\\frac{1}{2}", "0.5", 0, 0, 0.0, 1),   # 베낌(1/2 ≡ 0.5) → 0
    (0, "redirect", 0, "7", "7", 0, 0, 0.0, 1),       # 오답·선언만 → 0
    (0, "verify", 0, "7", "7", 0, 0, -1.0, 1),        # 오답·verify → −1
    (1, "verify", 0, "7", "7", 1, 0, 1.0, 1),         # 정답·verify → +1
    (1, "redirect", 1, "7", "7", 1, 0, -1.0, 1),      # 정답·재시도(추가 0 토큰) → −1
    (1, "redirect", 1, "9", "7", 0, 2000, -1.4, 1),   # 정답·재시도 2k 토큰 → −1 − 0.2·2
    (1, "redirect", 0, "7", "7", 1, 2000, -1.0, 1),   # 정답·선언만 → 길이 비용 없음
    (0, None, 0, "7", "7", 0, 0, 0.0, 0),             # decision 없음 → 미정의
])
def test_retry_judgment_truth_table(first_ok, decision, second, final, first, final_ok, extra, want, defined):
    got, d = M.retry_judgment_term(first_ok, decision, second, final, first, final_correct=final_ok,
                                   extra_tokens=extra, len_cost=0.2)
    assert d == defined and got == pytest.approx(want)


def test_copied_uses_math_equivalence_with_string_fallback(monkeypatch):
    # 동치 판정기가 죽어도 문자열 비교로 폴백한다(조용히 «변경»으로 읽히면 안 된다)
    def boom(*a, **k):
        raise RuntimeError("math_verify down")
    monkeypatch.setattr(M, "grade_math", boom)
    assert M.answers_equivalent("7", "7") is True
    assert M.answers_equivalent("7", "7.0") is False          # 폴백 = 문자열
    got, _ = M.retry_judgment_term(0, "redirect", 1, "9", "7", final_correct=0, extra_tokens=0, len_cost=0.2)
    assert got == pytest.approx(0.5)


def test_len_cost_env(monkeypatch):
    monkeypatch.setenv("MATH_RETRY_LEN_COST", "0.5")
    got, _ = M.retry_judgment_term(1, "redirect", 1, "9", "7", final_correct=0, extra_tokens=1000)
    assert got == pytest.approx(-1.5)
    monkeypatch.delenv("MATH_RETRY_LEN_COST")
    assert M.retry_len_cost() == 0.2 and M.retry_weight() == 0.5


def test_compute_rows_retry_answer_span_is_final_only(monkeypatch):
    monkeypatch.setenv("MATH_RETRY_W", "0.5")
    monkeypatch.delenv("MATH_RETRY_LEN_COST", raising=False)
    texts = [_row("7", "redirect", "42"),      # 오답→구제: 답 스팬 1, 메타 +0.5
             _row("42", "verify"),             # 정답·verify: 답 1, 메타 +0.5
             _row("7", "verify"),              # 오답·verify: 답 0, 메타 −0.5
             _row("42", "redirect", "7"),      # 정답→탈선: 답 0, 메타 −0.5 − cost
             _row("7", None),                  # decision 없음: 미정의
             "no meta at all \\boxed{42}",     # 미발화·정답: 답 1, 미정의
             _row("7", "redirect", "9"),       # ★3b 오답→다른 오답: 답 0, 메타 +0.5·W = 0.25
             _row("7", "redirect", "7.0")]     # ★수학 동치 베낌: 답 0, 메타 0
    rows = M.compute_rows(texts, ["42"] * 8, ["P"] * 8, "M_RETRY", uids=["u"] * 8)
    assert [r["first_correct"] for r in rows] == [0, 1, 0, 1, 0, 1, 0, 0]
    assert [r["final_correct"] for r in rows] == [1, 1, 0, 0, 0, 1, 0, 0]
    assert [r["answer_total"] for r in rows] == [1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    assert [r["meta_defined"] for r in rows] == [1, 1, 1, 1, 0, 0, 1, 1]
    assert rows[0]["meta_val"] == 0.5 and rows[1]["meta_val"] == 0.5 and rows[2]["meta_val"] == -0.5
    assert rows[3]["meta_val"] < -0.5 and rows[3]["redirected"] == 1
    assert rows[4]["meta_val"] == 0.0 and rows[5]["meta_val"] == 0.0
    assert rows[6]["meta_val"] == pytest.approx(0.25) and rows[6]["answer_changed"] == 1
    assert rows[7]["meta_val"] == 0.0 and rows[7]["answer_changed"] == 0
    assert rows[0]["n_tok_after_meta"] > 0 and rows[1]["n_tok_after_meta"] == 0
    # tok_len_fn 주입 시 그것을 쓴다
    rows2 = M.compute_rows(texts[:1], ["42"], ["P"], "M_RETRY", uids=["u"], tok_len_fn=lambda s: 4000)
    assert rows2[0]["n_tok_after_meta"] == 4000
    s, e = M.meta_char_spans(rows[0])[0]
    assert texts[0][s:e].startswith("<meta>")


# ── RAND: 그룹 단위 부호 반전 ─────────────────────────────────────────────────────
def test_retry_rand_flips_sign_per_group_not_per_row(monkeypatch):
    monkeypatch.setenv("MATH_RETRY_W", "1.0")
    texts = [_row("7", "redirect", "42"), _row("42", "verify")] * 4     # 두 행 모두 +1
    uids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    base = M.compute_rows(texts, ["42"] * 8, ["P"] * 8, "M_RETRY", uids=uids)
    assert all(r["meta_val"] == 1.0 for r in base)
    found_neg = False
    for seed in range(20):
        rows = M.compute_rows(texts, ["42"] * 8, ["P"] * 8, "M_RETRY_RAND", uids=uids, rng=random.Random(seed))
        for i in range(0, 8, 2):
            assert rows[i]["meta_val"] == rows[i + 1]["meta_val"]            # 같은 그룹 = 같은 부호
            assert abs(rows[i]["meta_val"]) == 1.0
        found_neg |= any(r["meta_val"] < 0 for r in rows)
        assert [r["answer_total"] for r in rows] == [r["answer_total"] for r in base]   # 답 스팬 불변
        assert [r["meta_defined"] for r in rows] == [r["meta_defined"] for r in base]
    assert found_neg
    # 같은 rng 시드면 재현
    a = M.compute_rows(texts, ["42"] * 8, ["P"] * 8, "M_RETRY_RAND", uids=uids, rng=random.Random(3))
    b = M.compute_rows(texts, ["42"] * 8, ["P"] * 8, "M_RETRY_RAND", uids=uids, rng=random.Random(3))
    assert [r["meta_val"] for r in a] == [r["meta_val"] for r in b]


def test_retry_rand_flips_judgment_sign_but_not_length_cost(monkeypatch):
    monkeypatch.setenv("MATH_RETRY_W", "1.0")
    monkeypatch.setenv("MATH_RETRY_LEN_COST", "0.2")
    texts = [_row("42", "verify"), _row("42", "redirect", "7")]       # +1 / −1 − 0.2·2 = −1.4
    base = M.compute_rows(texts, ["42"] * 2, ["P"] * 2, "M_RETRY", uids=["u"] * 2, tok_len_fn=lambda s: 2000)
    assert [r["meta_val"] for r in base] == pytest.approx([1.0, -1.4])
    seen = set()
    for seed in range(20):
        rows = M.compute_rows(texts, ["42"] * 2, ["P"] * 2, "M_RETRY_RAND", uids=["u"] * 2,
                              rng=random.Random(seed), tok_len_fn=lambda s: 2000)
        vals = tuple(round(r["meta_val"], 6) for r in rows)
        assert vals in ((1.0, -1.4), (-1.0, 0.6)), vals      # 길이 비용은 부호 반전 밖: +1.4 는 절대 없다
        seen.add(vals)
    assert len(seen) == 2


# ── 텔레메트리 ─────────────────────────────────────────────────────────────────
RETRY_TEL_KEYS = ("first_acc", "final_acc", "redirect_rate", "redirect_rate_given_wrong",
                  "redirect_rate_given_right", "judgment_acc", "second_attempt_rate",
                  "second_attempt_rate_given_redirect", "trunc_rate")


def test_retry_telemetry_keys_and_tel_line():
    texts = [_row("7", "redirect", "42"), _row("42", "verify"), _row("7", "verify"), _row("42", "redirect", "7")]
    rows = M.compute_rows(texts, ["42"] * 4, ["P"] * 4, "M_RETRY", uids=["u"] * 4, truncated=[0, 0, 1, 0])
    rep = M.telemetry(rows, arm="M_RETRY", step=7)
    for k in RETRY_TEL_KEYS:
        assert k in rep, k
    assert rep["first_acc"] == 0.5 and rep["final_acc"] == 0.5
    assert rep["redirect_rate"] == 0.5 and rep["second_attempt_rate"] == 0.5
    assert rep["redirect_rate_given_wrong"] == 0.5 and rep["redirect_rate_given_right"] == 0.5
    assert rep["judgment_acc"] == 0.5 and rep["trunc_rate"] == 0.25
    assert rep["second_attempt_rate_given_redirect"] == 1.0   # 두 redirect 행 모두 표식+boxed 실행
    line = M.format_tel(rep)
    for k in ("first_acc=0.500", "final_acc=0.500", "redirect=0.500", "redirect|wrong=0.500",
              "redirect|right=0.500", "judgment_acc=0.500", "second_attempt=0.500", "trunc=0.250"):
        assert k in line, line
    # 다른 팔의 TEL 줄엔 재시도 키가 없다(기존 계약 불변)
    rows_j = M.compute_rows(["t " + _meta("verify") + " \\boxed{1}"], ["1"], ["P"], "M_JUDGE", labels={"P": "verify"})
    rep_j = M.telemetry(rows_j, arm="M_JUDGE", step=1)
    assert "first_acc" not in rep_j and "first_acc=" not in M.format_tel(rep_j)


def test_redirect_rate_denominator_is_decided_rows():
    texts = [_row("7", "redirect", "42"), _row("42", "verify"), _row("7", None), "plain \\boxed{42}"]
    rows = M.compute_rows(texts, ["42"] * 4, ["P"] * 4, "M_RETRY", uids=["u"] * 4)
    rep = M.telemetry(rows, arm="M_RETRY", step=1)
    assert rep["n_decided"] == 2
    assert rep["redirect_rate"] == 0.5                 # 결정된 행 중(1/2) — 중단 규칙이 읽는 값
    assert rep["redirect_rate_all_rows"] == 0.25       # 전체 행 중(1/4) — 참고용
    rows0 = M.compute_rows(["plain \\boxed{42}"], ["42"], ["P"], "M_RETRY", uids=["u"])
    assert math.isnan(M.telemetry(rows0, arm="M_RETRY", step=1)["redirect_rate"])   # 분모 0 → NaN(missing)


def test_retry_telemetry_conditional_nan_when_empty():
    rows = M.compute_rows([_row("42", "verify")], ["42"], ["P"], "M_RETRY", uids=["u"])
    rep = M.telemetry(rows, arm="M_RETRY", step=1)
    assert math.isnan(rep["redirect_rate_given_wrong"]) and rep["redirect_rate_given_right"] == 0.0


# ── 중단 규칙 ──────────────────────────────────────────────────────────────────
def _rep(**kw):
    base = {"emit_rate": 0.5, "boxed_in_meta": 0.0, "boilerplate_rate": 0.01, "n_emitted": 100,
            "multi_block_rate": 0.0, "acc": 0.6, "redirect_rate": 0.3, "trunc_rate": 0.0, "step": 10}
    base.update(kw)
    return base


def _aborts(rep, arm):
    return {h["metric"] for h in M.check_abort(rep, arm=arm) if h["status"] == "abort"}


def test_retry_abort_rules(monkeypatch):
    monkeypatch.delenv("MATH_ACC_FLOOR", raising=False)
    assert M.check_abort(_rep(), arm="M_RETRY") == []
    assert _aborts(_rep(redirect_rate=0.01), "M_RETRY") == {"redirect_rate"}
    assert _aborts(_rep(redirect_rate=0.01), "M_RETRY_RAND") == {"redirect_rate"}
    assert _aborts(_rep(redirect_rate=0.01, step=5), "M_RETRY") == set()          # 워밍업 step≤5
    assert _aborts(_rep(redirect_rate=0.01, step=6), "M_RETRY") == {"redirect_rate"}
    # ★0914 수리: 문턱 0.15→0.25 + min_step 3(재시도 팔만) — 두 번째 시도가 응답을 늘려 smoke
    #   step 1 정상 정책도 .131 을 찍었다.
    assert _aborts(_rep(trunc_rate=0.26), "M_RETRY") == {"trunc_rate"}
    assert _aborts(_rep(trunc_rate=0.25), "M_RETRY") == set()
    assert _aborts(_rep(trunc_rate=0.9, step=3), "M_RETRY") == set()             # 워밍업 step≤3
    assert _aborts(_rep(trunc_rate=0.9, step=4), "M_RETRY") == {"trunc_rate"}
    # 기존 규칙(발화율·다중 블록·메타 안 \boxed)은 그대로 적용된다 — 단 발화율은 재시도 팔만 워밍업(step≤5)
    assert _aborts(_rep(emit_rate=0.1), "M_RETRY") == {"emit_rate"}
    assert _aborts(_rep(emit_rate=0.1, step=5), "M_RETRY") == set()
    assert _aborts(_rep(emit_rate=0.1, step=5), "M_RETRY_RAND") == set()
    assert _aborts(_rep(emit_rate=0.1, step=6), "M_RETRY") == {"emit_rate"}
    assert _aborts(_rep(emit_rate=0.1, step=1), "M_JUDGE") == {"emit_rate"}      # 다른 메타 팔은 워밍업 없음
    assert _aborts(_rep(multi_block_rate=0.2, boxed_in_meta=0.05), "M_RETRY") == {"multi_block_rate", "boxed_in_meta"}
    # 재시도 팔 전용 — 다른 팔엔 abort 도 missing 도 없다
    other = {"redirect_rate": 0.0, "trunc_rate": 0.9, "step": 10}
    for arm in ("M_G0", "M_G1", "M_JUDGE", "M_RAND", "M_PROBE"):
        hits = M.check_abort({**_rep(), **other}, arm=arm)
        assert not [h for h in hits if h["metric"] in ("redirect_rate", "trunc_rate")], arm
    # 재시도 팔에서 키가 빠지면 missing(«못 봤다»)
    r = _rep(); del r["trunc_rate"]
    assert [h["status"] for h in M.check_abort(r, arm="M_RETRY") if h["metric"] == "trunc_rate"] == ["missing"]


# ── 프롬프트 변형 ──────────────────────────────────────────────────────────────
def test_prompt_variant_math_retry():
    from src.metacot import math_meta_prompt as P
    s = P.MATH_PROMPT_VARIANTS["math_retry"]
    assert s == P.SOLVE_MATH_RETRY
    for frag in ("first answer in \\boxed", "EXACTLY ONE", "<meta>", "confidence:", "decision: verify",
                 "decision: redirect", "Second attempt:", "GENUINELY DIFFERENT", "LAST \\boxed"):
        assert frag in s, frag
    assert "You MAY" not in s and "At least once while solving" not in s
    msgs = P.build_math_prompt("What is 1+1?", "math_retry")
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == s and msgs[1]["content"] == "What is 1+1?"
    # 기존 변형 불변
    assert P.SOLVE_MATH_OPT.replace(P._MATH_PERMISSION, P._MATH_MANDATE, 1) == P.SOLVE_MATH_NEW
    # ★0914/0914b: math_retry_forced(강제탐색)·math_agree/math_agree_forced(형제 동의 예측, M_AGREE)
    #   추가 — 이 집합에 새 변형이 더해진 것은 의도된 확장이다(tests/test_math_agree.py 가 그 둘을 잰다).
    # ★0914c: math_crit(M_CRIT, 비평 정보이득 팔) 추가 — tests/test_math_crit.py 가 그 변형을 잰다.
    # ★0914d: math_dis(M_DIS, 불일치 진단 팔) 추가 — 시스템 프롬프트는 math_opt 와 바이트 동일이고
    #   갈리는 것은 **사용자 턴**(후보 4개 + 진단 지시)뿐이다. tests/test_math_dis.py 가 잰다.
    # ★0914e: math_diff(M_DIFF, 난이도 판단 팔) 추가 — 시스템 프롬프트는 math_opt 와 바이트
    #   동일이고 갈리는 것은 **사용자 턴 접미**(난이도 한 줄 + 이유 한 문장)뿐이다. 이 변형은
    #   행마다 다른 재료가 없어 build_math_prompt 하나로 조립된다. tests/test_math_diff.py 가 잰다.
    assert set(P.MATH_PROMPT_VARIANTS) == {"math_plain", "math_opt", "math_new", "math_retry",
                                            "math_retry_forced", "math_agree", "math_agree_forced",
                                            "math_crit", "math_dis", "math_diff"}
    assert M.MATH_ARM_SPECS["M_RETRY"]["variant"] == "math_retry" and M.MATH_ARM_SPECS["M_RETRY"]["require_meta"]
    assert M.MATH_ARM_SPECS["M_RETRY_RAND"]["variant"] == "math_retry"


# ── Stage-1 라벨 빌더 ──────────────────────────────────────────────────────────
def test_build_retry_labels_synthetic_group(tmp_path):
    import build_retry_labels as B
    # 그룹 g0: 4 롤아웃 중 2 정답 → 오답 행의 p_retry = 2/3 >0 (redirect_right), 정답 행은 verify_right
    # 그룹 g1: 전부 오답 → neither
    rows = ([{"group_id": "g0", "problem_id": 0, "gold": "1", "r_corr": c, "truncated": 0, "n_tok": 100}
             for c in (1, 1, 0, 0)]
            + [{"group_id": "g1", "problem_id": 1, "gold": "2", "r_corr": 0, "truncated": 1, "n_tok": 300}
               for _ in range(2)])
    lab = B.label_rows(rows)
    assert [r["p_retry"] for r in lab[:4]] == pytest.approx([1 / 3, 1 / 3, 2 / 3, 2 / 3])
    assert [r["redirect_right"] for r in lab] == [0, 0, 1, 1, 0, 0]
    assert [r["verify_right"] for r in lab] == [1, 1, 0, 0, 0, 0]
    assert [r["neither"] for r in lab] == [0, 0, 0, 0, 1, 1]
    st = B.summarize(lab)
    assert st["n_rows"] == 6 and st["n_groups"] == 2
    assert st["redirect_right_rate"] == pytest.approx(2 / 6) and st["verify_right_rate"] == pytest.approx(2 / 6)
    assert st["neither_rate"] == pytest.approx(2 / 6) and st["p_retry_given_redirect_right"] == pytest.approx(2 / 3)
    assert st["trunc_rate"] == pytest.approx(2 / 6)
    # 파일 경로로 실행 + 모델 태그
    p = tmp_path / "aime25_qwen35_plain/texts.jsonl"
    p.parent.mkdir()
    p.write_text("\n".join(json.dumps(r) for r in rows))
    out = tmp_path / "labels.jsonl"
    summ = B.run([("aime25", str(p))], str(out))
    assert "Qwen3.5" in summ["aime25"]["model"]
    assert len(out.read_text().splitlines()) == 6
    assert "aime25" in B.format_summary(summ)


# ── held-out 평가(모의 생성기) ─────────────────────────────────────────────────────
def test_math_retry_eval_summary_on_mock_generations():
    import math_retry_eval as E
    problems = [{"problem": "p0", "gold": "42"}, {"problem": "p1", "gold": "42"}]

    def gen(prompts):
        assert len(prompts) == 2 and prompts[0][0]["content"] == E.MATH_PROMPT_VARIANTS["math_retry"]
        return [
            [(_row("7", "redirect", "42"), 0, 900),      # 구제
             (_row("42", "verify"), 0, 400),             # 정답·verify
             (_row("7", "verify"), 0, 400),              # 오답·verify
             (_row("42", "redirect", "7"), 1, 6144)],    # 탈선·잘림
            [(_row("42", "verify"), 0, 300)] * 3
            + [(_row("7", "redirect", "7.0"), 0, 500)],  # 수학 동치 베낌(변경 아님)
        ]
    rows, tel = E.evaluate(problems, gen, num_samples=4)
    assert len(rows) == 8 and tel["n_groups"] == 2
    assert tel["first_acc"] == pytest.approx(5 / 8) and tel["final_acc"] == pytest.approx(5 / 8)
    assert tel["gain_from_retry"] == pytest.approx(0.0)
    assert tel["redirect_rate_given_wrong"] == pytest.approx(2 / 3) and tel["redirect_rate_given_right"] == pytest.approx(1 / 5)
    assert tel["redirect_rate"] == pytest.approx(3 / 8) and tel["redirect_rate_all_rows"] == pytest.approx(3 / 8)
    assert tel["judgment_acc"] == pytest.approx(6 / 8)
    assert tel["second_attempt_rate"] == pytest.approx(3 / 8) and tel["trunc_rate"] == 0.125
    assert tel["rescue_rate_given_redirected"] == pytest.approx(1 / 3)
    assert tel["derail_rate_given_redirected"] == pytest.approx(1 / 3)
    assert tel["changed_rate_given_redirected"] == pytest.approx(2 / 3)     # 7→7.0 은 변경이 아니다
    assert tel["mean_tokens"] == pytest.approx((900 + 400 + 400 + 6144 + 900 + 500) / 8)
    assert tel["pass_at_n_first"] == 1.0 and tel["pass_at_n_final"] == 1.0
    s = E.format_summary(tel)
    for k in ("first_acc", "final_acc", "gain_from_retry", "redirect_rate_given_wrong", "judgment_acc", "mean_tokens",
              "changed_rate_given_redirected"):
        assert k in s
    with pytest.raises(RuntimeError, match="생성기"):
        E.evaluate(problems, lambda ps: [[]], num_samples=1)


def test_math_retry_eval_includes_within_problem_selectivity_metrics():
    """retry_metrics.all_metrics 가 evaluate() 요약에 섞여 들어가고 summary 문자열에도 찍힌다."""
    import math_retry_eval as E
    problems = [{"problem": "p0", "gold": "42"}, {"problem": "p1", "gold": "42"}]

    def gen(prompts):
        return [
            # p0: 4 samples, 2 correct/2 wrong (mixed) — redirect exactly on the wrong ones.
            [(_row("7", "redirect", "42"), 0, 300), (_row("7", "redirect", "42"), 0, 300),
             (_row("42", "verify"), 0, 300), (_row("42", "verify"), 0, 300)],
            # p1: all correct (not mixed) — excluded from mixed metrics.
            [(_row("42", "verify"), 0, 300)] * 4,
        ]
    rows, tel = E.evaluate(problems, gen, num_samples=4)
    for k in ("n_mixed_problems", "redirect_rate_given_wrong_mixed", "redirect_rate_given_right_mixed",
             "selectivity_mixed", "mean_within_problem_auc", "frac_problems_auc_gt_half",
             "judgment_acc_mixed", "retry_lift_p0", "retry_lift_p0_50", "retry_lift_p50_100",
             "retry_lift_p100"):
        assert k in tel, k
    assert tel["n_mixed_problems"] == 1
    assert tel["redirect_rate_given_wrong_mixed"] == 1.0
    assert tel["redirect_rate_given_right_mixed"] == 0.0
    assert tel["selectivity_mixed"] == 1.0
    assert tel["mean_within_problem_auc"] == 1.0
    s = E.format_summary(tel)
    assert "selectivity_mixed" in s and "mean_within_problem_auc" in s


# ── verl_sdc 배선 ───────────────────────────────────────────────────────────────
class _Tok:
    def encode(self, s, add_special_tokens=False):
        return [0] * len(s)                       # 1 문자 = 1 토큰


# ── within-problem 선택성이 training telemetry 에 섞이는지(group_pass_rate 경유) ──────
def test_retry_telemetry_adds_mixed_metrics_when_group_pass_rate_present():
    texts = [_row("7", "redirect", "42"), _row("42", "verify"),
             _row("7", "redirect", "42"), _row("42", "verify")]
    rows = M.compute_rows(texts, ["42"] * 4, ["P"] * 4, "M_RETRY", uids=["u1", "u1", "u2", "u2"],
                          group_pass_rate=[0.5, 0.5, 1.0, 1.0])
    rep = M.retry_telemetry(rows)
    assert "selectivity_mixed" in rep and "mean_within_problem_auc" in rep
    assert "judgment_acc_mixed" in rep
    # u1 은 mixed(0.5), u2 는 not mixed(1.0) — mixed 지표는 u1 하나로만 잡힌다
    assert rep["n_mixed_problems"] == 1


def test_retry_telemetry_skips_mixed_metrics_when_group_pass_rate_absent():
    texts = [_row("7", "redirect", "42"), _row("42", "verify")]
    rows = M.compute_rows(texts, ["42"] * 2, ["P"] * 2, "M_RETRY", uids=["u1", "u1"])
    rep = M.retry_telemetry(rows)
    assert "selectivity_mixed" not in rep


def test_format_tel_prints_mixed_metrics_when_present():
    texts = [_row("7", "redirect", "42"), _row("42", "verify")]
    rows = M.compute_rows(texts, ["42"] * 2, ["P"] * 2, "M_RETRY", uids=["u1", "u1"],
                          group_pass_rate=[0.5, 0.5])
    rep = M.telemetry(rows, arm="M_RETRY", step=1)
    line = M.format_tel(rep)
    assert "selectivity_mixed=" in line and "auc_mixed=" in line and "judgment_acc_mixed=" in line


def _fake_trainer_data(texts, uids, arm, *, plen=4, resp_w=None, tokenizer=None, extra_info=None):
    import torch
    from types import SimpleNamespace as NS
    L = resp_w or (max(len(t) for t in texts) + 1)
    B = len(texts)
    am = torch.zeros(B, plen + L, dtype=torch.long)
    am[:, :plen] = 1
    for i, t in enumerate(texts):
        am[i, plen:plen + min(len(t), L)] = 1
    nt = {"problem": ["P"] * B, "gold": ["42"] * B, "uid": list(uids)}
    if extra_info is not None:
        nt["extra_info"] = extra_info
    data = NS(batch={"prompts": torch.zeros(B, plen, dtype=torch.long), "attention_mask": am,
                     "responses": torch.zeros(B, L, dtype=torch.long)},
              non_tensor_batch=nt)
    self = NS(config=NS(algorithm=NS(math_arm=arm), data=NS(max_prompt_length=plen)), tokenizer=tokenizer)
    return self, data


def test_stash_reads_group_pass_rate_from_extra_info(capsys):
    """verl_sdc stash 가 forced_redirect 와 같은 규약으로 extra_info.group_pass_rate 를 읽어
    compute_rows 에 그대로 넘기는지(0<rate<1 인 u1 이 mixed 로 잡히는지)."""
    from src.training import verl_sdc as V
    texts = [_row("7", "redirect", "42"), _row("42", "verify"),
             _row("7", "redirect", "42"), _row("42", "verify")]
    ei = [{"group_pass_rate": 0.5}, {"group_pass_rate": 0.5},
          {"group_pass_rate": 1.0}, {"group_pass_rate": 1.0}]
    self, data = _fake_trainer_data(texts, ["u1", "u1", "u2", "u2"], "M_RETRY",
                                    tokenizer=_Tok(), extra_info=ei)
    V._compute_math_arm_stash(self, data, texts, 4, 4, 1)
    rows = V._MATH_STASH["rows"]
    assert [r["group_pass_rate"] for r in rows] == [0.5, 0.5, 1.0, 1.0]


def test_stash_group_pass_rate_defaults_none_when_absent(capsys):
    from src.training import verl_sdc as V
    texts = [_row("7", "redirect", "42"), _row("42", "verify")]
    self, data = _fake_trainer_data(texts, ["u1", "u1"], "M_RETRY", tokenizer=_Tok())
    V._compute_math_arm_stash(self, data, texts, 2, 4, 1)
    rows = V._MATH_STASH["rows"]
    assert [r["group_pass_rate"] for r in rows] == [None, None]


def test_stash_wires_retry_arm_with_trunc_and_token_counter(capsys):
    from src.training import verl_sdc as V
    texts = [_row("7", "redirect", "42"), _row("42", "verify"), _row("42", "redirect", "999999 wrong"), "plain \\boxed{42}"]
    W = max(len(t) for t in texts)                       # 가장 긴 행(=행 2)이 응답 폭에 닿는다 → 잘림
    self, data = _fake_trainer_data(texts, ["u1", "u1", "u2", "u2"], "M_RETRY", resp_w=W, tokenizer=_Tok())
    totals = V._compute_math_arm_stash(self, data, texts, 4, 4, 1)
    assert totals == [1.0, 1.0, 0.0, 1.0]                # 답 스팬 = 최종 답 정오만
    st = V._MATH_REGION_STASH
    assert st["member"] == [1, 1, 1, 0]
    rows = V._MATH_STASH["rows"]
    assert [r["truncated"] for r in rows] == [0, 0, 1, 0]
    # tok_len_fn(1 문자 = 1 토큰): 메타 뒤 문자 수와 같다
    assert rows[0]["n_tok_after_meta"] == len(texts[0]) - rows[0]["meta_end"]
    assert rows[2]["meta_val"] < -0.5                    # 정답인데 redirect: −1 − cost, ×0.5
    out = capsys.readouterr().out
    assert "retry_w=0.50" in out and "trunc_rows=1" in out and "first_acc=" in out
    # M_RETRY_RAND 도 같은 스태시를 탄다
    self, data = _fake_trainer_data(texts, ["u1", "u1", "u2", "u2"], "M_RETRY_RAND", resp_w=W, tokenizer=_Tok())
    V._compute_math_arm_stash(self, data, texts, 4, 4, 1)
    assert "[MATH][RETRY_RAND]" in capsys.readouterr().out


def test_run_math_arm_retry_defaults_resp_len_6144_and_uses_retry_eval():
    import os
    import subprocess
    env = {k: v for k, v in os.environ.items() if k not in ("RESP_LEN", "MATH_RETRY_W", "MATH_RETRY_LEN_COST")}
    r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_RETRY", "1", "50", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    for want in ("RESP_LEN=6144", "data.max_response_length=6144", "LINEAGE=cd9_M_RETRY_s1_r6144",
                 "math_train_math_retry.parquet", "MATH_RETRY_W=0.5", "MATH_RETRY_LEN_COST=0.2",
                 "EVAL_SCRIPT=math_retry_eval.py"):
        assert want in out, (want, out)
    # 명시적 RESP_LEN 은 그대로 존중한다
    r2 = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_RETRY_RAND", "2", "--dry-run"],
                        cwd=REPO, capture_output=True, text=True, env={**env, "RESP_LEN": "4096"})
    assert r2.returncode == 0 and "RESP_LEN=4096" in r2.stdout and "EVAL_SCRIPT=math_retry_eval.py" in r2.stdout
    # 다른 팔은 4096 기본·math_rollout.py 평가 그대로
    r3 = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_G1", "1", "--dry-run"],
                        cwd=REPO, capture_output=True, text=True, env=env)
    assert r3.returncode == 0 and "RESP_LEN=4096" in r3.stdout and "EVAL_SCRIPT=math_rollout.py" in r3.stdout
    sh = (REPO / "scripts/local/run_math_arm.sh").read_text()
    assert "math_retry_eval.py" in sh
    header = sh.split("set -euo pipefail")[0]
    assert "M_RETRY" in header                                   # 헤더의 팔 목록에 재시도 팔이 있다


def test_run_math_arm_eval_max_tokens_independent_of_resp_len():
    """★EVAL_MAX_TOKENS(0914b 수리): docs/RESULTS_cd9.md 의 사전학습 베이스라인은 8192 로
    쟀는데 run_math_arm.sh 는 판정 스텝(30/50/100) math500 eval 을 RESP_LEN(M_G0/M_G1 = 4096)
    으로 돌려 M_G1 s30 eval 이 trunc_rate 11.5% 로 baseline 과 비교 불가였다. 수리: 사후 eval
    은 RESP_LEN 과 무관하게 EVAL_MAX_TOKENS(기본 8192, 오버라이드 가능)를 쓴다. --dry-run 은
    실제 학습·머지를 안 돌리므로 여기서는 스크립트가 찍는 eval 커맨드 템플릿(및 provenance
    줄)으로 --max_tokens 값을 검증한다."""
    import os
    import re
    import subprocess
    env = {k: v for k, v in os.environ.items()
           if k not in ("RESP_LEN", "EVAL_MAX_TOKENS", "MATH_RETRY_W", "MATH_RETRY_LEN_COST")}

    # M_G1(RESP_LEN 기본 4096) -> eval 은 그래도 EVAL_MAX_TOKENS 기본값 8192 를 쓴다.
    r1 = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_G1", "1", "--dry-run"],
                         cwd=REPO, capture_output=True, text=True, env=env)
    assert r1.returncode == 0, r1.stdout + r1.stderr
    out1 = r1.stdout
    assert "RESP_LEN=4096" in out1
    assert "EVAL_MAX_TOKENS=8192" in out1
    assert re.search(r"--max_tokens\s+8192", out1)
    assert "math500_8k" in out1

    # M_RETRY(RESP_LEN 기본 6144) -> eval 은 여전히 EVAL_MAX_TOKENS 기본값 8192, RESP_LEN 이 아니다.
    r2 = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_RETRY", "1", "50", "--dry-run"],
                         cwd=REPO, capture_output=True, text=True, env=env)
    assert r2.returncode == 0, r2.stdout + r2.stderr
    out2 = r2.stdout
    assert "RESP_LEN=6144" in out2
    assert "EVAL_MAX_TOKENS=8192" in out2
    assert re.search(r"--max_tokens\s+8192", out2)
    assert "math500_retry_8k" in out2               # 재시도 팔 eval 서브디렉토리에도 예산 태그

    # EVAL_MAX_TOKENS 오버라이드는 존중된다(RESP_LEN 과 독립적으로).
    r3 = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_G1", "1", "--dry-run"],
                         cwd=REPO, capture_output=True, text=True,
                         env={**env, "EVAL_MAX_TOKENS": "4096"})
    assert r3.returncode == 0, r3.stdout + r3.stderr
    out3 = r3.stdout
    assert "EVAL_MAX_TOKENS=4096" in out3
    assert re.search(r"--max_tokens\s+4096", out3)
    assert "math500_4k" in out3
    assert not re.search(r"--max_tokens\s+8192", out3)


def test_ray_env_forwarding_has_retry_names_and_launcher_accepts_arm():
    src = (REPO / "src/training/verl_sdc.py").read_text()
    for k in ("MATH_RETRY_W", "MATH_RETRY_LEN_COST"):
        assert f'"{k}"' in src, k
    # 런처는 MATH_ARM_SPECS 에서 변형을 읽으므로 M_RETRY 를 그대로 받는다(라벨표 불필요)
    sh = (REPO / "scripts/local/run_math_arm.sh").read_text()
    assert "MATH_ARM_SPECS" in sh and "M_RETRY" not in sh.split("case")[1].split("esac")[0]
