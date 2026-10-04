"""math_pmi_shift_probe 회귀 시험 (CPU, 모델 없음 — 왕복 토크나이저 + 가짜 채점기).

1. 문맥 조립 — CTX_OPEN/CTX_CLOSE 경계가 **의도한 박스의 닫는 `}` 바로 뒤**에 떨어진다.
2. 채움말(placebo) 토큰 길이가 실제 구간의 ±15% 안이고, 채움말에 추론 내용이 없다.
3. 채점 대상 토큰(`answer_target_ids`) — 답 토큰만, 닫는 `}` 는 빼고.
4. decoy / majority 선별(동률·건너뜀 포함).
5. PMI·shift 산술.
6. AUC 헬퍼(손으로 만든 경우) — pooled 와 문제 안.
7. VERDICT 진리표.
8. 재개(rows.jsonl 의 roll_id 건너뛰기).
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_pmi_shift_probe as M  # noqa: E402

PROBLEM = "What is 2 plus 2?"


class RoundTripTok:
    """공백을 보존하는 왕복 토크나이저 — encode/decode 가 정확히 되돌아간다(채움말 절단 검증용)."""

    _RE = re.compile(r"\s+|\S+")

    def __init__(self):
        self.vocab: dict = {}
        self.inv: dict = {}

    def _id(self, s: str) -> int:
        if s not in self.vocab:
            i = len(self.vocab) + 1
            self.vocab[s] = i
            self.inv[i] = s
        return self.vocab[s]

    def encode(self, text, add_special_tokens=False):
        return [self._id(m.group(0)) for m in self._RE.finditer(text or "")]

    def decode(self, ids, **kw):
        return "".join(self.inv[i] for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return "".join(f"<{m['role']}> {m['content']} " for m in msgs) + "<assistant> "


TOK = RoundTripTok()
# 첫 박스 → 긴 수정 구간 → 두 번째 박스
REVISED = ("First I get \\boxed{5}. " + "Hmm wait let me check this step again. " * 12
           + "So the answer is \\boxed{4}.")
PLAIN = "Straightforward: \\boxed{4}."


# ── 1·2. 문맥 조립 ──────────────────────────────────────────────────────────────
def test_context_boundaries_land_after_the_intended_box():
    ctx = M.row_contexts(TOK, "math_opt", PROBLEM, REVISED)
    prompt = M.render_generation_prompt(TOK, "math_opt", PROBLEM)
    assert ctx["first_answer"] == "5" and ctx["last_answer"] == "4"
    assert ctx["revised"] is True and ctx["n_boxes"] == 2
    assert ctx["ctx_open"] == prompt + "First I get \\boxed{5}"
    assert ctx["ctx_open"].endswith("\\boxed{5}")           # 닫는 `}` 포함, 그 뒤 문자는 없다
    # 기본 close_at=before_last_box — 마지막 \boxed{ 바로 앞까지, 답 Y("4")·닫는 `}` 는 없다.
    assert ctx["ctx_close"] == prompt + REVISED[:REVISED.rindex("\\boxed{4}")]
    assert not ctx["ctx_close"].endswith("\\boxed{4}")
    assert "\\boxed{4}" not in ctx["ctx_close"][len(prompt):]
    assert ctx["ctx_placebo"].startswith(ctx["ctx_open"])


def test_close_at_after_last_box_keeps_old_behaviour():
    ctx = M.row_contexts(TOK, "math_opt", PROBLEM, REVISED, close_at="after_last_box")
    prompt = M.render_generation_prompt(TOK, "math_opt", PROBLEM)
    assert ctx["ctx_close"] == prompt + REVISED[:REVISED.rindex("\\boxed{4}") + len("\\boxed{4}")]
    assert ctx["ctx_close"].endswith("\\boxed{4}")


def test_non_revising_row_is_kept_and_has_zero_length_segment():
    ctx = M.row_contexts(TOK, "math_opt", PROBLEM, PLAIN)
    assert ctx["revised"] is False
    assert ctx["ctx_close"] == ctx["ctx_open"]
    assert ctx["seg_tokens"] == 0 and ctx["ctx_placebo"] == ctx["ctx_open"]
    assert math.isnan(ctx["placebo_ratio"])
    # 같은 답이 두 번 나오면 수정이 아니다(수학적 동치).
    ctx2 = M.row_contexts(TOK, "math_opt", PROBLEM, "\\boxed{4} ... and again \\boxed{4.0}")
    assert ctx2["n_boxes"] == 2 and ctx2["revised"] is False
    assert ctx2["ctx_close"] != ctx2["ctx_open"]            # 귀무 모집단으로 남는다


def test_row_without_any_box_is_none():
    assert M.row_contexts(TOK, "math_opt", PROBLEM, "no box at all") is None


def test_placebo_is_length_matched_within_tolerance_and_contentless():
    ctx = M.row_contexts(TOK, "math_opt", PROBLEM, REVISED)
    assert ctx["seg_tokens"] > 0
    assert abs(ctx["placebo_ratio"] - 1.0) <= M.PLACEBO_TOL
    filler = ctx["ctx_placebo"][len(ctx["ctx_open"]):]
    assert "\\boxed" not in filler and not any(ch.isdigit() for ch in filler)
    assert M.PLACEBO_SENT.strip() in filler


def test_placebo_filler_hits_the_exact_token_budget():
    for n in (1, 7, 40, 137):
        assert len(TOK.encode(M.placebo_filler(TOK, n))) == n
    assert M.placebo_filler(TOK, 0) == "" and M.placebo_filler(TOK, -3) == ""


# ── 3. 채점 대상 토큰 ───────────────────────────────────────────────────────────
def test_answer_target_ids_scores_only_the_answer_tokens():
    assert M.answer_continuation("7") == M.ANSWER_HEAD + "7" + M.ANSWER_TAIL
    ctx = "<user> q <assistant> body"
    cid, tid = M.answer_target_ids(TOK, ctx, "42")
    # 문맥 + 대상 = ctx + HEAD + answer 이고, 닫는 `}` 는 **어디에도 없다**(채점하지 않는다).
    assert TOK.decode(cid) + TOK.decode(tid) == ctx + M.ANSWER_HEAD + "42"
    assert TOK.decode(tid).endswith("42") and M.ANSWER_TAIL not in TOK.decode(tid)
    assert TOK.decode(cid).startswith(ctx)
    # 대상은 «답이 붙으면서 달라지는 토큰»뿐이다 — 이 토크나이저는 `\boxed{` 와 답을 한
    # 토큰으로 합치므로 합친 토큰이 대상에 들어간다(모듈 docstring 의 규약).
    assert len(tid) == 1
    # 두 후보 답은 **같은 문맥 토큰**에서 채점된다(PMI 차가 자리 차이에 오염되지 않는다).
    cid2, tid2 = M.answer_target_ids(TOK, ctx, "17")
    assert cid2 == cid and TOK.decode(tid2).endswith("17")


def test_pair_target_ids_aligns_both_candidates_to_one_cut():
    """한쪽 답만 `{` 와 합쳐지는 경우에도 두 후보의 문맥 토큰열이 같아야 한다."""
    class MergeTok(RoundTripTok):
        """'{' 를 **뒤 글자와 합치는** 토크나이저 — 답에 따라 경계가 달라지는 실제 상황 모사."""
        _RE = re.compile(r"\{\S*|\s+|\S+")

    tok = MergeTok()
    ctx = "<user> q <assistant> body"
    (cp, tp), (cn, tn) = M.pair_target_ids(tok, ctx, "42", "-3")
    assert cp == cn                                     # 바이트 동일한 문맥
    # 같은 문자 구간을 채점한다 — 한쪽만 `{` 를 공짜로 얻는 비대칭이 없다.
    assert tok.decode(tp)[:-2] == tok.decode(tn)[:-2]
    assert tok.decode(tp).endswith("{42") and tok.decode(tn).endswith("{-3")
    assert M.ANSWER_TAIL not in tok.decode(tp) + tok.decode(tn)
    # 같은 유형의 두 답이면 자리가 이미 같으므로 그대로다.
    (cp2, tp2), (cn2, tn2) = M.pair_target_ids(TOK, ctx, "42", "17")
    assert cp2 == cn2 and TOK.decode(tp2).endswith("42") and TOK.decode(tn2).endswith("17")


# ── 4. 앵커 선별 ────────────────────────────────────────────────────────────────
def test_decoy_is_the_most_common_wrong_first_answer():
    assert M.decoy_answer(["4", "5", "5", "6"], "4") == "5"
    # 동률 → 먼저 나온 것
    assert M.decoy_answer(["6", "5", "5", "6", "4"], "4") == "6"
    # gold 와 동치인 표기는 오답 후보가 아니다
    assert M.decoy_answer(["4", "4.0", "\\frac{8}{2}"], "4") == ""
    assert M.decoy_answer(["", ""], "4") == ""


def test_majority_is_gold_free_with_first_seen_tiebreak():
    assert M.majority_answer(["5", "4", "4", ""]) == "4"
    assert M.majority_answer(["7", "9", "9", "7"]) == "7"     # 동률 → 먼저 나온 것
    assert M.majority_answer(["", ""]) == ""


def test_pair_answers_skip_cases():
    plan = {"gold": "4", "decoy": "5", "majority": "4"}
    want, skip = M.pair_answers(plan, "5")
    assert want["gold"] == ("4", "5") and want["self"] == ("4", "5") and skip == {}
    # decoy 가 없으면 GOLD 쌍만 건너뛴다
    want, skip = M.pair_answers({"gold": "4", "decoy": "", "majority": "9"}, "5")
    assert "gold" not in want and skip["gold"] == "no_decoy" and "self" in want
    # M ≡ X 면 SELF 쌍을 건너뛴다(동치 표기 포함)
    _, skip = M.pair_answers({"gold": "4", "decoy": "5", "majority": "4.0"}, "4")
    assert skip["self"] == "majority_equals_first"
    _, skip = M.pair_answers({"gold": "4", "decoy": "5", "majority": "4"}, "")
    assert skip["self"] == "no_first"


def test_goldx_pair_and_skip_when_first_answer_equals_gold():
    plan = {"gold": "4", "decoy": "5", "majority": "4"}
    want, skip = M.pair_answers(plan, "7", pairs=["goldx"])
    assert want["goldx"] == ("4", "7") and skip == {}
    # X ≡ gold(문자열 동일이든 동치든) 면 건너뛴다 — 이미 맞혀서 잴 게 없다.
    _, skip = M.pair_answers(plan, "4", pairs=["goldx"])
    assert skip["goldx"] == "x_equals_gold"
    _, skip = M.pair_answers(plan, "4.0", pairs=["goldx"])
    assert skip["goldx"] == "x_equals_gold"
    # gold 가 없으면 no_gold, 첫 답이 없으면 no_first
    _, skip = M.pair_answers({"gold": "", "decoy": "5", "majority": "4"}, "7", pairs=["goldx"])
    assert skip["goldx"] == "no_gold"
    _, skip = M.pair_answers(plan, "", pairs=["goldx"])
    assert skip["goldx"] == "no_first"


# ── 5. PMI·shift 산술 ───────────────────────────────────────────────────────────
def test_pmi_and_shift_arithmetic():
    v = M.pmi_values({"open": -1.0, "close": 2.0, "placebo": -0.5},
                     {"open": 1.0, "close": 0.5, "placebo": 0.5})
    assert v["pmi_open"] == -2.0 and v["pmi_close"] == 1.5 and v["pmi_placebo"] == -1.0
    assert v["shift_real"] == 3.5 and v["shift_placebo"] == 1.0
    bad = M.pmi_values({"open": -1.0, "close": None, "placebo": 0.0},
                       {"open": 1.0, "close": 0.5, "placebo": 0.5})
    assert math.isnan(bad["pmi_close"]) and math.isnan(bad["shift_real"])
    assert bad["pmi_open"] == -2.0


def test_score_row_end_to_end_with_a_fake_scorer():
    """가짜 채점기 = 대상 토큰 수에 비례하는 상수 — 배선(문맥×답 6회 채점)만 검증한다."""
    seen: list = []

    def scorer(reqs):
        seen.append(len(reqs))
        return [-float(len(t)) for _, t in reqs]

    plan = {"roll_id": "g0#1", "group_id": "g0", "problem_id": "0", "problem": PROBLEM,
            "gold": "4", "text": REVISED, "r_corr": 1, "truncated": 0, "decoy": "5",
            "majority": "4", "agree_state": "SPLIT"}
    row = M.score_row(TOK, plan, "math_opt", ["gold", "self"], scorer)
    # gold=(4,5), self=(4,5) → 답 두 개 × 세 문맥 = 6 요청(중복 제거가 작동한다)
    assert seen == [6]
    assert row["revised"] is True and row["first_correct"] == 0 and row["last_correct"] == 1
    assert row["gold_skip"] == "" and row["self_skip"] == ""
    for p in ("gold", "self"):
        assert row[f"{p}_pos"] == "4" and row[f"{p}_neg"] == "5"
        assert math.isfinite(row[f"{p}_shift_real"])
    noboxed = M.score_row(TOK, {**plan, "roll_id": "g0#2", "text": "nothing"}, "math_opt",
                          ["gold"], scorer)
    assert noboxed["skip"] == "no_box"


# ── 6. AUC ──────────────────────────────────────────────────────────────────────
def test_auc_hand_built():
    assert M.auc([3.0, 2.0, 1.0, 0.0], [1, 1, 0, 0])["auc"] == 1.0
    assert M.auc([0.0, 1.0], [1, 0])["auc"] == 0.0
    assert M.auc([1.0, 1.0], [1, 0])["auc"] == 0.5          # 동점 = 0.5
    assert math.isnan(M.auc([1.0, 2.0], [1, 1])["auc"])     # 한 부류뿐


def test_within_problem_auc_pools_over_problems_with_both_classes():
    rows = [  # g0: 완전 분리(AUC 1, 쌍 1) · g1: 뒤집힘(AUC 0, 쌍 1) · g2: 한 부류뿐(제외)
        {"group_id": "g0", "s": 2.0, "y": 1}, {"group_id": "g0", "s": 1.0, "y": 0},
        {"group_id": "g1", "s": 1.0, "y": 1}, {"group_id": "g1", "s": 3.0, "y": 0},
        {"group_id": "g2", "s": 5.0, "y": 1}, {"group_id": "g2", "s": 4.0, "y": 1}]
    got = M.within_problem_auc(rows, "s", "y")
    assert got["n_problems"] == 2 and got["n_pairs"] == 2 and got["auc"] == 0.5
    # 비유한 점수는 버린다
    rows2 = rows + [{"group_id": "g3", "s": float("nan"), "y": 1},
                    {"group_id": "g3", "s": 1.0, "y": 0}]
    assert M.within_problem_auc(rows2, "s", "y")["n_problems"] == 2


# ── 7. VERDICT 진리표 ───────────────────────────────────────────────────────────
def _ci(lo, hi, mean=None):
    return {"mean": (lo + hi) / 2 if mean is None else mean, "lo": lo, "hi": hi, "n": 10}


def test_verdict_truth_table():
    pos, neg, span = _ci(0.2, 0.9), _ci(-0.9, -0.2), _ci(-0.3, 0.4)
    ok = M.make_verdicts({"gold": pos, "self": pos}, {"gold": pos, "self": pos},
                         {"gold": 0.71, "self": 0.65})
    assert [ok[k]["pass"] for k in ("a_shift_real_positive_on_revised",
                                    "b_content_not_presence",
                                    "c_within_problem_auc_final_correct")] == [True] * 3
    assert ok["all_pass"] is True
    # (a) 한 쌍만 통과하면 FAIL
    v = M.make_verdicts({"gold": pos, "self": span}, {"gold": pos, "self": pos},
                        {"gold": 0.71, "self": 0.65})
    assert v["a_shift_real_positive_on_revised"]["pass"] is False and v["all_pass"] is False
    assert v["a_shift_real_positive_on_revised"]["by_pair"] == {"gold": True, "self": False}
    # (b) 음의 CI(플라시보가 더 크다)도 FAIL — «내용» 주장이 아니다
    v = M.make_verdicts({"gold": pos}, {"gold": neg}, {"gold": 0.9})
    assert v["b_content_not_presence"]["pass"] is False
    # (c) 경계값은 통과, 그 아래는 FAIL, NaN 도 FAIL
    assert M.make_verdicts({"g": pos}, {"g": pos}, {"g": M.AUC_MIN})[
        "c_within_problem_auc_final_correct"]["pass"] is True
    assert M.make_verdicts({"g": pos}, {"g": pos}, {"g": M.AUC_MIN - 1e-9})[
        "c_within_problem_auc_final_correct"]["pass"] is False
    assert M.make_verdicts({"g": pos}, {"g": pos}, {"g": float("nan")})[
        "c_within_problem_auc_final_correct"]["pass"] is False
    # 입력이 비면 FAIL(조용한 통과 금지)
    assert M.make_verdicts({}, {}, {})["all_pass"] is False


# ── 8. 재개 ─────────────────────────────────────────────────────────────────────
def _roll(gid, j, text, ans, corr):
    return {"group_id": gid, "problem_id": gid, "problem": PROBLEM, "gold": "4", "text": text,
            "final_answer": ans, "r_corr": corr, "truncated": 0, "n_tok": 100}


def test_plan_rows_and_resume_skipping(tmp_path):
    rolls = [_roll("g0", 0, REVISED, "4", 1), _roll("g0", 1, PLAIN, "4", 1),
             _roll("g0", 2, "I say \\boxed{5}.", "5", 0),
             _roll("g1", 0, PLAIN, "4", 1)]
    plans = M.plan_rows(rolls)
    assert [p["roll_id"] for p in plans] == ["g0#0", "g0#1", "g0#2", "g1#0"]
    assert plans[0]["decoy"] == "5" and plans[0]["majority"] == "4"
    assert plans[0]["agree_state"] in M.AGREE_STATES
    assert [p["roll_id"] for p in M.plan_rows(rolls, limit_problems=1)] == \
        ["g0#0", "g0#1", "g0#2"]

    path = tmp_path / "rows.jsonl"
    assert M.load_done(path) == {}
    path.write_text(json.dumps({"roll_id": "g0#1", "group_id": "g0"}) + "\n"
                    + json.dumps({"roll_id": "g1#0", "group_id": "g1"}) + "\n")
    done = M.load_done(path)
    assert set(done) == {"g0#1", "g1#0"}
    assert [p["roll_id"] for p in plans if p["roll_id"] not in done] == ["g0#0", "g0#2"]


# ── 9. 관문 (c) 재정의 — 수정 행만 ────────────────────────────────────────────────
def _sr(group_id, shift, last_correct, revised):
    return {"skip": "", "gold_skip": "", "group_id": group_id, "revised": revised,
            "last_correct": last_correct, "gold_shift_real": shift}


def test_gate_c_auc_uses_revised_rows_only_unrevised_rows_must_not_drive_it():
    rows = [_sr("g0", 1.0, 1, True), _sr("g0", -1.0, 0, True)]
    # 미수정 행(SHIFT_real ≡ 0) 을 많이 섞는다 — 포함되면 동점(0.5) 이 쌓여 AUC 를 깬다.
    for _ in range(10):
        rows.append(_sr("g0", 0.0, 1, False))
        rows.append(_sr("g0", 0.0, 0, False))
    s = M.pair_summary(rows, "gold")
    a = s["auc_final_correct"]
    # 수정 행만 보면 완전 분리(1.0 > -1.0) — 관문(c) 은 이 값을 써야 한다.
    assert a["within_problem_revised"]["auc"] == 1.0
    assert a["within_problem_revised"]["n_problems"] == 1
    # 미수정 행까지 섞은 «참고용» 전 행 AUC 는 그보다 뚜렷이 낮다(동점 희석).
    assert a["within_problem_all_rows_informational"]["auc"] < M.AUC_MIN
    # summarize()/make_verdicts 가 실제로 revised-only 값을 관문에 쓰는지도 확인한다.
    v = M.make_verdicts({"gold": _ci(0.1, 0.9)}, {"gold": _ci(0.1, 0.9)},
                        {"gold": a["within_problem_revised"]["auc"]})
    assert v["c_within_problem_auc_final_correct"]["pass"] is True
    v2 = M.make_verdicts({"gold": _ci(0.1, 0.9)}, {"gold": _ci(0.1, 0.9)},
                         {"gold": a["within_problem_all_rows_informational"]["auc"]})
    assert v2["c_within_problem_auc_final_correct"]["pass"] is False


# ── 10. summary meta ────────────────────────────────────────────────────────────
def test_summarize_records_close_at_default_and_override(tmp_path):
    rows = [M.score_row(TOK, {"roll_id": "g0#0", "group_id": "g0", "problem_id": "0",
                              "problem": PROBLEM, "gold": "4", "text": REVISED, "r_corr": 1,
                              "truncated": 0, "decoy": "5", "majority": "4",
                              "agree_state": "SPLIT"}, "math_opt", ["gold"],
                        lambda reqs: [-float(len(t)) for _, t in reqs])]
    summ = M.summarize(rows, ["gold"])
    # summarize() 자체는 close_at 을 모른다(main() 이 meta 에 적는다) — 여기서는 main() 이
    # 하는 일을 흉내내 검증한다.
    summ["meta"] = {"close_at": "before_last_box"}
    md = M.to_markdown(summ)
    assert "close_at = `before_last_box`" in md
    summ["meta"] = {"close_at": "after_last_box"}
    md2 = M.to_markdown(summ)
    assert "close_at = `after_last_box`" in md2
