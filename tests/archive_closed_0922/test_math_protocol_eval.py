"""math_protocol_eval 회귀 시험 (CPU, 모델 없음 — MockTok + 가짜 생성기).

1. 게이트는 ALL_SAME 을 절대 재시도하지 않는다.
2. always_notx 는 모든 상태를 재시도한다.
3. vote16 / vote_a2 산술(손으로 만든 경우).
4. flip_wrong / flip_right 계수.
5. 캐시 건너뛰기.
6. fact/notx/switch/pad 프롬프트가 `math_activation_gate` 의 F1 팔과 **바이트 동일**.
7. 토큰 회계 = 시도-1 n_tok + 추가 n_tok.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import math_activation_gate as A  # noqa: E402
import math_protocol_eval as M  # noqa: E402
import math_ruler_pivot as P  # noqa: E402

TOK = P.MockTok()
PROBLEM = "What is 2 plus 2?"
SOL = "I think it is five. \\boxed{5}"
GATE = list(M.GATE_STATES)
HABIT = dict(M.DEFAULT_HABIT_BY_STATE)


def _problem(pid, answers, corrs, ntok=100):
    rows = [{"roll_id": i, "answer": a, "r_corr": c, "n_tok": ntok, "text": SOL,
             "truncated": 0} for i, (a, c) in enumerate(zip(answers, corrs))]
    st = A.agreement_state(answers, k=len(answers))
    return {"problem_id": pid, "problem": PROBLEM, "gold": "4", "rows": rows,
            "state": st["state"], "a1_tokens": sum(r["n_tok"] for r in rows)}


ALL_SAME = _problem("g0", ["5"] * 4, [0] * 4)
SPLIT = _problem("g1", ["4", "4", "5", "5"], [1, 1, 0, 0])


# ── 1·2. 게이트 ────────────────────────────────────────────────────────────────
def test_gate_never_retries_all_same():
    assert ALL_SAME["state"] == "ALL_SAME"
    for proto in ("gated_notx", "gated_fact", "gated_pad", "gated_blind",
                  "gated_habit_by_state"):
        assert M.mode_for(proto, "ALL_SAME", GATE, HABIT) is None
    reqs = M.plan_requests([ALL_SAME, SPLIT], ["gated_notx"], gate=GATE, habit_by_state=HABIT)
    assert {r["problem_id"] for r in reqs} == {"g1"}


def test_gate_states_drop_all_same_even_if_asked():
    assert "ALL_SAME" not in M.resolve_states("ALL_SAME,SPLIT")


def test_always_notx_retries_everything():
    reqs = M.plan_requests([ALL_SAME, SPLIT], ["always_notx"], gate=GATE, habit_by_state=HABIT)
    assert {r["problem_id"] for r in reqs} == {"g0", "g1"}
    assert len(reqs) == 8                      # 문제당 행 4개
    assert all(r["mode"] == "notx" for r in reqs)
    # notx 의 X 는 그 행 **자기** 답이다
    assert [r["notx"] for r in reqs if r["problem_id"] == "g1"] == ["4", "4", "5", "5"]


def test_habit_by_state_picks_mode_per_state():
    assert M.mode_for("gated_habit_by_state", "SCATTER", GATE, HABIT) == "switch"
    assert M.mode_for("gated_habit_by_state", "NOANS", GATE, HABIT) == "fact"
    assert M.mode_for("gated_habit_by_state", "DOMINANT", GATE, HABIT) == "notx"


# ── 3. 투표 ────────────────────────────────────────────────────────────────────
def test_vote_majority_and_tie_goes_first():
    assert M.vote([("5", 0), ("4", 1), ("4", 1)]) == ("4", 1)
    assert M.vote([("5", 0), ("4", 1)]) == ("5", 0)          # 동률 → 먼저 나온 것
    assert M.vote([("", 0), ("", 0)]) == ("", 0)


def _cache(protocol, pid, answers, corrs, mode="notx", ntok=50):
    return {(protocol, pid, i, mode): {"protocol": protocol, "problem_id": pid, "roll_id": i,
                                       "mode": mode, "state": "SPLIT", "text": "",
                                       "final_answer": a, "r_corr": c, "truncated": 0,
                                       "n_tok": ntok}
            for i, (a, c) in enumerate(zip(answers, corrs))}


def test_vote16_and_vote_a2_on_hand_built_case():
    # 시도-1: 4,4,5,5 (2:2 동률 → 먼저 나온 "4" 가 이겨 maj8 은 맞는다)
    # 재시도: 5,5,5,4 → vote_a2 = "5"(오답) · vote16 = 4가 3표, 5가 5표 → "5"(오답)
    cache = _cache("gated_notx", "g1", ["5", "5", "5", "4"], [0, 0, 0, 1])
    rec = M.score_protocol([SPLIT], cache, "gated_notx", gate=GATE, habit_by_state=HABIT)[0]
    assert rec["answer"] == "5" and rec["correct"] == 0
    assert rec["answer_a2"] == "5" and rec["correct_a2"] == 0
    assert rec["retried"] == 1
    # 재시도가 정답 쪽이면 둘 다 뒤집힌다
    cache2 = _cache("gated_notx", "g1", ["4", "4", "4", "4"], [1, 1, 1, 1])
    rec2 = M.score_protocol([SPLIT], cache2, "gated_notx", gate=GATE, habit_by_state=HABIT)[0]
    assert (rec2["correct"], rec2["correct_a2"]) == (1, 1)


def test_vote_a2_falls_back_to_attempt1_when_not_retried():
    rec = M.score_protocol([ALL_SAME], {}, "gated_notx", gate=GATE, habit_by_state=HABIT)[0]
    assert rec["retried"] == 0
    assert rec["correct"] == rec["correct_a2"] == 0


# ── 4. flip 계수 ───────────────────────────────────────────────────────────────
def test_flip_wrong_and_flip_right_counts():
    probs = [SPLIT, ALL_SAME]                       # maj8: g1 맞음, g0 틀림
    base = M.score_protocol(probs, {}, "maj8", gate=GATE, habit_by_state=HABIT)
    cache = {**_cache("always_notx", "g1", ["5"] * 4, [0] * 4),        # 맞던 것 → 틀림
             **_cache("always_notx", "g0", ["4"] * 4, [1] * 4)}       # 틀리던 것 → 맞음
    recs = M.score_protocol(probs, cache, "always_notx", gate=GATE, habit_by_state=HABIT)
    summ = M.summarize({"maj8": base, "always_notx": recs}, ["maj8", "always_notx"], n_boot=50)
    # vote16 에서는 g0 의 4:4 동률이 **먼저 나온** 시도-1 답("5")로 풀려 뒤집히지 않는다.
    row = summ["table"]["always_notx"]
    assert (row["flip_wrong"], row["flip_right"]) == (1, 0)
    # 재시도만 보는 vote_a2 에서는 양쪽이 다 뒤집힌다.
    row2 = summ["table"]["always_notx" + M.A2_SUFFIX]
    assert (row2["flip_wrong"], row2["flip_right"]) == (1, 1)
    assert summ["table"]["maj8"]["acc"] == 0.5
    assert summ["by_state"]["always_notx"]["ALL_SAME"]["n"] == 1


# ── 5. 캐시 ────────────────────────────────────────────────────────────────────
def test_cache_skip(tmp_path):
    path = tmp_path / "gens.jsonl"
    rows = [{"protocol": "gated_notx", "problem_id": "g1", "roll_id": i, "mode": "notx",
             "state": "SPLIT", "text": "", "final_answer": "4", "r_corr": 1,
             "truncated": 0, "n_tok": 7} for i in range(2)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    cache = M.load_cache(path)
    assert len(cache) == 2
    reqs = M.plan_requests([SPLIT], ["gated_notx"], gate=GATE, habit_by_state=HABIT)
    todo = [r for r in reqs if M.key_of(r) not in cache]
    assert len(reqs) == 4 and len(todo) == 2
    assert {r["roll_id"] for r in todo} == {2, 3}


# ── 6. F1 프롬프트 바이트 동일 ────────────────────────────────────────────────
def test_prompts_are_byte_identical_to_f1_arms():
    assert (M.build_prompt(TOK, PROBLEM, "fact")
            == A.blind_external_prompt(TOK, "math_opt", PROBLEM))
    assert (M.build_prompt(TOK, PROBLEM, "notx", notx="5")
            == A.notx_prompt(TOK, "math_opt", PROBLEM, "5"))
    assert (M.build_prompt(TOK, PROBLEM, "switch", label="algebra")
            == A.switch_prompt(TOK, "math_opt", PROBLEM, "algebra"))
    assert (M.build_prompt(TOK, PROBLEM, "pad")
            == A.pad_prompt(TOK, "math_opt", PROBLEM))
    assert (M.build_prompt(TOK, PROBLEM, "blind")
            == A.blind_prompt(TOK, "math_opt", PROBLEM))
    assert (M.build_prompt(TOK, PROBLEM, "plain")
            == M.build_prompt(TOK, PROBLEM, "blind"))


def test_notx_without_answer_degrades_to_fact():
    assert M.build_prompt(TOK, PROBLEM, "notx", notx="") == M.build_prompt(TOK, PROBLEM, "fact")


# ── 7. 토큰 회계 ───────────────────────────────────────────────────────────────
def test_token_accounting_sums_attempt1_and_extra():
    cache = _cache("gated_notx", "g1", ["4"] * 4, [1] * 4, ntok=50)
    recs = M.score_protocol([SPLIT], cache, "gated_notx", gate=GATE, habit_by_state=HABIT)
    assert recs[0]["a1_tokens"] == 400 and recs[0]["extra_tokens"] == 200
    summ = M.summarize({"maj8": M.score_protocol([SPLIT], {}, "maj8", gate=GATE,
                                                 habit_by_state=HABIT),
                        "gated_notx": recs}, ["maj8", "gated_notx"], n_boot=50)
    assert summ["table"]["maj8"]["total_tokens"] == 400
    assert summ["table"]["gated_notx"]["total_tokens"] == 600
    assert summ["table"]["gated_notx"]["tokens_per_problem"] == 600.0
    assert "gated_notx" + M.A2_SUFFIX in summ["protocols"]


def test_label_prepass_tokens_count_but_do_not_vote():
    cache = {**_cache("gated_habit_by_state", "g1", ["4", "4"], [1, 1], mode="switch", ntok=10),
             ("gated_habit_by_state", "g1", 0, "label"): {
                 "protocol": "gated_habit_by_state", "problem_id": "g1", "roll_id": 0,
                 "mode": "label", "state": "SPLIT", "text": "algebra", "final_answer": "",
                 "r_corr": 0, "truncated": 0, "n_tok": 5}}
    rec = M.score_protocol([SPLIT], cache, "gated_habit_by_state", gate=GATE,
                           habit_by_state=HABIT)[0]
    assert rec["n_extra"] == 2                       # 라벨 행은 투표에 안 들어간다
    assert rec["extra_tokens"] == 25                 # 10+10+5


# ── 8. 행별 샘플링 시드 ────────────────────────────────────────────────────────
def test_row_seeds_are_distinct_within_a_problem_and_stable():
    # 같은 프롬프트를 쓰는 8개 표본(maj16)도 시드가 서로 달라야 «다수결»이 성립한다.
    probs = [_problem("g1", ["4", "4", "5", "5"], [1, 1, 0, 0])]
    reqs = M.plan_requests(probs, ["maj16", "gated_fact"], gate=GATE, habit_by_state=HABIT)
    seeds = [M.row_seed(r, 11) for r in reqs]
    assert len(set(seeds)) == len(seeds)                 # 쌍마다 다르다
    assert all(0 <= s < 2 ** 31 - 1 for s in seeds)
    # 재실행(= 이 함수를 다시 부르는 것)에도 같은 값 — zlib.crc32 는 소금이 없다.
    assert seeds == [M.row_seed(r, 11) for r in reqs]
    assert M.row_seed(reqs[0], 11) != M.row_seed(reqs[0], 12)
    # 라벨 사전 패스도 같은 규약(모드가 다르면 시드가 다르다)
    r0 = reqs[0]
    assert M.row_seed(r0, 11) != M.row_seed({**r0, "mode": "label"}, 11)


def test_row_seed_matches_known_value():
    """소금 없는 결정성의 고정 — 값이 바뀌면 재개 실행이 다른 표본을 만든다."""
    rec = {"protocol": "maj16", "problem_id": "g1", "roll_id": 3, "mode": "plain"}
    import zlib
    want = (11 * 100003 + zlib.crc32(b"maj16|g1|3|plain")) % (2 ** 31 - 1)
    assert M.row_seed(rec, 11) == want


# ── 9. long8 토큰 회계 ─────────────────────────────────────────────────────────
def test_long8_replaces_attempt1_in_token_accounting():
    cache = _cache("long8", "g1", ["4"] * 8, [1] * 8, mode="plain", ntok=1000)
    recs = M.score_protocol([SPLIT], cache, "long8", gate=GATE, habit_by_state=HABIT)
    assert recs[0]["a1_tokens"] == 0 and recs[0]["extra_tokens"] == 8000
    # maj16 은 시도 1 을 **쓰므로** 그 예산이 총계에 남는다.
    cache16 = _cache("maj16", "g1", ["4"] * 8, [1] * 8, mode="plain", ntok=1000)
    recs16 = M.score_protocol([SPLIT], cache16, "maj16", gate=GATE, habit_by_state=HABIT)
    assert recs16[0]["a1_tokens"] == 400
    base = M.score_protocol([SPLIT], {}, "maj8", gate=GATE, habit_by_state=HABIT)
    summ = M.summarize({"maj8": base, "long8": recs, "maj16": recs16},
                       ["maj8", "long8", "maj16"], n_boot=50)
    assert summ["table"]["long8"]["total_tokens"] == 8000
    assert summ["table"]["maj16"]["total_tokens"] == 8400


def test_long8_votes_only_over_fresh_samples():
    # 시도 1 은 "4"(정답 2표)인데 새 표본 8개가 전부 "5" 면 long8 은 틀려야 한다.
    cache = _cache("long8", "g1", ["5"] * 8, [0] * 8, mode="plain", ntok=10)
    rec = M.score_protocol([SPLIT], cache, "long8", gate=GATE, habit_by_state=HABIT)[0]
    assert rec["answer"] == "5" and rec["correct"] == 0


# ── 10. notx 퇴화 표식 ────────────────────────────────────────────────────────
NOANS = _problem("g2", ["4", "", "", ""], [1, 0, 0, 0])


def test_notx_rows_without_own_answer_are_marked_degraded():
    reqs = M.plan_requests([NOANS], ["always_notx"], gate=GATE, habit_by_state=HABIT)
    assert [r["degraded"] for r in reqs] == [False, True, True, True]
    # 퇴화한 행의 프롬프트는 실제로 fact 와 바이트 동일하다(그래서 표식이 필요하다).
    deg = next(r for r in reqs if r["degraded"])
    assert (M.build_prompt(TOK, PROBLEM, deg["mode"], notx=deg["notx"])
            == M.build_prompt(TOK, PROBLEM, "fact"))
    # 새 표본 프로토콜에는 퇴화 개념이 없다.
    assert all(not r["degraded"] for r in
               M.plan_requests([NOANS], ["maj16"], gate=GATE, habit_by_state=HABIT))


def test_degraded_counts_reach_summary_and_markdown():
    cache = _cache("always_notx", "g2", ["4"] * 4, [1] * 4)
    for i, k in enumerate(sorted(cache)):
        cache[k]["degraded"] = int(i > 0)            # 4행 중 3행이 퇴화
    recs = M.score_protocol([NOANS], cache, "always_notx", gate=GATE, habit_by_state=HABIT)
    assert recs[0]["n_degraded"] == 3
    base = M.score_protocol([NOANS], {}, "maj8", gate=GATE, habit_by_state=HABIT)
    summ = M.summarize({"maj8": base, "always_notx": recs}, ["maj8", "always_notx"], n_boot=50)
    row = summ["table"]["always_notx"]
    assert (row["n_degraded_rows"], row["n_degraded_problems"]) == (3, 1)
    assert summ["table"]["maj8"]["n_degraded_rows"] == 0
    md = M.to_markdown(summ)
    assert "n_degraded (rows/probs)" in md and "| 3/1 |" in md


# ── 11. 라벨 사전 키 ───────────────────────────────────────────────────────────
def test_label_key_is_protocol_qualified():
    r = {"protocol": "gated_habit_by_state", "problem_id": "g1", "roll_id": 2, "mode": "switch"}
    r2 = {**r, "protocol": "gated_notx"}
    assert M.label_key(r) != M.label_key(r2)
    # 캐시 키와 같은 폭(프로토콜 포함) — 한쪽 라벨이 다른 쪽에 새지 않는다.
    labels = {M.label_key(r): "algebra"}
    assert labels.get(M.label_key(r2), "") == ""
    assert M.label_key(r)[0] == M.key_of({**r, "mode": "label"})[0]


def test_two_protocols_resolving_to_switch_do_not_share_labels():
    scatter = _problem("g3", ["4", "5", "6", "7"], [1, 0, 0, 0])
    assert scatter["state"] == "SCATTER"
    habit2 = {**HABIT, "SCATTER": "switch"}
    reqs = M.plan_requests([scatter], ["gated_habit_by_state"], gate=GATE,
                           habit_by_state=habit2)
    assert all(r["needs_label"] for r in reqs)
    keys = {M.label_key(r) for r in reqs}
    other = {M.label_key({**r, "protocol": "gated_notx"}) for r in reqs}
    assert keys.isdisjoint(other)


# ── 손잡이 파싱 ────────────────────────────────────────────────────────────────
def test_resolve_protocols_always_includes_base_and_gates_long8():
    assert M.resolve_protocols("gated_notx") == ["maj8", "gated_notx"]
    assert "long8" not in M.resolve_protocols("maj8,long8")
    assert "long8" in M.resolve_protocols("maj8", include_long8=True)


def test_load_attempt1_smoke(tmp_path):
    path = tmp_path / "texts.jsonl"
    rows = [{"group_id": "g0", "problem_id": 0, "problem": PROBLEM, "gold": "4",
             "text": SOL, "r_corr": 0, "final_answer": "5", "truncated": 0, "n_tok": 11}
            for _ in range(4)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    probs = M.load_attempt1(str(path))
    assert len(probs) == 1 and probs[0]["state"] == "ALL_SAME"
    assert probs[0]["a1_tokens"] == 44
    assert [r["roll_id"] for r in probs[0]["rows"]] == [0, 1, 2, 3]


# ── 8. --attempt1 / --gen_chunk ────────────────────────────────────────────────
def test_attempt1_path_is_an_argument_with_the_eval_pool_as_default():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--attempt1", default=M.ATTEMPT1_PATH)
    ap.add_argument("--gen_chunk", type=int, default=2048)
    a = ap.parse_args([])
    assert a.attempt1 == M.ATTEMPT1_PATH and a.gen_chunk == 2048
    b = ap.parse_args(["--attempt1", "/x/train/texts.jsonl", "--gen_chunk", "3"])
    assert b.attempt1 == "/x/train/texts.jsonl" and b.gen_chunk == 3


def test_chunk_spans_covers_everything_exactly_once():
    assert M.chunk_spans(5, 2) == [(0, 2), (2, 4), (4, 5)]
    assert M.chunk_spans(4, 4) == [(0, 4)]
    assert M.chunk_spans(0, 4) == []
    assert M.chunk_spans(3, 0) == [(0, 1), (1, 2), (2, 3)]   # size<=0 → 1


class _FakeLLM:
    """llm.generate 를 흉내 내는 가짜 — 부른 판의 크기를 기록한다."""

    def __init__(self):
        self.calls = []

    def generate(self, prompts, sps):
        self.calls.append([(p, s) for p, s in zip(prompts, sps)])
        return [f"OUT::{p}::{s}" for p, s in zip(prompts, sps)]


def test_generate_chunked_splits_and_emits_per_chunk():
    llm = _FakeLLM()
    prompts = [f"p{i}" for i in range(5)]
    sps = [f"s{i}" for i in range(5)]
    seen = []
    M.generate_chunked(llm, prompts, sps, 2, lambda lo, outs: seen.append((lo, list(outs))))
    assert [len(c) for c in llm.calls] == [2, 2, 1]
    # 덩어리마다 즉시 방출되고, 오프셋이 원래 순서를 그대로 가리킨다
    assert [lo for lo, _ in seen] == [0, 2, 4]
    flat = [o for _, outs in seen for o in outs]
    assert flat == [f"OUT::p{i}::s{i}" for i in range(5)]


def test_generate_chunked_is_identical_to_one_shot():
    one, many = _FakeLLM(), _FakeLLM()
    prompts = [f"p{i}" for i in range(7)]
    sps = [M.row_seed({"protocol": "gated_notx", "problem_id": "g1", "roll_id": i,
                       "mode": "notx"}, 11) for i in range(7)]
    a, b = [], []
    M.generate_chunked(one, prompts, sps, 999, lambda lo, o: a.extend(o))
    M.generate_chunked(many, prompts, sps, 3, lambda lo, o: b.extend(o))
    assert len(one.calls) == 1 and len(many.calls) == 3
    assert a == b                         # 시드가 행마다 미리 정해져 판 크기와 무관하다
