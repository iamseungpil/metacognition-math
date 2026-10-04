"""설계 C «발화 = 결정, 실행 = 리셋» 회귀 시험 (CPU, 모델 없음).

1. 발화 왕복 — utterance → parse_decision(두 종류·공백·블록 없음·마지막 블록 승·
   \boxed 답 없는 restart 전용 문장).
2. decision_label(빈 답 포함, commit 불변).
3. build_decision_traces — 만장일치 전부 commit · 갈린 문제 혼합 · 마스킹 필드 ·
   gold 부재 · per_state/global 균형 상한 · \boxed 없는 시도-1 의 restart 문장.
4. math_decision_eval — 손으로 만든 4행 사례의 프로토콜 정확도·토큰 회계, AUC.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import build_decision_traces as B  # noqa: E402
import math_decision_eval as E  # noqa: E402
from src.metacot.prompt import META_END, META_START  # noqa: E402
from src.training.decision import (  # noqa: E402
    _COMMIT_TAILS, _RESTART_TAILS, decision_label, has_meta_block, parse_decision,
    utterance,
)
from src.training.sft import _should_mask_prefix  # noqa: E402

PROBLEM = "What is 2 plus 2?"
GOLD = "1729"          # ★일부러 어떤 답과도 겹치지 않는 값 — parquet 누출 검사에 쓴다.


# ── 1. 발화 왕복 ──────────────────────────────────────────────────────────────
def test_utterance_round_trip_both_kinds():
    for kind in ("commit", "restart"):
        u = utterance(kind, "42")
        got = parse_decision("solution work " + u)
        assert got == {"decision": kind, "x": "42"}


def test_parse_handles_nested_braces_and_loose_spacing():
    u = utterance("restart", "\\frac{1}{2}")
    assert parse_decision(u)["x"] == "\\frac{1}{2}"
    loose = (f"{META_START}  Decision :   my   answer  \\boxed {{7}}   may be wrong; "
             f"I restart from scratch, and the answer is not 7.{META_END}")
    assert parse_decision(loose) == {"decision": "restart", "x": "7"}


def test_parse_missing_block_is_none():
    assert parse_decision("just a solution \\boxed{4}") == {"decision": None, "x": None}
    assert parse_decision("") == {"decision": None, "x": None}
    # 메타 블록은 있으나 결정문이 아닌 경우
    assert parse_decision(f"{META_START}hmm, let me check.{META_END}")["decision"] is None


def test_parse_last_block_wins():
    text = "a" + utterance("commit", "4") + "\nmore work\n" + utterance("restart", "5")
    assert parse_decision(text) == {"decision": "restart", "x": "5"}


# ── 1c. D3: 꼬리 메타 블록 · 패러프레이즈 · 클래스 누수 ─────────────────────────
def _block(body: str) -> str:
    return META_START + body + META_END


def test_parse_skips_trailing_meta_block_without_decision():
    """결정 발화 **뒤에** 결정문이 아닌 메타 블록이 하나 더 붙어도 결정을 찾는다.
    (D3: 예전 구현은 마지막 블록만 보고 None → restart_score 0.5 중립으로 새고
    utter_gated 가 조용히 never 로 퇴화했다.)"""
    text = ("work " + utterance("commit", "42") + "\n"
            + _block("let me double-check the arithmetic once more."))
    assert parse_decision(text) == {"decision": "commit", "x": "42"}
    text_r = ("work " + utterance("restart", "42") + "\n" + _block("hmm.") + "\n"
              + _block("still thinking."))
    assert parse_decision(text_r) == {"decision": "restart", "x": "42"}


#: 패러프레이즈 목록과 **1:1** 로 대응하는 문장들(순서 동일). 목록에 표현을 추가하면
#: 여기에도 한 문장을 추가해야 테스트가 통과한다.
COMMIT_PARAPHRASES = (
    "is consistent; I commit to it.",
    "looks consistent, so I go with it.",
    "seems consistent; I commit.",
    "is correct.",
    "looks correct.",
    "seems correct.",
    "is right.",
    "; I'll keep it.",
    "; I commit to it.",
)
RESTART_PARAPHRASES = (
    "may be wrong; I restart from scratch, and the answer is not 42.",
    "might be wrong; I restart from scratch.",
    "could be wrong; I restart from scratch.",
    "may be incorrect; I start over.",
    "might be incorrect.",
    "is probably wrong.",
    "is incorrect.",
    "is not correct.",
    "is not consistent.",
    "; I'll restart from scratch.",
    "; I should restart from scratch.",
)
#: 선택 그룹(«probably|likely» 없는 꼴)까지 덮는 덧붙임.
RESTART_EXTRA = ("is wrong.", "is likely wrong.", "; I will restart.")


def test_paraphrase_lists_are_all_covered_by_tests():
    assert len(COMMIT_PARAPHRASES) == len(_COMMIT_TAILS)
    assert len(RESTART_PARAPHRASES) == len(_RESTART_TAILS)


def test_each_commit_paraphrase_parses_as_commit():
    for tail in COMMIT_PARAPHRASES:
        text = _block(f"Decision: my answer \\boxed{{42}}{'' if tail[0] == ';' else ' '}{tail}")
        assert parse_decision(text) == {"decision": "commit", "x": "42"}, tail


def test_each_restart_paraphrase_parses_as_restart():
    for tail in RESTART_PARAPHRASES + RESTART_EXTRA:
        text = _block(f"Decision: my answer \\boxed{{42}}{'' if tail[0] == ';' else ' '}{tail}")
        assert parse_decision(text) == {"decision": "restart", "x": "42"}, tail


def test_paraphrases_never_cross_classes():
    """commit 문장이 restart 로 파싱되면 게이트가 맞은 답을 버린다 — 양방향으로 막는다."""
    for tail in COMMIT_PARAPHRASES:
        text = _block(f"Decision: my answer \\boxed{{42}} {tail}")
        assert parse_decision(text)["decision"] != "restart", tail
    for tail in RESTART_PARAPHRASES + RESTART_EXTRA:
        text = _block(f"Decision: my answer \\boxed{{42}} {tail}")
        assert parse_decision(text)["decision"] != "commit", tail
    # 학습 문장 자체도(머리말·꼬리 전체) 서로 새지 않는다
    assert parse_decision(utterance("commit", "7"))["decision"] == "commit"
    assert parse_decision(utterance("restart", "7"))["decision"] == "restart"


def test_undecidable_tail_stays_none():
    """목록에 없는 꼬리는 억지로 분류하지 않는다 — None 그대로."""
    assert parse_decision(_block("Decision: my answer \\boxed{42} is 42."))["decision"] is None
    assert parse_decision(_block("I wonder about \\boxed{42} here."))["decision"] is None


def test_has_meta_block():
    assert has_meta_block(_block("anything")) is True
    assert has_meta_block("no block \\boxed{4}") is False
    assert has_meta_block("") is False


# ── 1b. 답 없는 restart (\boxed 없는 시도-1) ─────────────────────────────────────
def test_utterance_restart_empty_x_uses_nox_sentence_no_boxed():
    for empty in ("", "   "):
        u = utterance("restart", empty)
        assert "\\boxed{}" not in u and "\\boxed" not in u
        assert "did not reach a usable answer" in u
        assert u.startswith("\n\n" + META_START) and u.endswith(META_END)


def test_utterance_restart_empty_x_round_trips():
    u = utterance("restart", "")
    assert parse_decision("solution work " + u) == {"decision": "restart", "x": ""}


def test_utterance_commit_empty_x_raises():
    for bad in ("", "   ", None):
        try:
            utterance("commit", bad)
        except ValueError:
            continue
        raise AssertionError(f"commit with empty x={bad!r} should have raised ValueError")


# ── 2. 라벨 ───────────────────────────────────────────────────────────────────
def test_decision_label():
    assert decision_label("4", "4") == "commit"
    assert decision_label("4.0", "4") == "commit"        # 수학적 동치
    assert decision_label("5", "4") == "restart"
    assert decision_label("", "4") == "restart"          # 답 없는 행은 그대로 갈 수 없다
    assert decision_label("4", "") == "restart"          # 의사 라벨이 없으면 commit 불가


def test_decision_label_never_commits_on_empty_answer():
    # commit 은 utterance() 가 빈 x 를 거부하므로, 빌더가 안전하려면 decision_label 이
    # 빈 답을 절대 commit 으로 매기지 않는다는 불변이 어떤 의사 라벨에도 성립해야 한다.
    for pl in ("", "4", "0", "1729", "\\frac{1}{2}"):
        assert decision_label("", pl) == "restart"


# ── 3. 빌더 ───────────────────────────────────────────────────────────────────
def _prob(pid, answers):
    rows = [{"roll_id": i, "answer": a, "r_corr": int(a == GOLD), "n_tok": 100,
             "text": f"work{i} \\boxed{{{a}}}", "truncated": 0}
            for i, a in enumerate(answers)]
    return {"problem_id": pid, "problem": PROBLEM, "gold": GOLD, "rows": rows,
            "a1_tokens": sum(r["n_tok"] for r in rows)}


def _retry(answers):
    return {i: {"final_answer": a, "text": f"redo{i} \\boxed{{{a}}}", "n_tok": 50,
                "mode": "notx"}
            for i, a in enumerate(answers) if a is not None}


def test_unanimous_problem_gives_all_commit_one_row_per_rollout():
    rows, d = B.build_problem_rows(_prob("g0", ["4"] * 8), {})
    assert d["state"] == "ALL_SAME" and d["label"] == "4"
    assert len(rows) == 8 and {r["kind"] for r in rows} == {"commit"}
    tgt = rows[0]["messages"][-1]["content"]
    assert tgt.startswith("work0 \\boxed{4}")
    assert tgt.endswith(META_END) and "is consistent" in tgt
    assert parse_decision(tgt) == {"decision": "commit", "x": "4"}


def test_split_problem_gives_mixed_labels():
    # 시도 1: 5개 "4", 3개 "9" → 의사 라벨 "4"(재시도도 "4" 쪽을 민다)
    prob = _prob("g1", ["4", "4", "4", "4", "4", "9", "9", "9"])
    rows, d = B.build_problem_rows(prob, _retry([None] * 5 + ["4", "4", "4"]))
    assert d["state"] == "DOMINANT" and d["label"] == "4"
    kinds = [r["kind"] for r in rows]
    assert kinds == ["commit"] * 5 + ["restart"] * 3
    tgt = rows[-1]["messages"][-1]["content"]
    assert parse_decision(tgt) == {"decision": "restart", "x": "9"}


def test_masking_fields_put_loss_on_the_utterance_only():
    rows, _ = B.build_problem_rows(_prob("g0", ["4", "4"]), {})
    r = rows[0]
    assert r["scenario"] == "redirect"
    assert r["wrong_prefix"] == "work0 \\boxed{4}"
    # 손실이 붙는 부분은 target 에서 wrong_prefix 를 뺀 꼬리 = 발화뿐이다.
    assert r["messages"][-1]["content"][len(r["wrong_prefix"]):] == utterance("commit", "4")
    assert _should_mask_prefix(r["wrong_prefix"], r["scenario"]) is True


def test_no_gold_anywhere_in_rows():
    rows, _ = B.build_problem_rows(_prob("g2", ["4", "9"]), _retry(["4", "4"]))
    blob = repr(rows)
    assert GOLD not in blob
    for r in rows:
        assert set(r) == {"messages", "wrong_prefix", "scenario", "kind", "extra_info"}
        assert set(r["extra_info"]) >= {"agree_state", "pseudo_label", "share",
                                        "problem_id", "label"}


def test_build_problem_rows_empty_a1_answer_gets_nox_sentence():
    # roll 0 은 \boxed 답이 없는(잘리거나 못 낸) 시도-1 — 실제 빌드에서 SCATTER/SPLIT/
    # DOMINANT/NOANS 에 섞여 나온 1,157행짜리 결함의 축소판.
    prob = {"problem_id": "gnox", "problem": PROBLEM, "gold": GOLD,
            "rows": [{"roll_id": 0, "answer": "", "r_corr": 0, "n_tok": 100,
                      "text": "I ran out of steps before finishing.", "truncated": 1},
                     {"roll_id": 1, "answer": "4", "r_corr": 0, "n_tok": 100,
                      "text": "work1 \\boxed{4}", "truncated": 0}],
            "a1_tokens": 200}
    rows, diag = B.build_problem_rows(prob, {})
    empty_row = next(r for r in rows if r["extra_info"]["roll_id"] == 0)
    assert empty_row["kind"] == "restart"
    tgt = empty_row["messages"][-1]["content"]
    utter = tgt[len(empty_row["wrong_prefix"]):]
    assert "\\boxed" not in utter                    # 말이 안 되는 \boxed{} 가 없다
    assert "did not reach a usable answer" in utter
    assert parse_decision(tgt) == {"decision": "restart", "x": ""}
    other_row = next(r for r in rows if r["extra_info"]["roll_id"] == 1)
    assert other_row["kind"] == "commit"              # 답 있는 행은 그대로 라벨링된다


def test_balance_per_state_caps_commit_at_three_times_restart():
    rows: list[dict] = []
    # DOMINANT 상태: commit 7 · restart 1 → commit 3 만 남는다.
    rows += B.build_problem_rows(_prob("g3", ["4"] * 7 + ["9"]), _retry([None] * 7 + ["4"]))[0]
    kept, rep = B.balance_rows(rows, "per_state", seed=0)
    assert rep["DOMINANT"] == {"restart": 1, "kept_commit": 3, "dropped_commit": 4}
    assert sum(1 for r in kept if r["kind"] == "commit") == 3
    assert sum(1 for r in kept if r["kind"] == "restart") == 1
    # 결정적이어야 한다 — 같은 시드면 같은 행 집합
    kept2, _ = B.balance_rows(rows, "per_state", seed=0)
    assert [r["extra_info"]["roll_id"] for r in kept] == \
           [r["extra_info"]["roll_id"] for r in kept2]
    assert B.balance_rows(rows, "none")[0] == rows


# ── 3b. --balance global ─────────────────────────────────────────────────────
def _grow(pid: str, state: str, kind: str, roll: int = 0) -> dict:
    return {"kind": kind, "extra_info": {"agree_state": state, "problem_id": pid,
                                         "roll_id": roll}}


def _global_case() -> list[dict]:
    """ALL_SAME(restart 0, commit 30/10문제) · DOMINANT(commit 8/4문제, restart 4) ·
    SCATTER(commit 2/2문제, restart 4) → 총 commit 40, 총 restart 8."""
    rows: list[dict] = []
    for i in range(10):
        for j in range(3):
            rows.append(_grow(f"as{i}", "ALL_SAME", "commit", j))
    for i in range(4):
        rows.append(_grow(f"dom{i}", "DOMINANT", "commit", 0))
        rows.append(_grow(f"dom{i}", "DOMINANT", "commit", 1))
        rows.append(_grow(f"dom{i}", "DOMINANT", "restart", 2))
    for i in range(2):
        rows.append(_grow(f"sc{i}", "SCATTER", "commit", 0))
        rows.append(_grow(f"sc{i}", "SCATTER", "restart", 1))
        rows.append(_grow(f"sc{i}", "SCATTER", "restart", 2))
    return rows


def test_global_balance_keeps_all_restart_rows():
    rows = _global_case()
    kept, rep = B.balance_rows(rows, "global", seed=0)
    n_before = sum(1 for r in rows if r["kind"] == "restart")
    n_after = sum(1 for r in kept if r["kind"] == "restart")
    assert n_before == n_after == 8 == rep["n_restart"]


def test_global_balance_caps_total_commit_at_ratio_times_restart():
    rows = _global_case()
    kept, rep = B.balance_rows(rows, "global", seed=0, balance_ratio=3.0)
    n_commit = sum(1 for r in kept if r["kind"] == "commit")
    assert n_commit == rep["n_commit_kept"] == 24         # cap = int(3.0 * 8)
    assert n_commit <= 3.0 * rep["n_restart"]
    assert rep["n_commit_before"] == 40


def test_global_balance_allocates_proportional_to_commit_share():
    rows = _global_case()
    _, rep = B.balance_rows(rows, "global", seed=0)
    bs = rep["by_state"]
    # 점유율 ALL_SAME 30/40, DOMINANT 8/40, SCATTER 2/40 → 최대잔여법으로 18/5/1.
    assert bs["ALL_SAME"] == {"commit_before": 30, "commit_kept": 18, "restart": 0}
    assert bs["DOMINANT"] == {"commit_before": 8, "commit_kept": 5, "restart": 4}
    assert bs["SCATTER"] == {"commit_before": 2, "commit_kept": 1, "restart": 4}


def test_global_balance_deterministic_and_seed_sensitive():
    rows = _global_case()

    def ids(kept):
        return sorted((r["extra_info"]["problem_id"], r["extra_info"]["roll_id"])
                      for r in kept)

    kept_a, _ = B.balance_rows(rows, "global", seed=0)
    kept_a2, _ = B.balance_rows(rows, "global", seed=0)
    assert ids(kept_a) == ids(kept_a2)                    # 같은 시드 → 같은 집합
    kept_b, _ = B.balance_rows(rows, "global", seed=1)
    assert ids(kept_b) != ids(kept_a)                     # 다른 시드 → 다른 집합


def test_global_balance_spreads_across_problems_before_seconds():
    rows = _global_case()
    kept, _ = B.balance_rows(rows, "global", seed=0)

    def counts(state):
        c: dict[str, int] = {}
        for r in kept:
            if r["extra_info"]["agree_state"] == state and r["kind"] == "commit":
                pid = r["extra_info"]["problem_id"]
                c[pid] = c.get(pid, 0) + 1
        return c

    dom = counts("DOMINANT")
    assert len(dom) == 4                                  # 문제 4개 전부 최소 1개는 남는다
    assert sorted(dom.values(), reverse=True) == [2, 1, 1, 1]
    all_same = counts("ALL_SAME")
    assert len(all_same) == 10                             # 문제 10개 전부 최소 1개는 남는다
    assert sorted(all_same.values(), reverse=True) == [2] * 8 + [1] * 2


def test_global_balance_stats_internally_consistent():
    rows = _global_case()
    kept, rep = B.balance_rows(rows, "global", seed=0)
    bs = rep["by_state"]
    assert sum(d["commit_kept"] for d in bs.values()) == rep["n_commit_kept"]
    assert sum(d["restart"] for d in bs.values()) == rep["n_restart"]
    assert rep["n_commit_kept"] <= rep["ratio"] * rep["n_restart"]
    assert rep["n_commit_kept"] <= rep["n_commit_before"]
    assert len(kept) == rep["n_commit_kept"] + rep["n_restart"]


def test_global_balance_noop_when_cap_exceeds_commit():
    rows = _global_case()
    kept, rep = B.balance_rows(rows, "global", seed=0, balance_ratio=100.0)
    assert rep["n_commit_kept"] == rep["n_commit_before"] == 40
    assert len(kept) == len(rows)


def test_none_and_per_state_unaffected_by_global_addition():
    rows = _global_case()
    assert "global" in B.BALANCE_MODES
    assert B.balance_rows(rows, "none")[0] == rows
    kept, rep = B.balance_rows(rows, "per_state", seed=0)
    # ALL_SAME 은 자기 상태 restart 가 0 이라 per_state 에서는 통째로 비워진다(기존 동작).
    assert rep["ALL_SAME"] == {"restart": 0, "kept_commit": 0, "dropped_commit": 30}
    assert sum(1 for r in kept if r["extra_info"]["agree_state"] == "ALL_SAME") == 0


# ── 4. 평가 회계 ──────────────────────────────────────────────────────────────
def _row(roll, a1, ok1, dec, a2, ok2, t1=100, t2=50):
    return {"roll_id": roll, "a1_answer": a1, "a1_correct": ok1, "a1_tokens": t1,
            "decision": dec, "x": a1, "a2_answer": a2, "a2_correct": ok2, "a2_tokens": t2}


def _case():
    """2문제 × 2행. P1 = ALL_SAME(둘 다 정답), P2 = SCATTER(둘 다 오답, 재시도가 구제)."""
    return [
        {"problem_id": "p1", "state": "ALL_SAME", "rows": [
            _row(0, "4", 1, "commit", "5", 0), _row(1, "4", 1, "restart", "5", 0)]},
        {"problem_id": "p2", "state": "SCATTER", "rows": [
            _row(0, "3", 0, "restart", "4", 1), _row(1, "5", 0, "commit", "4", 1)]},
    ]


def test_protocol_accounting_and_tokens():
    p = _case()
    r = {name: E.score_protocol(p, name) for name in E.PROTOCOLS}
    assert r["never"]["pass1"] == 0.5 and r["never"]["tokens_per_problem"] == 200
    assert r["never"]["maj_k"] == 0.5 and r["never"]["n_retried_rows"] == 0
    # 발화 게이트: 맞은 행 하나를 잘못 버리고(flip_wrong) 틀린 행 하나를 구한다
    assert r["utter_gated"]["pass1"] == 0.5
    assert r["utter_gated"]["tokens_per_problem"] == 250
    assert r["utter_gated"]["flip_wrong"] == 1
    assert r["utter_gated"]["maj_k"] == 1.0          # 동률이면 먼저 나온 군집이 이긴다
    assert r["always"]["pass1"] == 0.5 and r["always"]["tokens_per_problem"] == 300
    assert r["always"]["flip_wrong"] == 2 and r["always"]["maj_k"] == 0.5
    assert r["agree_gated"]["pass1"] == 1.0
    assert r["agree_gated"]["tokens_per_problem"] == 250
    assert r["agree_gated"]["flip_wrong"] == 0


def test_no_retry_row_falls_back_to_attempt1():
    p = _case()
    for r in p[1]["rows"]:
        r["a2_answer"] = None
    assert E.score_protocol(p, "always")["pass1"] == 0.0     # p1 재시도만 쓰고 둘 다 오답
    assert E.score_protocol(p, "agree_gated")["n_retried_rows"] == 0


def test_auc_within_problem_and_pooled():
    probs = [
        # 완벽한 판별 — 맞은 행은 commit, 틀린 행은 restart
        {"problem_id": "a", "state": "SPLIT", "rows": [
            _row(0, "4", 1, "commit", "4", 1), _row(1, "9", 0, "restart", "4", 1)]},
        # 완전히 뒤집힌 판별
        {"problem_id": "b", "state": "SPLIT", "rows": [
            _row(0, "4", 1, "restart", "9", 0), _row(1, "9", 0, "commit", "9", 0)]},
    ]
    rep = E.auc_report(probs)
    assert rep["n_problems_scored"] == 2
    assert rep["within_problem_auc"] == 0.5
    assert rep["pooled_auc"] == 0.5


def test_auc_is_half_when_no_utterance_and_nan_without_both_classes():
    probs = [{"problem_id": "a", "state": "SPLIT", "rows": [
        _row(0, "4", 1, None, "4", 1), _row(1, "9", 0, None, "4", 1)]}]
    assert E.auc_report(probs)["within_problem_auc"] == 0.5     # 동점 → .5
    one = [{"problem_id": "a", "state": "ALL_SAME", "rows": [
        _row(0, "4", 1, "commit", "4", 1), _row(1, "4", 1, "restart", "4", 1)]}]
    rep = E.auc_report(one)
    assert rep["n_problems_scored"] == 0
    assert rep["within_problem_auc"] != rep["within_problem_auc"]   # nan


def test_emission_report():
    p = _case()
    p[0]["rows"][0]["decision"] = None
    e = E.emission_report(p)
    assert e["emission_rate"] == 0.75
    assert e["by_state"]["ALL_SAME"]["restart_rate_of_all"] == 0.5
    assert e["by_state"]["SCATTER"]["emission_rate"] == 1.0


# ── 4b. D3 버킷 계수 ─────────────────────────────────────────────────────────
def test_parse_report_buckets_split_emitted_from_parsed():
    """세 행: ①블록+결정 파싱 ②블록은 있는데 결정 없음(위험 버킷) ③아무것도 없음."""
    rows = [dict(_row(0, "4", 1, "commit", "5", 0), has_block=1),
            dict(_row(1, "4", 0, None, "5", 0), has_block=1),
            dict(_row(2, "4", 0, None, "5", 0), has_block=0)]
    rep = E.parse_report([{"problem_id": "p", "state": "SCATTER", "rows": rows}])
    assert rep["n_rows"] == 3
    assert rep["n_emitted_block"] == 2
    assert rep["n_decision_parsed"] == 1
    assert rep["n_block_without_decision"] == 1
    assert rep["emitted_block_rate"] == 2 / 3
    assert rep["decision_parsed_rate"] == 1 / 3
    assert rep["block_without_decision_rate"] == 1 / 3
    # 위험 버킷은 여전히 중립 0.5 로 채점된다(보고만 늘린다)
    assert E.restart_score(rows[1]) == 0.5


# ── 4c. D4 degraded 표식 ─────────────────────────────────────────────────────
def test_notx_marks_degraded_only_when_x_and_last_boxed_are_both_empty():
    assert E.notx_for_row({"x": "7", "a1_answer": "4"}) == ("7", 0)
    assert E.notx_for_row({"x": "", "a1_answer": "4"}) == ("4", 0)    # last_boxed 로 되돌림
    assert E.notx_for_row({"x": None, "a1_answer": "4"}) == ("4", 0)
    assert E.notx_for_row({"x": "", "a1_answer": ""}) == ("", 1)      # 배제 절이 사라진다
    assert E.notx_for_row({"x": None, "a1_answer": None}) == ("", 1)


def test_degraded_report_counts_overall_and_per_state():
    p = _case()
    p[0]["rows"][0]["a2_degraded"] = 1        # ALL_SAME 한 행이 degraded
    rep = E.degraded_report(p)
    assert rep["n_retries"] == 4 and rep["n_degraded_retries"] == 1
    assert rep["degraded_rate"] == 0.25
    assert rep["by_state"]["ALL_SAME"] == {"n_retry": 2, "n_degraded": 1,
                                           "degraded_rate": 0.5}
    assert rep["by_state"]["SCATTER"]["n_degraded"] == 0


def test_score_protocol_counts_degraded_retries_it_actually_used():
    p = _case()
    for r in p[0]["rows"] + p[1]["rows"]:
        r["a2_degraded"] = 1
    assert E.score_protocol(p, "never")["n_degraded_retries"] == 0
    assert E.score_protocol(p, "always")["n_degraded_retries"] == 4
    assert E.score_protocol(p, "utter_gated")["n_degraded_retries"] == 2
    assert E.score_protocol(p, "utter_gated")["degraded_rate"] == 1.0


# ── 4d. D5 빈 재시도 되돌림 vs strict ────────────────────────────────────────
def _empty_retry_case():
    """맞은 시도-1(정답 "4") + restart 발화 + **답이 없는**(절단) 재시도 한 행."""
    row = _row(0, "4", 1, "restart", "", 0, t1=100, t2=50)
    row["a2_truncated"] = 1
    return [{"problem_id": "p1", "state": "SCATTER", "rows": [row]}]


def test_empty_retry_falls_back_to_attempt1_but_strict_marks_it_wrong():
    p = _empty_retry_case()
    fb = E.score_protocol(p, "utter_gated")
    st = E.score_protocol(p, "utter_gated", strict=True)
    assert fb["pass1"] == 1.0 and fb["flip_wrong"] == 0     # 맞은 답을 지키다
    assert st["pass1"] == 0.0 and st["flip_wrong"] == 1     # 옛 동작: 덮어써서 오답
    assert fb["n_fallback_to_a1"] == 1 and st["n_fallback_to_a1"] == 0
    for r in (fb, st):
        assert r["n_retried_rows"] == 1                     # 재시도는 어느 쪽이든 굴렸다
        assert r["n_a2_no_answer"] == 1 and r["n_a2_truncated"] == 1
    # ★토큰 회계는 두 변종이 **동일**하다
    assert fb["total_tokens"] == st["total_tokens"] == 150
    assert fb["tokens_per_problem"] == st["tokens_per_problem"] == 150
    assert fb["protocol"] == "utter_gated" and st["protocol"] == "utter_gated__strict"


def test_missing_retry_is_not_counted_as_empty_retry():
    """재시도를 **굴리지 않은** 행(a2_answer is None)은 no_answer 버킷이 아니다."""
    p = _empty_retry_case()
    p[0]["rows"][0]["a2_answer"] = None
    r = E.score_protocol(p, "utter_gated")
    assert r["n_retried_rows"] == 0 and r["n_a2_no_answer"] == 0
    assert r["pass1"] == 1.0 and r["total_tokens"] == 100


def test_summarize_reports_both_variants_with_equal_tokens():
    summ = E.summarize(_empty_retry_case())
    names = set(summ["protocols"])
    assert names == set(E.PROTOCOLS) | {f"{p}__strict" for p in E.PROTOCOLS if p != "never"}
    assert "never__strict" not in names
    for name in ("utter_gated", "always", "agree_gated"):
        a, b = summ["protocols"][name], summ["protocols"][f"{name}__strict"]
        assert a["total_tokens"] == b["total_tokens"]
    assert summ["parse"]["n_block_without_decision"] == 0
    assert summ["degraded"]["n_retries"] == 1
    md = E.to_markdown(summ)
    assert "utter_gated__strict" in md and "블록 있는데 결정 없음" in md
