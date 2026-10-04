"""build_self_traces 회귀 시험 (CPU, 모델 없음).

1. 만장일치 문제 → direct 한 줄(메타 없음).
2. 갈린 문제 + 자기 구조 → rescue(시도-1 + 다리 + 재시도) + direct.
3. 재시도가 투표와 어긋나면 rescue 없음.
4. 의사 라벨 동률 규칙(재시도 표가 많은 쪽).
5. 문제당 rescue 상한(최단 우선).
6. parquet 에 gold 가 한 글자도 없다.
7. 감사 수치.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import build_self_traces as B  # noqa: E402

PROBLEM = "What is 2 plus 2?"
GOLD = "4"


def _a1_row(roll_id, ans, text=None, ntok=100, truncated=0):
    return {"roll_id": roll_id, "answer": ans, "r_corr": int(ans == GOLD), "n_tok": ntok,
            "text": text if text is not None else f"work{roll_id} \\boxed{{{ans}}}",
            "truncated": truncated}


def _prob(pid, answers, texts=None):
    rows = [_a1_row(i, a, None if texts is None else texts[i]) for i, a in enumerate(answers)]
    return {"problem_id": pid, "problem": PROBLEM, "gold": GOLD, "rows": rows,
            "a1_tokens": sum(r["n_tok"] for r in rows)}


def _retry(answers, texts=None):
    return {i: {"final_answer": a, "text": (f"redo{i} \\boxed{{{a}}}" if texts is None
                                            else texts[i]), "r_corr": int(a == GOLD),
                "n_tok": 50, "mode": "notx"}
            for i, a in enumerate(answers) if a is not None}


# ── 1. 만장일치 ───────────────────────────────────────────────────────────────
def test_unanimous_gives_one_direct_without_meta():
    rows, d = B.build_problem_traces(_prob("g0", ["4"] * 8), _retry([]))
    assert d["state"] == "ALL_SAME"
    assert [r["kind"] for r in rows] == ["direct"]
    tgt = rows[0]["messages"][-1]["content"]
    assert B.META_START not in tgt and B.META_END not in tgt
    assert rows[0]["wrong_prefix"] == "" and rows[0]["scenario"] == ""
    assert rows[0]["extra_info"]["pseudo_label"] == "4"


def test_unanimous_direct_picks_shortest_non_truncated():
    p = _prob("g0", ["4"] * 3, texts=["looooooong \\boxed{4}", "s \\boxed{4}", "mid \\boxed{4}"])
    p["rows"][1]["truncated"] = 1              # 최단이지만 절단 → 뒤로
    rows, _ = B.build_problem_traces(p, {})
    assert rows[0]["messages"][-1]["content"].startswith("mid")


# ── 2·3. 갈린 문제 ────────────────────────────────────────────────────────────
def test_split_with_self_rescue_emits_rescue_and_direct():
    # 시도 1: 4,4,5,5 → 비-만장일치. 재시도: 틀린 두 행이 4 로 돌아온다 → 의사 라벨 4.
    p = _prob("g1", ["4", "4", "5", "5"])
    rows, d = B.build_problem_traces(p, _retry([None, None, "4", "4"]), max_rescue_per_problem=2)
    assert d["state"] == "SCATTER"   # K=4, top=2 → SPLIT 문턱(3) 미만이라 SCATTER
    assert d["label"] == "4"
    kinds = [r["kind"] for r in rows]
    assert kinds == ["rescue", "rescue", "direct"]
    tgt = rows[0]["messages"][-1]["content"]
    assert tgt.startswith("work2 ")                       # 시도-1 텍스트 그대로(자르지 않는다)
    assert "\\boxed{5}" in tgt.split(B.META_START)[0]     # 시도-1 의 답이 남아 있다
    assert B.bridge_text("5") in tgt                      # 다리 = 자기 답 5 를 부정
    assert tgt.endswith("redo2 \\boxed{4}")
    assert rows[0]["wrong_prefix"] == ""                  # mask_prefix=none 기본


def test_rescue_is_skipped_when_retry_disagrees_with_the_vote():
    # 재시도가 의사 라벨(4)이 아니라 7 을 내면 자기 구조가 아니다 → rescue 0.
    p = _prob("g2", ["4", "4", "4", "5"])
    rows, d = B.build_problem_traces(p, _retry([None, None, None, "7"]))
    assert d["label"] == "4" and d["n_rescue"] == 0
    assert [r["kind"] for r in rows] == ["direct"]


def test_min_a2_share_gates_rescue():
    p = _prob("g3", ["4", "4", "5", "5"])
    retr = _retry([None, None, "4", "9"])       # 재시도 중 라벨 몫 = 0.5
    assert B.build_problem_traces(p, retr, min_a2_share=0.5)[1]["n_rescue"] == 1
    assert B.build_problem_traces(p, retr, min_a2_share=0.9)[1]["n_rescue"] == 0


def test_mask_prefix_a1_sets_wrong_prefix_and_scenario():
    p = _prob("g1", ["4", "4", "5", "5"])
    rows, _ = B.build_problem_traces(p, _retry([None, None, "4", "4"]), mask_prefix="a1")
    r = rows[0]
    assert r["wrong_prefix"] == "work2 \\boxed{5}" and r["scenario"] == "redirect"
    assert rows[-1]["kind"] == "direct" and rows[-1]["wrong_prefix"] == ""


# ── 4. 동률 ───────────────────────────────────────────────────────────────────
def test_pseudo_label_tie_prefers_cluster_with_more_retry_votes():
    # 시도 1 에서 5 가 먼저 나오고 개수는 같다 → 재시도 표가 더 많은 4 가 이긴다.
    pl = B.pseudo_label(["5", "5", "4"], ["4"])
    assert pl["label"] == "4" and pl["n_a1"] == 1 and pl["n_a2"] == 1
    # 재시도 표도 같으면 먼저 나온 군집.
    assert B.pseudo_label(["5", "4"], [])["label"] == "5"
    assert B.pseudo_label([], [])["label"] == ""


def test_pseudo_label_ignores_empty_answers_and_reports_share():
    pl = B.pseudo_label(["4", "", "5"], ["4"])
    assert pl["label"] == "4" and pl["n_total"] == 3 and abs(pl["share"] - 2 / 3) < 1e-9


# ── 5. 상한 ───────────────────────────────────────────────────────────────────
def test_max_rescue_per_problem_caps_and_prefers_shortest():
    p = _prob("g4", ["4", "5", "5", "5"],
              texts=["ok \\boxed{4}", "x" * 200 + " \\boxed{5}", "short \\boxed{5}",
                     "x" * 100 + " \\boxed{5}"])
    rows, d = B.build_problem_traces(p, _retry([None, "4", "4", "4"]),
                                     max_rescue_per_problem=2)
    assert d["n_rescue"] == 2
    got = [r["messages"][-1]["content"][:5] for r in rows if r["kind"] == "rescue"]
    assert got == ["short", "x" * 5]            # 최단 두 개(short, 100자짜리)
    assert B.build_problem_traces(p, _retry([None, "4", "4", "4"]),
                                  max_rescue_per_problem=0)[1]["n_rescue"] == 0


# ── 6·7. 세 문제 합·gold 부재·감사 ────────────────────────────────────────────
def _three_problems():
    return [_prob("g0", ["4"] * 8),
            _prob("g1", ["4", "4", "5", "5"]),
            _prob("g2", ["4", "4", "4", "5"])]


def _three_retries():
    return {"g1": _retry([None, None, "4", "4"]),
            "g2": _retry([None, None, None, "7"])}


def test_no_gold_anywhere_in_parquet(tmp_path):
    import pandas as pd

    probs, retr = _three_problems(), _three_retries()
    rows, diags = [], []
    for p in probs:
        rs, d = B.build_problem_traces(p, retr.get(p["problem_id"], {}))
        for r in rs:
            r["extra_info"]["problem_id"] = p["problem_id"]
        rows.extend(rs)
        diags.append(d)
    out = tmp_path / "traces.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    df = pd.read_parquet(out)
    assert "gold" not in df.columns and "r_corr" not in df.columns
    blob = df.to_json()
    for r in df.to_dict("records"):
        assert "gold" not in r["extra_info"] and "r_corr" not in r["extra_info"]
    assert '"gold"' not in blob
    st = B.build_stats(diags, rows)
    assert st["kind_counts"] == {"rescue": 2, "direct": 3}
    assert st["by_state"]["ALL_SAME"]["direct"] == 1
    assert st["by_state"]["SCATTER"]["rescue"] == 2


def test_audit_numbers():
    probs, retr = _three_problems(), _three_retries()
    rows, diags = [], []
    for p in probs:
        rs, d = B.build_problem_traces(p, retr.get(p["problem_id"], {}))
        for r in rs:
            r["extra_info"]["problem_id"] = p["problem_id"]
        rows.extend(rs)
        diags.append(d)
    au = B.audit_stats(probs, diags, rows, retr)
    # 세 문제 모두 의사 라벨 = 4 = gold
    assert au["pseudo_label_acc_by_state"]["ALL_SAME"]["pseudo_acc"] == 1.0
    assert au["target_true_correct_by_kind"]["rescue"]["target_true_correct"] == 1.0
    assert au["target_true_correct_by_kind"]["direct"]["target_true_correct"] == 1.0
    assert au["n_retry_rows"] == 3


def test_load_retries_filters_protocol_and_labels(tmp_path):
    p = tmp_path / "gens.jsonl"
    recs = [{"protocol": "gated_notx", "problem_id": "g1", "roll_id": 2, "mode": "notx",
             "final_answer": "4", "text": "t"},
            {"protocol": "gated_notx", "problem_id": "g1", "roll_id": 2, "mode": "label",
             "final_answer": "", "text": "L"},
            {"protocol": "gated_fact", "problem_id": "g1", "roll_id": 3, "mode": "fact",
             "final_answer": "9", "text": "t"}]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    got = B.load_retries(str(p), "gated_notx")
    assert list(got) == ["g1"] and list(got["g1"]) == [2]
    assert got["g1"][2]["mode"] == "notx"


def test_bridge_template_uses_repo_meta_tokens():
    assert B.META_START == "<|meta|>" and B.META_END == "<|/meta|>"
    t = B.bridge_text("5")
    assert t.startswith("\n\n<|meta|>") and t.endswith("<|/meta|>\n\n")
    assert "\\boxed{5}" in t and "not 5" in t
    t2 = B.bridge_text("5", "[m]", "[/m]")
    assert t2.startswith("\n\n[m]") and t2.endswith("[/m]\n\n")
