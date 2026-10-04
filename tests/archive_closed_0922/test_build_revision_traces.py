"""build_revision_traces 회귀 시험 (CPU, 모델 없음).

1. 깔때기가 제외 부류를 **각각 제 이유로** 떨어뜨린다(박스 1개·첫=끝·변경점 2개·상태·절단·마커).
2. majority 모드는 gold 를 한 번도 읽지 않는다.
3. 마스크 경계 — target 이 wrong_prefix 로 시작하고, wrong_prefix 는 **첫 박스 끝**에서
   정확히 끝나며, sft._should_mask_prefix 가 True 다.
4. 문제당 상한 + 최단 우선.
5. parquet 왕복에 gold·r_corr 가 (직렬화된 messages 안까지) 한 글자도 없다.
6. 감사 수치(gold 는 여기서만 읽힌다).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

import build_revision_traces as B  # noqa: E402
from src.training import sft  # noqa: E402

PROBLEM = "What is 2 plus 2?"
GOLD = "4"
MARK = " wait, let me re-check. "
NOMARK = " and then, continuing on. "


def _mk_text(answers, mid=MARK, pad=""):
    parts = []
    for i, a in enumerate(answers):
        parts.append(f"step{i} \\boxed{{{a}}}")
        if i == 0:
            parts.append(mid + pad)
    return "".join(parts)


def _roll(roll_id, answers, *, truncated=0, mid=MARK, pad=""):
    text = _mk_text(answers, mid, pad)
    return {"roll_id": roll_id, "answer": str(answers[-1]), "r_corr": int(answers[-1] == GOLD),
            "n_tok": 100, "text": text, "truncated": truncated}


def _prob(pid, rolls, gold=GOLD):
    return {"problem_id": pid, "problem": PROBLEM, "gold": gold, "rows": rolls}


class _GoldTrap(dict):
    """`gold` 를 읽으면 즉사하는 문제 dict — majority 모드의 gold 무접촉을 강제한다."""

    def __getitem__(self, key):
        assert key != "gold", "majority 모드가 gold 를 읽었다"
        return super().__getitem__(key)

    def get(self, key, default=None):
        assert key != "gold", "majority 모드가 gold 를 읽었다"
        return super().get(key, default)


# ── 0. 답 계열 · 다수결 · 마커 ────────────────────────────────────────────────
def test_answer_runs_collapses_equivalent_neighbours():
    assert B.answer_runs(["4", "4", "5", "5"]) == ["4", "5"]          # 변경점 1
    assert B.answer_runs(["5", "6", "4"]) == ["5", "6", "4"]          # 변경점 2
    assert B.answer_runs(["4", "4.0"]) == ["4"]                       # 동치는 변경이 아니다
    assert B.answer_runs(["4"]) == ["4"]


def test_majority_prefers_first_cluster_on_tie_and_ignores_empty():
    # majority_answer 는 dedup 되어 agreement_state(...)["dominant_answer"] 를 직접 쓴다.
    dom = lambda xs: B.agreement_state(xs, k=len(xs))["dominant_answer"]  # noqa: E731
    assert dom(["5", "4", "4", ""]) == "4"
    assert dom(["5", "4"]) == "5"
    assert dom(["", ""]) == ""


def test_find_marker_is_case_insensitive_and_documented():
    assert B.find_marker("Wait, that is off") == "wait"
    assert B.find_marker("nothing to see here") == ""
    assert "double-check" in B.REVISION_MARKERS and "hold on" in B.REVISION_MARKERS


# ── 1. 깔때기 ─────────────────────────────────────────────────────────────────
def _funnel_problem():
    """DOMINANT(답 4 가 7/8) 문제 — 제외 부류를 한 줄씩 심는다."""
    return _prob("g0", [
        _roll(0, ["5", "4"]),                       # 교사 (유지)
        _roll(1, ["4"]),                            # 박스 1개
        _roll(2, ["4", "4"]),                       # 첫 == 끝
        _roll(3, ["5", "6", "4"]),                  # 변경점 2 (두 번째 수정)
        _roll(4, ["5", "4"], truncated=1),          # 절단
        _roll(5, ["5", "9"]),                       # 다수결(4)로 나아지지 않았다
        _roll(6, ["4"]), _roll(7, ["4"]),           # 박스 1개
    ])


def test_funnel_drops_each_class_for_its_own_reason():
    rows, d = B.build_problem_rows(_funnel_problem(), label_mode="majority")
    assert d["state"] == "DOMINANT"
    assert d["n_rollouts"] == 8
    assert d["drop_truncated"] == 1
    assert d["drop_lt2_boxes"] == 3
    assert d["drop_first_equiv_last"] == 1
    assert d["drop_multi_change"] == 1 and d["second_revisions"] == 1
    assert d["drop_not_improved"] == 1
    assert d["drop_state"] == 0 and d["drop_no_marker"] == 0 and d["drop_cap"] == 0
    assert d["n_kept"] == 1 and len(rows) == 1
    e = rows[0]["extra_info"]
    assert (e["roll_id"], e["first_answer"], e["last_answer"], e["n_boxes"]) == (0, "5", "4", 2)
    assert e["agree_state"] == "DOMINANT" and e["label_mode"] == "majority"
    assert rows[0]["kind"] == "revision" and rows[0]["scenario"] == "redirect"


def test_excluded_state_drops_every_row_and_is_counted():
    # 4×3, 5×3, 6, 7 → top=3 → SPLIT (수정이 순해로운 상태) → 기본 --states 에서 제외.
    rolls = [_roll(0, ["5", "4"]), _roll(1, ["4"]), _roll(2, ["4"]),
             _roll(3, ["5"]), _roll(4, ["5"]), _roll(5, ["5"]),
             _roll(6, ["6"]), _roll(7, ["7"])]
    rows, d = B.build_problem_rows(_prob("g1", rolls), label_mode="majority")
    assert d["state"] == "SPLIT"
    assert d["drop_state"] == 1 and d["n_kept"] == 0 and rows == []
    # 그 상태를 명시적으로 허용하면 같은 행이 살아난다.
    rows2, d2 = B.build_problem_rows(_prob("g1", rolls), label_mode="majority",
                                     states=("SPLIT",))
    assert d2["drop_state"] == 0 and d2["n_kept"] == 1 and len(rows2) == 1


def test_require_marker_filters_but_counts_are_reported_either_way():
    rolls = [_roll(0, ["5", "4"], mid=MARK), _roll(1, ["5", "4"], mid=NOMARK)] + \
            [_roll(i, ["4"]) for i in range(2, 8)]
    rows, d = B.build_problem_rows(_prob("g2", rolls), label_mode="majority")
    assert d["n_candidates"] == 2 and d["n_candidates_with_marker"] == 1
    assert d["drop_no_marker"] == 0 and d["n_kept"] == 2
    assert sorted(r["extra_info"]["has_marker"] for r in rows) == [False, True]
    rows_m, dm = B.build_problem_rows(_prob("g2", rolls), label_mode="majority",
                                      require_marker=True)
    assert dm["n_candidates"] == 2 and dm["n_candidates_with_marker"] == 1
    assert dm["drop_no_marker"] == 1 and dm["n_kept"] == 1
    assert rows_m[0]["extra_info"]["marker"] == "wait"
    st = B.build_stats([dm], rows_m)
    assert st["marker"] == {**st["marker"], "n_candidates": 2, "n_with_marker": 1,
                            "rate": 0.5, "n_kept_if_required": 1, "n_kept_if_not_required": 2}


def test_gold_label_mode_uses_grading_not_the_vote():
    # 다수결은 9 지만 gold 는 4 — gold 모드는 5→4 를 잡고 4→9 를 버린다.
    rolls = [_roll(0, ["5", "4"]), _roll(1, ["4", "9"])] + [_roll(i, ["9"]) for i in range(2, 8)]
    rows_g, dg = B.build_problem_rows(_prob("g3", rolls), label_mode="gold",
                                      states=("ALL_SAME", "DOMINANT"))
    assert dg["state"] == "DOMINANT" and dg["n_kept"] == 1
    assert rows_g[0]["extra_info"]["last_answer"] == "4"
    assert rows_g[0]["extra_info"]["label_mode"] == "gold"
    rows_m, dm = B.build_problem_rows(_prob("g3", rolls), label_mode="majority")
    assert dm["n_kept"] == 1 and rows_m[0]["extra_info"]["last_answer"] == "9"


# ── 2. gold 무접촉 ────────────────────────────────────────────────────────────
def test_majority_mode_never_reads_gold():
    p = _GoldTrap(_funnel_problem())
    rows, d = B.build_problem_rows(p, label_mode="majority")
    assert d["n_kept"] == 1
    assert "gold" not in rows[0]["extra_info"]
    # 같은 dict 를 gold 모드로 부르면 덫이 터진다 — 덫이 실제로 작동함을 보인다.
    try:
        B.build_problem_rows(_GoldTrap(_funnel_problem()), label_mode="gold")
    except AssertionError as exc:
        assert "gold" in str(exc)
    else:
        raise AssertionError("gold 모드가 gold 를 안 읽었다 — 덫이 죽어 있다")


def test_unknown_label_mode_dies():
    try:
        B.build_problem_rows(_funnel_problem(), label_mode="oracle")
    except ValueError as exc:
        assert "oracle" in str(exc)
    else:
        raise AssertionError("알 수 없는 --label 이 통과했다")


# ── 3. 마스크 경계 ────────────────────────────────────────────────────────────
def test_mask_boundary_is_exactly_the_end_of_the_first_box():
    rows, _ = B.build_problem_rows(_funnel_problem(), label_mode="majority")
    r = rows[0]
    target = r["messages"][-1]["content"]
    prefix = r["wrong_prefix"]
    assert target == _mk_text(["5", "4"])                 # 원문 그대로(자르지 않는다)
    assert target.startswith(prefix)
    assert prefix == "step0 \\boxed{5}"                   # 첫 박스의 닫는 중괄호까지
    assert prefix.endswith("\\boxed{5}")
    # 경계 = boxed_spans 의 첫 박스 끝 오프셋
    from src.training.math_meta import boxed_spans
    assert len(prefix) == boxed_spans(target)[0][2]
    # 손실은 수정 구간부터 — sft 가 실제로 이 머리를 마스크한다.
    assert sft._should_mask_prefix(prefix, r["scenario"]) is True
    assert target[len(prefix):].startswith(MARK)


def test_seg_length_unit_falls_back_to_chars_without_a_tokenizer():
    rows, _ = B.build_problem_rows(_funnel_problem(), label_mode="majority")
    e = rows[0]["extra_info"]
    assert "seg_tokens" not in e
    target = rows[0]["messages"][-1]["content"]
    assert e["seg_chars"] == len(target) - len(rows[0]["wrong_prefix"])

    class _Tok:
        def encode(self, text, add_special_tokens=False):
            return text.split()

    rows2, _ = B.build_problem_rows(_funnel_problem(), label_mode="majority", tokenizer=_Tok())
    e2 = rows2[0]["extra_info"]
    assert "seg_chars" not in e2 and e2["seg_tokens"] == len(MARK.split()) + 2


# ── 4. 상한 · 최단 우선 ───────────────────────────────────────────────────────
def test_cap_keeps_the_shortest_responses_first():
    rolls = [_roll(0, ["5", "4"], pad="x" * 300), _roll(1, ["5", "4"], pad="x" * 100),
             _roll(2, ["5", "4"]), _roll(3, ["5", "4"], pad="x" * 200)] + \
            [_roll(i, ["4"]) for i in range(4, 8)]
    rows, d = B.build_problem_rows(_prob("g4", rolls), label_mode="majority", max_per_problem=2)
    assert d["n_candidates"] == 4 and d["drop_cap"] == 2 and d["n_kept"] == 2
    assert [r["extra_info"]["roll_id"] for r in rows] == [2, 1]      # 최단 → 다음 최단
    rows0, d0 = B.build_problem_rows(_prob("g4", rolls), label_mode="majority",
                                     max_per_problem=0)
    assert rows0 == [] and d0["drop_cap"] == 4


# ── 5. parquet 왕복 ───────────────────────────────────────────────────────────
def _two_problems():
    good = _prob("g0", [_roll(0, ["5", "4"])] + [_roll(i, ["4"]) for i in range(1, 8)])
    bad = _prob("g1", [_roll(0, ["4", "9"])] + [_roll(i, ["9"]) for i in range(1, 8)])
    return [good, bad]


def _build_all(problems, **kw):
    mode = kw.pop("label_mode", "majority")
    rows, diags, lengths = [], [], []
    for p in problems:
        rs, d = B.build_problem_rows(p, label_mode=mode, **kw)
        rows.extend(rs)
        diags.append(d)
        lengths.extend(len(r["messages"][-1]["content"]) for r in rs)
    return rows, diags, lengths


def test_parquet_round_trip_has_no_gold_or_r_corr(tmp_path):
    import pandas as pd

    rows, diags, lengths = _build_all(_two_problems())
    assert len(rows) == 2
    out = tmp_path / "traces.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    df = pd.read_parquet(out)
    assert set(df.columns) == {"messages", "wrong_prefix", "scenario", "kind", "extra_info"}
    assert "gold" not in df.columns and "r_corr" not in df.columns
    for rec in df.to_dict("records"):
        assert "gold" not in rec["extra_info"] and "r_corr" not in rec["extra_info"]
        assert rec["messages"][-1]["content"].startswith(rec["wrong_prefix"])
        assert sft._should_mask_prefix(rec["wrong_prefix"], rec["scenario"]) is True
    blob = df.to_json()
    assert "gold" not in blob and "r_corr" not in blob     # 직렬화된 messages 안까지

    st = B.build_stats(diags, rows, unit="chars", lengths=lengths)
    assert st["n_problems"] == 2 and st["n_rows"] == 2 and st["n_problems_represented"] == 2
    assert st["kept_by_state"] == {"ALL_SAME": 2}   # 8개 답이 모두 같은 문제들
    assert st["funnel"]["n_rollouts"] == 16 and st["funnel"]["n_kept"] == 2
    assert st["funnel"]["drop_lt2_boxes"] == 14
    assert st["response_len"]["unit"] == "chars" and st["response_len"]["p50"] >= 1
    assert B.funnel_line(st, label_mode="majority", require_marker=False).startswith("[revision]")


# ── 6. 감사 ───────────────────────────────────────────────────────────────────
def test_audit_reads_gold_and_scores_kept_traces():
    problems = _two_problems()               # g0: 5→4 (gold 4) · g1: 4→9 (gold 4)
    rows, _, _ = _build_all(problems)
    au = B.audit_stats(problems, rows)
    c = au["by_state"]["ALL_SAME"]
    assert c["n"] == 2 and c["n_last_correct"] == 1 and c["n_first_correct"] == 1
    assert c["precision"] == 0.5 and c["true_improvement"] == 0.5
    assert au["precision"] == 0.5 and au["n_rows"] == 2
    # gold 모드로 뽑으면 정밀도는 구성상 1.0 이다(감사가 라벨과 어긋나면 그게 버그다).
    rows_g, _, _ = _build_all(_two_problems(), label_mode="gold")
    au_g = B.audit_stats(problems, rows_g)
    assert au_g["n_rows"] == 1 and au_g["precision"] == 1.0
    assert au_g["by_state"]["ALL_SAME"]["true_improvement"] == 1.0
