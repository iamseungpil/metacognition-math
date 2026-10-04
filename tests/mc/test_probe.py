"""수정 17 도구 — 습관 탐침(mc/probe.py) 채점·선택·요약 · 병합 경로 · 학습 가능성 풀의 다수결 사건."""
from __future__ import annotations

import json

import pytest

from mc import probe as PR

WRONG_PREFIX = r"so the answer is \boxed{5}"          # gold 42 기준 첫 답 오답
RIGHT_PREFIX = r"so the answer is \boxed{42}"


@pytest.mark.parametrize("prefix,cont,want", [
    (WRONG_PREFIX, " done.", dict(reopen=False, revised=False, final_correct=False, broke=True)),
    (WRONG_PREFIX, r" wait, recheck: \boxed{42}", dict(reopen=True, revised=True,
                                                         final_correct=True, broke=False)),
    (WRONG_PREFIX, r" check \boxed{5}", dict(reopen=True, revised=False,
                                              final_correct=False, broke=True)),
    (RIGHT_PREFIX, " done.", dict(reopen=False, revised=False, final_correct=True, broke=False)),
    (RIGHT_PREFIX, r" hmm \boxed{41}", dict(reopen=True, revised=True,
                                             final_correct=False, broke=True)),
])
def test_score_on_synthetic_continuations(prefix, cont, want):
    got = PR.score(prefix, cont, "42")
    assert {k: got[k] for k in want} == want
    assert got["format"] == (got["reopen"] and not got["meta"])


@pytest.mark.parametrize("cont,meta,cats", [
    (" Wait — let’s double-check the sum. \\boxed{5}", True, {"verify"}),
    (" Let’s verify: 2+3 = 5.", True, {"verify"}),                         # ’ 아포스트로피
    (" ### Final Check: why valid?", True, set()),
    (" I made a mistake in step 2.", True, {"error"}),
    (" Alternatively, use symmetry. Let me try again.", False, {"switch"}),
    (" ✅ Final Answer: \\boxed{5}", False, set()),
    ("x" * 250 + " wait", False, {"verify"}),                             # 메타는 200자, 갈래는 600자
])
def test_meta_phrase_and_categories(cont, meta, cats):
    got = PR.score(WRONG_PREFIX, cont, "42")
    assert got["meta"] is meta and {k for k in PR.META_CATS if got[f"m_{k}"]} == cats


def test_meta_metrics_conditional_on_meta(tmp_path):
    rows = [{"uid": "a", "cls": "wrong", "meta": True, "format": False, "final_correct": 1, "m_error": True},
            {"uid": "a", "cls": "wrong", "meta": False, "format": True, "final_correct": 0, "m_error": False},
            {"uid": "b", "cls": "right", "meta": True, "format": False, "broke": 1, "m_error": True}]
    per = PR.per_problem(rows)
    assert per["meta_wrong"] == {"a": 0.5} and per["fix_after_meta"] == {"a": 1.0} and per["format_wrong"] == {"a": 0.5}
    assert per["fix_after_error"] == {"a": 1.0} and per["break_after_error"] == {"b": 1.0} and per["reopen_wrong"] == {}


def _prefixes():
    out = []
    for u in range(5):                                  # 문제 5개 × (오답 3 + 정답 2)
        out += [{"uid": f"p{u}", "first_correct": False, "prefix": f"w{u}{j}"} for j in range(3)]
        out += [{"uid": f"p{u}", "first_correct": True, "prefix": f"r{u}{j}"} for j in range(2)]
    return out


def test_select_is_stratified_seeded_and_deterministic():
    sel = PR.select(_prefixes(), 7, 4, seed=11)
    assert len(sel) == 11 and [r["pid"] for r in sel] == list(range(11))
    wrong = [r for r in sel if not r["first_correct"]]
    assert len(wrong) == 7 and len({r["uid"] for r in wrong[:5]}) == 5   # 한 바퀴 = 문제마다 하나
    assert sum(r["first_correct"] for r in sel) == 4
    assert sel == PR.select(_prefixes(), 7, 4, seed=11)                    # 같은 시드 = 같은 선택
    assert PR.select(_prefixes(), 99, 99, seed=1).__len__() == 25          # 모자라면 있는 만큼


def test_summarize_is_problem_clustered_and_vs_ref_pairs(tmp_path):
    rows = [{"uid": "a", "cls": "wrong", "reopen": 1, "revised": 1, "final_correct": 1,
             "broke": 0, "n_tok": 10, "trunc": 0},
            {"uid": "a", "cls": "wrong", "reopen": 0, "revised": 0, "final_correct": 0,
             "broke": 1, "n_tok": 20, "trunc": 1},
            {"uid": "b", "cls": "wrong", "reopen": 0, "revised": 0, "final_correct": 0,
             "broke": 1, "n_tok": 30, "trunc": 0},
            {"uid": "b", "cls": "right", "reopen": 1, "revised": 1, "final_correct": 0,
             "broke": 1, "n_tok": 40, "trunc": 0}]
    s = PR.summarize(rows)
    assert s["per_problem"]["reopen_wrong"] == {"a": 0.5, "b": 0.0}
    assert s["reopen_wrong"]["delta"] == pytest.approx(0.25)             # 문제 평균
    assert s["break_right"]["delta"] == 1.0 and s["break_right"]["n"] == 1
    assert s["tokens"]["delta"] == pytest.approx((15 + 35) / 2)
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"per_problem": {**s["per_problem"],
                                               "reopen_wrong": {"a": 0.0, "b": 0.0}}}))
    d = PR.vs_ref(s, str(ref))
    assert d["reopen_wrong"]["delta"] == pytest.approx(0.25) and d["reopen_wrong"]["n"] == 2
    assert d["fix_wrong"]["delta"] == 0.0


def test_merged_model_paths(tmp_path, monkeypatch):
    from mc import rollout as R
    assert R.merged_model("/some/hf/model") == "/some/hf/model"      # 병합본·HF 경로는 그대로
    monkeypatch.setenv("WORK", str(tmp_path))
    done = tmp_path / "models" / "merged_mc_X_s2_step10"
    done.mkdir(parents=True)
    (done / "config.json").write_text("{}")
    (done / "model.safetensors").write_text("")
    actor = tmp_path / "ck" / "mc_X_s2" / "global_step_10" / "actor"
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: pytest.fail("재병합하면 안 된다"))
    assert R.merged_model(str(actor)) == str(done)                   # 있으면 재사용
    (done / "model.safetensors").unlink()
    with pytest.raises(RuntimeError):                                # 빈 산출물은 즉사
        R.merged_model(str(actor))


# ── 학습 가능성 풀: 사건은 LOO 다수결, gold 는 저장만 · 앞부분은 비선별 문제에서 ────────────
def test_pool_event_uses_loo_majority_not_gold():
    from mc import pool as P
    src = {"problem": "q", "gold": "7"}
    texts = [r"a \boxed{3} redo \boxed{9}",            # 첫 답 3: 형제 다수(9)와 다름 → 사건
             r"b \boxed{9}", r"c \boxed{9}",
             r"d \boxed{7} hmm \boxed{9}"]             # 첫 답 7 = gold 정답이지만 다수결로는 오답 → 사건
    recs = P.rollout_records(src, 0, texts, "plain")
    assert [r["event"] for r in recs] == [True, False, False, True]
    assert [r["first_correct_gold"] for r in recs] == [False, False, False, True]
    assert recs[3]["first_correct_major"] == 0.0 and recs[1]["first_correct_major"] == 1.0
    assert recs[0]["first_answer"] == "3" and recs[0]["final_answer"] == "9"
    assert len({r["uid"] for r in recs}) == 1 and recs[0]["prompt_variant"] == "plain"
    tie = P.rollout_records(src, 1, [r"\boxed{1} x \boxed{2}", r"\boxed{3}"], "plain")
    assert tie[0]["first_correct_major"] is None and tie[0]["event"] is False   # 라벨 없음


def test_pool_prefixes_cut_at_first_box_and_skip_selected():
    from mc import pool as P
    recs = (P.rollout_records({"problem": "q0", "gold": "5"}, 0, [r"x \boxed{5} y \boxed{6}"], "plain")
            + P.rollout_records({"problem": "q1", "gold": "5"}, 1,
                                [r"x \boxed{4} more \boxed{5}", "no box"], "plain"))
    pre = P.prefix_records(recs, selected={0})
    assert [r["prefix"] for r in pre] == [r"x \boxed{4}"]            # 무박스 행·선별 문제는 없다
    assert pre[0]["first_correct"] is False and pre[0]["problem"] == "q1"


def test_pool_prompt_column_passes_check_prompt_lengths(tmp_path, monkeypatch):
    """pool 이 굽는 `prompt` 컬럼(`build_math_prompt(q, PROMPT_VARIANT)`)을 트레이너 관문이 받는다."""
    import inspect

    import pandas as pd

    from mc import context as C
    from mc import pool as P
    from mc import trainer as T
    assert "build_math_prompt(str(q), variant)" in inspect.getsource(P.main)

    class _Tok:
        def encode(self, s, add_special_tokens=False):
            return list(s)

        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
            return "".join(m["content"] for m in msgs)
    qs = ["What is 1+1?", "Find x."]
    f = tmp_path / "pool.parquet"
    pd.DataFrame({"problem": qs, "prompt": [C.build_math_prompt(q, "plain") for q in qs]}).to_parquet(f)
    monkeypatch.setenv("PROMPT_VARIANT", "plain")
    assert T.check_prompt_lengths(_Tok(), str(f), 99999)["n"] == 2
    monkeypatch.setenv("PROMPT_VARIANT", "math_opt")
    with pytest.raises(ValueError):
        T.check_prompt_lengths(_Tok(), str(f), 99999)


@pytest.mark.parametrize("text,want", [
    ("so the answer is \\boxed{5}\n\nWait — check.", True),                   # 종결 표지 ∧ 줄 끝
    ("Therefore, the measure is $\\boxed{130^\\circ}$\n\nWait", True),
    ("units digit is:\n$$\n\\boxed{1}\n$$\n\n---\n\n### Step 2: tens digit", False),   # 다음 단계 제목
    ("Thus the minimum in this case is $ \\boxed{2} $ at $ x = 0 $.", False),        # 같은 줄에 본문이 이어짐
    ("Sum: 63 + 80 = \\boxed{215}\n\nBetter!", False),                           # 종결 표지 없음(탐색 중 후보)
    ("no box here", None),
])
def test_committed_first_box_rule(text, want):
    assert PR.committed_first_box(text) is want


def test_resummarize_weighted_by_prefix_key(tmp_path):
    sel = [{"pid": 0, "uid": "a", "prefix": "so \\boxed{5}", "gold": "42", "first_correct": False, "w_commit": 0.75},
           {"pid": 1, "uid": "a", "prefix": "Hence the answer is \\boxed{5}", "gold": "42", "first_correct": False,
            "w_commit": 0.25},
           {"pid": 2, "uid": "b", "prefix": "so \\boxed{5}", "gold": "42", "first_correct": False}]   # 열 없음 = 무게 0
    (tmp_path / "selection.json").write_text(json.dumps(sel))
    (tmp_path / "summary.json").write_text(json.dumps({"k": 16}))
    conts = {0: " wait \\boxed{42}", 1: " done.", 2: " wait \\boxed{42}"}
    (tmp_path / "texts.jsonl").write_text("".join(json.dumps({"pid": p, "uid": sel[p]["uid"], "cls": "wrong", "n_tok": 3,
                                                               "trunc": False, "text": c}) + "\n" for p, c in conts.items()))
    assert PR.resummarize(tmp_path, weight_key="w_commit") == 0
    s = json.loads((tmp_path / "summary_w_commit.json").read_text())
    assert s["per_problem"]["fix_wrong"] == {"a": 0.75} and s["n_rows_weighted"] == 1.0 and s["k"] == 16
    assert PR.resummarize(tmp_path) == 0 and json.loads((tmp_path / "summary.json").read_text())["per_problem"][
        "fix_wrong"] == {"a": 0.5, "b": 1.0}


def test_intermediate_box_veto():
    t = "units digit is:\n$$\n\\boxed{1}\n$$\n\n### Step 2: tens"
    _, a, b = PR.boxed_spans(t)[0]
    assert PR.intermediate_box(t, a, b)                              # 표지 없음 ∧ 다음 단계 제목
    t2 = "Therefore the answer is \\boxed{7}\n\n### Step 2"
    _, a, b = PR.boxed_spans(t2)[0]
    assert not PR.intermediate_box(t2, a, b)                         # 종결 표지가 있으면 거부 안 함
    t3 = "sum = \\boxed{5}\n\nWait — let me double-check."
    _, a, b = PR.boxed_spans(t3)[0]
    assert not PR.intermediate_box(t3, a, b)                         # 메타 말 ∧ 제목 없음


def test_score_uses_last_prefix_box_as_committed_answer():
    pre = "step \\boxed{7}\n\nso the answer is \\boxed{5}"          # 중간 박스 뒤 약속 박스에서 자른 앞부분
    assert PR.score(pre, " done.", "42")["reopen"] is False             # 이어쓰기에 박스 없음 = 안 다시 열었다
    s = PR.score(pre, " wait \\boxed{42}", "42")
    assert s["reopen"] and s["revised"] and s["final_correct"]
    assert PR.score(pre, " check \\boxed{5}", "42")["revised"] is False  # A0 = 5(마지막 앞부분 박스)
