"""pool.filter_pass_rate — 중간 난이도 필터 단위 테스트."""
import pandas as pd
import pytest
from mc.pool import filter_pass_rate


def test_filter_pass_rate_bounds_and_default():
    df = pd.DataFrame({"extra_info": [{"group_pass_rate": p} for p in (0.1, 0.5, 0.9)]})
    assert filter_pass_rate(df, None, None) is df               # 기본값 무변화(동일 객체)
    out = filter_pass_rate(df, 0.2, 0.75)
    assert list(out["extra_info"].apply(lambda e: e["group_pass_rate"])) == [0.5]


def test_learnability_score_is_event_variance():
    from mc.pool import learnability
    k = 4
    texts = ([{"event": e} for e in (True, True, True, True)]      # p=1 → 0
             + [{"event": e} for e in (False, False, False, False)]  # p=0 → 0
             + [{"event": e} for e in (True, True, False, False)]  # p=.5 → .25
             + [{"event": e} for e in (True, False, False, False)])  # p=.25 → .1875
    sc = learnability([0, 1, 2, 3], texts, k)
    assert sc == pytest.approx([0.0, 0.0, 0.25, 0.1875])
    assert max(range(4), key=lambda i: sc[i]) == 2      # 상위 N 은 «갈리는» 문제다


def test_pfx_commit_records_cut_weight_and_choice():
    from mc.pool import pfx_commit_records
    t = lambda uid, text, sel=True: {"uid": uid, "problem_idx": 0 if uid == "u" else 1, "problem": "Q (A) 1 (B) 2 (C) 3 (D) 4"  # noqa: E731
                                     if uid == "v" else "Q", "gold": "42" if uid == "u" else "2", "text": text, "selected": sel}
    texts = [t("u", "a \\boxed{7} b \\boxed{42} c"), t("u", "no box"), t("u", "x \\boxed{5}"), t("v", "so \\boxed{(B)}"),
             t("w", "\\boxed{1}", False)]
    box = lambda uid, bs, sel=True: {"uid": uid, "selected": sel, "boxes": [  # noqa: E731
        {"k": k, "cut": c, "answer": a, "commit_weight": w} for k, (a, c, w) in enumerate(bs)]}
    rows = [box("u", [("7", 11, .2), ("42", 26, .9)]), box("u", [("5", 11, .1)]), box("v", [("(B)", 14, .8)]),
            box("w", [("1", 9, .9)], False)]
    recs, summ = pfx_commit_records(texts, rows, veto=False)
    assert [r["pfx_id"] for r in recs] == ["u:0:1", "v:0:0"] and recs[0]["prefix"] == texts[0]["text"][:26]
    assert recs[0]["first_wrong"] is False and recs[1]["first_wrong"] is False and recs[1]["choice"]  # (B) ≡ 2 선택지 채점
    assert summ["n_no_committed_box"] == 1 and summ["n_no_box"] == 1 and summ["n_rows"] == 2
    texts[2]["text"], rows[1]["boxes"][0]["commit_weight"] = "x \\boxed{5}", .7
    recs, summ = pfx_commit_records(texts, rows, veto=False)
    w = {r["pfx_id"]: r["weight"] for r in recs}
    assert w["u:2:0"] == 3 / 2 and w["u:0:1"] == 3 / 4 and summ["weight"]["sum"] == 3.0   # 유효 반반, 가지치기 없음


def test_pfx_commit_veto_rejects_intermediate_boxes():
    from mc.pool import pfx_commit_records
    txt = "a = \\boxed{7}\n\n### Step 2: next\nso the answer is \\boxed{42}\n\nWait — check."
    sp1 = txt.index("}") + 1
    sp2 = txt.index("}", sp1) + 1
    texts = [{"uid": "u", "problem_idx": 0, "problem": "Q", "gold": "42", "text": txt, "selected": True}]
    rows = [{"uid": "u", "selected": True, "boxes": [{"k": 0, "cut": sp1, "answer": "7", "commit_weight": .9},
                                                     {"k": 1, "cut": sp2, "answer": "42", "commit_weight": .9}]}]
    recs, summ = pfx_commit_records(texts, rows)                     # 첫 박스는 무게가 높아도 글자 규칙이 거부
    assert [r["box_k"] for r in recs] == [1] and summ["n_rows_without_veto"] == 1 and summ["veto"]
    assert [r["box_k"] for r in pfx_commit_records(texts, rows, veto=False)[0]] == [0]
    assert pfx_commit_records(texts, rows, selected=False)[0] == []  # 탐침 쪽(선별 안 된) 롤아웃만 고를 때


def test_commit_probe_selection_cli(tmp_path):
    import json

    from mc.pool import main
    texts, rows = [], []
    for u in range(4):
        for j in range(2):
            ans = "42" if j else "7"
            txt = f"so the answer is \\boxed{{{ans}}}\n\nWait — check."
            texts.append({"uid": f"u{u}", "problem_idx": u, "problem": "Q", "gold": "42", "text": txt, "selected": u == 3})
            rows.append({"uid": f"u{u}", "selected": u == 3, "boxes": [{"k": 0, "cut": txt.index("}") + 1, "answer": ans,
                                                                        "commit_weight": .8}]})
    (tmp_path / "t.jsonl").write_text("".join(json.dumps(t) + "\n" for t in texts))
    (tmp_path / "b.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = tmp_path / "sel.json"
    assert main(["--pfx_from", str(tmp_path / "t.jsonl"), "--commit_scores", str(tmp_path / "b.jsonl"), "--probe_out", str(out),
                 "--n_wrong", "2", "--n_right", "5", "--out", str(tmp_path / "x")]) == 0
    sel = json.loads(out.read_text())
    assert len(sel) == 2 + 3 and {e["uid"] for e in sel} == {"u0", "u1", "u2"} and all(e["committed"] == 1.0 for e in sel)
    assert [e["pid"] for e in sel] == list(range(5)) and sum(not e["first_correct"] for e in sel) == 2
