"""mc.shift_check — 박스별 내부 신호 채점(수정 22): 후보 집합·트리 채점·약속 무게(순수 부분만, vLLM 없음)."""
from __future__ import annotations

import math

import pytest

from mc import shift_check as S


def test_score_trees_chunks_blocks_in_order(monkeypatch):
    import mc.trainer as T
    calls = []
    devs = []
    monkeypatch.setattr(T, "tree_score", lambda m, pre, bl, dev=None: devs.append(dev) or calls.append(len(bl)) or [
        float(b[0]) for b in bl])
    trees = [([0] * 10, [(t, [1] * 6, 2) for t in range(1, 6)]), ([0] * 3, [(2, [1, 1], 1)])]
    assert S.score_trees(None, trees, max_tok=22) == [1.0, 2.0, 3.0, 4.0, 5.0, 2.0]
    assert calls == [2, 2, 1, 1]                            # 10 + 6·2 ≤ 22 — 넘기 전에 나눈다
    assert set(devs) == {None} and S.score_trees(None, trees[1:], dev="cuda:0") == [2.0] and devs[-1] == "cuda:0"


def test_score_trees_splits_on_oom(monkeypatch):
    import torch

    import mc.trainer as T

    def ts(m, pre, bl, dev=None):
        if len(bl) > 1:
            raise torch.cuda.OutOfMemoryError("fake")
        return [float(bl[0][0])]
    monkeypatch.setattr(T, "tree_score", ts)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    assert S.score_trees(None, [([0] * 4, [(t, [1, 1], 1) for t in (1, 2, 3)])]) == [1.0, 2.0, 3.0]


def test_candidates_decoys_siblings_and_forms():
    cand, st = S.v1p_candidates("7", {"gold": "42", "maj": "7"}, ["42", "9", "9", "", "7", "11"],
                                ["7", "42", "42.0", "9"])
    L, alts = st["gold"]
    assert L == "42" and cand["42"] == ["42", "42.0"]                  # 동치 다른 표기 하나를 더 잰다
    assert len(alts) == 4 and "9" in alts and "11" in alts             # 가짜 답 2 + 형제 2(흔한 것부터)
    assert not {"7", "42"} & set(alts[:2])                             # 가짜 답은 L·A0 와 다르다
    assert st["maj"][0] == "7"                                         # 맞은 첫 답(L ≡ A0)은 한 이름
    assert "7" not in st["maj"][1]


def test_v5_candidates_fixed_per_group():
    fins = ["5", "7", "42.0", "", "9", "7"]
    c1, s1 = S.v5_candidates("5", {"gold": "42", "maj": "7"}, fins, ["5", *fins, "42", "7"])
    L, C = s1["gold"]
    assert L == "42" and C[0] == "42" and {"5", "7", "9"} <= set(C) and len(C) == len(set(C)) == 6   # L·A0·fin·가짜 2
    assert "42.0" not in C and c1["42"] == ["42", "42.0"]                     # 동치 최종 답은 L 로 묶인다
    assert s1["maj"][0] == "7" and "42" in s1["maj"][1]                        # gold 도 최종 답이면 C 에 있다
    assert S.v5_candidates("5", {"gold": "42", "maj": "7"}, fins, ["5", *fins, "42", "7"]) == (c1, s1)   # 행과 무관


def test_w_ans_share_and_committed_rule_reexport():
    h0 = {0: {"A0": math.log(0.6), "L": math.log(0.3), "d": math.log(0.1)}}
    h1 = {0: {"A0": math.log(0.2), "L": math.log(0.2), "d": math.log(0.1)}}
    hn = {0: {"A0": math.log(0.2), "L": math.log(0.2), "d": float("nan")}}      # 후보가 NaN 인 머리는 빠진다
    assert S.w_ans([h0], "A0", ["A0", "L", "d"]) == pytest.approx(0.6)
    assert S.w_ans([h0, h1], "A0", ["A0", "L", "d"]) == pytest.approx((0.6 + 0.4) / 2)
    assert S.w_ans([h0, hn], "A0", ["A0", "L", "d"]) == pytest.approx(0.6)
    assert math.isnan(S.w_ans([{0: {"A0": float("nan"), "L": 0.0}}], "A0", ["A0", "L"]))


def test_commit_weight_logistic_on_logit():
    assert S.commit_weight(0.5) == pytest.approx(1 / (1 + math.exp(-S.COMMIT_A)))
    assert S.commit_weight(1.0) > S.commit_weight(1 - 1e-6) > S.commit_weight(0.9) and S.commit_weight(1.0) < 1
    assert S.commit_weight(0.3, a=0.0, b=1.0) == pytest.approx(0.3)                 # a=0,b=1 이면 항등
    assert math.isnan(S.commit_weight(float("nan")))


class CharTok:                                            # 글자 = 토큰(HF 처럼 문자열 하나면 평평한 목록)
    def __call__(self, texts, add_special_tokens=False, return_offsets_mapping=True):
        if isinstance(texts, str):
            return {k: v[0] for k, v in self([texts]).items()}
        return {"input_ids": [[ord(c) for c in t] for t in texts],
                "offset_mapping": [[(i, i + 1) for i in range(len(t))] for t in texts]}

    def encode(self, t, add_special_tokens=False):
        return [ord(c) for c in t]


def test_box_requests_and_rows(monkeypatch):
    monkeypatch.setattr(S.ctx, "turn1_prompt", lambda tok, prob, v: "P:" + prob)
    recs = [{"uid": "u", "problem_idx": 0, "problem": "Q", "gold": "42", "selected": True, "final_answer": "42",
             "text": "step \\boxed{7} then so the answer is \\boxed{42}."},
            {"uid": "u", "problem_idx": 0, "problem": "Q", "gold": "42", "selected": True, "final_answer": "", "text": "no box"}]
    trees, meta, items = S.box_requests(CharTok(), recs, heads=S.V5_HEADS[:2], cap=8)
    assert len(trees) == len(items) == 1 and [b[0] for b in items[0]["boxes"]] == [0, 1]
    pre, blocks = trees[0]
    cut1 = recs[0]["text"].index("}") + 1
    assert pre == [ord(c) for c in "P:Q" + recs[0]["text"][:recs[0]["text"].rindex("}") + 1]]
    assert blocks[meta[(0, 0, "7", 0)]][0] == 3 + cut1 and items[0]["boxes"][0][1] == cut1   # 박스 끝 토큰 자리
    assert len(meta) == len(blocks) and {k[1] for k in meta} == {0, 1} and {k[3] for k in meta} == {0, 1}
    sc = [(-0.1 if blocks[j][1][-2] == ord("7") and blocks[j][0] == 3 + cut1 else -5.0) for j in range(len(blocks))]
    rows = S.box_rows(items, sc, meta, heads=2)
    b0, b1 = rows[0]["boxes"]
    assert b0["answer"] == "7" and b1["answer"] == "42" and b0["w_ans"] > b1["w_ans"] - 1 and 0 <= b0["commit_weight"] <= 1
    assert rows[0]["selected"] is True and b1["cut"] == recs[0]["text"].rindex("}") + 1


def test_box_requests_prior_scores_at_response_start(monkeypatch):
    """수정 27a T1: prior=True 면 같은 후보·박스 구조를 응답 시작 자리(문제만)에서 — 트리 앞부분 = 머리뿐, 블록 자리 = len(머리)."""
    monkeypatch.setattr(S.ctx, "turn1_prompt", lambda tok, prob, v: "P:" + prob)
    recs = [{"uid": "u", "problem_idx": 0, "problem": "Q", "gold": "42", "selected": True, "final_answer": "42",
             "text": "step \\boxed{7} then so the answer is \\boxed{42}."}]
    post = S.box_requests(CharTok(), recs, heads=S.V5_HEADS[:2], cap=8)
    trees, meta, items = S.box_requests(CharTok(), recs, heads=S.V5_HEADS[:2], cap=8, prior=True)
    assert trees[0][0] == [ord(c) for c in "P:Q"] and {b[0] for b in trees[0][1]} == {3}
    assert meta == post[1] and items[0]["boxes"] == post[2][0]["boxes"]          # 행 맞춤(k·자름 글자·답·후보) 동일
