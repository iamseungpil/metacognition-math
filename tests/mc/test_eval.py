"""mc.eval — 짝 부트스트랩 · 단일 패스 수정 계측 · 엔진 스위치."""
from __future__ import annotations

import json

from mc import eval as E


def test_paired_bootstrap_sign_and_ci():
    up = [(0.0, 1.0)] * 30
    out = E.paired_bootstrap(up, n=200)
    assert out["delta"] == 1.0 and out["lo"] == 1.0 and out["excludes_zero"]
    flat = [(0.5, 0.5)] * 30
    assert E.paired_bootstrap(flat, n=200)["delta"] == 0.0
    assert E.paired_bootstrap([], n=10)["n"] == 0


def test_single_summary_revision_metrics():
    rows = [
        # 박스 2개, 오답 → 정답 (w→r)
        {"text": r"try \boxed{5} hmm wait \boxed{42}", "gold": "42", "n_tok": 10,
         "group_id": "g0", "final_answer": "42"},
        # 박스 2개, 정답 → 오답 (r→w)
        {"text": r"try \boxed{42} hmm \boxed{5}", "gold": "42", "n_tok": 10,
         "group_id": "g0", "final_answer": "5"},
        # 박스 2개인데 같은 답 = 재확인(수정 아님)
        {"text": r"\boxed{42} check \boxed{42}", "gold": "42", "n_tok": 10,
         "group_id": "g0", "final_answer": "42"},
        # 박스 1개
        {"text": r"\boxed{42}", "gold": "42", "n_tok": 10, "group_id": "g0",
         "final_answer": "42"},
    ]
    s = E.summarize_single(rows)
    assert s["revised"] == 2 and s["w2r"] == 1 and s["r2w"] == 1
    assert s["confirmed_same_answer"] == 1
    assert s["precision"] == 0.5
    assert s["acc"] == 0.75
    assert s["revised_pct"] == 50.0


def test_vs_ref_pairs_by_problem(tmp_path):
    ref = {"per_problem": {"a": [0.0, 0.5], "b": [0.0, 0.5]}}
    p = tmp_path / "ref.json"
    p.write_text(json.dumps(ref))
    summ = {"per_problem": {"a": [0.0, 1.0], "b": [0.0, 1.0]}}
    out = E.vs_ref(summ, str(p))
    assert out["n"] == 2 and out["delta"] == 0.5
    assert E.vs_ref({"per_problem": {"z": [0, 0]}}, str(p))["n"] == 0


def test_eval_max_tokens_default_is_12288(monkeypatch, tmp_path):
    """단일 프로토콜 토큰 예산 기본값 — 오늘의 동작(암묵 12,288) 그대로."""
    import mc.rollout as R
    captured = {}
    def fake_run_eval(a):
        captured["mt"] = a.max_tokens
        return []
    monkeypatch.setattr(R, "run_eval", fake_run_eval)
    from mc.eval import main
    main(["--model_path", "m", "--dataset", "d", "--out_dir", str(tmp_path)])
    assert captured["mt"] == 12288
    assert json.loads((tmp_path / "summary.json").read_text())["protocol"] == "single"   # 옛 기본 twoturn 함정 제거


def test_build_engine_eager_switch(monkeypatch):
    """MC_EAGER 미설정 = 옛 평가 엔진(eager) 그대로, 0 일 때만 CUDA 그래프(수정 41)."""
    import sys, types
    got = []
    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(
        LLM=lambda **kw: got.append(kw["enforce_eager"]) or types.SimpleNamespace(get_tokenizer=lambda: None)))
    from mc.rollout import build_engine
    monkeypatch.delenv("MC_EAGER", raising=False)
    build_engine("m", max_tokens=8)
    monkeypatch.setenv("MC_EAGER", "0")
    build_engine("m", max_tokens=8)
    assert got == [True, False]


def test_load_problems_hf_spec_and_bare_id_split(monkeypatch):
    """`hf:<id>:<split>` 은 요청한 split 을(HMMT25="train"), 맨 HF id 는 여전히 "test"."""
    captured = {}

    def fake_load_dataset(hf_id, split=None):
        captured[hf_id] = split
        return [{"problem": "p1", "answer": "1"}]

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    from mc.rollout import load_problems
    rows = load_problems("hf:MathArena/hmmt_feb_2025:train")
    load_problems("math-ai/aime25")
    assert captured == {"MathArena/hmmt_feb_2025": "train", "math-ai/aime25": "test"}
    assert rows == [{"problem": "p1", "gold": "1"}]

