"""cd9 MATH_META 회귀 시험 — 판단 항 진리표 / 영역 라우팅 / 중단 규칙 / parquet 빌더 / 런처."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "local"))

from src.training import math_meta as M  # noqa: E402


# ── 판단 항 진리표 ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("emitted,decision,best,want", [
    (1, "verify", "verify", 1.0),
    (1, "redirect", "redirect", 1.0),
    (1, "verify", "redirect", -1.0),
    (1, "redirect", "verify", -1.0),
    (1, "verify", "tie", 0.0),
    (1, None, "verify", 0.0),
    (0, "verify", "verify", 0.0),
    (1, "verify", None, 0.0),
])
def test_judgment_term_truth_table(emitted, decision, best, want):
    assert M.judgment_term(emitted, decision, best) == want


def _meta(decision=None, conf="0.4", body="My approach is substitution; it seems fine."):
    d = f"decision: {decision}\n" if decision else ""
    return f"<meta>\nconfidence: {conf}\n{body}\n{d}</meta>"


def test_compute_rows_separates_answer_and_meta(monkeypatch):
    monkeypatch.setenv("MATH_JUDGE_W", "0.5")
    texts = ["x=1 " + _meta("verify") + " so \\boxed{42}",       # 정답·판단 일치
             "y " + _meta("redirect") + " \\boxed{7}",           # 오답·판단 불일치
             "no meta here \\boxed{42}"]                          # 미발화·정답
    rows = M.compute_rows(texts, ["42"] * 3, ["P"] * 3, "M_JUDGE", labels={"P": "verify"})
    assert [r["r_corr"] for r in rows] == [1, 0, 1]
    assert [r["answer_total"] for r in rows] == [1.0, 0.0, 1.0]      # 답 스팬 = 정답만
    assert [r["meta_val"] for r in rows] == [0.5, -0.5, 0.0]         # 메타 스팬 = 판단 항만
    assert rows[0]["emitted"] == 1 and rows[2]["emitted"] == 0
    s, e = M.meta_char_spans(rows[0])[0]
    assert texts[0][s:e].startswith("<meta>") and texts[0][s:e].endswith("</meta>")
    assert M.meta_char_spans(rows[2]) == []


def test_g0_g1_have_no_meta_term():
    texts = ["a " + _meta("verify") + " \\boxed{1}"]
    for arm in ("M_G0", "M_G1"):
        r = M.compute_rows(texts, ["1"], ["P"], arm, labels={"P": "verify"})[0]
        assert r["meta_val"] == 0.0 and r["answer_total"] == 1.0


def test_rand_shuffles_labels_but_keeps_distribution():
    import random
    texts = ["t " + _meta("verify") + " \\boxed{1}"] * 6
    probs = [f"P{i}" for i in range(6)]
    labels = {f"P{i}": ("verify" if i < 3 else "redirect") for i in range(6)}
    judge = M.compute_rows(texts, ["1"] * 6, probs, "M_JUDGE", labels=labels)
    assert [r["judge"] for r in judge] == [1, 1, 1, -1, -1, -1]
    seen = set()
    for seed in range(20):
        rnd = M.compute_rows(texts, ["1"] * 6, probs, "M_RAND", labels=labels, rng=random.Random(seed))
        assert sorted(r["best_decision"] for r in rnd) == sorted(labels.values())
        seen.add(tuple(r["judge"] for r in rnd))
    assert len(seen) > 1                       # 실제로 섞인다


def test_rand_permutes_per_group_and_excludes_tie():
    """★감사 4: M_RAND 는 uid→라벨 «표»를 섞는다(행이 아니라 프롬프트 그룹 단위). 같은
    그룹의 8 롤아웃은 여전히 같은(섞인) 라벨을 받고, tie/무라벨 그룹은 순열 풀에서 빠진다."""
    import random
    texts = ["t " + _meta("verify") + " \\boxed{1}"] * 9
    probs = ["A"] * 3 + ["B"] * 3 + ["C"] * 3
    uids = ["ua"] * 3 + ["ub"] * 3 + ["uc"] * 3
    labels = {"A": "verify", "B": "redirect", "C": "tie"}
    flipped = 0
    for seed in range(30):
        rnd = M.compute_rows(texts, ["1"] * 9, probs, "M_RAND", labels=labels, uids=uids,
                             rng=random.Random(seed))
        bd = [r["best_decision"] for r in rnd]
        assert len(set(bd[0:3])) == 1 and len(set(bd[3:6])) == 1, bd     # 그룹 안 동질
        assert bd[6:9] == ["tie"] * 3                                     # tie 는 풀 밖
        assert sorted(bd[0:3] + bd[3:6]) == ["redirect"] * 3 + ["verify"] * 3
        flipped += int(bd[0] == "redirect")
    assert 0 < flipped < 30                                               # 실제로 섞인다
    # 그룹당 문제 하나 규약: 같은 uid 에 다른 문제가 섞이면 즉사
    with pytest.raises(RuntimeError):
        M.compute_rows(texts[:2], ["1"] * 2, ["A", "B"], "M_RAND", labels=labels, uids=["u", "u"])


def test_rand_logs_n_lab_after_shuffle(capsys):
    texts = ["t " + _meta("verify") + " \\boxed{1}"] * 4
    M.compute_rows(texts, ["1"] * 4, ["A", "A", "B", "B"], "M_RAND", labels={"A": "verify", "B": "redirect"},
                   uids=["u1", "u1", "u2", "u2"])
    out = capsys.readouterr().out
    assert "[MATH][RAND]" in out and "n_lab=4" in out


def test_probe_default_scorer_zero_and_registered_scorer_used(capsys):
    M.set_meta_scorer(None)
    texts = ["t " + _meta() + " \\boxed{1}", "plain \\boxed{1}"]
    rows = M.compute_rows(texts, ["1", "1"], ["P", "Q"], "M_PROBE")
    assert [r["meta_val"] for r in rows] == [0.0, 0.0]
    assert "[MATH][PROBE][WARN]" in capsys.readouterr().out
    M.set_meta_scorer(lambda rs: [0.7] * len(rs))
    try:
        rows = M.compute_rows(texts, ["1", "1"], ["P", "Q"], "M_PROBE")
        assert rows[0]["meta_val"] == 0.7 and rows[1]["meta_val"] == 0.0   # 미발화 행은 0
    finally:
        M.set_meta_scorer(None)


def test_boxed_in_meta_flag():
    r = M.parse_row("<meta>\nconfidence: 0.5\nthe answer is \\boxed{3}\n</meta> \\boxed{3}", "3", "P")
    assert r["boxed_in_meta"] == 1 and r["emitted"] == 1
    r2 = M.parse_row(_meta() + " \\boxed{3}", "3", "P")
    assert r2["boxed_in_meta"] == 0


def test_unknown_arm_fails_closed():
    with pytest.raises(ValueError):
        M.compute_rows(["x"], ["1"], ["P"], "M_NOPE")


def test_load_judge_labels_both_shapes(tmp_path):
    p1 = tmp_path / "a.json"
    p1.write_text(json.dumps({"What  is 1+1?": "verify", "Q2": "bogus"}))
    assert M.load_judge_labels(str(p1)) == {"What is 1+1?": "verify"}
    p2 = tmp_path / "b.json"
    p2.write_text(json.dumps([{"problem": "Q", "best_decision": "tie"}, {"problem": "R", "best_decision": None}]))
    assert M.load_judge_labels(str(p2)) == {"Q": "tie"}
    assert M.load_judge_labels("") == {}


# ── 영역 라우팅: 메타 스팬만 판단 항을 받고 답 스팬은 안 받는다 ───────────────────
class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def _fake_batch(texts, *, plen=3, resp_mask_clip=0):
    """responses=문자 코드 토큰, attention_mask=[prompt(plen) | response]. resp_mask_clip>0 이면
    response_mask 만 꼬리를 그만큼 잘라 «response_mask 가 짧은» 상황을 흉내 낸다."""
    import torch
    L = max(len(t) for t in texts)
    pad = lambda t: [ord(ch) for ch in t] + [32] * (L - len(t))
    ids = torch.tensor([pad(t) for t in texts])
    B = len(texts)
    am = torch.zeros(B, plen + L, dtype=torch.long)
    am[:, :plen] = 1
    rm = torch.zeros(B, L, dtype=torch.long)
    for i, t in enumerate(texts):
        am[i, plen:plen + len(t)] = 1
        rm[i, :max(0, len(t) - resp_mask_clip)] = 1

    class _D:
        pass
    d = _D()
    d.batch = {"advantages": torch.full((B, L), 0.3), "responses": ids, "response_mask": rm,
               "prompts": torch.zeros(B, plen, dtype=torch.long), "attention_mask": am}
    return d


def test_meta_region_advantage_lands_only_on_meta_tokens():
    import torch
    from src.training import verl_sdc as V
    text_a = "ab<meta>\nconfidence: 0.5\nok\ndecision: verify\n</meta>cd \\boxed{1}"
    text_b = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEF"
    text_c = "ab<meta>\nconfidence: 0.5\nok\ndecision: redirect\n</meta>cd \\boxed{1}"
    d = _fake_batch([text_a, text_b, text_c])
    base = d.batch["advantages"].clone()
    rows = M.compute_rows([text_a, text_b, text_c], ["1"] * 3, ["P"] * 3, "M_JUDGE", labels={"P": "verify"})
    spans = [M.meta_char_spans(r) for r in rows]
    assert spans[0] and not spans[1] and spans[2]
    # ★member: 판단 항이 정의된 행(발화·decision·라벨 모두 있음)만 1. 미발화 행은 0 이며
    #   그룹 평균에 들어가지 않는다 → 평균은 (0.5 − 0.5)/2 = 0, 중심화 값 [0.5, 0, −0.5].
    V._MATH_REGION_STASH.update({"step": 1, "bs": 3, "uid": ["g"] * 3, "meta": [0.5, 0.0, -0.5],
                                 "member": [1, 0, 1], "spans": spans})
    out = V._math_add_meta_region_advantage(d, tokenizer=_Tok())
    a = out.batch["advantages"]
    assert torch.allclose(a[1], base[1])                   # 미발화 행: 답 스팬 그대로
    s0, e0 = spans[0][0]
    assert torch.allclose(a[0, s0:e0], torch.full((e0 - s0,), 0.3 + 0.5))
    assert torch.allclose(a[0, :s0], base[0, :s0]) and torch.allclose(a[0, e0:], base[0, e0:])
    s2, e2 = spans[2][0]
    assert torch.allclose(a[2, s2:e2], torch.full((e2 - s2,), 0.3 - 0.5))
    assert V._MATH_REGION_STASH["meta"] == []              # 한 번 쓰고 비운다


def test_meta_region_centering_excludes_undefined_rows():
    """★감사 3: 항이 «정의되지 않은» 행(tie / decision 없음 / 미발화)은 그룹 평균을 밀지도,
    중심화 값을 받지도 않는다. 옛 동작(전 행 평균)은 무결정 발화 행에 −mean 을 얹었다."""
    import torch
    from src.training import verl_sdc as V
    mk = lambda dec: "ab<meta>\nconfidence: 0.5\nok\n" + (f"decision: {dec}\n" if dec else "") + "</meta>cd \\boxed{1}"
    texts = [mk("verify"), mk("redirect"), mk("verify"), mk(None)]
    d = _fake_batch(texts)
    base = d.batch["advantages"].clone()
    rows = M.compute_rows(texts, ["1"] * 4, ["P"] * 4, "M_JUDGE", labels={"P": "verify"})
    assert [r["meta_defined"] for r in rows] == [1, 1, 1, 0]
    assert [r["meta_val"] for r in rows] == [0.5, -0.5, 0.5, 0.0]
    spans = [M.meta_char_spans(r) for r in rows]
    assert all(spans)                                       # 넷 다 발화(무결정 행 포함)
    V._MATH_REGION_STASH.update({"step": 1, "bs": 4, "uid": ["g"] * 4, "meta": [r["meta_val"] for r in rows],
                                 "member": [r["meta_defined"] for r in rows], "spans": spans})
    a = V._math_add_meta_region_advantage(d, tokenizer=_Tok()).batch["advantages"]
    mean = (0.5 - 0.5 + 0.5) / 3                            # 정의된 세 행만의 평균
    for i, want in enumerate([0.5 - mean, -0.5 - mean, 0.5 - mean]):
        s, e = spans[i][0]
        assert torch.allclose(a[i, s:e], torch.full((e - s,), 0.3 + want)), i
    assert torch.allclose(a[3], base[3])                    # 무결정 발화 행: 메타 어드밴티지 정확히 0


def test_meta_region_length_from_attention_mask_not_response_mask():
    """★감사 5: 응답 길이 L 은 attention_mask[:, prompt_length:] 에서 온다(_decode_response 와
    같은 출처). response_mask 가 꼬리를 잘라도 맨 끝 메타 블록이 잘리지 않는다."""
    import torch
    from src.training import verl_sdc as V
    mk = lambda dec: "answer \\boxed{1} <meta>\nconfidence: 0.5\nok\ndecision: " + dec + "\n</meta>"
    texts = [mk("verify"), mk("redirect")]
    d = _fake_batch(texts, resp_mask_clip=12)               # response_mask 가 </meta> 꼬리를 자른다
    rows = M.compute_rows(texts, ["1"] * 2, ["P"] * 2, "M_JUDGE", labels={"P": "verify"})
    spans = [M.meta_char_spans(r) for r in rows]
    V._MATH_REGION_STASH.update({"step": 1, "bs": 2, "uid": ["g"] * 2, "meta": [0.5, -0.5],
                                 "member": [1, 1], "spans": spans})
    a = V._math_add_meta_region_advantage(d, tokenizer=_Tok()).batch["advantages"]
    s0, e0 = spans[0][0]
    assert e0 == len(texts[0])                              # 블록이 응답의 맨 끝까지 간다
    assert torch.allclose(a[0, s0:e0], torch.full((e0 - s0,), 0.3 + 0.5))   # 끝 토큰까지 얹힌다


def test_math_arm_reward_refuses_stale_stash():
    from src.training import verl_sdc as V
    V._MATH_STASH.update({"total": None, "n": 0})
    with pytest.raises(RuntimeError):
        V.math_arm_reward(["a", "b"])
    V._MATH_STASH.update({"total": [1.0, 0.0], "n": 2})
    assert V.math_arm_reward(["a", "b"]) == [1.0, 0.0]
    assert V.REWARD_CONFIGS["MATH_META"]["keys"] == ["math_arm"]
    assert "MATH_META" not in V._REGION_ROUTED_MODES and "MATH_META" not in V._VANILLA_MODES
    assert V.REWARD_CONFIGS["COUNTDOWN_6ARM"]["keys"] == ["countdown_arm"]   # 불변


# ── 중단 규칙 ───────────────────────────────────────────────────────────────────
def _rep(**kw):
    base = {"emit_rate": 0.5, "boxed_in_meta": 0.0, "boilerplate_rate": 0.01, "n_emitted": 100,
            "multi_block_rate": 0.0, "acc": 0.6}
    base.update(kw)
    return base


def _aborts(rep, arm):
    return {h["metric"] for h in M.check_abort(rep, arm=arm) if h["status"] == "abort"}


def test_check_abort_rules(monkeypatch):
    monkeypatch.delenv("MATH_ACC_FLOOR", raising=False)
    assert M.check_abort(_rep(), arm="M_JUDGE") == []
    assert _aborts(_rep(emit_rate=0.1), "M_JUDGE") == {"emit_rate"}
    # 메타 항 없는 팔은 발화율 규칙을 안 본다
    assert not _aborts(_rep(emit_rate=0.0), "M_G1")
    assert _aborts(_rep(boxed_in_meta=0.03, boilerplate_rate=0.6), "M_G1") == {"boxed_in_meta", "boilerplate_rate"}
    miss = [h for h in M.check_abort(_rep(boilerplate_rate=float("nan")), arm="M_JUDGE")]
    assert miss and miss[0]["status"] == "missing"


def test_boilerplate_threshold_is_half_and_small_n_is_missing(monkeypatch):
    """★감사 1: boilerplate_rate 는 «최빈 메타 문장의 점유율»(바닥 1/n_emitted)이라 0.05 는
    n_emitted ≤ 20 이면 무조건 발화한다. Countdown 과 같은 0.5, n_emitted<30 이면 missing."""
    monkeypatch.delenv("MATH_ACC_FLOOR", raising=False)
    assert M.ABORT_RULES["boilerplate_rate"]["thr"] == 0.5
    assert _aborts(_rep(boilerplate_rate=0.3, n_emitted=100), "M_JUDGE") == set()
    assert _aborts(_rep(boilerplate_rate=0.6, n_emitted=100), "M_JUDGE") == {"boilerplate_rate"}
    hits = M.check_abort(_rep(boilerplate_rate=0.6, n_emitted=10), arm="M_JUDGE")
    assert [h["status"] for h in hits if h["metric"] == "boilerplate_rate"] == ["missing"]
    # n_emitted=8 → 최빈 문장 하나가 0.125: 옛 문턱 0.05 였으면 여기서 죽었다.
    hits = M.check_abort(_rep(boilerplate_rate=0.125, n_emitted=8), arm="M_JUDGE")
    assert not [h for h in hits if h["status"] == "abort"]


def test_multi_block_is_format_violation(monkeypatch):
    """★감사 6: n_blocks>1 → 그 행의 메타 항 0(정의 안 됨), 텔레메트리 multi_block_rate, 중단 규칙 >0.10."""
    monkeypatch.setenv("MATH_JUDGE_W", "0.5")
    two = _meta("verify") + " then " + _meta("verify") + " \\boxed{1}"
    one = _meta("verify") + " \\boxed{1}"
    rows = M.compute_rows([two, one], ["1", "1"], ["P", "P"], "M_JUDGE", labels={"P": "verify"})
    assert rows[0]["n_blocks"] == 2 and rows[0]["meta_val"] == 0.0 and rows[0]["meta_defined"] == 0
    assert rows[1]["meta_val"] == 0.5 and rows[1]["meta_defined"] == 1
    rep = M.telemetry(rows, arm="M_JUDGE", step=1)
    assert rep["multi_block_rate"] == 0.5
    assert "multi_block=" in M.format_tel(rep)
    assert M.ABORT_RULES["multi_block_rate"] == {**M.ABORT_RULES["multi_block_rate"], "op": ">", "thr": 0.10}
    assert _aborts(_rep(multi_block_rate=0.11), "M_G1") == {"multi_block_rate"}
    assert _aborts(_rep(multi_block_rate=0.09), "M_G1") == set()


def test_acc_floor_rule_from_env(monkeypatch):
    """★감사 7: 사전등록 «acc < M_G0 − 1pp» — MATH_ACC_FLOOR 가 설정된 때만 3-스텝 연속 규칙에 참여."""
    monkeypatch.delenv("MATH_ACC_FLOOR", raising=False)
    assert not [h for h in M.check_abort(_rep(acc=0.1), arm="M_JUDGE") if h["metric"] == "acc"]
    monkeypatch.setenv("MATH_ACC_FLOOR", "0.55")
    assert _aborts(_rep(acc=0.50), "M_JUDGE") == {"acc"}
    assert _aborts(_rep(acc=0.56), "M_JUDGE") == set()
    assert _aborts(_rep(acc=0.50), "M_G0") == {"acc"}                    # 팔 무관
    st = {}
    hit = [h for h in M.check_abort(_rep(acc=0.5), arm="M_JUDGE") if h["status"] == "abort"]
    assert [M.update_streak(st, "k", hit, patience=3) for _ in range(3)] == [False, False, True]


def test_abort_streak_triggers_after_patience(monkeypatch):
    monkeypatch.delenv("COUNTDOWN_ABORT_PATIENCE", raising=False)
    st = {}
    hit = [{"metric": "emit_rate", "status": "abort"}]
    assert M.update_streak(st, "k", hit) is False
    assert M.update_streak(st, "k", hit) is False
    assert M.update_streak(st, "k", hit) is True          # 3 연속
    assert M.update_streak(st, "k", []) is False and st["k"] == 0   # 리셋
    assert M.update_streak(st, "k", hit, patience=1) is True


def test_telemetry_line_format():
    texts = ["t " + _meta("verify") + " \\boxed{1}", "plain \\boxed{2}"]
    rows = M.compute_rows(texts, ["1", "1"], ["P", "Q"], "M_JUDGE", labels={"P": "verify"})
    rep = M.telemetry(rows, arm="M_JUDGE", step=3)
    line = M.format_tel(rep)
    for k in ("[MATH][TEL]", "step=3", "arm=M_JUDGE", "acc=0.500", "emit=0.500", "n_blocks_mean=",
              "decision_rate=1.000", "judge_match=1.000", "boxed_in_meta=0.000", "len_mean="):
        assert k in line, line


# ── parquet 빌더 ────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("sol,want", [
    ("so the answer is $\\boxed{42}$.", "42"),
    ("first \\boxed{1} then finally \\boxed{\\frac{1}{2}}", "\\frac{1}{2}"),
    ("nested \\boxed{\\{a, b\\}} end", "\\{a, b\\}"),
    ("unclosed \\boxed{42", ""),
    ("no box at all", ""),
])
def test_last_boxed(sol, want):
    assert M.last_boxed(sol) == want


def test_split_records_holds_out_and_extracts_gold():
    import build_math_parquet as B
    rows = [{"problem": "P1", "solution": "\\boxed{1}", "level": "Level 1", "type": "algebra"},
            {"problem": "P2", "solution": "no gold"},
            {"problem": "  P3 ", "solution": "\\boxed{3}"},          # held-out (공백 정규화 일치)
            {"problem": "P1", "solution": "\\boxed{9}"},              # 중복
            {"problem": "P4", "solution": "x \\boxed{4}"}]
    train, val, st = B.split_records(rows, {"P3"}, val_n=1, seed=0, variant="math_opt")
    assert st == {"n_in": 5, "n_heldout_removed": 1, "n_dup": 1, "n_nogold": 1, "n_train": 1, "n_val": 1}
    rec = (train + val)[0]
    assert rec["data_source"] == "math_meta" and rec["gold"] in ("1", "4")
    assert rec["extra_info"]["problem"] == rec["problem"] and rec["extra_info"]["gold"] == rec["gold"]
    assert rec["reward_model"]["ground_truth"] == rec["gold"]
    assert rec["prompt"][0]["role"] == "system" and "<meta>" in rec["prompt"][0]["content"]
    assert rec["prompt"][1]["content"] == rec["problem"]


# ── 런처 ───────────────────────────────────────────────────────────────────────
def test_run_math_arm_dry_run_has_expected_overrides():
    env = dict(os.environ, MATH_JUDGE_LABELS="/nonexistent/labels.json", MATH_JUDGE_W="0.7",
               MATH_ACC_FLOOR="0.55")
    r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_JUDGE", "3", "50", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    out = r.stdout
    for want in ("++mode=MATH_META", "++algorithm.math_arm=M_JUDGE", "LINEAGE=cd9_M_JUDGE_s3",
                 "Qwen3-4B-Instruct-2507", "data.max_response_length=4096", "math_train_math_opt.parquet",
                 "++trainer.total_training_steps=50", "++data.seed=3", "MATH_JUDGE_W=0.7",
                 "MATH_ACC_FLOOR=0.55",
                 "--dry-run: not executing"):
        assert want in out, (want, out)


def test_run_math_arm_rejects_unknown_arm():
    r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_NOPE", "1", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode != 0 and "not in src.training.math_meta.MATH_ARM_SPECS" in r.stderr


def test_g0_uses_plain_variant_in_dry_run():
    r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_G0", "1", "--dry-run"],
                       cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "math_train_math_plain.parquet" in r.stdout and "++algorithm.math_arm=M_G0" in r.stdout


def test_ray_env_forwarding_list_contains_audited_names():
    src = (REPO / "src/training/verl_sdc.py").read_text()
    for k in ("MATH_JUDGE_LABELS", "MATH_JUDGE_W", "MATH_ACC_FLOOR", "CHK_AMP", "CHK_AMP_NEG", "PLAN_NG_W", "W_PERSIST",
              "LEN_BONUS_CHARS", "NOSURR_FRAC", "VTR_TAU", "VTR_WHEN_W"):
        assert f'"{k}"' in src, k


# ── 감사 2: 라벨 키가 학습 문제와 교집합이 없으면 조용히 M_G1 이 된다 — 즉사 ─────────────
def _labels_json(tmp_path, d, name="labels.json"):
    p = tmp_path / name
    p.write_text(json.dumps(d))
    return str(p)


def test_check_labels_cover_counts_and_fails_loud(tmp_path):
    import pandas as pd
    pq = tmp_path / "train.parquet"
    pd.DataFrame({"problem": ["What is 1+1?", "What  is 2+2?", "Q3"], "gold": ["2", "4", "9"]}).to_parquet(pq)
    st = M.check_labels_cover(str(pq), _labels_json(tmp_path, {"What is 2+2?": "verify", "ZZZ": "redirect"}))
    assert st == {"n_train": 3, "n_labels": 2, "n_cover": 1}
    with pytest.raises(RuntimeError, match="교집합"):
        M.check_labels_cover(str(pq), _labels_json(tmp_path, {"ZZZ": "verify"}))


def test_check_labels_cover_reads_extra_info_fallback(tmp_path):
    import pandas as pd
    pq = tmp_path / "train.parquet"
    pd.DataFrame({"extra_info": [{"problem": "P1", "gold": "1"}, {"problem": "P2", "gold": "2"}]}).to_parquet(pq)
    st = M.check_labels_cover(str(pq), _labels_json(tmp_path, {"P2": "tie"}))
    assert st["n_cover"] == 1


def _fake_trainer_data(texts, problems, uids, arm, plen=4):
    import torch
    from types import SimpleNamespace as NS
    L = max(len(t) for t in texts) + 1
    B = len(texts)
    am = torch.zeros(B, plen + L, dtype=torch.long)
    am[:, :plen] = 1
    for i, t in enumerate(texts):
        am[i, plen:plen + len(t)] = 1
    data = NS(batch={"prompts": torch.zeros(B, plen, dtype=torch.long), "attention_mask": am},
              non_tensor_batch={"problem": list(problems), "gold": ["1"] * B, "uid": list(uids)})
    self = NS(config=NS(algorithm=NS(math_arm=arm), data=NS(max_prompt_length=plen)))
    return self, data


def test_stash_raises_when_labels_do_not_intersect_training_problems(tmp_path, monkeypatch):
    from src.training import verl_sdc as V
    monkeypatch.setenv("MATH_JUDGE_LABELS", _labels_json(tmp_path, {"ZZZ": "verify", "YYY": "redirect"}))
    V._MATH_JUDGE_LABELS.update({"path": None, "table": None})
    V._MATH_LABEL_COVER.clear()
    texts = ["t " + _meta("verify") + " \\boxed{1}"] * 4
    self, data = _fake_trainer_data(texts, ["A", "A", "B", "B"], ["u1", "u1", "u2", "u2"], "M_JUDGE")
    with pytest.raises(RuntimeError, match="교집합"):
        V._compute_math_arm_stash(self, data, texts, 4, 4, 1)
    # 교집합이 있으면 정상 — 그리고 스태시에 member 가 실린다
    monkeypatch.setenv("MATH_JUDGE_LABELS", _labels_json(tmp_path, {"A": "verify", "B": "tie"}, "l2.json"))
    V._MATH_LABEL_COVER.clear()
    V._compute_math_arm_stash(self, data, texts, 4, 4, 1)
    assert V._MATH_REGION_STASH["member"] == [1, 1, 0, 0]
    assert V._MATH_REGION_STASH["meta"][:2] == [0.5, 0.5]


def test_run_math_arm_checks_label_cover_before_launch(tmp_path):
    """런처: 판단 팔은 발사 전에 check_labels_cover 를 돈다(교집합 0 → rc≠0, 카운트 출력).
    실제 학습 parquet 대신 임시 parquet 을 DATA_TRAIN 으로 넘긴다."""
    import pandas as pd
    work = tmp_path / "work"
    (work / "data").mkdir(parents=True)
    pd.DataFrame({"problem": ["P1", "P2"], "gold": ["1", "2"]}).to_parquet(work / "data" / "math_train_math_opt.parquet")
    pd.DataFrame({"problem": ["V1"], "gold": ["1"]}).to_parquet(work / "data" / "math_val_math_opt.parquet")
    # env.sh 가 WORK 를 고정하므로 DATA_TRAIN/DATA_VAL 재지정으로 임시 parquet 을 쓴다.
    env = dict(os.environ, MATH_JUDGE_LABELS=_labels_json(tmp_path, {"NOPE": "verify"}),
               DATA_TRAIN=str(work / "data" / "math_train_math_opt.parquet"),
               DATA_VAL=str(work / "data" / "math_val_math_opt.parquet"))
    r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_JUDGE", "1", "5", "--check-labels-only"],
                       cwd=REPO, capture_output=True, text=True, env=env)
    assert r.returncode != 0 and "교집합" in (r.stdout + r.stderr) and "n_cover=0" in (r.stdout + r.stderr)
    env["MATH_JUDGE_LABELS"] = _labels_json(tmp_path, {"P2": "redirect"})
    r = subprocess.run(["bash", "scripts/local/run_math_arm.sh", "M_JUDGE", "1", "5", "--check-labels-only"],
                       cwd=REPO, capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "n_cover=1" in (r.stdout + r.stderr), r.stdout + r.stderr
