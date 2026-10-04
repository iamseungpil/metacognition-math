"""cd9 M_RETRY_SL — 결정 토큰 자기지도(self-distilled judgment) 회귀 시험.

결정 단어 문자 구간 / 네 경우의 부호(맞음·verify +W, 틀림·verify −W, 틀림·redirect +W,
맞음·redirect −W) / 강제 행 제외 / 그룹 중심화 안 함 / 워밍업 2배 / 텔레메트리 키 /
중단 규칙 min_step 15 / 런처 dry-run.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.training import math_meta as M  # noqa: E402


def _meta(decision, conf="0.4", body="The substitution step might be off."):
    d = f"decision: {decision}\n" if decision else ""
    return f"<meta>\nconfidence: {conf}\n{body}\n{d}</meta>"


def _text(first, decision, second=None):
    t = f"work \\boxed{{{first}}}\n" + _meta(decision) + "\n"
    if second is not None:
        t += f"Second attempt: other method \\boxed{{{second}}}"
    return t


def _rows(texts, golds, *, forced=None, arm="M_RETRY_SL", step=99):
    rows = M.compute_rows(texts, golds, ["P"] * len(texts), arm,
                          uids=["g"] * len(texts), forced_redirect=forced)
    M.annotate_sl_rows(rows, step=step)
    return rows


# ── 팔 명세 ────────────────────────────────────────────────────────────────────
def test_arm_spec_is_retry_like():
    spec = M.require_arm("M_RETRY_SL")
    assert spec["variant"] == "math_retry" and spec["require_meta"]
    assert "M_RETRY_SL" in M._RETRY_ARMS and "M_RETRY_SL" in M._RETRY_LIKE_ARMS
    assert "M_RETRY_SL" in M._SL_ARMS


def test_sl_arm_rows_are_byte_identical_to_m_retry_except_sl_fields():
    """★SL 항 말고는 M_RETRY 와 같은 행이어야 한다(보상·판단 항 불변)."""
    texts = [_text("1", "verify"), _text("2", "redirect", "1")]
    a = M.compute_rows(texts, ["1"] * 2, ["P"] * 2, "M_RETRY", uids=["g"] * 2)
    b = M.compute_rows(texts, ["1"] * 2, ["P"] * 2, "M_RETRY_SL", uids=["g"] * 2)
    for x, y in zip(a, b):
        assert (x["meta_val"], x["meta_defined"], x["answer_total"], x["judge"]) == \
               (y["meta_val"], y["meta_defined"], y["answer_total"], y["judge"])


# ── 결정 단어 문자 구간 ────────────────────────────────────────────────────────
def test_decision_char_span_points_at_the_decision_word():
    t = _text("1", "redirect", "1")
    r = _rows([t], ["1"])[0]
    (c0, c1), = M.decision_char_spans(r)
    assert t[c0:c1] == "redirect"
    # 메타 블록 안이다(= 메타 스팬의 부분집합)
    (m0, m1), = M.meta_char_spans(r)
    assert m0 <= c0 < c1 <= m1


def test_decision_span_empty_without_decision_line():
    r = _rows([_text("1", None)], ["1"])[0]
    assert M.decision_char_spans(r) == [] and r["sl_defined"] == 0 and r["sl_val"] == 0.0


# ── 네 경우의 부호 ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("first,gold,decision,want_match", [
    ("1", "1", "verify", 1),      # 맞음 + verify  → +W
    ("2", "1", "verify", 0),      # 틀림 + verify  → −W  (★p(verify) 를 내려 redirect 를 키운다)
    ("2", "1", "redirect", 1),    # 틀림 + redirect → +W
    ("1", "1", "redirect", 0),    # 맞음 + redirect → −W
])
def test_sl_sign_four_cases(first, gold, decision, want_match, monkeypatch):
    monkeypatch.setenv("MATH_SL_W", "0.5")
    monkeypatch.setenv("MATH_SL_WARMUP_STEPS", "0")
    r = _rows([_text(first, decision, "9")], [gold], step=99)[0]
    assert r["sl_defined"] == 1
    assert r["sl_target"] == ("verify" if first == gold else "redirect")
    assert r["sl_match"] == want_match
    assert r["sl_val"] == pytest.approx(0.5 if want_match else -0.5)


def test_forced_rows_excluded(monkeypatch):
    monkeypatch.setenv("MATH_SL_WARMUP_STEPS", "0")
    rows = _rows([_text("2", "redirect", "1"), _text("2", "redirect", "1")], ["1"] * 2,
                 forced=[0, 1])
    assert [r["sl_defined"] for r in rows] == [1, 0]
    assert rows[1]["sl_val"] == 0.0 and rows[1]["sl_spans"] == []


def test_warmup_doubles_weight(monkeypatch):
    monkeypatch.setenv("MATH_SL_W", "0.5")
    monkeypatch.setenv("MATH_SL_WARMUP_STEPS", "10")
    assert M.sl_step_weight(1) == pytest.approx(1.0)
    assert M.sl_step_weight(10) == pytest.approx(1.0)
    assert M.sl_step_weight(11) == pytest.approx(0.5)
    r = _rows([_text("1", "verify")], ["1"], step=3)[0]
    assert r["sl_val"] == pytest.approx(1.0)


# ── 텔레메트리·중단 ────────────────────────────────────────────────────────────
def test_telemetry_has_sl_keys(monkeypatch):
    monkeypatch.setenv("MATH_SL_WARMUP_STEPS", "0")
    rows = _rows([_text("1", "verify"), _text("2", "verify"), _text("1", None)], ["1"] * 3)
    rep = M.telemetry(rows, arm="M_RETRY_SL", step=20)
    assert rep["sl_rows"] == 2 and rep["sl_match_rate"] == pytest.approx(0.5)
    # 다른 팔엔 안 붙는다(기존 계약 불변)
    assert "sl_rows" not in M.telemetry(rows, arm="M_RETRY", step=20)


def test_format_tel_prints_sl_segment(monkeypatch):
    monkeypatch.setenv("MATH_SL_WARMUP_STEPS", "0")
    rows = _rows([_text("1", "verify"), _text("2", "verify")], ["1"] * 2)
    line = M.format_tel(M.telemetry(rows, arm="M_RETRY_SL", step=20))
    assert "sl_rows=2" in line and "sl_match=0.500" in line
    assert "sl_rows" not in M.format_tel(M.telemetry(rows, arm="M_RETRY", step=20))


def test_sl_match_rate_nan_when_no_defined_rows():
    rows = _rows([_text("1", None)], ["1"])
    assert math.isnan(M.telemetry(rows, arm="M_RETRY_SL", step=20)["sl_match_rate"])


def test_redirect_rate_abort_min_step_15_for_sl_arm():
    rep = {"step": 12, "redirect_rate": 0.0, "emit_rate": 1.0, "boxed_in_meta": 0.0,
           "boilerplate_rate": 0.0, "n_emitted": 100, "multi_block_rate": 0.0,
           "trunc_rate": 0.0}
    hits = {h["metric"] for h in M.check_abort(rep, arm="M_RETRY_SL") if h["status"] == "abort"}
    assert "redirect_rate" not in hits                    # ★15 스텝까지는 봐준다
    hits = {h["metric"] for h in M.check_abort(rep, arm="M_RETRY") if h["status"] == "abort"}
    assert "redirect_rate" in hits                        # M_RETRY 는 5 스텝 규약 그대로
    rep["step"] = 16
    hits = {h["metric"] for h in M.check_abort(rep, arm="M_RETRY_SL") if h["status"] == "abort"}
    assert "redirect_rate" in hits


# ── 어드밴티지 주입(합성 토크나이저: 1 문자 = 1 토큰) ──────────────────────────
class _Tok:
    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(i) for i in ids)


def _fake_batch(texts, *, plen=3):
    import torch
    L = max(len(t) for t in texts)
    pad = lambda t: [ord(ch) for ch in t] + [32] * (L - len(t))
    ids = torch.tensor([pad(t) for t in texts])
    B = len(texts)
    am = torch.zeros(B, plen + L, dtype=torch.long)
    am[:, :plen] = 1
    for i, t in enumerate(texts):
        am[i, plen:plen + len(t)] = 1

    class _D:
        pass
    d = _D()
    d.batch = {"advantages": torch.full((B, L), 0.3), "responses": ids,
               "prompts": torch.zeros(B, plen, dtype=torch.long), "attention_mask": am}
    return d


def test_sl_advantage_lands_only_on_decision_word_tokens(monkeypatch):
    import torch
    from src.training import verl_sdc as V
    monkeypatch.setenv("MATH_SL_W", "0.5")
    monkeypatch.setenv("MATH_SL_WARMUP_STEPS", "0")
    texts = [_text("2", "verify"), _text("2", "redirect", "1")]   # 둘 다 첫 답 오답
    rows = _rows(texts, ["1"] * 2)
    assert [r["sl_val"] for r in rows] == [-0.5, 0.5]
    d = _fake_batch(texts)
    base = d.batch["advantages"].clone()
    V._MATH_REGION_STASH.update({"step": 20, "bs": 2, "uid": ["g"] * 2, "meta": [], "member": [1, 1],
                                 "spans": [], "sl": [r["sl_val"] for r in rows],
                                 "sl_spans": [r["sl_spans"] for r in rows]})
    a = V._math_add_decision_sl_advantage(d, tokenizer=_Tok()).batch["advantages"]
    for i, want in enumerate([-0.5, 0.5]):
        (c0, c1), = rows[i]["sl_spans"]
        assert torch.allclose(a[i, c0:c1], torch.full((c1 - c0,), 0.3 + want)), i
        assert torch.allclose(a[i, :c0], base[i, :c0]) and torch.allclose(a[i, c1:], base[i, c1:])
    assert V._MATH_REGION_STASH["sl"] == []               # 한 번 쓰고 비운다


def test_sl_advantage_is_not_group_centered(monkeypatch):
    """★그룹이 전부 verify 여도(지금 실측) 신호가 0 으로 지워지면 안 된다 — 중심화 금지."""
    import torch
    from src.training import verl_sdc as V
    monkeypatch.setenv("MATH_SL_W", "0.5")
    monkeypatch.setenv("MATH_SL_WARMUP_STEPS", "0")
    texts = [_text("2", "verify"), _text("3", "verify")]          # 같은 그룹·전원 verify·전원 오답
    rows = _rows(texts, ["1"] * 2)
    d = _fake_batch(texts)
    V._MATH_REGION_STASH.update({"step": 20, "bs": 2, "uid": ["g"] * 2, "meta": [], "member": [1, 1],
                                 "spans": [], "sl": [r["sl_val"] for r in rows],
                                 "sl_spans": [r["sl_spans"] for r in rows]})
    a = V._math_add_decision_sl_advantage(d, tokenizer=_Tok()).batch["advantages"]
    for i in range(2):
        (c0, c1), = rows[i]["sl_spans"]
        assert torch.allclose(a[i, c0:c1], torch.full((c1 - c0,), 0.3 - 0.5)), i


def test_sl_advantage_noop_for_other_arms():
    import torch
    from src.training import verl_sdc as V
    texts = [_text("1", "verify")]
    d = _fake_batch(texts)
    base = d.batch["advantages"].clone()
    V._MATH_REGION_STASH.update({"step": 20, "bs": 1, "uid": ["g"], "meta": [], "member": [1],
                                 "spans": [], "sl": [], "sl_spans": []})
    a = V._math_add_decision_sl_advantage(d, tokenizer=_Tok()).batch["advantages"]
    assert torch.allclose(a, base)


# ── 런처 배선 ──────────────────────────────────────────────────────────────────
def test_launcher_dry_run_uses_retry_conventions():
    out = subprocess.run(["bash", str(REPO / "scripts/local/run_math_arm.sh"),
                          "M_RETRY_SL", "2", "30", "--dry-run"],
                         capture_output=True, text=True, cwd=str(REPO))
    assert out.returncode == 0, out.stderr[-2000:]
    txt = out.stdout
    assert "ARM=M_RETRY_SL" in txt and "VARIANT=math_retry" in txt
    assert "RESP_LEN=6144" in txt
    assert "EVAL_SCRIPT=math_retry_eval.py" in txt
    assert "MATH_SL_W=0.5" in txt and "MATH_SL_WARMUP_STEPS=10" in txt


def test_ray_env_forwards_sl_knobs():
    src = (REPO / "src/training/verl_sdc.py").read_text()
    assert '"MATH_SL_W", "MATH_SL_WARMUP_STEPS"' in src
