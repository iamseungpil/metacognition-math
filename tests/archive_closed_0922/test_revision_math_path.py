r"""수정(revision) 팔의 **MATH_META 트레이너 경로** — 팔 명세 · 스태시 · 구간 가산기(CPU).

`tests/test_revision.py` 가 «무엇이 수정인가»를, `tests/test_revision_rmeta.py` 가 크레딧
계산을 고정한다면, 이 파일은 그 크레딧이 verl 배치 위에서 **어느 토큰에** 얹히는지를
고정한다 — 형제 중심화가 다시 들어오는 것, 구간 밖 토큰이 물드는 것, 결과(정답) 어드밴티지가
바뀌는 것을 막는 것이 목적이다. GPU 없이 돈다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import math_meta as mm  # noqa: E402
from src.training import verl_sdc as V  # noqa: E402
from src.training.revision import revision_zone  # noqa: E402

torch = pytest.importorskip("torch")


class CharTok:
    """1 문자 = 1 토큰인 가짜 토크나이저(문자 오프셋 ↔ 토큰 인덱스가 1:1)."""

    def encode(self, s, add_special_tokens=False):
        return [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i)) for i in ids)


TOK = CharTok()
REV = r"first \boxed{3}. Wait, recheck. So \boxed{7}"
PLAIN = r"the answer is \boxed{7} and again \boxed{7}"


# ── 1) 팔 명세 (run_math_arm.sh 가 읽는 단일 진실 원천) ──────────────────────
def test_three_revision_arms_resolve_with_the_right_terms_and_anchors():
    assert mm.require_arm("M_REV_CF")["meta_term"] == "revision_cf"
    assert mm.require_arm("M_REV_PMI_GOLD")["meta_term"] == "revision_pmi"
    assert mm.require_arm("M_REV_PMI_COMBO")["meta_term"] == "revision_pmi"
    for a in ("M_REV_CF", "M_REV_PMI_GOLD", "M_REV_PMI_COMBO"):
        spec = mm.require_arm(a)
        assert spec["variant"] == "math_opt"      # 메타 프롬프트 주입 없음
        assert spec["require_meta"] is False      # 발화 중단 규칙에 걸리면 안 된다
        assert a in mm._REV_ARMS
    # 앵커는 팔 이름이 정한다(런처가 MATH_REV_ANCHOR 를 잊어도 두 PMI 팔이 갈린다)
    assert mm.rev_anchor("M_REV_PMI_GOLD") == "gold_x"
    assert mm.rev_anchor("M_REV_PMI_COMBO") == "combo"


def test_outcome_only_control_is_m_g1_not_a_new_arm():
    g1 = mm.require_arm("M_G1")
    assert (g1["variant"], g1["meta_term"], g1["require_meta"]) == ("math_opt", None, False)
    assert "M_REV_OUTCOME" not in mm.MATH_ARM_SPECS


def test_emit_rate_abort_rule_does_not_apply_to_revision_arms():
    """★이 팔들은 <meta> 를 한 번도 내지 않는다 — emit_rate<0.2 규칙에 걸리면 즉사한다."""
    rep = {"step": 50, "arm": "M_REV_CF", "emit_rate": 0.0, "boxed_in_meta": 0.0,
           "boilerplate_rate": float("nan"), "multi_block_rate": 0.0, "n_emitted": 0}
    hits = [h for h in mm.check_abort(rep, arm="M_REV_CF") if h["status"] == "abort"]
    assert hits == []


def test_rev_weight_env(monkeypatch):
    monkeypatch.delenv("MATH_REV_W", raising=False)
    assert mm.rev_weight() == 1.0
    monkeypatch.setenv("MATH_REV_W", "0.25")
    assert mm.rev_weight() == 0.25
    monkeypatch.setenv("MATH_REV_ANCHOR", "self_mx")
    assert mm.rev_anchor("M_REV_PMI_COMBO") == "self_mx"


# ── 2) 스태시 (문자 구간 + 가중치) ──────────────────────────────────────────
def _fake_batch(texts, plen=3):
    """prompts/responses/attention_mask 만 든 최소 배치(문자=토큰)."""
    B = len(texts)
    rlen = max(len(t) for t in texts)
    resp = torch.zeros(B, rlen, dtype=torch.long)
    am = torch.zeros(B, plen + rlen, dtype=torch.long)
    am[:, :plen] = 1
    for i, t in enumerate(texts):
        ids = TOK.encode(t)
        resp[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        am[i, plen:plen + len(ids)] = 1
    return _FakeData({"prompts": torch.zeros(B, plen, dtype=torch.long),
                      "responses": resp, "attention_mask": am})


class _FakeData:
    """DataProto 의 최소 흉내 — `.batch[key]` 와 행 인덱싱(`data[i].batch[key]`)만."""

    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return int(self.batch["responses"].shape[0])

    def __getitem__(self, i):
        return SimpleNamespace(batch={k: v[i] for k, v in self.batch.items()})


@pytest.fixture(autouse=True)
def _clean():
    V._REV_FIRST_CORRECT_HISTORY.clear()
    V._MATH_REGION_STASH.update({"rev": [], "rev_spans": [], "rev_tel": {}})
    yield
    V._REV_FIRST_CORRECT_HISTORY.clear()
    V._MATH_REGION_STASH.update({"rev": [], "rev_spans": [], "rev_tel": {}})


def test_stash_gives_char_spans_weighted_credit_and_telemetry(monkeypatch):
    monkeypatch.setenv("MATH_REV_W", "0.5")
    texts = [REV] + [PLAIN] * 7
    shim = SimpleNamespace(tokenizer=TOK, config=SimpleNamespace(algorithm=None))
    vals, spans, tel = V._math_revision_stash(
        shim, _fake_batch(texts), texts, ["7"] * 8, ["g"] * 8, 8, 3, 0,
        "M_REV_CF", mm.require_arm("M_REV_CF"))
    z = revision_zone(REV)
    assert spans[0] == [(z["zone_start"], z["zone_end"])]
    assert vals[0] == pytest.approx(0.5)          # 오답→정답 save 1.0 × MATH_REV_W 0.5
    assert all(not sp for sp in spans[1:]) and all(v == 0.0 for v in vals[1:])
    assert tel["revision_rate_batch"] == pytest.approx(1 / 8)
    assert tel["rev_member_rate"] == pytest.approx(1 / 8)
    assert "first_correct_mean" in tel and tel["rev_anchor"] == "gold_x"


# ── 3) 구간 가산기 ──────────────────────────────────────────────────────────
def test_advantage_lands_only_on_the_zone_and_is_not_centered():
    texts = [REV] + [PLAIN] * 3
    data = _fake_batch(texts)
    B = len(texts)
    T = data.batch["responses"].shape[1]
    # 결과(GRPO) 어드밴티지: 모든 유효 토큰에 0.4 — cap = 0.4 이므로 0.3 크레딧은 안 잘린다
    adv = torch.zeros(B, T)
    adv[:, :] = 0.4 * data.batch["attention_mask"][:, 3:].float()
    data.batch["advantages"] = adv.clone()
    before = adv.clone()
    z = revision_zone(REV)
    V._MATH_REGION_STASH.update({
        "step": 7, "bs": B, "rev": [0.3, 0.0, 0.0, 0.0],
        "rev_spans": [[(z["zone_start"], z["zone_end"])], [], [], []]})
    out = V._math_add_revision_advantage(data, tokenizer=TOK)
    got = out.batch["advantages"]
    delta = (got - before)[0]
    on = torch.nonzero(delta.abs() > 1e-6).flatten().tolist()
    assert on, "수정 구간에 아무것도 안 얹혔다"
    # 문자=토큰이므로 구간 경계가 그대로 토큰 경계다
    assert min(on) == z["zone_start"] and max(on) == z["zone_end"] - 1
    assert torch.allclose(delta[on], torch.full((len(on),), 0.3), atol=1e-5)
    # 마지막 \boxed 토큰(구간 밖)과 첫 답 토큰은 손대지 않는다
    assert delta[z["zone_end"]].abs().item() < 1e-6
    assert delta[0].abs().item() < 1e-6
    # ★중심화 없음: 수정하지 않은 형제 세 행은 한 토큰도 안 바뀐다(중심화면 −r/4 가 생긴다)
    assert torch.allclose(got[1:], before[1:])
    assert V._MATH_REGION_STASH["rev"] == []      # 소비 후 비운다(다음 스텝 재사용 금지)


def test_adder_is_a_noop_without_a_stash():
    data = _fake_batch([PLAIN, PLAIN])
    data.batch["advantages"] = torch.ones(2, data.batch["responses"].shape[1])
    out = V._math_add_revision_advantage(data, tokenizer=TOK)
    assert torch.allclose(out.batch["advantages"], torch.ones_like(out.batch["advantages"]))


def test_credit_is_clipped_by_mean_abs_answer_advantage():
    texts = [REV]
    data = _fake_batch(texts)
    T = data.batch["responses"].shape[1]
    adv = 0.05 * data.batch["attention_mask"][:, 3:].float()
    data.batch["advantages"] = adv.clone()
    z = revision_zone(REV)
    V._MATH_REGION_STASH.update({"step": 1, "bs": 1, "rev": [9.0],
                                 "rev_spans": [[(z["zone_start"], z["zone_end"])]]})
    got = V._math_add_revision_advantage(data, tokenizer=TOK).batch["advantages"]
    assert float((got - adv)[0, z["zone_start"]].item()) == pytest.approx(0.05, abs=1e-3)


def test_zero_cap_batch_adds_nothing():
    data = _fake_batch([REV])
    data.batch["advantages"] = torch.zeros(1, data.batch["responses"].shape[1])
    z = revision_zone(REV)
    V._MATH_REGION_STASH.update({"step": 1, "bs": 1, "rev": [1.0],
                                 "rev_spans": [[(z["zone_start"], z["zone_end"])]]})
    got = V._math_add_revision_advantage(data, tokenizer=TOK).batch["advantages"]
    assert float(got.abs().sum().item()) == 0.0


# ── 4) 사행 가드가 MATH_META 경로에서도 터진다 ──────────────────────────────
def test_sandbagging_guard_raises_countdown_abort_on_the_math_path():
    W = V.REV_GUARD_WARMUP
    for _ in range(W):
        V._rev_guard_check(0.60, step=1)
    for _ in range(W - 1):
        V._rev_guard_check(0.40, step=2)
    with pytest.raises(V._CountdownAbort):
        V._rev_guard_check(0.40, step=3)


# ── 5) PMI 팔: MATH_META 에서도 frozen-ref forward 가 실제로 불린다 ─────────
def test_pmi_arm_uses_the_math_ref_scorer_and_decoded_prompts(monkeypatch):
    seen = {}

    def fake_build(prompts, resps, pad_unit):
        seen["prompts"] = [TOK.decode(p) for p in prompts]
        seen["resps"] = [TOK.decode(r) for r in resps]
        return object(), len(prompts)

    def fake_ref(trainer, tensors):
        seen["ref_called"] = True
        n = len(seen["prompts"])
        out = torch.zeros(n, max(len(r) for r in seen["resps"]))
        for k in range(n):
            plus, close = (k % 2 == 0), ((k % 4) >= 2)
            out[k, :len(seen["resps"][k])] = 1.0 if plus == close else -1.0
        return out

    monkeypatch.setattr(V, "_build_pmi_score_batches", fake_build)
    monkeypatch.setattr(V, "_math_ref_logprobs", fake_ref)
    monkeypatch.setenv("MATH_REV_W", "1.0")
    monkeypatch.setattr(V, "_decode_prompt_only", lambda *a, **k: "P: ")
    texts = [REV] + [PLAIN] * 7
    shim = SimpleNamespace(tokenizer=TOK, config=SimpleNamespace(algorithm=None))
    vals, spans, _tel = V._math_revision_stash(
        shim, _fake_batch(texts), texts, ["7"] * 8, ["g"] * 8, 8, 3, 0,
        "M_REV_PMI_GOLD", mm.require_arm("M_REV_PMI_GOLD"))
    assert seen.get("ref_called") is True
    assert vals[0] > 0.0 and spans[0]
    # A+ = gold, A- = 첫 답 / CLOSE 는 최종 \boxed 를 쓰기 **전** 까지다
    assert seen["resps"][:2] == [r"\boxed{7}", r"\boxed{3}"]
    assert seen["prompts"][0].endswith(r"\boxed{3}")
    assert seen["prompts"][2].count(r"\boxed") == 1 and "recheck" in seen["prompts"][2]


# ── 6) 비만장일치 학습 풀 빌더(scripts/local/build_revision_pool_parquet.py) ─
def _pool_mod():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "local"))
    import build_revision_pool_parquet as bp
    return bp


def _group(finals, toks):
    return {"problem": "p", "gold": "7", "finals": list(finals), "n_tok": list(toks)}


def test_pool_select_filters_by_state_and_length():
    bp = _pool_mod()
    groups = {
        "all_same": _group(["7"] * 8, [100] * 8),
        "dominant": _group(["7"] * 6 + ["3", "5"], [100] * 8),
        "split": _group(["7"] * 4 + ["3"] * 4, [100] * 8),
        "scatter": _group([str(i) for i in range(8)], [100] * 8),
        "dom_too_long": _group(["7"] * 6 + ["3", "5"], [100] * 7 + [9000]),
    }
    keys_a, st_a = bp.select(groups, bp.VARIANT_A, 8192)
    assert set(keys_a) == {"dominant"}                 # ALL_SAME/SPLIT/SCATTER 제외
    assert st_a["dropped_len"] == 1                    # 8192 초과 그룹은 버리고 **센다**
    assert st_a["by_state"]["ALL_SAME"] == 1 and st_a["by_state"]["DOMINANT"] == 2
    keys_b, _st_b = bp.select(groups, bp.VARIANT_B, 8192)
    assert set(keys_b) == {"dominant", "split"}        # 변형 B 는 SPLIT 도 받는다


def test_pool_rows_match_the_s3c_schema():
    bp = _pool_mod()
    groups = {"k": _group(["7"] * 6 + ["3", "5"], [100] * 8)}
    df = bp.build(["k"], groups, {"k": {"level": "Level 5", "subject": "Algebra",
                                        "pass_rate": 0.75, "state": "DOMINANT"}})
    assert list(df.columns) == ["data_source", "prompt", "problem", "gold",
                                "reward_model", "extra_info"]
    r = df.iloc[0]
    assert r["data_source"] == "math_meta"
    assert r["reward_model"] == {"style": "rule", "ground_truth": "7"}
    assert set(r["extra_info"]) == {"problem", "gold", "level", "subject",
                                    "prompt_variant", "group_pass_rate", "agree_state"}
    assert r["extra_info"]["prompt_variant"] == "math_opt"
    assert r["extra_info"]["agree_state"] == "DOMINANT"


# ── 7) 크레딧 상한 배수(dcpo_rev_cap_mult / MATH_REV_CAP_MULT) ──────────────
def test_cap_mult_scales_the_applied_clip(monkeypatch):
    """★0919: mult=3 이면 상한이 3배가 되어 잘린 크레딧도 3배로 얹힌다."""
    z = revision_zone(REV)

    def _applied(mult):
        monkeypatch.setenv("MATH_REV_CAP_MULT", str(mult))
        data = _fake_batch([REV])
        adv = 0.05 * data.batch["attention_mask"][:, 3:].float()
        data.batch["advantages"] = adv.clone()
        V._MATH_REGION_STASH.update({"step": 1, "bs": 1, "rev": [9.0],
                                     "rev_spans": [[(z["zone_start"], z["zone_end"])]]})
        got = V._math_add_revision_advantage(data, tokenizer=TOK).batch["advantages"]
        return float((got - adv)[0, z["zone_start"]].item())

    one = _applied(1.0)
    three = _applied(3.0)
    assert one == pytest.approx(0.05, abs=1e-3)
    assert three == pytest.approx(3.0 * one, rel=1e-3)


def test_cap_mult_defaults_to_one(monkeypatch):
    monkeypatch.delenv("MATH_REV_CAP_MULT", raising=False)
    assert V._rev_cap_mult() == 1.0
    monkeypatch.setenv("MATH_REV_CAP_MULT", "2.5")
    assert V._rev_cap_mult() == 2.5


# ── 8) 질량 몫 모드(dcpo_rev_mass_share / MATH_REV_MASS_SHARE) ──────────────
def _mass_batch(monkeypatch, share, cap_mult, credit=0.3):
    """수정 행 하나 + 평범한 형제 셋. 반환 (얹힌 델타, m_ans, 출력)."""
    monkeypatch.setenv("MATH_REV_MASS_SHARE", str(share))
    monkeypatch.setenv("MATH_REV_CAP_MULT", str(cap_mult))
    texts = [REV] + [PLAIN] * 3
    data = _fake_batch(texts)
    valid = data.batch["attention_mask"][:, 3:].float()
    adv = 0.4 * valid
    data.batch["advantages"] = adv.clone()
    z = revision_zone(REV)
    V._MATH_REGION_STASH.update({
        "step": 3, "bs": 4, "rev": [credit, 0.0, 0.0, 0.0],
        "rev_spans": [[(z["zone_start"], z["zone_end"])], [], [], []]})
    got = V._math_add_revision_advantage(data, tokenizer=TOK).batch["advantages"]
    return (got - adv), float((adv.abs() * valid).sum().item())


def test_mass_share_unset_is_byte_identical(monkeypatch):
    monkeypatch.delenv("MATH_REV_MASS_SHARE", raising=False)
    assert V._rev_mass_share() == 0.0
    delta, _m = _mass_batch(monkeypatch, 0, 1.0)
    on = delta[0][delta[0].abs() > 1e-9]
    # 종전 경로: 크레딧 0.3 이 cap(=0.4) 아래라 그대로 얹힌다
    assert torch.allclose(on, torch.full_like(on, 0.3), atol=1e-5)
    assert torch.allclose(delta[1:], torch.zeros_like(delta[1:]))


def test_mass_share_hits_the_target_share_when_clip_does_not_bind(monkeypatch, capsys):
    delta, m_ans = _mass_batch(monkeypatch, 0.05, 100.0)   # cap_mult 크게 → 안전 클립 무력
    realized = float(delta.abs().sum().item()) / m_ans
    assert realized == pytest.approx(0.05, abs=1e-6)
    out = capsys.readouterr().out
    assert "mass_share_target=0.0500" in out and "scale_s=" in out
    assert "clipped_rows=0" in out


def test_mass_share_is_invariant_to_the_raw_credit_size(monkeypatch):
    """★핵심: 원시 크레딧 크기가 달라도 정책에 닿는 **몫**은 같다."""
    d1, m1 = _mass_batch(monkeypatch, 0.05, 100.0, credit=0.3)
    d2, m2 = _mass_batch(monkeypatch, 0.05, 100.0, credit=9.0)
    assert (float(d1.abs().sum().item()) / m1) == pytest.approx(
        float(d2.abs().sum().item()) / m2, abs=1e-6)


def test_safety_clip_binds_and_is_counted(monkeypatch, capsys):
    delta, m_ans = _mass_batch(monkeypatch, 0.05, 1e-4)     # 안전 클립을 아주 작게
    realized = float(delta.abs().sum().item()) / m_ans
    assert realized < 0.05                                  # 클립이 물려 목표에 못 미친다
    out = capsys.readouterr().out
    assert "clipped_rows=1" in out
    assert "mass_share_realized=" in out


# ── 9) 혼합 팔 명세 ─────────────────────────────────────────────────────────
def test_hybrid_arm_spec_resolves():
    spec = mm.require_arm("M_REV_PMI_CF")
    assert spec["meta_term"] == "revision_pmi_cf"
    assert spec["variant"] == "math_opt" and spec["require_meta"] is False
    assert "M_REV_PMI_CF" in mm._REV_ARMS
    assert mm.rev_anchor("M_REV_PMI_CF") == "gold_x"


# ── 10) 쉬운 문제 섞기(--all_same_frac) ─────────────────────────────────────
def test_all_same_frac_appends_easy_problems_to_the_target_share():
    bp = _pool_mod()
    groups = {f"dom{i}": _group(["7"] * 6 + ["3", "5"], [100] * 8) for i in range(8)}
    groups.update({f"easy{i}": _group(["7"] * 8, [100] * 8) for i in range(50)})
    keys, st = bp.select(groups, bp.VARIANT_A, 8192, all_same_frac=0.20)
    n_easy = sum(1 for k in keys if k.startswith("easy"))
    assert len(keys) == 10 and n_easy == 2          # 8 hard / (1-0.2) → 2 easy
    assert st["all_same_added"] == 2
    assert n_easy / len(keys) == pytest.approx(0.20, abs=0.02)
    # 기본(0)은 종전과 동일 — 쉬운 문제를 한 개도 안 붙인다
    keys0, st0 = bp.select(groups, bp.VARIANT_A, 8192)
    assert len(keys0) == 8 and st0["all_same_added"] == 0


def test_all_same_frac_respects_the_length_gate_and_is_seeded():
    bp = _pool_mod()
    groups = {"dom0": _group(["7"] * 6 + ["3", "5"], [100] * 8)}
    groups.update({f"easy{i}": _group(["7"] * 8, [100] * 8) for i in range(9)})
    groups["easy_long"] = _group(["7"] * 8, [100] * 7 + [9000])
    k1, st1 = bp.select(groups, bp.VARIANT_A, 8192, all_same_frac=0.5)
    k2, _st2 = bp.select(groups, bp.VARIANT_A, 8192, all_same_frac=0.5)
    assert k1 == k2                                  # seed 0 → 재현 가능
    assert "easy_long" not in k1 and st1["all_same_dropped_len"] == 1
