"""S3 2-시도 팔의 **트레이너 경로** 테스트 (docs/DESIGN_S3_trial2_0915.md §2·§3·§5).

`tests/test_trial2.py` 가 순수 함수 층을 고정한다면 이 파일은 그 층이 verl 배치 위에서
실제로 하는 일을 고정한다 — 0915 감사 A(비활성 슬롯이 실재 행의 복제라 KL·엔트로피·
token-mean 분모를 오염시킨다)가 다시 들어오는 것을 막는 것이 주목적이다. GPU 없이 돈다.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training import trial2 as t2  # noqa: E402

torch = pytest.importorskip("torch")

P, K = 2, 4                       # 문제 2 × K=4 → 3K=12 슬롯/문제, 배치 24행
PLEN, RLEN = 5, 6
# 문제 0: 샘플 0·2 오답 / 문제 1: 전부 정답(RETRY_ONLY_WRONG=1 이면 재시도 0)
R1 = [0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
RETRIED = [i for i, v in enumerate(R1) if v <= 0.0]


def _plan(note_on: bool = True):
    return t2.reassembly_plan(P, K, RETRIED, note_on=note_on)


# ── 슬롯 배치 ────────────────────────────────────────────────────────────────
def test_reassembly_plan_slot_order_and_activity():
    perm, stages, active = _plan()
    assert len(perm) == len(stages) == len(active) == 3 * K * P
    # 슬롯 규약: 문제 p 의 블록 안에서 0…K−1=a1, K…2K−1=note, 2K…3K−1=a2
    assert stages == [t2.stage_of(i, K) for i in range(3 * K * P)]
    # 순열은 concat 블록(a1|note|a2, 각 K·P 행)에서 뽑는 전역 인덱스다
    assert perm[:K] == [0, 1, 2, 3]                     # 문제 0 의 a1
    assert perm[K:2 * K] == [K * P + 0, K * P + 1, K * P + 2, K * P + 3]
    # 시도-1 은 전부 활성, 재시도가 없는 trial 의 note/a2 는 비활성
    for i in range(3 * K * P):
        st, tr = t2.stage_of(i, K), t2.trial_index(i, K)
        pi = i // (3 * K)
        want = 1 if st == "a1" else int((pi * K + tr) in RETRIED)
        assert active[i] == want, (i, st)
    # RETRY_ONLY_WRONG=1 → R1=1 문제(=문제 1)에는 note/a2 행이 하나도 없다
    assert sum(active[3 * K:]) == K


def test_reassembly_plan_note_mode_none_kills_note_slots():
    """NOTE_MODE=none 이면 반성문은 생성되지 않는다 — 그 슬롯이 활성이면 크레딧이 유령
    행에 실린다(설계 §2 의 각주)."""
    _, stages, active = _plan(note_on=False)
    assert all(a == 0 for s, a in zip(stages, active) if s == "note")
    assert sum(a for s, a in zip(stages, active) if s == "a2") == len(RETRIED)


def test_reassembly_puts_each_trials_a2_next_to_its_own_a1():
    """★r2≈0 의 원인 후보 «인덱스 어긋남»을 고정한다. `_s3_generate_sequences` 의 재조립을
    가짜 배치로 그대로 재연한다 — concat 블록(a1|note|a2) → `perm` → 슬롯 순서. 각 슬롯 행에
    «어느 trial 의 어느 단계 출력인가»를 태그로 싣고, a2 슬롯의 태그가 **형제 a1 슬롯과 같은
    trial** 인지, 그리고 `trial_id` 키가 세 행에서 같은지를 본다."""
    perm, stages, active = _plan()
    wrong = list(RETRIED)
    pos = {i: j for j, i in enumerate(wrong)}
    kp = K * P
    # 생성 블록: a1 은 전 행, note/a2 는 wrong 행만 만들고 나머지는 0 번 행을 빌린다
    blk = ([("a1", i) for i in range(kp)]
           + [("note", wrong[pos.get(i, 0)]) for i in range(kp)]
           + [("a2", wrong[pos.get(i, 0)]) for i in range(kp)])
    rows = [blk[p] for p in perm]
    uids = [f"u{i // (3 * K)}" for i in range(3 * K * P)]
    tkeys = [t2.trial_id(i, K, uids[(i // (3 * K)) * 3 * K]) for i in range(3 * K * P)]
    for s in range(3 * K * P):
        st, trial = rows[s]
        assert st == stages[s], (s, st, stages[s])
        sib = t2.sibling_rows(s, K)
        assert tkeys[sib["a1"]] == tkeys[sib["note"]] == tkeys[sib["a2"]] == tkeys[s]
        if not active[s]:
            continue
        # 활성 슬롯은 **자기 trial** 의 출력이어야 한다(빌린 0번 행이 아니다)
        a1_row = rows[sib["a1"]][1]
        assert trial == a1_row, (s, st, trial, a1_row)
        assert a1_row == (s // (3 * K)) * K + t2.trial_index(s, K)


# ── 보상·어드밴티지 (CREDIT vs OUTCOME) ──────────────────────────────────────
def _recredit(term, r2_by_trial):
    """트레이너의 **실제** 훅(`verl_sdc._math_trial2_recredit`)을 돈다 — `own_correct` 는
    `_compute_math_arm_stash` 가 내는 행별 정오 스칼라다. 반환 = (rewards, 갈린 uid, tel)."""
    from src.training import verl_sdc as vs  # noqa: PLC0415

    _, stages, active = _plan()
    uids, tkeys, own = [], [], []
    for i in range(3 * K * P):
        pi, j, st = i // (3 * K), t2.trial_index(i, K), t2.stage_of(i, K)
        uids.append(f"u{pi}")
        tkeys.append(f"u{pi}#t{j}")
        if st == "a1":
            own.append(R1[pi * K + j])
        elif st == "a2":
            own.append(r2_by_trial.get(f"u{pi}#t{j}", 0.0))
        else:
            own.append(0.0)   # 반성문 행의 채점값은 쓰이지 않는다(반성문은 답이 아니다)
    data = SimpleNamespace(non_tensor_batch={
        "uid": list(uids), "s3_stage": list(stages),
        "s3_active": list(active), "s3_trial": list(tkeys)})
    vs._ACTIVE_SDC_CONTEXT["trial2_term"] = term
    vs._ACTIVE_SDC_CONTEXT["trial2_note_generic"] = 0.2
    vs._TRIAL2_HISTORY.clear()
    try:
        rew = vs._math_trial2_recredit(data, own, 1)
    finally:
        vs._ACTIVE_SDC_CONTEXT.pop("trial2_term", None)
        vs._TRIAL2_HISTORY.clear()
    keys = [str(x) for x in data.non_tensor_batch["uid"]]
    tel = {"n_trials": float(K * P), "n_retried": float(len(RETRIED)),
           "r1_mean": sum(R1) / len(R1),
           "two_trial_acc": sum(1.0 for i, v in enumerate(R1)
                                if v > 0 or r2_by_trial.get(
                                    f"u{i // K}#t{i % K}", 0.0) > 0) / len(R1)}
    return rew, keys, tel, stages, active, uids, tkeys


def test_recredit_is_noop_without_stage_marks():
    """다른 팔·검증 배치는 이 훅을 지나도 **바이트 동일**이어야 한다."""
    from src.training import verl_sdc as vs  # noqa: PLC0415

    data = SimpleNamespace(non_tensor_batch={"uid": ["a", "b"]})
    totals = [0.5, 1.0]
    assert vs._math_trial2_recredit(data, totals, 3) is totals
    assert data.non_tensor_batch["uid"] == ["a", "b"]


def test_credit_vs_outcome_row_by_row():
    """u0#t0 는 재시도 성공(R2=1), u0#t2 는 실패(R2=0). γ=.6."""
    r2 = {"u0#t0": 1.0, "u0#t2": 0.0}
    g = 0.6
    for term in (t2.CREDIT_TERM, t2.OUTCOME_TERM):
        rew, keys, tel, stages, active, uids, tkeys = _recredit(term, r2)
        for i in range(3 * K * P):
            st, tk, on = stages[i], tkeys[i], active[i]
            r1 = R1[(i // (3 * K)) * K + t2.trial_index(i, K)]
            r2v = r2.get(tk, 0.0)
            if not on:
                want = 0.0
            elif st == "a1":
                want = r1 + (g * r2v if term == t2.CREDIT_TERM else 0.0)
            elif st == "note":
                want = g * r2v if term == t2.CREDIT_TERM else 0.0
            else:
                want = r2v
            assert rew[i] == pytest.approx(want), (term, i, st)
        # 중심화 그룹 키가 단계별로 갈려 있는가
        for i in range(3 * K * P):
            assert keys[i] == (f"{uids[i]}#{stages[i]}" if active[i]
                               else f"{uids[i]}#dead{i}")
        assert tel["n_trials"] == K * P
        assert tel["n_retried"] == len(RETRIED)
        assert tel["r1_mean"] == pytest.approx(sum(R1) / len(R1))
        assert tel["two_trial_acc"] == pytest.approx(7 / 8)   # 8 trial 중 u0#t2 만 실패


def test_note_span_advantage_is_gamma_centered_r2_in_credit_and_zero_in_outcome():
    """설계 §3 표의 핵심 칸: 반성문 스팬 = γ·(R2 − mean R2) (CREDIT) / 0 (OUTCOME)."""
    r2 = {"u0#t0": 1.0, "u0#t2": 0.0}
    _, stages, active = _plan()
    uids = [f"u{i // (3 * K)}" for i in range(3 * K * P)]
    tkeys = [f"u{i // (3 * K)}#t{t2.trial_index(i, K)}" for i in range(3 * K * P)]
    r1v = [R1[(i // (3 * K)) * K + t2.trial_index(i, K)] for i in range(3 * K * P)]
    r2v = [r2.get(tk, 0.0) for tk in tkeys]
    mean_r2 = (1.0 + 0.0) / 2                       # 문제 0 의 활성 note 두 행
    for term, want in ((t2.CREDIT_TERM, lambda v: 0.6 * (v - mean_r2)),
                       (t2.OUTCOME_TERM, lambda v: 0.0)):
        adv = t2.trial2_advantages(term, uids, stages, active, r1v, r2v, 0.6)
        for i in range(3 * K * P):
            if stages[i] == "note" and active[i]:
                assert adv[i] == pytest.approx(want(r2v[i])), (term, i)
            elif not active[i]:
                assert adv[i] == 0.0


# ── 비활성 슬롯 = 불활성 더미 (감사 A) ───────────────────────────────────────
class _FakeOut:
    """`_s3_blank_dead_rows` 가 만지는 최소 표면 — DataProto.batch 는 dict 처럼 쓰인다."""

    def __init__(self, n):
        self.batch = {
            "prompts": torch.full((n, PLEN), 7, dtype=torch.long),
            "responses": torch.full((n, RLEN), 3, dtype=torch.long),
            "input_ids": torch.full((n, PLEN + RLEN), 3, dtype=torch.long),
            "attention_mask": torch.ones(n, PLEN + RLEN, dtype=torch.long),
            "response_mask": torch.ones(n, RLEN, dtype=torch.long),
            "loss_mask": torch.ones(n, PLEN + RLEN, dtype=torch.long),
            "rollout_log_probs": torch.full((n, RLEN), -0.5),
        }


def _blank(out, dead):
    from src.training.verl_sdc import SDCRayPPOTrainer  # noqa: PLC0415
    shim = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    return SDCRayPPOTrainer._s3_blank_dead_rows(shim, out, dead)


def test_dead_rows_are_inert_for_loss_kl_entropy_and_token_counts():
    n = 3 * K * P
    _, _, active = _plan()
    dead = [i for i in range(n) if not active[i]]
    live = [i for i in range(n) if active[i]]
    out = _FakeOut(n)
    _blank(out, dead)
    b = out.batch
    d = torch.tensor(dead)
    # ① 손실/KL/엔트로피가 보는 마스크가 전부 0 (verl 0.9: agg_loss·apply_kl_penalty 모두
    #    response_mask 로 곱하고 분모에서도 제외한다)
    assert b["response_mask"][d].sum() == 0
    assert b["attention_mask"][d, PLEN:].sum() == 0
    assert b["loss_mask"][d, PLEN:].sum() == 0
    # ② 토큰 수에 들어가지 않는다 — token-mean 분모 = response_mask.sum()
    assert int(b["response_mask"].sum()) == len(live) * RLEN
    # ③ 내용이 남지 않는다(복제된 오답 본문이 디코드 경로로 새지 않게)
    assert int(b["responses"][d].abs().sum()) == 0
    assert b["rollout_log_probs"][d].abs().sum() == 0
    # ④ 프롬프트 attention 은 **살아 있다** — 전 구간 0 인 행은 varlen 경로에서 죽는다
    assert int(b["attention_mask"][d, :PLEN].sum()) == len(dead) * PLEN
    # ⑤ 살아 있는 행은 한 값도 바뀌지 않는다
    l = torch.tensor(live)
    assert int(b["response_mask"][l].sum()) == len(live) * RLEN
    assert int(b["responses"][l].sum()) == len(live) * RLEN * 3


def test_blank_dead_rows_noop_when_nothing_dead():
    out = _FakeOut(4)
    before = {k: v.clone() for k, v in out.batch.items()}
    _blank(out, [])
    for k, v in out.batch.items():
        assert torch.equal(v, before[k]), k


# ── 시도-2 프롬프트에 오답 본문이 없다 (설계 §4 마지막 항) ────────────────────
class _Tok:
    """chat 템플릿 없이 역할을 그대로 적는 최소 토크나이저(렌더러가 요구하는 표면만)."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                            **kw):
        s = "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in msgs)
        return s + ("<assistant>" if add_generation_prompt else "")


A1_TEXT = "I set x equal to 41 and concluded \\boxed{41}. The parity argument fails here."


def test_attempt2_prompt_contains_no_substring_of_attempt1():
    tok = _Tok()
    prob = "What is the smallest prime above 40?"
    note, why = t2.clean_note("assumed the parity argument applies to primes", "41")
    assert why == "ok"
    p2 = t2.attempt2_prompt(tok, "math_opt", prob, note)
    # 오답 본문의 어떤 조각도(4단어 이상 연속) 시도-2 프롬프트에 없다
    w = A1_TEXT.split()
    for i in range(len(w) - 3):
        assert " ".join(w[i:i + 4]) not in p2
    assert "\\boxed{41}" not in p2 and "41" not in p2.split("Note:")[-1]
    # 반성문 생성 프롬프트에는 **들어 있다**(본문이 들어가는 유일한 자리)
    assert A1_TEXT in t2.note_ask_prompt(tok, "math_opt", prob, A1_TEXT)


def test_none_mode_attempt2_prompt_is_byte_identical_to_fact_reference():
    tok = _Tok()
    prob = "p"
    assert t2.attempt2_prompt(tok, "math_opt", prob, "") == t2.fact_prompt(tok, "math_opt", prob)


# ── 중단 규칙 (설계 §4) ──────────────────────────────────────────────────────
def test_stop_rule_fires_on_sustained_r1_drop():
    h = [{"r1_mean": 0.70, "note_generic": 0.2} for _ in range(5)]
    assert t2.stop_reason(h) is None
    h += [{"r1_mean": 0.688, "note_generic": 0.2}] * 2          # 1.2pp 낮지만 2스텝
    assert t2.stop_reason(h) is None
    h += [{"r1_mean": 0.688, "note_generic": 0.2}]              # 3스텝 연속
    assert "r1_mean" in t2.stop_reason(h)


def test_stop_rule_ignores_drop_smaller_than_1pp_and_recovery():
    h = [{"r1_mean": 0.70, "note_generic": 0.2} for _ in range(5)]
    assert t2.stop_reason(h + [{"r1_mean": 0.695, "note_generic": 0.2}] * 5) is None
    bad = [{"r1_mean": 0.60, "note_generic": 0.2}] * 2
    assert t2.stop_reason(h + bad + [{"r1_mean": 0.71, "note_generic": 0.2}]) is None


def test_stop_rule_fires_on_degenerate_notes():
    h = [{"r1_mean": 0.70, "note_generic": 1.0}] * 3
    assert "note_generic" in t2.stop_reason(h)
    assert t2.stop_reason([{"r1_mean": 0.70, "note_generic": 1.0}] * 2) is None
    # 측정 못 한 스텝(None)은 위반이 아니다 — «못 봤다»가 «통과»가 되면 안 되듯 그 역도 아니다
    assert t2.stop_reason([{"r1_mean": 0.70, "note_generic": None}] * 5) is None


# ── cf_prefix_agent 워커 등록 (0915 스모크 실패 회귀) ─────────────────────────
# 실패 재현: `agent_loop_config_path` 가 null 인 채로 `agent_name="cf_prefix_agent"`
# 를 실으면 Ray 워커가
#   AssertionError: Agent loop cf_prefix_agent not registered,
#   registered agent loops: dict_keys(['single_turn_agent', 'tool_agent'])
# 로 잡 전체를 죽인다(agent_loop.py:692). 드라이버의 @register 는 그 프로세스에
# 아무 영향이 없다 — 그래서 yaml 경로가 **config 에** 있어야 한다.
def test_cf_agent_yaml_exists_and_targets_the_loop_class():
    import importlib

    from omegaconf import OmegaConf

    entries = OmegaConf.load(t2.cf_agent_config_path())
    names = {e.name: e._target_ for e in entries}
    assert "cf_prefix_agent" in names, names
    mod, _, cls = names["cf_prefix_agent"].rpartition(".")
    assert hasattr(importlib.import_module(mod), cls)


def test_ensure_cf_agent_registered_fills_empty_path():
    rollout = SimpleNamespace(agent=SimpleNamespace(agent_loop_config_path=None))
    path = t2.ensure_cf_agent_registered(rollout)
    assert path.endswith(t2.CF_AGENT_CONFIG_REL)
    assert rollout.agent.agent_loop_config_path == path


def test_ensure_cf_agent_registered_respects_explicit_path():
    rollout = SimpleNamespace(agent=SimpleNamespace(agent_loop_config_path="/x/other.yaml"))
    assert t2.ensure_cf_agent_registered(rollout) == "/x/other.yaml"
    assert rollout.agent.agent_loop_config_path == "/x/other.yaml"


# ── 단계별 판 폭 맞추기 (`_s3_harmonize`) ────────────────────────────────────
# 0915 스모크: 반성문 호출의 프롬프트(시도-1 본문이 들어간다)가 rollout.prompt_length 를
# 넘고 verl 의 agent-loop 은 cf 경로 프롬프트를 **자르지 않고 그 길이로 패딩**하므로 판 폭이
# 시도-1 과 어긋났다(attention_mask [(5120,), (8788,)]). 여기서 그 수리를 고정한다.
PAD_ID = 11


class _FakeProto:
    """`_s3_harmonize` 가 만지는 최소 표면(DataProto: .batch / .non_tensor_batch / .pop)."""

    def __init__(self, n, plen, rlen, *, with_logprobs=True, valid_p=None, valid_r=None):
        vp = plen if valid_p is None else valid_p     # 왼쪽 패딩 뒤 유효 프롬프트 토큰 수
        vr = rlen if valid_r is None else valid_r     # 오른쪽 패딩 앞 유효 응답 토큰 수
        self.plen, self.rlen, self.vp, self.vr = plen, rlen, vp, vr
        pm = torch.zeros(n, plen, dtype=torch.long)
        pm[:, plen - vp:] = 1
        rm = torch.zeros(n, rlen, dtype=torch.long)
        rm[:, :vr] = 1
        attn = torch.cat([pm, rm], dim=1)
        self.batch = {
            "prompts": torch.where(pm.bool(), torch.full_like(pm, 5), torch.full_like(pm, PAD_ID)),
            "responses": torch.where(rm.bool(), torch.full_like(rm, 3), torch.full_like(rm, PAD_ID)),
            "input_ids": torch.cat([
                torch.where(pm.bool(), torch.full_like(pm, 5), torch.full_like(pm, PAD_ID)),
                torch.where(rm.bool(), torch.full_like(rm, 3), torch.full_like(rm, PAD_ID))], dim=1),
            "attention_mask": attn,
            "position_ids": torch.clip(torch.cumsum(attn, dim=-1) - 1, min=0),
            "response_mask": rm.clone(),
            "loss_mask": attn.clone(),
        }
        if with_logprobs:
            self.batch["rollout_log_probs"] = torch.where(
                rm.bool(), torch.full_like(rm, -1, dtype=torch.float), torch.zeros_like(rm, dtype=torch.float))
        self.non_tensor_batch = {"uid": ["u"] * n, "extra": [1] * n}
        self.meta_info = {}

    def pop(self, batch_keys=(), non_tensor_batch_keys=()):
        for k in batch_keys or ():
            self.batch.pop(k, None)
        for k in non_tensor_batch_keys or ():
            self.non_tensor_batch.pop(k, None)


def _harmonize(outs, max_prompt_length=0):
    from src.training.verl_sdc import SDCRayPPOTrainer  # noqa: PLC0415
    shim = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=PAD_ID),
                           config=SimpleNamespace(data=SimpleNamespace(
                               max_prompt_length=max_prompt_length)),
                           _S3_ID_KEYS=SDCRayPPOTrainer._S3_ID_KEYS,
                           _S3_PROMPT_KEYS=SDCRayPPOTrainer._S3_PROMPT_KEYS,
                           _S3_RESP_KEYS=SDCRayPPOTrainer._S3_RESP_KEYS,
                           _S3_FULL_KEYS=SDCRayPPOTrainer._S3_FULL_KEYS,
                           _s3_unify_meta_info=SDCRayPPOTrainer._s3_unify_meta_info)
    return SDCRayPPOTrainer._s3_harmonize(shim, outs)


def _stages():
    # 시도 1(짧은 프롬프트) / 반성문(프롬프트가 3배 길다) / 시도 2(응답이 더 길다)
    return [_FakeProto(4, 5, 6, valid_p=3, valid_r=4),
            _FakeProto(4, 15, 6, with_logprobs=False, valid_p=14, valid_r=2),
            _FakeProto(4, 7, 9, with_logprobs=False, valid_p=7, valid_r=9)]


def test_harmonize_equalizes_every_width():
    outs = _harmonize(_stages())
    keys = set(outs[0].batch)
    assert keys == set.intersection(*[set(o.batch) for o in outs])
    assert "rollout_log_probs" not in keys        # 교집합 — cf 경로는 안 싣는다
    assert set(outs[0].non_tensor_batch) == {"uid", "extra"}
    for k in keys:
        assert len({tuple(o.batch[k].shape) for o in outs}) == 1, k
    b = outs[0].batch
    assert b["prompts"].shape[-1] == 15 and b["responses"].shape[-1] == 9
    assert b["attention_mask"].shape[-1] == 15 + 9 == b["input_ids"].shape[-1]
    assert b["response_mask"].shape[-1] == 9


def test_harmonize_pads_left_on_prompt_right_on_response_with_zero_masks():
    stages = _stages()
    outs = _harmonize(stages)
    for o, src in zip(outs, stages):
        b = o.batch
        dp, dr = 15 - src.plen, 9 - src.rlen
        # 유효 토큰 수가 보존된다(패딩은 마스크 0)
        assert int(b["attention_mask"].sum()) == 4 * (src.vp + src.vr)
        assert int(b["response_mask"].sum()) == 4 * src.vr
        assert int(b["loss_mask"].sum()) == 4 * (src.vp + src.vr)
        # 프롬프트는 **왼쪽**, 응답은 **오른쪽**으로 늘어난다
        if dp:
            assert int(b["attention_mask"][:, :dp].sum()) == 0
            assert (b["prompts"][:, :dp] == PAD_ID).all()
            assert (b["input_ids"][:, :dp] == PAD_ID).all()
        if dr:
            assert int(b["attention_mask"][:, -dr:].sum()) == 0
            assert (b["responses"][:, -dr:] == PAD_ID).all()
            assert (b["input_ids"][:, -dr:] == PAD_ID).all()
        # 프롬프트|응답 경계가 옮겨졌어도 판 = [프롬프트, 응답] 이다
        assert torch.equal(b["input_ids"], torch.cat([b["prompts"], b["responses"]], dim=1))
        # 응답 마스크는 판의 응답 구간과 일치한다
        assert torch.equal(b["response_mask"], b["attention_mask"][:, 15:])


def test_harmonize_position_ids_follow_verl_convention():
    outs = _harmonize(_stages())
    for o in outs:
        b = o.batch
        attn = b["attention_mask"]
        assert torch.equal(b["position_ids"],
                           torch.clip(torch.cumsum(attn, dim=-1) - 1, min=0))
        for i in range(attn.shape[0]):
            valid = b["position_ids"][i][attn[i].bool()]
            assert valid[0] == 0
            assert torch.equal(valid, torch.arange(len(valid)))   # 유효 토큰에서 단조 증가


def test_harmonize_is_a_noop_when_widths_already_agree():
    outs = [_FakeProto(3, 5, 6, with_logprobs=False) for _ in range(3)]
    before = [{k: v.clone() for k, v in o.batch.items()} for o in outs]
    _harmonize(outs)
    for o, b0 in zip(outs, before):
        for k, v in o.batch.items():
            assert torch.equal(v, b0[k]), k


def test_harmonize_then_concat_and_blank_dead_rows_stay_correct():
    outs = _harmonize(_stages())
    n = outs[0].batch["prompts"].shape[0]
    cat = _FakeProto(3 * n, 15, 9, with_logprobs=False)
    for k in cat.batch:
        cat.batch[k] = torch.cat([o.batch[k] for o in outs], dim=0)
    dead = [0, 5, 11]
    _blank(cat, dead)
    b, d = cat.batch, torch.tensor(dead)
    assert int(b["response_mask"][d].sum()) == 0
    assert int(b["attention_mask"][d, 15:].sum()) == 0
    assert int(b["loss_mask"][d, 15:].sum()) == 0
    assert int(b["responses"][d].sum()) == 0      # _blank 의 pad id = 0 (shim)
    # 프롬프트 attention 은 살아 있다(전 구간 0 인 행은 varlen 경로에서 죽는다)
    assert int(b["attention_mask"][d, :15].sum()) > 0
    live = [i for i in range(3 * n) if i not in dead]
    assert int(b["response_mask"][torch.tensor(live)].sum()) > 0


def test_harmonize_refuses_a_width_it_cannot_place():
    outs = _stages()
    outs[0].batch["mystery"] = torch.zeros(4, 2)
    outs[1].batch["mystery"] = torch.zeros(4, 2)
    outs[2].batch["mystery"] = torch.zeros(4, 2)
    with pytest.raises(RuntimeError, match="패딩 규약"):
        _harmonize(outs)


def test_harmonize_unifies_meta_info_so_concat_does_not_conflict():
    """`DataProto.concat` 은 `metrics` 를 뺀 겹치는 meta_info 키에 `assert merged == v` 를
    걸고 죽는다(protocol.py:936-952) — 호출마다 자기 `timing` 이 달려 오므로 세 파트가
    반드시 충돌한다(0915 스모크: Conflicting values for meta_info key 'timing')."""
    outs = _stages()
    for i, o in enumerate(outs):
        o.meta_info = {"timing": {"gen": 1.5 * (i + 1), "n": i + 1, "tag": f"t{i}"},
                       "validate": False, "who": f"stage{i}"}
    _harmonize(outs)
    # 세 파트가 **같은** meta_info 를 들고 있다 → concat 의 assert 가 닿지 않는다
    assert all(o.meta_info == outs[0].meta_info for o in outs)
    assert outs[0].meta_info["timing"]["gen"] == pytest.approx(1.5 + 3.0 + 4.5)
    assert outs[0].meta_info["timing"]["n"] == 6           # 수치 항목은 합
    assert outs[0].meta_info["timing"]["tag"] == "t0"      # 수치가 아니면 시도-1 값
    assert outs[0].meta_info["who"] == "stage0"            # 충돌 키는 시도-1 값
    assert outs[0].meta_info["validate"] is False          # 일치 키는 그대로
    # 진짜 DataProto 로 concat 해서 assert 가 사라졌음을 확인한다
    from verl import DataProto  # noqa: PLC0415
    parts = [DataProto.from_dict(tensors=dict(o.batch), meta_info=o.meta_info) for o in outs]
    cat = DataProto.concat(parts)
    assert len(cat) == sum(len(p) for p in parts)
    assert cat.meta_info["timing"]["gen"] == pytest.approx(9.0)


def test_unify_meta_info_tolerates_missing_and_empty_timing():
    outs = _stages()
    outs[1].meta_info = {"timing": {}}
    outs[2].meta_info = {}
    _harmonize(outs)
    assert all(o.meta_info == outs[0].meta_info for o in outs)
    assert "timing" not in outs[0].meta_info


# ── E-131 불변식: 프롬프트 폭 == data.max_prompt_length (0916) ───────────────
def test_harmonize_pads_prompts_up_to_the_config_width():
    """하류 스태시는 «배치 프롬프트 폭 == data.max_prompt_length» 를 불변식으로 쓴다
    (E-131). 관측 최대(15)가 아니라 **config 폭**으로 맞춰야 한다."""
    outs = _harmonize(_stages(), max_prompt_length=20)
    for o in outs:
        b = o.batch
        assert b["prompts"].shape[-1] == 20
        assert b["attention_mask"].shape[-1] == 20 + 9
        assert torch.equal(b["input_ids"], torch.cat([b["prompts"], b["responses"]], dim=1))
        assert torch.equal(b["response_mask"], b["attention_mask"][:, 20:])
        assert torch.equal(b["position_ids"],
                           torch.clip(torch.cumsum(b["attention_mask"], dim=-1) - 1, min=0))
    # 유효 토큰 수는 보존된다(늘어난 폭은 전부 마스크 0)
    assert int(outs[0].batch["attention_mask"].sum()) == 4 * (3 + 4)


def test_harmonize_dies_when_a_stage_prompt_exceeds_the_config_width():
    with pytest.raises(RuntimeError, match="data.max_prompt_length"):
        _harmonize(_stages(), max_prompt_length=10)   # 반성문 단계가 15 > 10


# ── E-131 길이-0 검사와 비활성 슬롯 (0916) ──────────────────────────────────
def _guard(valid, active=None, width=8, expected=8, step=1):
    from src.training.verl_sdc import _countdown_batch_geometry_guard  # noqa: PLC0415
    return _countdown_batch_geometry_guard(prompts_width=width, expected_width=expected,
                                           response_valid_lengths=valid, step=step,
                                           active_mask=active)


def test_e131_ignores_zero_length_rows_that_are_inactive_slots():
    # S3 의 비활성 슬롯은 설계상 유효 응답 토큰 0 이다(_s3_blank_dead_rows)
    _guard([5, 0, 0, 7], active=[1, 0, 0, 1])


def test_e131_still_trips_on_an_active_zero_length_row():
    with pytest.raises(RuntimeError, match="길이 0 응답"):
        _guard([5, 0, 0, 7], active=[1, 0, 1, 1])


def test_e131_semantics_unchanged_without_an_active_mask():
    _guard([5, 7])
    with pytest.raises(RuntimeError, match="길이 0 응답"):
        _guard([5, 0])
    with pytest.raises(RuntimeError, match="배치 프롬프트 폭"):
        _guard([5, 7], width=9, expected=8)


def test_e131_rejects_a_mask_of_the_wrong_length():
    with pytest.raises(RuntimeError, match="active_mask"):
        _guard([5, 0], active=[1])


# ── 반성문 요청 프롬프트의 시도-1 본문 상한 (NOTE_CTX_MAX_TOKENS) ─────────────
class _CountTok(_Tok):
    """공백 단위 토크나이저 — encode/decode 왕복이 항등이다."""

    def encode(self, s, add_special_tokens=False):
        return s.split(" ")

    def decode(self, ids):
        return " ".join(ids)


def test_note_ask_prompt_keeps_only_the_tail_of_attempt1():
    tok = _CountTok()
    a1 = " ".join(f"w{i}" for i in range(200)) + " so \\boxed{41}"
    kept = t2.cap_attempt1_text(tok, a1, max_tokens=8)
    assert kept.startswith(t2.NOTE_CTX_ELLIPSIS)
    assert kept.endswith("\\boxed{41}")            # 꼬리(최종 답)를 남긴다
    assert len(kept.split(" ")) == 8 + 1           # 8 토큰 + "…" 조각
    assert "w0 " not in kept
    assert t2.cap_attempt1_text(tok, "a b c", max_tokens=8) == "a b c"


def test_note_ask_prompt_respects_the_env_cap(monkeypatch):
    tok = _CountTok()
    a1 = " ".join(f"w{i}" for i in range(500))
    assert t2.note_ctx_max_tokens() == 1536          # MAX_PROMPT=3072 예산 안에 든다
    monkeypatch.setenv("NOTE_CTX_MAX_TOKENS", "16")
    assert t2.note_ctx_max_tokens() == 16
    p = t2.note_ask_prompt(tok, "math_opt", "Q?", a1)
    assert "w499" in p and "w0 " not in p
    assert t2.NOTE_ASK in p


# ── NOTE_MODE=notx: 노트 자리가 죽는다(0916) ─────────────────────────────────
def test_notx_mode_reassembly_has_no_active_note_rows():
    """`notx` 는 사전 패스가 없다 → `note_on=False` 와 같은 재조립이고 활성 note 행이 0.
    그래서 CREDIT 의 «반성문 스팬 γ·R2» 항이 사라지고, CREDIT 과 OUTCOME 은 시도-1 행의
    γ·R2 항 하나만으로 갈린다."""
    assert not t2.note_stage_on("notx") and not t2.note_stage_on("none")
    _, stages, active = _plan(note_on=t2.note_stage_on("notx"))
    assert sum(a for s, a in zip(stages, active) if s == "note") == 0
    uids = [f"u{i // (3 * K)}" for i in range(3 * K * P)]
    keys = t2.row_group_keys(uids, stages, active)
    assert not any(k.endswith("#note") for k in keys)
    r1 = [1.0 if s == "a1" else 0.0 for s in stages]
    cr = t2.build_row_rewards(t2.CREDIT_TERM, stages, active,
                              [0.0] * len(stages), [1.0] * len(stages), 0.6)
    ou = t2.build_row_rewards(t2.OUTCOME_TERM, stages, active,
                              [0.0] * len(stages), [1.0] * len(stages), 0.6)
    del r1
    # 두 팔이 다른 곳은 **활성 시도-1 행뿐**(= γ·R2), note 행은 양쪽 다 0.
    diff = {i for i in range(len(stages)) if cr[i] != ou[i]}
    assert diff and all(stages[i] == "a1" for i in diff)
    assert all(cr[i] == 0.0 == ou[i] for i, s in enumerate(stages) if s == "note")
