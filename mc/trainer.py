#!/usr/bin/env python
r"""verl 진입점 `python -u -m mc.trainer <hydra overrides…>`. 한 스텝: ①`compute_advantage` 앞 — 결과
보상만 ②verl GRPO 중심화 ③뒤 — `add_span_credit`(CH-Fork 곱 재배분 · 멈춤 순위 크레딧 가산)."""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

import ray

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ABORT_EXIT_CODE = 75        #: 중단 규칙 = 의도된 정지. 큐·재시도 래퍼가 이 rc 는 재시도 안 함


class MCAbort(RuntimeError):
    """사전등록 중단(PFX 파괴 가드 — 맞은 첫 답을 깬다). 크래시가 아니라 의도된 정지다."""


#: 훅들이 읽는 프로세스 전역(트레이너·토크나이저). Ray 워커가 아니라 **메인프로세스** 전용.
CTX: dict = {}

#: 워커에 실을 환경변수 — Ray 워커는 드라이버 env 를 **상속하지 않는다**(사고 2회).
WORKER_ENV_KEYS = ("LABEL", "MC_CKPT_DIR", "PROMPT_VARIANT", "MC_DUMP_ADV",
                   "OUTCOME_MODE", "PFX_WEIGHT_KEY", "PFX_TRUNC", "PFX_FORK", "PFX_REP",
                   "PFX_BREAK_W", "PFX_ADV_CAP", "PFX_DISTILL", "PFX_DISTILL_SHUF",
                   "PFX_ALLOC", "PFX_ALLOC_SHUF", "PFX_KEEP", "PFX_GUARD_ABS")


def reward_loop_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kw):
    """agent-loop 자리표시자 0 — 실제 보상은 `populate_token_rewards` 가 그룹 구조를 보고 쓴다."""
    return 0.0


# ── 보상 훅(compute_advantage 자리) ──────────────────────────────────────────
def decode_active(tokenizer, data, active=None) -> list[str]:
    """활성 행만 디코드한다(죽은 슬롯은 유효 응답 토큰이 0 — pad 만 든 행을 채점하지 않는다)."""
    plen = data.batch["prompts"].shape[-1]
    am, resp = data.batch["attention_mask"], data.batch["responses"]
    out = []
    for i in range(len(data)):
        if active is not None and not active[i]:
            out.append("")
            continue
        n = int(am[i, plen:].sum())
        out.append(tokenizer.decode([int(t) for t in resp[i, :n]], skip_special_tokens=True))
    return out


def check_prompt_lengths(tokenizer, parquet_path: str, max_prompt_length: int) -> dict:
    r"""학습 parquet 의 **모든** 1턴 프롬프트가 판 폭 안에 드는지(넘치면 즉사 — RLHFDataset 은 넘치는
    행을 **조용히 버려** 어려운 문제만 사라진다, 0921)와 system 프롬프트가 PROMPT_VARIANT 인지 본다.
    SPONT_PFX 행은 `extra_info.prefix` 토큰까지 센다(`prefix_agent_loop` 과 같은 따로 인코딩)."""
    import pandas as pd

    from mc.context import VARIANTS, default_variant, turn1_prompt  # noqa: PLC0415
    cap = int(max_prompt_length or 0)
    df = pd.read_parquet(parquet_path)
    # ★프롬프트는 parquet 에 구워져 들어온다 — 손잡이와 갈리면 대조군이 되므로 fail-closed.
    want, vname = VARIANTS[default_variant()], default_variant()
    bad = sum(1 for p in df["prompt"]
              if str((list(p)[0] or {}).get("content", "")) != want)
    if bad:
        raise ValueError(
            f"[MC] PROMPT_VARIANT={vname} 인데 {parquet_path} 의 {bad}/{len(df)} 행 system "
            "프롬프트가 그 변형과 다르다 — 변형은 parquet 빌드 시점에 굽는다(mc/run.sh 머리말).")
    pre = [str((e or {}).get("prefix") or "") for e in df.get("extra_info", [None] * len(df))]
    lens = [len(tokenizer.encode(turn1_prompt(tokenizer, str(q)), add_special_tokens=False))
            + (len(tokenizer.encode(p, add_special_tokens=False)) if p else 0)
            for q, p in zip(df["problem"], pre)]
    srt = sorted(lens)
    stat = {"n": len(lens), "max": max(lens, default=0),
            "p99": srt[min(len(lens) - 1, int(0.99 * len(lens)))] if lens else 0,
            "cap": cap, "n_over": sum(1 for x in lens if cap and x > cap)}
    print(f"[MC] 프롬프트 길이 n={stat['n']} max={stat['max']} p99={stat['p99']} "
          f"cap={cap} over={stat['n_over']}", flush=True)
    if stat["n_over"]:
        raise ValueError(f"[MC] 학습 프롬프트 {stat['n_over']}/{stat['n']} 행이 data.max_prompt_length="
                         f"{cap} 를 넘는다(최대 {stat['max']}) — verl 은 조용히 버리므로 어려운 문제만 빠진다. "
                         f"MAX_PROMPT 를 {stat['max'] + 64} 이상으로 올려라.")
    return stat


@functools.cache
def prefix_agent_loop_cls():
    r"""SPONT_PFX 에이전트 루프 — verl `SingleTurnAgentLoop` 에서 **프롬프트 토큰만** 바꾼다: chat 템플릿
    (assistant 머리 포함) 토큰 + `extra_info.prefix` 토큰 = 탐침 `turn1_prompt(…) + prefix` 와 같은 자리. 앞부분은
    prompt_ids 라 응답·손실 마스크 밖이다(verl single_turn_agent_loop.py `run`: response_mask = 생성 토큰만)."""
    from verl.experimental.agent_loop.single_turn_agent_loop import SingleTurnAgentLoop

    class PrefixAgentLoop(SingleTurnAgentLoop):
        async def run(self, sampling_params, **kwargs):
            self.mc_prefix = str((kwargs.get("extra_info") or {}).get("prefix") or "")
            return await super().run(sampling_params, **kwargs)

        async def apply_chat_template(self, messages, **kwargs):
            ids = list(await super().apply_chat_template(messages, **kwargs))
            pre = getattr(self, "mc_prefix", "")
            ids += self.tokenizer.encode(pre, add_special_tokens=False) if pre else []
            if len(ids) > self.prompt_length:        # verl 은 왼쪽을 조용히 자른다 — 즉사가 낫다
                raise ValueError(f"[MC] PFX 프롬프트 {len(ids)} > prompt_length {self.prompt_length}")
            return ids
    return PrefixAgentLoop


def prefix_agent_loop(**kw):
    """hydra `_target_`(mc/agent_loop.yaml — mc/run.sh 가 PFX 에만 건다). 앞부분이 없는 행은 원판과 같다."""
    return prefix_agent_loop_cls()(**kw)


def col(nt, name: str) -> list:
    """`non_tensor_batch` 컬럼 — flat 컬럼이 없으면 `extra_info` 로 폴백, 둘이 어긋나면 즉사."""
    v = nt.get(name)
    ei = nt.get("extra_info")
    alt = None
    if ei is not None:
        cand = [(e or {}).get(name) for e in list(ei)]
        if all(c is not None for c in cand):
            alt = cand
    if v is None and alt is None:
        raise RuntimeError(f"[MC] 배치에 '{name}' 컬럼이 없다(extra_info 에도 없다).")
    if v is not None and alt is not None and list(v) != list(alt):
        raise RuntimeError(f"[MC] '{name}' 이 flat 컬럼과 extra_info 에서 다르다.")
    return list(v if v is not None else alt)


def populate_token_rewards(data, algo_config):
    """★mc 의 **유일한** 보상 훅 → (data, 구간 크레딧, 계기). 크레딧은 `add_span_credit` 가 얹는다."""
    import torch
    from mc.train_hook import token_rewards  # noqa: PLC0415

    trainer = CTX.get("trainer")
    step = int(getattr(trainer, "global_steps", 0) or 0)
    tlr, credit, tel = token_rewards(data, CTX["tokenizer"], trainer, step)
    data.batch["token_level_rewards"] = torch.as_tensor(
        tlr, dtype=torch.float32, device=data.batch["responses"].device)
    return data, credit, tel


def _split_returns(data):
    """(advantages, returns) — verl GRPO 는 같은 텐서를 두 키에 꽂는다(alias → returns=None, 2배 방지)."""
    adv = data.batch["advantages"]
    ret = data.batch.get("returns")
    return adv, (None if ret is not None and ret.data_ptr() == adv.data_ptr() else ret)


def add_span_credit(data, credit: dict, tel: dict, step=0):
    r"""구간 크레딧을 **advantages 의 그 자리에** 더한다(중심화 뒤·마스크 밖 버림, 0921). CH-Fork(수정 28)
    는 먼저 행 안 합 보존 재배분(곱), 멈춤(수정 33) 같은 딕셔너리 크레딧은 더한다(합). `mass_share`(로그) =
    Σ|크레딧|/(Σ|크레딧|+Σ|결과 adv|) — token-mean(유효 토큰마다 1/batch_num_tokens) 기준, seq-mean-token-mean
    이면 경고. critic 용 returns 도."""
    import torch

    from mc.train_hook import outcome_mode  # noqa: PLC0415
    dump_n = int(float(os.environ.get("MC_DUMP_ADV") or 0))
    om = outcome_mode()
    if not credit and dump_n <= 0 and not any(tel.get(k) for k in ("pfx_weights", "ds_credit", "keep_credit")) \
            and not os.environ.get("PFX_ADV_CAP"):
        return data
    adv, ret = _split_returns(data)
    mask = data.batch.get("response_mask")
    T = adv.shape[-1]
    before = adv.detach().clone() if dump_n > 0 else None
    cnum = cden = 0.0
    for i, ws in (tel.get("pfx_weights") or {}).items():   # CH-Fork(수정 28): 행 안 합 보존 재배분(곱) — 결과 adv 유지
        k = min(len(ws), T)
        w = torch.as_tensor(ws[:k], dtype=adv.dtype, device=adv.device)
        if ds := (tel.get("ds_credit") or {}).get(i):      # 56c: CH 이동 adv·(w−1) 과 리셋 가산의 부호 충돌 몫(계기만)
            j0, v = int(ds[0]), torch.as_tensor(ds[1], dtype=adv.dtype, device=adv.device)
            n = max(0, min(k, j0 + len(v)) - j0)
            mv, v = adv[i, j0:j0 + n] * (w[j0:j0 + n] - 1), v[:n]
            cnum, cden = cnum + float(v.abs()[mv * v < 0].sum()), cden + float(v.abs().sum())
        adv[i, :k] *= w
        if ret is not None:
            ret[i, :k] *= w
    if cden > 0:
        tel["ch_ds_conflict"] = cnum / cden
        print(f"[MC][CONFLICT] step={step} ch_ds_conflict={tel['ch_ds_conflict']:.4f}", flush=True)
    if (cap := os.environ.get("PFX_ADV_CAP", "")):     # 수정 48: CH×BREAK_W 말 단위 adv 스파이크 상한(첫 풀이 보호 가설)
        cap = float(cap)
        if cap <= 0:
            raise ValueError(f"[MC] PFX_ADV_CAP={cap} — 양수만(미설정 = 상한 없음)")
        live = mask.bool() if mask is not None else torch.ones_like(adv, dtype=torch.bool)
        tel["adv_absmax"] = float(adv[live].abs().max().item()) if live.any() else 0.0
        tel["adv_capped_frac"] = float((adv[live].abs() > cap).float().mean().item()) if live.any() else 0.0
        print(f"[MC][CAP] step={step} adv_absmax={tel['adv_absmax']:.2f} capped_frac={tel['adv_capped_frac']:.4f}", flush=True)
        adv.clamp_(-cap, cap)
        if ret is not None:
            ret.clamp_(-cap, cap)
    drop = list(tel.get("pfx_drop") or [])
    if drop and mask is not None:   # 수정 32c: 잘린 행은 token-mean 분모·KL 에서도 뺀다(DAPO). 사본(attention_mask 의 view
        orig = mask                 # 일 수 있음) · 첫 토큰(adv 0)은 남겨 전부 잘린 미니배치의 0/0 을 막는다
        mask = data.batch["response_mask"] = mask.clone()
        mask[drop, 1:] = 0
        for i, j0 in (tel.get("pfx_keep_from") or {}).items():   # 수정 43: 잘린 행의 반복 구간은 손실에 남긴다(반복 벌만 받게)
            mask[i, int(j0):] = orig[i, int(j0):]
    outcome_mass = float((adv.abs() * (mask if mask is not None else 1.0)).sum().item())
    if credit and (cap := float(tel.get("credit_cap") or 0.0)) > 0:   # 수정 34: 가산 크레딧 총량 ≤ cap/(1−cap)·결과 adv 총량
        tot = sum(abs(v) for _, vals in credit.values() for v in vals)
        tel["credit_scale"] = k = min(1.0, cap / (1.0 - cap) * outcome_mass / max(tot, 1e-9))
        credit = {i: (j0, [k * v for v in vals]) for i, (j0, vals) in credit.items()}
    credit_mass, extra = 0.0, {k: 0.0 for k in ("ds_credit", "keep_credit")}   # 수정 51–53 증류·유지는 상한 밖(가산)
    spans = [(k, i, j0, vals) for k in extra for i, (j0, vals) in (tel.get(k) or {}).items()]
    for src, i, j0, vals in [*spans, *((None, i, j0, vals) for i, (j0, vals) in credit.items())]:
        for t, v in enumerate(vals):
            j = int(j0) + t
            if j >= T or (mask is not None and float(mask[i, j]) <= 0):
                continue
            adv[i, j] += float(v)
            if ret is not None:
                ret[i, j] += float(v)
            credit_mass += abs(float(v))
            if src:
                extra[src] += abs(float(v))
    mode = str(getattr(getattr(getattr(getattr(CTX.get("trainer"), "config", None),
                                       "actor_rollout_ref", None), "actor", None),
                       "loss_agg_mode", "token-mean") or "token-mean")
    if mode not in ("token-mean", "token-sum", "seq-mean-token-sum",
                    "seq-mean-token-sum-norm"):
        print(f"[MC][WARN] loss_agg_mode={mode} — 토큰 가중이 행 길이에 반비례하므로 "
              "mass_share 가 «손실에 실제 들어가는 몫»과 갈린다(token-mean 기준 정의).",
              flush=True)
    tel["mass_share"] = credit_mass / max(1e-9, credit_mass + outcome_mass)
    for k, m in extra.items():                 # ds_share(수정 52 검출력 ≥ .02) · keep_share(수정 53 ≤ .30)
        tel[k.replace("credit", "share")] = m / max(1e-9, credit_mass + outcome_mass)
    print(f"[MC] step={step} credit_mass={credit_mass:.4f} credit_scale={tel.get('credit_scale', 1.0):.3f} "
          f"outcome_mass={outcome_mass:.4f} mass_share={tel['mass_share']:.4f} ds_share={tel['ds_share']:.4f} keep_share={tel['keep_share']:.4f} "
          f"outcome_mode={om}", flush=True)
    if dump_n > 0 and int(step) <= dump_n:
        dump_advantages(data, credit, tel, step, before)
    return data


def dump_advantages(data, credit: dict, tel: dict, step, before) -> str | None:
    r"""`MC_DUMP_ADV=N` → step ≤ N 의 adv 전/후를 `${MC_CKPT_DIR}/adv_dump_step{step}.{npz,json}` 로."""
    d = os.environ.get("MC_CKPT_DIR")
    if not d or before is None:
        return None
    import json

    import numpy as np
    try:
        Path(d).mkdir(parents=True, exist_ok=True)
        base = Path(d) / f"adv_dump_step{int(step)}"
        mask = data.batch.get("response_mask")
        # ★자리 보기: 크레딧 행(≤64) + 대조 8행에만 본문·토큰별 크레딧을 붙인다(값 불변).
        rows = {str(i): dict(v) for i, v in (tel.get("rows") or {}).items()}
        cr = sorted(credit, key=lambda i: -abs(sum(credit[i][1])))[:64]
        rest = [i for i in range(len(data)) if i not in set(cr)]
        pick = list(cr) + rest[::max(1, len(rest) // 8 or 1)][:8]
        try:
            texts = decode_active(CTX["tokenizer"], data)
        except Exception:                      # 토크나이저 없음 → 본문만 빠진다
            texts = None
        arrs = {}
        for i in pick:
            r = rows.setdefault(str(i), {})
            if texts is not None:
                r["text"] = texts[i][:6000]
            r["credited"] = bool(i in credit)
            if i in credit:
                j0, vals = credit[i]
                arrs[f"credit_{i}"] = np.asarray(vals, dtype="float16")
                r.setdefault("zone", [int(j0), int(j0) + len(vals)])
        np.savez_compressed(
            str(base) + ".npz",
            adv_before=before.float().cpu().numpy(),
            adv_after=data.batch["advantages"].float().cpu().numpy(),
            response_mask=(mask.float().cpu().numpy() if mask is not None
                           else np.zeros(1, dtype="float32")), **arrs)
        meta = {"step": int(step),
                "credit": {str(i): [int(j0), [float(x) for x in vals]]
                           for i, (j0, vals) in credit.items()},
                "rows": rows, "detail_rows": [int(i) for i in pick],
                "mass_share": float(tel.get("mass_share", float("nan"))),
                "uid": [str(u) for u in data.non_tensor_batch["uid"]],
                "r_first": tel.get("r_first_all"), "r_last": tel.get("r_last_all"),
                "outcome_mode": tel.get("outcome_mode", "group")}
        Path(str(base) + ".json").write_text(json.dumps(meta, ensure_ascii=False))
        print(f"[MC] adv dump -> {base}.npz(+.json)", flush=True)
        return str(base) + ".npz"
    except Exception as e:
        print(f"[MC] adv dump 실패: {type(e).__name__}: {e}", flush=True)
        return None


def build_advantage_hook(original):
    """verl `compute_advantage` 감싸개 — 앞: 결과 보상 · 뒤: `add_span_credit`."""
    def patched(data, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1,
                norm_adv_by_std_in_grpo=True, config=None):
        step = int(getattr(CTX.get("trainer"), "global_steps", 0) or 0)
        data, credit, tel = populate_token_rewards(data, config)
        data = original(data, adv_estimator=adv_estimator, gamma=gamma, lam=lam,
                        num_repeat=num_repeat,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo, config=config)
        data = add_span_credit(data, credit, tel, step)
        if tel.get("abort"):
            raise MCAbort(f"[MC][ABORT] step={step} {tel['abort']}")
        return data
    return patched


def patch_compute_advantage():
    import verl.trainer.ppo.ray_trainer as rt
    if getattr(rt, "_mc_patched", False):
        return
    rt.compute_advantage = build_advantage_hook(rt.compute_advantage)
    rt._mc_patched = True
    print("[MC] compute_advantage 훅 설치", flush=True)


def pfx_sampler(data_config, dataset):
    r"""SPONT_PFX 앞부분 **표집 무게**(수정 25 — 배치 실효 오답:정답 50:50, 고칠 수 있는 앞부분을 더 자주) =
    `extra_info[PFX_WEIGHT_KEY]`(기본 `weight`, 없으면 같은 이름 flat 컬럼)에 비례하는 복원 표집. 시드 = data.seed,
    에폭 길이 = 행 수(verl 스텝 수 불변). 무게 없는 parquet → None(verl 기본 표집기). 손실 배율은 없다(두 번 무게
    금지). PFX-U/B/const 가 같은 표집기를 쓴다. verl 0.9 `create_rl_sampler` 엔 사용자 표집기 걸이가 없어 main_task
    가 `train_sampler` 로 넘긴다."""
    import torch
    key, df = os.environ.get("PFX_WEIGHT_KEY") or "weight", dataset.dataframe
    ws = list(df[key]) if key in df.column_names else [
        (e or {}).get(key) for e in df["extra_info"]] if "extra_info" in df.column_names else []
    if all(w is None for w in ws):
        return None
    if any(w is None or not 0 <= float(w) < float("inf") for w in ws) or sum(map(float, ws)) <= 0:
        raise ValueError(f"[MC] PFX 표집 무게 '{key}' 가 비었거나 음수·무한이다({len(ws)}행).")
    ws = [float(w) for w in ws]
    g = torch.Generator()
    g.manual_seed(int(data_config.get("seed") or 0))
    print(f"[MC] PFX 표집 무게 '{key}': {len(ws)}행 · 합 {sum(ws):.3f} · 최대/최소 {max(ws):.3f}/{min(ws):.3f}", flush=True)
    return torch.utils.data.WeightedRandomSampler(torch.as_tensor(ws, dtype=torch.double),
                                                  len(ws), replacement=True, generator=g)


# ── 참조(frozen ref) 접두 공유 채점 — mc/shift_check.box_requests · mc/train_hook.pfx_fork 가 쓴다 ──
def tree_inputs(prefix, blocks, align: int = 16):
    r"""한 행의 트리 입력 — 접두(프롬프트+응답[:t_K]) 뒤에 블록 (t, 토큰열, 대상 수 m) 을 잇는다. 블록 토큰은
    접두[:t] 와 블록 안 앞 토큰만 본다(bool 4D 마스크, 위치 t 부터) → 자리마다 따로 넣은 것과 같은 조건부.
    → (ids, pos, ok[S,S], keep(대상 예측 자리), lab(대상 토큰), own(블록 번호)). S 는 `align` 배수(자기만 보는 채움)."""
    import torch
    n = len(prefix)
    S = n + sum(len(b) for _, b, _ in blocks)
    S += -S % align
    ids, pos = torch.zeros(S, dtype=torch.long), torch.zeros(S, dtype=torch.long)
    ok = torch.eye(S, dtype=torch.bool)
    ids[:n], pos[:n] = torch.as_tensor(list(prefix), dtype=torch.long), torch.arange(n)
    ok[:n, :n] = torch.ones(n, n, dtype=torch.bool).tril()
    keep, lab, own, s = [], [], [], n
    for q, (t, toks, m) in enumerate(blocks):
        L = len(toks)
        assert 0 < m < L and 0 < t <= n, (t, L, m, n)
        ids[s:s + L], pos[s:s + L] = torch.as_tensor(list(toks), dtype=torch.long), torch.arange(t, t + L)
        ok[s:s + L, :t] = True
        ok[s:s + L, s:s + L] = torch.ones(L, L, dtype=torch.bool).tril()
        keep += range(s + L - m - 1, s + L - 1)
        lab += list(toks[L - m:])
        own += [q] * m
        s += L
    return ids, pos, ok, keep, lab, own


def tree_score(model, prefix, blocks, dev=None, per_token: bool = False) -> list:
    r"""블록마다 Σ log p(대상 | 접두[:t] + 블록 앞부분) — forward **한 번**(접두 공유), 로짓은 대상 자리만
    (`logits_to_keep`), 온도 1. sdpa 전용(bool 4D 마스크 — flash 는 임의 마스크를 못 받는다). `dev` = 계산 장치
    (ref 워커는 현재 cuda 를 준다 — 매개변수 오프로드면 `next(parameters())` 가 cpu 다, 0924 스모크 gather 즉사)."""
    import torch
    impl = getattr(model.config, "_attn_implementation", None)
    assert impl == "sdpa", f"[MC] 트리 채점은 sdpa 전용이다: {impl!r}"
    ids, pos, ok, keep, lab, own = tree_inputs(prefix, blocks)
    dev = dev or next(model.parameters()).device
    out = model(input_ids=ids[None].to(dev), position_ids=pos[None].to(dev), attention_mask=ok[None, None].to(dev),
                logits_to_keep=torch.as_tensor(keep, device=dev), use_cache=False).logits[0]
    lp = torch.log_softmax(out.float(), -1).gather(-1, torch.as_tensor(lab, device=out.device)[:, None])[:, 0]
    if per_token:                           # 블록마다 대상 토큰별 log p(수정 28 CH-Fork)
        v, own = lp.double().cpu(), torch.as_tensor(own)
        return [v[own == q].tolist() for q in range(len(blocks))]
    return torch.zeros(len(blocks), dtype=torch.float64).index_add_(
        0, torch.as_tensor(own), lp.double().cpu()).tolist()


def ref_tree_score(trainer, trees, per_token: bool = False) -> list:
    """trees = [(접두 ids, [(t, 토큰열, m)])] → 이어 붙인 블록 점수(동결 ref 워커 `mc.ref_worker.MCRefWorker`);
    per_token 이면 블록마다 토큰별 목록."""
    return (trainer.ref_policy_wg.mc_tree_score(trees, True) if per_token else trainer.ref_policy_wg.mc_tree_score(trees))[0]


# ── 트레이너 ─────────────────────────────────────────────────────────────────
def build_trainer_cls():
    """verl import 를 함수 안에 둔다(CPU 테스트가 이 모듈을 import 할 수 있게)."""
    from verl.trainer.ppo.ray_trainer import RayPPOTrainer

    class MCTrainer(RayPPOTrainer):
        """verl 0.9 에선 사실상 안 불린다(agent-loop 가 rm_scores 를 얹는다) — zero-fallback 안전망."""

        def _compute_reward_colocate(self, batch):
            import torch
            from verl import DataProto
            rm = torch.zeros_like(batch.batch["responses"], dtype=torch.float32)
            return DataProto.from_dict(tensors={"rm_scores": rm})

    return MCTrainer


@ray.remote
def main_task(config):
    from omegaconf import OmegaConf, open_dict
    from verl.single_controller.ray import RayWorkerGroup
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.rl_dataset import collate_fn
    from verl.utils.fs import copy_to_local
    from verl.workers.engine_workers import ActorRolloutRefWorker, TrainingWorker

    from mc.ref_worker import MCRefWorker  # noqa: PLC0415 — 최상위 클래스(참조 피클)
    try:
        from verl.trainer.ppo.utils import create_rl_dataset, create_rl_sampler
    except ImportError:                                   # verl 0.7.x
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

    OmegaConf.resolve(config)
    from mc.train_hook import MC_ARM_SPECS, describe  # noqa: PLC0415
    arm = str(getattr(getattr(config, "algorithm", None), "math_arm", "") or "").upper()
    if arm not in MC_ARM_SPECS:
        raise ValueError(f"[MC] algorithm.math_arm={arm!r} 이 {sorted(MC_ARM_SPECS)} 에 없다.")
    alg = getattr(config, "algorithm", None)
    if str(getattr(alg, "adv_estimator", "")).lower() != "grpo":
        raise ValueError("[MC] algorithm.adv_estimator=grpo 여야 한다(시퀀스 GRPO + 스팬 후처리).")
    print(f"[MC] arm={arm} {describe()}", flush=True)

    logger_cfg = [x for x in list(config.trainer.get("logger", [])) if x != "wandb"] or ["console"]  # wandb 금지
    with open_dict(config.trainer):
        config.trainer.logger = logger_cfg

    patch_compute_advantage()
    local_path = copy_to_local(config.actor_rollout_ref.model.path,
                              use_shm=config.actor_rollout_ref.model.get("use_shm", False))
    trust = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust)
    processor = hf_processor(local_path, trust_remote_code=trust, use_fast=True)
    if (processor is not None and not getattr(processor, "chat_template", None)
            and getattr(tokenizer, "chat_template", None)):
        # VL 프로세서의 chat_template 이 None 이면 verl 의 데이터셋이 전 행을 버린다(빈 데이터셋).
        processor.chat_template = tokenizer.chat_template

    pool = "global_pool"
    rpm = ResourcePoolManager(
        resource_pool_spec={pool: [config.trainer.n_gpus_per_node] * config.trainer.nnodes},
        mapping={Role.ActorRollout: pool, Role.Critic: pool, Role.RefPolicy: pool})
    # ★학습 데이터셋은 **한 번만** 만든다(샘플러가 다른 인스턴스를 보면 셔플 순서가 갈린다).
    check_prompt_lengths(tokenizer, str(config.data.train_files),
                        int(config.data.max_prompt_length))
    train_ds = create_rl_dataset(config.data.train_files, config.data, tokenizer,
                                processor, is_train=True)
    sampler = pfx_sampler(config.data, train_ds) if arm == "SPONT_PFX" else None   # 수정 25 표집 무게
    trainer = build_trainer_cls()(
        config=config, tokenizer=tokenizer, processor=processor,
        role_worker_mapping={Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
                             Role.Critic: ray.remote(TrainingWorker),
                             Role.RefPolicy: ray.remote(MCRefWorker)},
        resource_pool_manager=rpm, ray_worker_group_cls=RayWorkerGroup,
        train_dataset=train_ds,
        val_dataset=create_rl_dataset(config.data.val_files, config.data, tokenizer,
                                     processor, is_train=False),
        collate_fn=collate_fn,
        train_sampler=sampler or create_rl_sampler(config.data, train_ds))
    CTX["trainer"], CTX["tokenizer"] = trainer, tokenizer
    trainer.init_workers()
    trainer.use_rm = True    # RM 워커 없이 use_rm 만 켠다 → fit() 이 우리 훅으로
    trainer.fit()


def _ray_env() -> dict:
    env = {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN",
           "PYTHONPATH": os.environ.get("PYTHONPATH", str(Path(__file__).resolve().parents[1])),
           "VERL_DISABLE_FLASH_XENT": os.environ.get("VERL_DISABLE_FLASH_XENT", "1"),
           # vllm 0.20 FP8 워밍업 즉사 방지 · 8k×151k 로짓 단편화 OOM 완화(0921 실측).
           "VLLM_USE_DEEP_GEMM": os.environ.get("VLLM_USE_DEEP_GEMM", "0"),
           "VLLM_MOE_USE_DEEP_GEMM": os.environ.get("VLLM_MOE_USE_DEEP_GEMM", "0"),
           "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
           # 0928: 잡마다 각자 Ray → job id 가 같아 vLLM 가중치 소켓(/tmp/rl-colocate-zmq-…)이 겹친다 → 잡 꼬리표
           # (verl09 vllm_rollout.py·utils.py 가 job id 뒤에 붙임, docs/ENV_verl09.md).
           "MC_ZMQ_TAG": os.environ.get("MC_ZMQ_TAG") or f"-mc{os.getpid()}"}
    for k in WORKER_ENV_KEYS:
        if os.environ.get(k) is not None:      # 미설정 ≠ 빈 문자열
            env[k] = os.environ[k]
    return env


#: Ray 워커 선기동 끄기 — raylet env 로만 먹으므로 ray.init **전에** 심는다(2.58 에 실재).
PRESTART_OFF = {"RAY_enable_worker_prestart": "0", "RAY_prestart_worker_first_driver": "0"}
#: CPU 수요 = agent-loop 워커(기본 8) + main_task + 워커그룹 + 여유. 그보다 넉넉한 하한.
MIN_NUM_CPUS = 16


def ray_init_kwargs(config=None) -> dict:
    r"""`ray.init` 인자 — num_cpus 필수(0921: 선기동 워커 124개가 NFS import 타임아웃으로 죽어 대기)."""
    nw = 8
    try:
        nw = int(config.actor_rollout_ref.rollout.agent.num_workers)
    except Exception:
        pass
    return {
        "include_dashboard": False,
        "_node_ip_address": "127.0.0.1",
        "num_cpus": max(MIN_NUM_CPUS, nw + 8),
        # 등록 대기: 대시보드 기동(20초+)·NFS import 동안 죽지 않게(위 사고).
        "_system_config": {"agent_register_timeout_ms": 600000,
                           "worker_register_timeout_seconds": 600},
        "object_store_memory": 20_000_000_000,        # 기본 /dev/shm 200GB mmap 은 기동이 느리다
        "_temp_dir": os.environ.get("RAY_TMPDIR") or None,   # 같은 박스 여러 잡 → 잡별 임시
        "runtime_env": {"env_vars": _ray_env()},
    }


import hydra  # noqa: E402
from omegaconf import OmegaConf, open_dict  # noqa: E402


def fill_verl_defaults(config):
    """단독 yaml(`defaults:` 없음)에 verl 0.9 완전판 기본값을 깐다 — 없는 키만 채우고 있는 키는 산다."""
    import verl
    path = os.path.join(os.path.dirname(verl.__file__), "trainer/config/_generated_ppo_trainer.yaml")
    print(f"[MC] verl 기본값 병합: {path}", flush=True)
    return OmegaConf.merge(OmegaConf.load(path), config)


@hydra.main(config_path="../configs", config_name="countdown_6arm", version_base=None)
def main(config):
    config = fill_verl_defaults(config)
    with open_dict(config.reward.custom_reward_function):     # agent-loop 보상 자리표시자 등록
        config.reward.custom_reward_function.path = str(Path(__file__).resolve())
        config.reward.custom_reward_function.name = "reward_loop_score"
    if not ray.is_initialized():
        for k, v in PRESTART_OFF.items():
            os.environ.setdefault(k, v)      # mc/run.sh 가 이미 export 하지만 단독 실행도 막는다
        kw = ray_init_kwargs(config)
        print(f"[MC] ray.init num_cpus={kw['num_cpus']} prestart="
              f"{os.environ.get('RAY_enable_worker_prestart')}", flush=True)
        ray.init(**kw)
    try:
        ray.get(main_task.remote(config))
    except Exception as e:                     # noqa: BLE001
        if "MCAbort" in type(e).__name__ or "[MC][ABORT]" in str(e):
            try:
                d = str(config.trainer.default_local_dir)
                os.makedirs(d, exist_ok=True)
                Path(d, "ABORTED.txt").write_text(str(e)[:2000])
            except Exception:
                pass
            print(f"[MC][ABORT] 사전등록 중단 — rc {ABORT_EXIT_CODE}", flush=True)
            sys.exit(ABORT_EXIT_CODE)
        raise


if __name__ == "__main__":
    main()
