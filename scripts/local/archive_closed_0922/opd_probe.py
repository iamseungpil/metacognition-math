#!/usr/bin/env python
r"""OPD 힌트 교사 — 오프라인 타당성 탐침 (`docs/DESIGN_opd_hint_teacher.md` §7).

학습 코드를 건드리지 않고, 설계 문서의 힌트(계열 생사·살아있는 첫수 목록 — **정답
식·witness 는 절대 없음**)를 학생 정책 자신에게 조건화했을 때:

  (a) META 구간 + 첫 post-meta attempt 줄 위에서 힌트-有/無 다음-토큰 분포의
      평균 토큰당 KL(hint||plain) 이 얼마인가.
  (b) 같은 길이의 무작위 비-META 구간에서도 같은 크기의 KL 이 나오는가(통제 —
      힌트의 영향이 META 에 특이적인지, 아니면 문맥 전반에 퍼지는 잡음인지).
  (c) 힌트가 메타 직후 `live_new_moves`(오라클이 준 "아직 안 가본, 해로 이어지는
      첫수") 문자열의 확률질량을 실제로 올리는가.

사용:
    source scripts/local/env.sh
    python scripts/local/opd_probe.py \
        --conts $WORK/conts_v1/judge_cd7_OPT_opt_s1_step30.parquet \
        --sites $WORK/data/sites_v1/sites_judge.parquet \
        --model_path $WORK/models/Qwen3-4B \
        --n_sites 200 --out $WORK/opd_probe

GPU 선택: 과제 지시대로 GPU 1·2 만 확인해 ≥20GB 여유가 있으면 그 카드를 쓰고,
없으면 CPU 로 내려가며 `--n_sites`를 30 으로 강제한다(`--force_cpu`로 수동 지정도
가능). 이 스크립트는 어떤 학습 파일도 import/수정하지 않는다 — `src.rulers.*`(자
패키지, 읽기 전용 인터페이스)와 `src.training.countdown_sites`(오라클 라벨 재사용,
순수함수) 만 가져온다.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.rulers.base import live_new_moves_from_json  # noqa: E402
from src.rulers.hf_ctx import HfCtx, MAX_CTX_TOKENS  # noqa: E402
from src.rulers.table import sample_from_row, site_from_row  # noqa: E402
from src.training.countdown_selfcontrol import _ARITH_EQ  # noqa: E402

MIN_FREE_MB = 20_000
CANDIDATE_GPUS = (1, 2)


# ══════════════════════════════════════════════════════════════════════════════
# 0. GPU 선택 — 과제 지시: GPU 1·2 만 보고, ≥20GB 여유가 없으면 CPU·30사이트.
# ══════════════════════════════════════════════════════════════════════════════

def pick_device(force_cpu: bool = False) -> tuple[str, dict]:
    info = {"checked": list(CANDIDATE_GPUS), "free_mb": {}, "chosen": None}
    if force_cpu:
        info["chosen"] = "cpu"
        info["reason"] = "--force_cpu"
        return "cpu", info
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"], text=True, timeout=10)
    except Exception as e:  # noqa: BLE001 — nvidia-smi 없으면 CPU로
        info["chosen"] = "cpu"
        info["reason"] = f"nvidia-smi failed: {e}"
        return "cpu", info
    free_by_idx = {}
    for line in out.strip().splitlines():
        idx_s, free_s = [x.strip() for x in line.split(",")]
        free_by_idx[int(idx_s)] = int(free_s)
    for idx in CANDIDATE_GPUS:
        info["free_mb"][idx] = free_by_idx.get(idx)
    best = max(CANDIDATE_GPUS, key=lambda i: free_by_idx.get(i, -1))
    if free_by_idx.get(best, 0) >= MIN_FREE_MB:
        info["chosen"] = f"cuda:{best}"
        info["reason"] = f"gpu{best} free={free_by_idx[best]}MiB >= {MIN_FREE_MB}MiB"
        return f"cuda:{best}", info
    info["chosen"] = "cpu"
    info["reason"] = (f"neither gpu in {CANDIDATE_GPUS} has >= {MIN_FREE_MB}MiB free "
                       f"(free={info['free_mb']})")
    return "cpu", info


# ══════════════════════════════════════════════════════════════════════════════
# 1. 힌트 텍스트 — DESIGN §1.1. 정답 식·witness 를 절대 담지 않는다.
# ══════════════════════════════════════════════════════════════════════════════

def build_hint(family_dead, live_moves: list[str]) -> str | None:
    """`family_dead is None`(메타 앞 시도 없음)이면 힌트를 아예 만들지 않는다 —
    DESIGN §1.1: "만들 판정이 없다"."""
    if family_dead is None:
        return None
    dead_or_alive = "dead" if int(family_dead) == 1 else "alive"
    moves_str = ", ".join(live_moves) if live_moves else "none found"
    return (f"Hint: your current line of attack is {dead_or_alive}.\n"
            f"Hint: first moves that still reach the target: {moves_str}.")


def inject_hint(prompt_messages: list[dict], hint: str) -> list[dict]:
    """힌트를 **마지막 user 메시지 뒤에 이어붙인** 새 메시지 리스트(얕은 복사).
    DESIGN §1.2 — 프리픽스 바로 앞, user 턴의 연장으로."""
    msgs = [dict(m) for m in prompt_messages]
    for i in range(len(msgs) - 1, -1, -1):
        if msgs[i].get("role") == "user":
            msgs[i] = dict(msgs[i])
            msgs[i]["content"] = str(msgs[i]["content"]) + "\n\n" + hint
            return msgs
    raise ValueError("inject_hint: user 메시지가 없다 — 프롬프트 형식이 예상과 다르다.")


# ══════════════════════════════════════════════════════════════════════════════
# 2. 전체 분포 forward — HfCtx.token_logprobs 는 실현 토큰의 logprob만 gather
#    한다(§DESIGN 4.1 이 인용하는 그 함수). KL 은 vocab 전체 분포가 필요하므로
#    여기서 ctx.model/ctx.tokenizer 를 직접 불러 별도로 짠다(hf_ctx.py 는
#    학습 코드가 아니지만 수정하지 않는다 — 읽기 전용으로 재사용).
# ══════════════════════════════════════════════════════════════════════════════

def full_log_probs(ctx: HfCtx, ctx_ids: list[int], target_ids: list[int]):
    """`ctx.token_logprobs`(hf_ctx.py:150-172)와 같은 teacher-forced 정렬이되,
    gather 대신 vocab 전체 log_softmax 를 돌려준다. shape [len(target_ids), V]."""
    import torch  # noqa: PLC0415
    ctx._ensure_loaded()  # noqa: SLF001 — 지연 로드는 이 클래스의 공개 계약(모듈 docstring)
    if len(ctx_ids) > MAX_CTX_TOKENS:
        ctx_ids = ctx_ids[-MAX_CTX_TOKENS:]
    full = ctx_ids + target_ids
    with torch.inference_mode():
        x = torch.tensor([full], device=ctx.model.device)
        logits = ctx.model(input_ids=x).logits[0]
    s = len(ctx_ids)
    return torch.log_softmax(logits[s - 1:s - 1 + len(target_ids)].float(), dim=-1)


def kl_per_token(logp_hint, logp_plain):
    """KL(hint || plain) per position. shape [T]."""
    p_hint = logp_hint.exp()
    return (p_hint * (logp_hint - logp_plain)).sum(dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
# 3. 구간 추출 — META, 첫 post-meta attempt 줄, 무작위 통제 구간 (DESIGN §2.1)
# ══════════════════════════════════════════════════════════════════════════════

def first_attempt_line_after(cont: str, meta_end: int) -> tuple[int, int] | None:
    """메타 뒤 첫 `a op b = c` 매치가 끝나는 줄의 끝까지. 없으면 None.
    `countdown_sites.cut_attempt_boundary`와 같은 "줄 끝" 정의(그 함수를 그대로
    부르지 않는 이유: 그건 프리픽스 **전체**에서 층화 샘플링을 하는 함수이고,
    여기 필요한 건 메타 **뒤** 첫 매치 하나뿐이다 — 다른 문제)."""
    tail = cont[meta_end:]
    m = _ARITH_EQ.search(tail)
    if m is None:
        return None
    nl = tail.find("\n", m.end())
    line_end = (nl + 1) if nl != -1 else len(tail)
    return meta_end, meta_end + line_end


def pick_control_span(cont: str, span_len_chars: int, exclude: tuple[int, int],
                      rng: random.Random) -> tuple[int, int] | None:
    """`exclude`(메타+첫줄 구간)와 겹치지 않는, 같은 문자 길이의 무작위 구간.
    못 찾으면 None(응답이 너무 짧음)."""
    n = len(cont)
    if span_len_chars <= 0 or span_len_chars >= n:
        return None
    candidates = [s for s in range(0, n - span_len_chars)
                  if s + span_len_chars <= exclude[0] or s >= exclude[1]]
    if not candidates:
        return None
    s = rng.choice(candidates)
    return s, s + span_len_chars


# ══════════════════════════════════════════════════════════════════════════════
# 4. 사이트 하나 채점
# ══════════════════════════════════════════════════════════════════════════════

def score_site(ctx: HfCtx, site_row, cont_row, rng: random.Random) -> dict | None:
    site = site_from_row(site_row)
    sample = sample_from_row(cont_row)
    hint = build_hint(site.family_dead, list(site.live_new_moves))
    if hint is None:
        return None
    cont = sample.continuation
    ms, me = sample.meta_start, sample.meta_end
    if ms < 0 or me < 0 or me <= ms:
        return None
    attempt_span = first_attempt_line_after(cont, me)
    target_end = attempt_span[1] if attempt_span is not None else me
    target_text = cont[ms:target_end]
    if not target_text:
        return None

    prompt_plain_msgs = site.prompt_messages
    prompt_hint_msgs = inject_hint(site.prompt_messages, hint)
    prompt_plain = ctx.tokenizer.apply_chat_template(
        prompt_plain_msgs, tokenize=False, add_generation_prompt=True)
    prompt_hint = ctx.tokenizer.apply_chat_template(
        prompt_hint_msgs, tokenize=False, add_generation_prompt=True)

    pre_text = site.prefix + cont[:ms]
    pre_ids_plain = list(ctx.encode(prompt_plain)) + list(ctx.encode(pre_text))
    pre_ids_hint = list(ctx.encode(prompt_hint)) + list(ctx.encode(pre_text))
    target_ids, target_offsets = ctx.encode_with_offsets(target_text)
    if not target_ids:
        return None
    n_meta_tok = sum(1 for a, b in target_offsets if b <= (me - ms))

    logp_plain = full_log_probs(ctx, pre_ids_plain, target_ids)
    logp_hint = full_log_probs(ctx, pre_ids_hint, target_ids)
    kl = kl_per_token(logp_hint, logp_plain).cpu().numpy()
    kl_meta = float(np.mean(kl[:n_meta_tok])) if n_meta_tok > 0 else float("nan")
    kl_line = float(np.mean(kl[n_meta_tok:])) if n_meta_tok < len(kl) else float("nan")

    # (b) 통제 — 같은 토큰 길이의 무작위 비-META 구간(메타+첫줄과 안 겹침).
    ctrl_span = pick_control_span(cont, len(target_text), (ms, target_end), rng)
    kl_ctrl_mean = None
    if ctrl_span is not None:
        cs, ce = ctrl_span
        ctrl_pre_text = site.prefix + cont[:cs]
        ctrl_target = cont[cs:ce]
        ctrl_ids, _ = ctx.encode_with_offsets(ctrl_target)
        if ctrl_ids:
            cpre_plain = list(ctx.encode(prompt_plain)) + list(ctx.encode(ctrl_pre_text))
            cpre_hint = list(ctx.encode(prompt_hint)) + list(ctx.encode(ctrl_pre_text))
            clp_plain = full_log_probs(ctx, cpre_plain, ctrl_ids)
            clp_hint = full_log_probs(ctx, cpre_hint, ctrl_ids)
            kl_ctrl_mean = float(kl_per_token(clp_hint, clp_plain).mean().item())

    # (c) live_new_moves 확률질량 — 메타 바로 뒤(target position = me) 부터, 각
    # 후보 첫수 문자열(예: "5+19")을 있는 그대로 이어 쓸 teacher-forced 로그확률의
    # 합을 exp 해서 "이 후보들 중 하나로 시작할 질량"을 근사한다(최대 5개, 오라클이
    # 준 순서 그대로 — 순위 매기지 않는다).
    live_moves = list(site.live_new_moves)[:5]
    mass_plain = mass_hint = None
    if live_moves:
        move_ids_list = [ctx.encode(mv) for mv in live_moves]
        move_ids_list = [ids for ids in move_ids_list if ids]
        if move_ids_list:
            post_meta_pre = site.prefix + cont[:me]
            pre_plain = list(ctx.encode(prompt_plain)) + list(ctx.encode(post_meta_pre))
            pre_hint = list(ctx.encode(prompt_hint)) + list(ctx.encode(post_meta_pre))
            mp = mh = 0.0
            for ids in move_ids_list:
                lp_p = ctx.token_logprobs(pre_plain, ids)
                lp_h = ctx.token_logprobs(pre_hint, ids)
                mp += float(np.exp(np.sum(lp_p)))
                mh += float(np.exp(np.sum(lp_h)))
            mass_plain, mass_hint = mp, mh

    return {
        "site_id": site.site_id,
        "family_dead": site.family_dead,
        "n_live_moves": len(site.live_new_moves),
        "n_meta_tok": n_meta_tok,
        "n_line_tok": len(kl) - n_meta_tok,
        "kl_meta_mean": kl_meta,
        "kl_line_mean": kl_line,
        "kl_ctrl_mean": kl_ctrl_mean,
        "mass_plain": mass_plain,
        "mass_hint": mass_hint,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conts", required=True)
    ap.add_argument("--sites", required=True)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--n_sites", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--force_cpu", action="store_true")
    args = ap.parse_args()

    device, gpu_info = pick_device(args.force_cpu)
    n_sites = args.n_sites
    if device == "cpu" and n_sites > 30:
        print(f"[opd_probe] CPU 로 내려감({gpu_info['reason']}) — n_sites {n_sites}->30")
        n_sites = 30
    print(f"[opd_probe] device={device} ({gpu_info['reason']}), n_sites={n_sites}")

    model_path = args.model_path or os.path.join(os.environ.get("WORK", ""), "models/Qwen3-4B")
    conts = pd.read_parquet(args.conts)
    sites = pd.read_parquet(args.sites)

    sub = conts[(conts["mode"] == "meta") & (conts["emitted"] == 1)]
    sub = sub.drop_duplicates(subset="site_id", keep="first")
    sites_idx = sites.set_index("site_id")
    sub = sub[sub["site_id"].isin(sites_idx.index)]

    rng = random.Random(args.seed)
    site_ids = list(sub["site_id"])
    rng.shuffle(site_ids)
    site_ids = site_ids[: min(n_sites, len(site_ids))]
    print(f"[opd_probe] {len(sub)} sites with a complete student meta available; "
          f"scoring {len(site_ids)}")

    dtype = "bfloat16" if device.startswith("cuda") else None
    ctx = HfCtx(model_path=model_path, device=device, dtype=dtype)

    rows = []
    cont_by_site = sub.set_index("site_id")
    for i, sid in enumerate(site_ids):
        try:
            r = score_site(ctx, sites_idx.loc[sid], cont_by_site.loc[sid], rng)
        except Exception as e:  # noqa: BLE001 — 탐침 스크립트: 한 사이트 실패로 전체를 죽이지 않는다
            print(f"[opd_probe] site {sid} failed: {type(e).__name__}: {e}")
            r = None
        if r is not None:
            rows.append(r)
        if (i + 1) % 10 == 0:
            print(f"[opd_probe] {i + 1}/{len(site_ids)} sites processed, {len(rows)} scored")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        return float(np.mean(vals)) if vals else None

    def _frac_positive(key_a, key_b):
        pairs = [(r[key_a], r[key_b]) for r in rows
                if r.get(key_a) is not None and r.get(key_b) is not None]
        if not pairs:
            return None
        return float(np.mean([1.0 if b > a else 0.0 for a, b in pairs]))

    summary = {
        "device": device, "gpu_info": gpu_info, "n_sites_requested": n_sites,
        "n_sites_scored": len(rows),
        "kl_meta_mean": _mean("kl_meta_mean"),
        "kl_line_mean": _mean("kl_line_mean"),
        "kl_ctrl_mean": _mean("kl_ctrl_mean"),
        "kl_meta_minus_ctrl": (
            None if _mean("kl_meta_mean") is None or _mean("kl_ctrl_mean") is None
            else _mean("kl_meta_mean") - _mean("kl_ctrl_mean")),
        "mass_plain_mean": _mean("mass_plain"),
        "mass_hint_mean": _mean("mass_hint"),
        "mass_hint_minus_plain": (
            None if _mean("mass_plain") is None or _mean("mass_hint") is None
            else _mean("mass_hint") - _mean("mass_plain")),
        "frac_sites_mass_increased": _frac_positive("mass_plain", "mass_hint"),
        "frac_sites_kl_meta_gt_ctrl": _frac_positive("kl_ctrl_mean", "kl_meta_mean"),
        "rows": rows,
    }
    out_path = out_dir / "summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[opd_probe] wrote {out_path}")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2, default=str))


if __name__ == "__main__":
    main()
