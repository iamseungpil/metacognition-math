"""미끼(오답식) 종류별 PMI-shift 비교 — 「명백히 틀린 오답」이 더 나은가.

문제의식: 현행 미끼는 증인식의 연산자 하나만 바꾼 «근접 오답»이라 두 식의 토큰
대부분이 글자까지 같다. 같은 토큰은 정답·미끼의 logp 가 정확히 같아 차이가 0 이고,
실제 정보는 갈린 칸과 그 뒤 몇 칸에만 남는다(08-28 실측: min/max 집계 전부 무신호,
미끼 난수만 바꿔도 AUC 0.587↔0.537 로 흔들림).

그렇다면 «다른 문제의 정답식» 처럼 명백히 틀린 미끼를 쓰면 대비가 커져 신호가
살아날까? 네 종류를 같은 사이트·같은 문맥에서 정면 비교한다.

  near    연산자 1개 교체 (현행 swap_op_decoy)          — 글자 1개 차이
  family  연산자 가족 교체 (* ↔ + , / ↔ −)              — 연산 구조가 다름
  other   ★다른 문제의 증인식                            — 숫자부터 전부 다름
  shuffle 같은 숫자를 다른 순서·연산으로 (목표 빗나감)     — 숫자는 같고 배열만 다름

각 미끼로:  PMI(ctx) = mean logp(gold|ctx) − mean logp(decoy|ctx)
            shift    = PMI(close) − PMI(open)
또한 미끼를 쓰지 않는 OSD 계열도 같은 표본에서 함께 재어 비교한다:
  osd_self  = mean logp(모델 자신의 이어쓰기 200토큰 | close) − (| open)
  osd_gold  = mean logp(\\boxed{gold} | close) − (| open)      ← 미끼 없음

출력: decoy_variants.parquet [site_id, shift_near, shift_family, shift_other,
      shift_shuffle, osd_gold, (osd_self)]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"))
from src.training.countdown_task import eval_countdown, swap_op_decoy  # noqa: E402

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

_FAM_MULDIV = str.maketrans({"*": "+", "/": "-"})
_FAM_ADDSUB = str.maketrans({"+": "*", "-": "/"})


def family_decoy(witness: str, target) -> str | None:
    t = int(target)
    for cand in (witness.translate(_FAM_MULDIV), witness.translate(_FAM_ADDSUB)):
        if cand != witness and eval_countdown(cand) != t:
            return cand
    cand = f"({witness})+1"
    return cand if eval_countdown(cand) != t else None


def shuffle_decoy(nums, target, rng: random.Random) -> str | None:
    """같은 숫자를 쓰되 목표를 빗나가는 식 (숫자는 같고 배열만 다름)."""
    ns = [int(v) for v in nums]
    for _ in range(60):
        p = ns[:]
        rng.shuffle(p)
        ops = [rng.choice("+-*") for _ in range(len(p) - 1)]
        expr = f"(({p[0]}{ops[0]}{p[1]}){ops[1]}{p[2]}){ops[2]}{p[3]}" if len(p) == 4 else None
        if expr is None:
            return None
        v = eval_countdown(expr)
        if v is not None and v != int(target):
            return expr
    return None


def score_logp_hf(model, reqs, batch_size=4, max_len=6144):
    import torch
    out = [None] * len(reqs)
    todo = [i for i, (c, t) in enumerate(reqs)
            if t and len(c) >= 1 and len(c) + len(t) <= max_len]
    todo.sort(key=lambda i: len(reqs[i][0]) + len(reqs[i][1]))
    for lo in range(0, len(todo), batch_size):
        idxs = todo[lo:lo + batch_size]
        seqs = [list(reqs[i][0]) + list(reqs[i][1]) for i in idxs]
        L = max(len(s) for s in seqs)
        ids = torch.zeros((len(seqs), L), dtype=torch.long)
        att = torch.zeros((len(seqs), L), dtype=torch.long)
        for j, sq in enumerate(seqs):
            ids[j, :len(sq)] = torch.tensor(sq)
            att[j, :len(sq)] = 1
        with torch.no_grad():
            logits = model(input_ids=ids.to(model.device),
                           attention_mask=att.to(model.device)).logits
        for j, i in enumerate(idxs):
            c, t = reqs[i]
            pos = torch.arange(len(c) - 1, len(c) - 1 + len(t), device=logits.device)
            lsm = torch.log_softmax(logits[j, pos, :].float(), dim=-1)
            tid = torch.tensor(list(t), device=logits.device)
            out[i] = lsm[torch.arange(len(t), device=logits.device), tid].double().cpu().numpy()
        del logits
        done = min(lo + batch_size, len(todo))
        if done % 120 < batch_size or done == len(todo):
            print(f"[decoy] scored {done}/{len(todo)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_glob", required=True)
    ap.add_argument("--sites_file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--self_tok", type=int, default=200, help="osd_self 창 길이")
    args = ap.parse_args()

    sites = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(args.sites_glob))],
                      ignore_index=True).drop_duplicates("site_id")
    if args.sites_file:
        keep = {l.strip() for l in open(args.sites_file) if l.strip()}
        sites = sites[sites.site_id.isin(keep)]
    sites = sites.reset_index(drop=True)
    print(f"[decoy] 대상 {len(sites)} 사이트", flush=True)

    # «다른 문제»의 증인식 풀 (자기 문제와 겹치지 않게 짝짓는다)
    wit_pool = [(str(r.witness), int(r.target)) for _, r in sites.iterrows() if str(r.witness)]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    rng = random.Random(0)
    reqs, index, rows, skip = [], {}, [], 0
    for j, r in sites.iterrows():
        wit = str(r["witness"])
        if not wit:
            skip += 1
            continue
        d_near = swap_op_decoy(wit, list(r["nums"]), int(r["target"]), random.Random(0))
        if isinstance(d_near, tuple):
            d_near = d_near[0]
        d_fam = family_decoy(wit, r["target"])
        d_shuf = shuffle_decoy(list(r["nums"]), int(r["target"]), rng)
        # other: 목표값이 다른 «다른 문제»의 증인식
        d_other = None
        for _ in range(20):
            w2, t2 = wit_pool[rng.randrange(len(wit_pool))]
            if w2 != wit and t2 != int(r["target"]):
                d_other = w2
                break
        cands = {"near": d_near, "family": d_fam, "other": d_other, "shuffle": d_shuf}
        if d_near is None or d_other is None:
            skip += 1
            continue

        msgs = json.loads(r["prompt_json"]) if isinstance(r["prompt_json"], str) else r["prompt_json"]
        try:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = str(r["response_text"])
        ctx_open = tok(head + text[:int(r["meta_start"])], add_special_tokens=False).input_ids
        ctx_close = tok(head + text[:int(r["meta_end"])], add_special_tokens=False).input_ids
        # 자기 이어쓰기 창 (OSD 계열): 메타 뒤 원문 200토큰
        self_ids = tok(text[int(r["meta_end"]):], add_special_tokens=False).input_ids[:args.self_tok]

        targets = {"gold": tok("\\boxed{" + wit + "}", add_special_tokens=False).input_ids}
        for k, v in cands.items():
            if v:
                targets[k] = tok("\\boxed{" + v + "}", add_special_tokens=False).input_ids
        if self_ids:
            targets["self"] = self_ids
        for cn, ctx in (("open", ctx_open), ("close", ctx_close)):
            for tn, tg in targets.items():
                index[(j, cn, tn)] = len(reqs)
                reqs.append((ctx, tg))
        rows.append(dict(idx=j, site_id=r["site_id"], has=set(targets)))
    print(f"[decoy] 시퀀스 {len(reqs)}개 · skip={skip}", flush=True)

    lps = score_logp_hf(model, reqs, batch_size=args.batch_size)

    out = []
    for rw in rows:
        j = rw["idx"]
        def get(c, t):
            i = index.get((j, c, t))
            return None if i is None else lps[i]
        g_o, g_c = get("open", "gold"), get("close", "gold")
        if g_o is None or g_c is None:
            continue
        rec = {"site_id": rw["site_id"],
               "osd_gold": float(g_c.mean() - g_o.mean())}
        for k in ("near", "family", "other", "shuffle"):
            d_o, d_c = get("open", k), get("close", k)
            rec[f"shift_{k}"] = (float((g_c.mean() - d_c.mean()) - (g_o.mean() - d_o.mean()))
                                 if (d_o is not None and d_c is not None) else np.nan)
        s_o, s_c = get("open", "self"), get("close", "self")
        rec["osd_self"] = (float(s_c.mean() - s_o.mean())
                           if (s_o is not None and s_c is not None) else np.nan)
        out.append(rec)
    df = pd.DataFrame(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"[decoy] wrote {args.out} ({len(df)}행)", flush=True)


if __name__ == "__main__":
    main()
