"""측정 «시점» 실험 — 혼잣말 직후가 아니라, 조금 더 진행한 뒤 재면 자가 살아나는가.

문제의식: 현행 PMI-shift 는 혼잣말이 끝난 «바로 그 자리»에서 정답식을 강제로 읽힌다.
즉 «지금 당장 답을 불쑥 내뱉는다면» 을 묻는다. 그러나 실제 롤아웃은 혼잣말 뒤로도
수백 토큰을 더 탐색한 뒤에야 답을 쓴다 — 시점이 어긋나 있다. 값싼 자(AUC 0.59)와
비싼 자(실제로 끝까지 풀려서 비교, 0.71)의 격차가 여기서 나올 수 있다.

설계(생성 불필요): 원본 롤아웃의 «메타 뒤 다음 D 토큰»을 두 문맥에 **똑같이** 붙인다.
    ctx_open+D  = 프롬프트 + 프리픽스            + (원본 다음 D토큰)
    ctx_close+D = 프롬프트 + 프리픽스 + 혼잣말     + (원본 다음 D토큰)
이어붙이는 텍스트가 동일하므로 «혼잣말 유무»만 다르다. D=0 이면 현행 자와 같다.

    PMI_D(ctx) = mean logp(gold|ctx) − mean logp(decoy|ctx)
    shift_D    = PMI_D(close+D) − PMI_D(open+D)

D ∈ {0, 100, 200, 400} 를 한 번에 재어 시점 의존성을 그린다.
미끼는 08-28 비교에서 가장 나았던 «다른 문제의 정답식»과 현행 «근접 오답» 둘 다.
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
from src.training.countdown_task import swap_op_decoy  # noqa: E402

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

DELAYS = (0, 100, 200, 400)


def score_logp_hf(model, reqs, batch_size=2, max_len=7168):
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
        if done % 200 < batch_size or done == len(todo):
            print(f"[tp] scored {done}/{len(todo)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_glob", required=True)
    ap.add_argument("--sites_file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    args = ap.parse_args()

    sites = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(args.sites_glob))],
                      ignore_index=True).drop_duplicates("site_id").reset_index(drop=True)
    if args.sites_file:
        keep = {l.strip() for l in open(args.sites_file) if l.strip()}
        sites = sites[sites.site_id.isin(keep)].reset_index(drop=True)
    print(f"[tp] 대상 {len(sites)} 사이트 · D={DELAYS}", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    wit_pool = [str(r.witness) for _, r in sites.iterrows() if str(r.witness)]
    rng = random.Random(0)
    reqs, index, rows, skip = [], {}, [], 0
    for j, r in sites.iterrows():
        wit = str(r["witness"])
        d_near = swap_op_decoy(wit, list(r["nums"]), int(r["target"]), random.Random(0))
        if isinstance(d_near, tuple):
            d_near = d_near[0]
        d_other = next((w for w in (wit_pool[rng.randrange(len(wit_pool))] for _ in range(20))
                        if w != wit), None)
        if not wit or d_near is None or d_other is None:
            skip += 1
            continue
        msgs = json.loads(r["prompt_json"]) if isinstance(r["prompt_json"], str) else r["prompt_json"]
        try:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = str(r["response_text"])
        pre_ids = tok(head + text[:int(r["meta_start"])], add_special_tokens=False).input_ids
        cls_ids = tok(head + text[:int(r["meta_end"])], add_special_tokens=False).input_ids
        # ★두 문맥에 «똑같이» 붙일 원본 이어쓰기
        cont_ids = tok(text[int(r["meta_end"]):], add_special_tokens=False).input_ids
        tg = {"gold": tok("\\boxed{" + wit + "}", add_special_tokens=False).input_ids,
              "near": tok("\\boxed{" + d_near + "}", add_special_tokens=False).input_ids,
              "other": tok("\\boxed{" + d_other + "}", add_special_tokens=False).input_ids}
        for D in DELAYS:
            if D > len(cont_ids):
                continue
            tail = cont_ids[:D]
            for cn, base in (("open", pre_ids), ("close", cls_ids)):
                ctx = base + tail
                for tn, t in tg.items():
                    index[(j, D, cn, tn)] = len(reqs)
                    reqs.append((ctx, t))
        rows.append(dict(idx=j, site_id=r["site_id"], n_cont=len(cont_ids)))
    print(f"[tp] 시퀀스 {len(reqs)}개 · skip={skip}", flush=True)

    lps = score_logp_hf(model, reqs, batch_size=args.batch_size)

    out = []
    for rw in rows:
        j = rw["idx"]
        rec = {"site_id": rw["site_id"], "n_cont": rw["n_cont"]}
        for D in DELAYS:
            def g(c, t):
                i = index.get((j, D, c, t))
                return None if i is None else lps[i]
            go, gc = g("open", "gold"), g("close", "gold")
            if go is None or gc is None:
                continue
            for dn in ("near", "other"):
                do, dc = g("open", dn), g("close", dn)
                if do is None or dc is None:
                    continue
                rec[f"shift_{dn}_D{D}"] = float((gc.mean() - dc.mean()) - (go.mean() - do.mean()))
            rec[f"osd_gold_D{D}"] = float(gc.mean() - go.mean())
        out.append(rec)
    df = pd.DataFrame(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"[tp] wrote {args.out} ({len(df)}행)", flush=True)


if __name__ == "__main__":
    main()
