"""프리픽스 상태 자 — «지금 이 길이 막혔는가»를 메타를 보지 않고 읽는다.

동기: Δ_abl 라벨이 잡은 축은 «막힌 자리에서 계속 가자(해로움) vs 방향을 튼다(도움)»
였다. 그런데 가장 잘 맞히는 신호가 메타 자신이 쓴 decision 토큰(AUC 0.664)이라,
그걸 그대로 보상하면 내용과 무관하게 "redirect"만 쓰면 되는 구트하트가 된다.

그래서 «막힘»은 메타 앞의 프리픽스에서만 읽고(메타를 보지 않는다), 메타는 결정만
제공하게 한다. 보상은 둘의 «정합»이다.

계산 (전부 프리픽스 = 메타 직전까지의 텍스트만 사용):
  doom_gold = mean logP(\boxed{witness}      | prompt + prefix)
  doom_fam  = mean logP(\boxed{family decoy} | prompt + prefix)
  doom      = doom_gold − doom_fam      (높을수록 «정답 가족 쪽에 서 있다» = 안 막힘)
  doom_abs  = doom_gold                 (절대 수준)

산출: prefix_state.parquet [site_id, doom, doom_gold, doom_fam, decision, confidence]
채점은 HF forward (vLLM prompt_logprobs 는 이 환경에서 EngineCore segfault).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"))
from src.training.countdown_task import eval_countdown  # noqa: E402

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

_FAM_MULDIV = str.maketrans({"*": "+", "/": "-"})
_FAM_ADDSUB = str.maketrans({"+": "*", "-": "/"})


def make_decoy_family(witness: str, target) -> str | None:
    t = int(target)
    for cand in (witness.translate(_FAM_MULDIV), witness.translate(_FAM_ADDSUB)):
        if cand != witness and eval_countdown(cand) != t:
            return cand
    cand = f"({witness})+1"
    return cand if eval_countdown(cand) != t else None


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
        if done % 80 < batch_size or done == len(todo):
            print(f"[state] scored {done}/{len(todo)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_glob", required=True)
    ap.add_argument("--sites_file", default=None, help="site_id 목록(줄당 1개)으로 제한")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    sites = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(args.sites_glob))],
                      ignore_index=True).drop_duplicates("site_id")
    if args.sites_file:
        keep = {l.strip() for l in open(args.sites_file) if l.strip()}
        sites = sites[sites.site_id.isin(keep)]
    print(f"[state] 대상 {len(sites)} 사이트", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    reqs, index, rows, skip = [], {}, [], 0
    for j, r in sites.iterrows():
        wit = str(r["witness"])
        dec = make_decoy_family(wit, r["target"])
        if not wit or dec is None:
            skip += 1
            continue
        msgs = json.loads(r["prompt_json"]) if isinstance(r["prompt_json"], str) else r["prompt_json"]
        try:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # ★메타 «직전»까지만 — 메타는 보지 않는다
        ctx = tok(head + str(r["response_text"])[:int(r["meta_start"])],
                  add_special_tokens=False).input_ids
        for nm, expr in (("gold", wit), ("fam", dec)):
            index[(j, nm)] = len(reqs)
            reqs.append((ctx, tok("\\boxed{" + expr + "}", add_special_tokens=False).input_ids))
        rows.append(dict(idx=j, site_id=r["site_id"], decision=r.get("decision"),
                         confidence=r.get("confidence"), prob_idx=r.get("prob_idx")))
    print(f"[state] 시퀀스 {len(reqs)}개 · skip={skip}", flush=True)

    lps = score_logp_hf(model, reqs, batch_size=args.batch_size)
    out = []
    for rw in rows:
        g, f = lps[index[(rw["idx"], "gold")]], lps[index[(rw["idx"], "fam")]]
        if g is None or f is None:
            continue
        out.append(dict(site_id=rw["site_id"], decision=rw["decision"],
                        confidence=rw["confidence"], prob_idx=rw["prob_idx"],
                        doom_gold=float(g.mean()), doom_fam=float(f.mean()),
                        doom=float(g.mean() - f.mean())))
    df = pd.DataFrame(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"[state] wrote {args.out} ({len(df)}행)", flush=True)


if __name__ == "__main__":
    main()
