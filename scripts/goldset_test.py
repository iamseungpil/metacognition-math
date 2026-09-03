"""정답 «집합» 실험 — 증인식 하나가 아니라 그 문제의 모든 해를 정답으로 보면 자가 나아지는가.

문제의식: 지금 자는 데이터셋에 적힌 증인식 **하나만** 정답으로 놓고 채점한다. Countdown
4수 문제는 보통 해가 여러 개인데, 롤아웃이 «다른 유효한 해»를 향해 잘 가고 있어도
"정답 확률이 낮다"고 벌점을 받는다 — 좋은 혼잣말이 벌을 받는 구조다.

해결: 네 수로 만들 수 있는 모든 식을 열거해 목표에 도달하는 것을 전부 모으고,
      «어떤 해든» 을 정답으로 본다.

  logp_set(ctx) = logsumexp_s [ mean logp(\\boxed{s} | ctx) ]   (해가 여러 개일 때)
  또는 max_s 도 함께 계산해 비교한다.

  shift_set = [logp_set(close) − logp_decoy(close)] − [logp_set(open) − logp_decoy(open)]

미끼는 08-28 비교에서 가장 좋았던 «다른 문제의 정답식»을 쓴다.
해가 너무 많으면 상위 N개(짧은 순)만 쓴다 — 토큰 예산 때문.
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import random
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


def all_solutions(nums, target, cap=6):
    """네 수 전부를 한 번씩 써서 target 이 되는 식들 (문자열). 중간값 양의 정수 규칙 준수."""
    t = int(target)
    found, seen = [], set()
    for perm in set(itertools.permutations([int(v) for v in nums])):
        a, b, c, d = perm
        for o1 in "+-*/":
            for o2 in "+-*/":
                for o3 in "+-*/":
                    for tmpl in (f"((({a}{o1}{b}){o2}{c}){o3}{d})",
                                 f"(({a}{o1}{b}){o2}({c}{o3}{d}))",
                                 f"({a}{o1}(({b}{o2}{c}){o3}{d}))"):
                        if tmpl in seen:
                            continue
                        seen.add(tmpl)
                        if eval_countdown(tmpl) == t:
                            found.append(tmpl)
                            if len(found) >= cap * 4:
                                return sorted(found, key=len)[:cap]
    return sorted(found, key=len)[:cap]


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
        if done % 200 < batch_size or done == len(todo):
            print(f"[gs] scored {done}/{len(todo)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_glob", required=True)
    ap.add_argument("--sites_file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=6)
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    sites = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(args.sites_glob))],
                      ignore_index=True).drop_duplicates("site_id").reset_index(drop=True)
    if args.sites_file:
        keep = {l.strip() for l in open(args.sites_file) if l.strip()}
        sites = sites[sites.site_id.isin(keep)].reset_index(drop=True)
    print(f"[gs] 대상 {len(sites)} 사이트", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    wit_pool = [str(r.witness) for _, r in sites.iterrows() if str(r.witness)]
    rng = random.Random(0)
    reqs, index, rows = [], {}, []
    for j, r in sites.iterrows():
        sols = all_solutions(list(r["nums"]), int(r["target"]), cap=args.cap)
        wit = str(r["witness"])
        if wit and wit not in sols:
            sols = [wit] + sols[:args.cap - 1]
        if not sols:
            continue
        d_other = next((w for w in (wit_pool[rng.randrange(len(wit_pool))] for _ in range(20))
                        if w != wit), None)
        if d_other is None:
            continue
        msgs = json.loads(r["prompt_json"]) if isinstance(r["prompt_json"], str) else r["prompt_json"]
        try:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = str(r["response_text"])
        ctx_o = tok(head + text[:int(r["meta_start"])], add_special_tokens=False).input_ids
        ctx_c = tok(head + text[:int(r["meta_end"])], add_special_tokens=False).input_ids
        tgs = {f"s{i}": tok("\\boxed{" + s + "}", add_special_tokens=False).input_ids
               for i, s in enumerate(sols)}
        tgs["dec"] = tok("\\boxed{" + d_other + "}", add_special_tokens=False).input_ids
        for cn, ctx in (("open", ctx_o), ("close", ctx_c)):
            for tn, tg in tgs.items():
                index[(j, cn, tn)] = len(reqs)
                reqs.append((ctx, tg))
        rows.append(dict(idx=j, site_id=r["site_id"], n_sol=len(sols)))
    print(f"[gs] 시퀀스 {len(reqs)}개 · 해 개수 중앙값 "
          f"{int(np.median([x['n_sol'] for x in rows])) if rows else 0}", flush=True)

    lps = score_logp_hf(model, reqs, batch_size=args.batch_size)

    out = []
    for rw in rows:
        j, n = rw["idx"], rw["n_sol"]
        def g(c, t):
            i = index.get((j, c, t))
            return None if i is None else lps[i]
        dec_o, dec_c = g("open", "dec"), g("close", "dec")
        so = [g("open", f"s{i}") for i in range(n)]
        sc = [g("close", f"s{i}") for i in range(n)]
        if dec_o is None or dec_c is None or any(v is None for v in so + sc):
            continue
        mo = np.array([v.mean() for v in so]); mc = np.array([v.mean() for v in sc])
        lse = lambda a: float(np.log(np.exp(a - a.max()).sum()) + a.max())
        out.append(dict(site_id=rw["site_id"], n_sol=n,
                        shift_gold1=float((mc[0] - dec_c.mean()) - (mo[0] - dec_o.mean())),
                        shift_setmax=float((mc.max() - dec_c.mean()) - (mo.max() - dec_o.mean())),
                        shift_setlse=float((lse(mc) - dec_c.mean()) - (lse(mo) - dec_o.mean())),
                        shift_setmean=float((mc.mean() - dec_c.mean()) - (mo.mean() - dec_o.mean()))))
    df = pd.DataFrame(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"[gs] wrote {args.out} ({len(df)}행)", flush=True)


if __name__ == "__main__":
    main()
