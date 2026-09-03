"""PMI-shift 집계 방식 비교 — mean vs sum vs min vs max vs 분위수.

배경: 현행 R1 은 gold/decoy 의 «발산 토큰»(대부분 1개)만 비교해 낙제(AUC 0.522)했고,
전체 평균으로 고친 R2_full 만 살아남았다(0.587). 그렇다면 평균이 최선인가?
RLT 계열처럼 «가장 어려운 토큰»(min) 이나 «가장 쉬운 토큰»(max) 이 더 나을 수 있다.

계산 (문맥 2종 × 대상 2종 = 4벌의 per-token logp):
    ctx_open  = 프롬프트 + 메타 «직전»까지          (혼잣말 없음)
    ctx_close = 프롬프트 + 메타 «끝»까지            (혼잣말 있음)
    target    = "\\boxed{witness}" (gold) / "\\boxed{decoy}" (오답, 연산자 1개 교체)

    PMI_agg(ctx) = agg(logp(gold_t | ctx)) − agg(logp(decoy_t | ctx))
    R2_agg       = PMI_agg(close) − PMI_agg(open)      ← 혼잣말이 만든 이동

집계: mean · sum · min · max · median · top25(상위25% 평균) · bot25(하위25% 평균)
      · minmax(min+max 평균) · range(max−min)

출력: pmi_agg_scores.parquet  [site_id, R2_<agg> …]  + 예시 1건의 per-token 덤프.
채점 백엔드는 HF forward (vLLM prompt_logprobs 는 이 환경에서 segfault).
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
from src.training.countdown_task import eval_countdown, swap_op_decoy  # noqa: E402

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

AGGS = {
    "mean": lambda a: float(a.mean()),
    "sum": lambda a: float(a.sum()),
    "min": lambda a: float(a.min()),
    "max": lambda a: float(a.max()),
    "median": lambda a: float(np.median(a)),
    "top25": lambda a: float(np.sort(a)[-max(1, len(a) // 4):].mean()),
    "bot25": lambda a: float(np.sort(a)[:max(1, len(a) // 4)].mean()),
    "minmax": lambda a: float((a.min() + a.max()) / 2),
    "range": lambda a: float(a.max() - a.min()),
}

_SWAP = {"+": "*", "-": "+", "*": "-", "/": "+"}


def swap_one_op(witness: str, target) -> str | None:
    """연산자 하나만 바꾼 오답 (값이 target 과 달라야 한다) — R1/R2 가 쓰는 미끼 계열."""
    t = int(target)
    for i, ch in enumerate(witness):
        if ch in _SWAP:
            cand = witness[:i] + _SWAP[ch] + witness[i + 1:]
            if eval_countdown(cand) != t:
                return cand
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
        if done % 100 < batch_size or done == len(todo):
            print(f"[agg] scored {done}/{len(todo)}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_glob", required=True)
    ap.add_argument("--sites_file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump_example", default=None, help="per-token 덤프 저장 경로(json)")
    ap.add_argument("--batch_size", type=int, default=4)
    args = ap.parse_args()

    sites = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(args.sites_glob))],
                      ignore_index=True).drop_duplicates("site_id")
    if args.sites_file:
        keep = {l.strip() for l in open(args.sites_file) if l.strip()}
        sites = sites[sites.site_id.isin(keep)]
    print(f"[agg] 대상 {len(sites)} 사이트", flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    reqs, index, rows, skip = [], {}, [], 0
    for j, r in sites.iterrows():
        wit = str(r["witness"])
        # ★배터리·학습과 «동일한» 미끼 생성기를 쓴다 (자체 구현은 규칙 위반식을 만들어
        #   대비가 과대해지고 결과가 어긋난다 — 08-28 실측).
        import random as _rnd
        dec = swap_op_decoy(wit, list(r["nums"]), int(r["target"]), _rnd.Random(0))
        if isinstance(dec, tuple):
            dec = dec[0]
        if not wit or dec is None:
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
        g_ids = tok("\\boxed{" + wit + "}", add_special_tokens=False).input_ids
        d_ids = tok("\\boxed{" + dec + "}", add_special_tokens=False).input_ids
        for cn, ctx in (("open", ctx_open), ("close", ctx_close)):
            for tn, tg in (("gold", g_ids), ("decoy", d_ids)):
                index[(j, cn, tn)] = len(reqs)
                reqs.append((ctx, tg))
        rows.append(dict(idx=j, site_id=r["site_id"], witness=wit, decoy=dec,
                         gold_tok=[tok.decode([t]) for t in g_ids],
                         decoy_tok=[tok.decode([t]) for t in d_ids]))
    print(f"[agg] 시퀀스 {len(reqs)}개 · skip={skip}", flush=True)

    lps = score_logp_hf(model, reqs, batch_size=args.batch_size)

    out, dumped = [], None
    for rw in rows:
        j = rw["idx"]
        vals = {k: lps[index[(j, c, t)]] for k, (c, t) in
                {"go": ("open", "gold"), "do": ("open", "decoy"),
                 "gc": ("close", "gold"), "dc": ("close", "decoy")}.items()}
        if any(v is None for v in vals.values()):
            continue
        rec = {"site_id": rw["site_id"]}
        for an, af in AGGS.items():
            rec[f"R2_{an}"] = ((af(vals["gc"]) - af(vals["dc"]))
                               - (af(vals["go"]) - af(vals["do"])))
        out.append(rec)
        if dumped is None and len(vals["gc"]) >= 6:
            dumped = dict(site_id=rw["site_id"], witness=rw["witness"], decoy=rw["decoy"],
                          gold_tokens=rw["gold_tok"], decoy_tokens=rw["decoy_tok"],
                          gold_open=[round(float(x), 3) for x in vals["go"]],
                          gold_close=[round(float(x), 3) for x in vals["gc"]],
                          decoy_open=[round(float(x), 3) for x in vals["do"]],
                          decoy_close=[round(float(x), 3) for x in vals["dc"]])
    df = pd.DataFrame(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out)
    print(f"[agg] wrote {args.out} ({len(df)}행)", flush=True)
    if args.dump_example and dumped:
        Path(args.dump_example).write_text(json.dumps(dumped, ensure_ascii=False, indent=1))
        print(f"[agg] 예시 덤프 → {args.dump_example}", flush=True)


if __name__ == "__main__":
    main()
