"""역방향 자(reverse ruler) 검정 — p(메타 | 정답 힌트) 계열.

기존 배터리(ruler_battery.py)의 자들은 모두 «메타를 조건으로 → 정답식의 logP»
방향이었고 전부 낙제했다(AUC 0.45~0.55). 이 스크립트는 방향을 뒤집는다:

  정답(witness)을 프롬프트에 힌트로 주었을 때 «그 메타 문장 자체»의 logP 가
  얼마나 오르는가 = 정답을 아는 선생 눈에 이 혼잣말이 얼마나 그럴듯해지는가.

문맥 3종 (프리픽스·메타는 동일, user 메시지 끝에 힌트 한 줄만 다름):
  plain  : 힌트 없음
  gold   : "Hint: one valid solution is <witness>."
  decoy  : "Hint: one valid solution is <family decoy>."   (연산 가족을 바꾼 오답)

집계 변형 (메타 토큰 per-token logP 배열에 대해):
  mean / max / min / top10 (상위 10% 평균) / sum

자:
  V1_gold_mean 등  = agg(gold) − agg(plain)          «정답 조건화 이득»
  V2_gvd_mean  등  = agg(gold) − agg(decoy)          «정답 vs 오답 대비»
  V3_prose_*       = V1 을 프로즈 토큰만으로 (conf/decision 줄 제외)

주: 합(sum) 형태의 V1 은 PMI 대칭성 때문에 이론상 log p(gold|meta)−log p(gold)
와 같지만, 길이 정규화(mean/max)를 하면 대칭이 깨져 별개의 자가 된다 —
낙제한 R2/R4 와 다른 결과가 나올 수 있는 이유가 여기에 있다.

채점 백엔드는 HF forward (vLLM prompt_logprobs 는 이 환경에서 EngineCore segfault).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"))
from src.training.countdown_task import eval_countdown  # noqa: E402

W = "/home/jovyan/beomi/splee"
MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

_FAM_MULDIV = str.maketrans({"*": "+", "/": "-"})
_FAM_ADDSUB = str.maketrans({"+": "*", "-": "/"})


def make_decoy_family(witness: str, target) -> str | None:
    """ruler_battery.make_decoy_family 와 동일 규칙 (가족을 바꾼 «값이 다른» 식)."""
    t = int(target)
    for cand in (witness.translate(_FAM_MULDIV), witness.translate(_FAM_ADDSUB)):
        if cand == witness:
            continue
        if eval_countdown(cand) != t:
            return cand
    cand = f"({witness})+1"
    return cand if eval_countdown(cand) != t else None


_CONF_LINE = re.compile(r"^\s*confidence\s*:", re.I)
_DEC_LINE = re.compile(r"^\s*decision\s*:", re.I)


def prose_span(meta_raw: str) -> tuple[int, int] | None:
    """meta_raw 안에서 프로즈(= conf/decision/태그 줄이 아닌 본문) 문자 구간."""
    lines, off, spans = meta_raw.split("\n"), 0, []
    for ln in lines:
        s, e = off, off + len(ln)
        off = e + 1
        if not ln.strip() or ln.strip().startswith("<") or _CONF_LINE.match(ln) or _DEC_LINE.match(ln):
            continue
        spans.append((s, e))
    if not spans:
        return None
    return spans[0][0], spans[-1][1]


def chat_prefix(tok, prompt_json, hint: str | None) -> str:
    """수확 당시 저장된 원본 prompt_json 사용 (학습과 바이트 동일).
    hint 가 있으면 user 메시지 끝에 한 줄만 덧붙인다."""
    raw = json.loads(prompt_json) if isinstance(prompt_json, str) else prompt_json
    msgs = [dict(m) for m in raw]
    if hint:
        for m in reversed(msgs):
            if m["role"] == "user":
                m["content"] = m["content"] + "\n\nHint: one valid solution is " + hint + "."
                break
    try:
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def score_logp_hf(model, reqs, batch_size=4, max_len=6144):
    """[(ctx_ids, tgt_ids)] → 대상 구간 per-token logprob 배열 (HF forward)."""
    import torch
    out = [None] * len(reqs)
    todo = [i for i, (c, t) in enumerate(reqs) if t and len(c) >= 1 and len(c) + len(t) <= max_len]
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
        if done % 60 < batch_size or done == len(todo):
            print(f"[rev] scored {done}/{len(todo)}", flush=True)
    return out


AGGS = {
    "mean": lambda a: float(a.mean()),
    "max": lambda a: float(a.max()),
    "min": lambda a: float(a.min()),
    "top10": lambda a: float(np.sort(a)[-max(1, len(a) // 10):].mean()),
    "sum": lambda a: float(a.sum()),
}


def auc(y, s):
    from scipy.stats import rankdata
    y = np.asarray(y, float)
    s = np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(set(y.tolist())) < 2:
        return float("nan")
    r = rankdata(s)
    n1, n0 = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc_ci(y, s, n=4000, seed=0):
    rng = np.random.RandomState(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(set(y[i].tolist())) < 2:
            continue
        vals.append(auc(y[i], s[i]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--sites_glob", default=f"{W}/cd6_work/probe/sites_shard*.parquet")
    ap.add_argument("--out_dir", default=f"{W}/cd6_work/probe")
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    lab = pd.read_parquet(args.labels)
    sites = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(args.sites_glob))],
                      ignore_index=True).drop_duplicates("site_id")
    df = lab.merge(sites, on="site_id", how="inner", suffixes=("", "_s"))
    if args.limit:
        # 층화: good/null 을 절반씩 (연기시험에서 한 클래스만 뽑히는 것 방지)
        h = max(1, args.limit // 2)
        df = pd.concat([df[df.label == "good"].head(h), df[df.label == "null"].head(h)])
    print(f"[rev] 채점 대상 {len(df)} 사이트 (label 분포: {df.label.value_counts().to_dict()})",
          flush=True)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

    reqs, index, rows, skip = [], {}, [], 0
    for j, r in df.iterrows():
        wit = str(r["witness"])
        decoy = make_decoy_family(wit, r["target"])
        if decoy is None:
            skip += 1
            continue
        pre = str(r["response_text"])[:int(r["meta_start"])]
        meta = str(r["meta_raw"])
        ps = prose_span(meta)
        for cond, hint in (("plain", None), ("gold", wit), ("decoy", decoy)):
            ctx_txt = chat_prefix(tok, r["prompt_json"], hint) + pre
            ctx = tok(ctx_txt, add_special_tokens=False).input_ids
            tgt = tok(meta, add_special_tokens=False).input_ids
            index[(j, cond, "all")] = len(reqs)
            reqs.append((ctx, tgt))
            if ps:
                index[(j, cond, "prose")] = len(reqs)
                reqs.append((ctx, tok(meta[ps[0]:ps[1]], add_special_tokens=False).input_ids))
        rows.append(dict(idx=j, site_id=r["site_id"], label=r["label"],
                         delta=float(r.get("delta_ctl_B", np.nan))))
    print(f"[rev] 시퀀스 {len(reqs)}개 · decoy 실패 skip={skip}", flush=True)

    lps = score_logp_hf(model, reqs, batch_size=args.batch_size)

    out = []
    for rw in rows:
        j = rw["idx"]
        rec = dict(site_id=rw["site_id"], label=rw["label"], delta=rw["delta"])
        for scope in ("all", "prose"):
            g = index.get((j, "gold", scope))
            p = index.get((j, "plain", scope))
            d = index.get((j, "decoy", scope))
            if g is None or p is None or d is None:
                continue
            G, P, D = lps[g], lps[p], lps[d]
            if G is None or P is None or D is None:
                continue
            for an, af in AGGS.items():
                rec[f"V1_{scope}_{an}"] = af(G) - af(P)
                rec[f"V2_{scope}_{an}"] = af(G) - af(D)
        out.append(rec)
    sc = pd.DataFrame(out)
    outdir = Path(args.out_dir)
    sc.to_parquet(outdir / "reverse_scores.parquet")

    y = (sc.label == "good").astype(int).to_numpy()
    cols = [c for c in sc.columns if c.startswith(("V1_", "V2_"))]
    report = []
    for c in cols:
        s = sc[c].to_numpy(float)
        ok = ~np.isnan(s)
        a = auc(y[ok], s[ok])
        lo, hi = boot_auc_ci(y[ok], s[ok])
        report.append(dict(ruler=c, auc=a, lo=lo, hi=hi, n=int(ok.sum())))
    rep = pd.DataFrame(report).sort_values("auc", ascending=False)
    rep.to_csv(outdir / "reverse_report.csv", index=False)
    print("\n## 역방향 자 판별력 (good vs null)\n")
    print("| 자 | AUC | 95% CI | n |")
    print("|---|---|---|---|")
    for _, r in rep.iterrows():
        print(f"| {r.ruler} | {r.auc:.3f} | [{r.lo:.3f}, {r.hi:.3f}] | {int(r.n)} |")
    print(f"\n[rev] wrote {outdir}/reverse_scores.parquet · reverse_report.csv", flush=True)


if __name__ == "__main__":
    main()
