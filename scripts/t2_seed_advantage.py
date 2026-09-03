"""T2 — 프롬프트가 심은 전략 어휘(P5) × GRPO 0스텝 이득 부호.

질문: 학습 없이 프롬프트만으로 «다음 전략»을 닫힌 어휘로 말하게 하면(P5), 같은 문제의
n 롤아웃 안에서 어휘별로 집단상대 이득 adv = corr − mean(corr) 의 부호가 갈리는가?
갈린다면 GRPO 첫 스텝이 «그 어휘를 말하는 것» 자체를 밀거나 당긴다(전략 옵션의 씨앗).
대조 P0 은 verify/redirect 이진 어휘.

  python scripts/t2_seed_advantage.py --data hf_data/metacot-sdc-data/countdown_val_4num.parquet \
      --limit 300 --n 16 --cells P0,P5 --seed 71 --out cd6_work/probe/t2_s71.parquet

이득은 std 정규화 없이 평균만 뺀다(우리 verl 설정과 동일). 부트스트랩은 문제 단위 클러스터.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
REPO = os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.training.countdown_task import extract_expr  # noqa: E402
from countdown_gs0_eval import _solves  # noqa: E402
import steer_prompts as SP  # noqa: E402

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
_META = re.compile(r"<meta>(.*?)</meta>", re.S)
_CONF = re.compile(r"confidence\s*:\s*([01](?:\.\d+)?|\.\d+)", re.I)
_DEC0 = re.compile(r"^\s*decision\s*:\s*[`'\"<\[(*]*\s*(verify|redirect)", re.I | re.M)
DECS = ("backward", "decompose", "constrain", "answer", "continue", "redirect", "verify")
STRATA = (("pass≤.25", 0.0, 0.25), (".25–.75", 0.25, 0.75), ("pass≥.75", 0.75, 1.01))


def parse_decision(cell: str, body: str):
    if cell.startswith("P5"):
        return SP.parse_decision5(body)
    SP.parse_fields(body)                       # P0 계열: 필드 파서는 decision 을 안 돌려준다
    m = _DEC0.search(body)
    return m.group(1).lower() if m else None


def rollout(args, df, cells):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_util, max_model_len=args.max_tokens + 1800,
              enforce_eager=True)

    def chat(sys_txt, u):
        msgs = [{"role": "system", "content": sys_txt}, {"role": "user", "content": u}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    users, golds = [], []
    for _, r in df.iterrows():
        p = r["prompt"]
        users.append(p[-1]["content"] if isinstance(p, (list, np.ndarray)) else str(p))
        golds.append(([int(v) for v in r["nums"]], int(r["target"])))

    stride = max(100, args.n)                   # 프롬프트별 씨앗 간격 ≥ n → 자식 씨앗 전부 서로 다름
    seeds = {}
    for ci in range(len(cells)):
        for i in range(len(users)):
            seeds[(ci, i)] = args.seed * 10**7 + (ci * len(users) + i) * stride
    child = {seeds[k] + j for k in seeds for j in range(args.n)}
    assert len(child) == len(seeds) * args.n, "씨앗 충돌 — (프롬프트, 자식) 쌍이 겹친다"

    rows, t0, ntok = [], time.time(), 0
    for ci, cell in enumerate(cells):
        prompts = [chat(SP.VARIANTS[cell], u) for u in users]
        sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, n=args.n,
                              seed=seeds[(ci, i)]) for i in range(len(prompts))]
        outs = llm.generate(prompts, sps)
        for i, o in enumerate(outs):
            nums, tgt = golds[i]
            for k, g in enumerate(o.outputs):
                t = g.text
                e = extract_expr(t)
                m = _META.search(t)                       # 첫 메타 블록
                body = m.group(1) if m else ""
                c = _CONF.search(body) if m else None
                ntok += len(g.token_ids)
                rows.append(dict(
                    cell=cell, prob=i, k=k, seed=seeds[(ci, i)],
                    nums=str(nums), target=tgt,
                    corr=int(bool(e) and _solves(e, nums, tgt)),
                    has_meta=int(m is not None), has_boxed=int("\\boxed" in t),
                    decision=parse_decision(cell, body) if m else None,
                    confidence=float(c.group(1)) if c else np.nan,
                    meta_pos=(m.start() / max(len(t), 1)) if m else np.nan,
                    n_meta=len(_META.findall(t)),                                   # 감사: 다중 메타 기록
                    meta_before_boxed=int(m is not None and ("\\boxed" not in t or m.start() < t.index("\\boxed"))),
                    n_tok=len(g.token_ids), finish_reason=g.finish_reason, text=t[:12000]))
        print(f"[t2] 칸 {cell} 완료  ({time.time()-t0:.0f}s 누적)", flush=True)
    dt = time.time() - t0
    print(f"[t2] 생성 {ntok} 토큰 / {dt:.0f}s = {ntok/dt:.0f} tok/s", flush=True)
    return pd.DataFrame(rows)


def add_adv(out):
    out["adv"] = out["corr"] - out.groupby(["cell", "prob"])["corr"].transform("mean")
    out["pass_rate"] = out.groupby(["cell", "prob"])["corr"].transform("mean")
    return out


def contrast(g, d, B=1000, rng=None):
    """E[adv|dec=d] − E[adv|dec≠d], 문제 클러스터 부트스트랩 95% CI."""
    probs = g.prob.unique()
    byp = {p: (g.adv.values[g.prob.values == p], (g.decision.values[g.prob.values == p] == d))
           for p in probs}

    def est(ps):
        a = np.concatenate([byp[p][0] for p in ps]); m = np.concatenate([byp[p][1] for p in ps])
        return a[m].mean() - a[~m].mean() if 0 < m.sum() < len(m) else np.nan
    pt = est(probs)
    if np.isnan(pt) or B == 0:
        return pt, np.nan, np.nan
    bs = [est(rng.choice(probs, len(probs), replace=True)) for _ in range(B)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return pt, lo, hi


def summarize(out, cells, B=1000, seed=0):
    rng = np.random.default_rng(seed)
    for cell in cells:
        g = out[out.cell == cell]
        print(f"\n══ {cell}: 정답률 {g['corr'].mean():.4f} · 발화 {g.has_meta.mean():.3f} · "
              f"boxed {g.has_boxed.mean():.3f} · 문제 {g.prob.nunique()} × {g.k.nunique()}")
        em = g.decision.value_counts(dropna=False, normalize=True)
        print("  발화율:", "  ".join(f"{k}={v:.3f}" for k, v in em.items()))
        print(f"  {'decision':<10}{'emit':>7}{'Δadv':>9}{'CI lo':>9}{'CI hi':>9}    | 층별 Δadv (n)")
        for d in DECS:
            if (g.decision == d).sum() == 0:
                continue
            pt, lo, hi = contrast(g, d, B, rng)
            parts = []
            for name, a, b in STRATA:
                s = g[(g.pass_rate >= a) & (g.pass_rate < b)]
                sp, _, _ = contrast(s, d, 0, rng) if len(s) else (np.nan, 0, 0)
                parts.append(f"{name} {sp:+.3f} ({(s.decision == d).sum()})")
            nprob = g.loc[g.decision == d, "prob"].nunique()          # 감사: 기여 문제 수 (게이트 ≥ 30)
            print(f"  {d:<10}{(g.decision == d).mean():>7.3f}{pt:>+9.4f}{lo:>+9.4f}{hi:>+9.4f}"
                  f"    | " + " · ".join(parts) + f"    | 문제 {nprob}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--cells", default="P0,P5")
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--out", required=True)
    ap.add_argument("--summary_only", action="store_true", help="--out 을 읽어 요약만")
    args = ap.parse_args()
    cells = [c.strip() for c in args.cells.split(",")]
    if args.summary_only:
        out = pd.read_parquet(args.out)
    else:
        df = pd.read_parquet(args.data).head(args.limit)
        print(f"[t2] 문제 {len(df)} × 롤아웃 {args.n} × 칸 {cells} · 씨앗 {args.seed} · "
              f"max_tokens {args.max_tokens}", flush=True)
        out = add_adv(rollout(args, df, cells))
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(args.out)
        print(f"[t2] wrote {args.out} ({len(out)}행, 열 {list(out.columns)})", flush=True)
    summarize(out, cells)


if __name__ == "__main__":
    main()
