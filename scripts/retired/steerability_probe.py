"""조향 가능성 검사 — 「지금 상태」가 「최종 결과」를 애초에 결정하는가.

동기(2026-09-01): 455 사이트 전수 재측정에서 혼잣말 지점의 어떤 상태 변수도 최종
정답을 예측하지 못했다 —
    confidence → 정답  AUC 0.499
    doom       → 정답  AUC 0.511   (혼잣말을 안 읽고 잰 «실제 막힘»)
    novelty    → 정답  AUC 0.453   (혼잣말 뒤에 실제로 새 조합을 뒤졌는가)
즉 이 과제에서는 중간 지점의 상태가 결과와 무관하다. 그렇다면 **어떤 혼잣말도
결과를 바꿀 수 없다** — 조향할 대상 자체가 없기 때문이다. 자를 고치는 문제가 아니다.

이 스크립트는 그 전제를 과제별로 직접 잰다. 혼잣말도, 자도, 보상도 쓰지 않는다.

  각 (문제, 프리픽스) 에서 N번 이어 쓰고 성공률 p̂ 를 잰다.
  프리픽스가 정보를 담는다면 p̂ 의 프리픽스간 분산이 이항잡음보다 커야 한다.

      ICC = (프리픽스간 참분산) / (프리픽스간 참분산 + 이항잡음)
      ICC ≈ 0  →  어느 프리픽스에서 출발하든 똑같다 = 조향 불가 = 계열 종료
      ICC 크다 →  좋은 프리픽스/나쁜 프리픽스가 실재 = 혼잣말이 바꿀 여지 있음

  ★같은 문제 안에서만 비교한다(문제 난이도가 분산을 부풀리므로 문제를 그룹으로
    잡고 within-problem 성분만 센다). 이것이 이 검사의 핵심 통제다.

과제를 인자로 받아 4수 · 5수를 같은 절차로 재고 나란히 놓는다. 5수에서 ICC 가 크고
4수에서 0 이면, 문제는 보상이 아니라 «과제»이며 5수로 옮겨야 한다는 뜻이다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
sys.path.insert(0, os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"))
from src.training.countdown_task import PROMPT_VARIANTS, grade  # noqa: E402

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")


def icc_within(df: pd.DataFrame, n: int) -> dict:
    """문제를 그룹으로 잡고 프리픽스간 «참»분산 성분만 뽑는다 (일원 분산분석)."""
    out = {}
    parts = []
    for _, g in df.groupby("prob"):
        if len(g) < 2:
            continue
        p = g.succ.values / n
        # 프리픽스간 관측분산에서 이항잡음 기대치를 뺀다
        obs = p.var(ddof=1)
        noise = float(np.mean(p * (1 - p) / n))
        parts.append((obs, noise, len(g)))
    if not parts:
        return dict(icc=np.nan, n_prob=0)
    w = np.array([x[2] for x in parts], float)
    obs = float(np.average([x[0] for x in parts], weights=w))
    noise = float(np.average([x[1] for x in parts], weights=w))
    true = max(0.0, obs - noise)
    out.update(var_obs=obs, var_noise=noise, var_true=true,
               icc=true / obs if obs > 0 else np.nan, n_prob=len(parts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True, help="보고용 과제 이름 (예: 4num, 5num)")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--n_prefix", type=int, default=4, help="문제당 서로 다른 프리픽스 수")
    ap.add_argument("--n_cont", type=int, default=8, help="프리픽스당 이어쓰기 수")
    ap.add_argument("--cut_tokens", type=int, default=900)
    ap.add_argument("--seed", type=int, default=21)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    args = ap.parse_args()

    df = pd.read_parquet(args.data).head(args.limit)
    print(f"[steer] {args.tag} · 문제 {len(df)} · 프리픽스 {args.n_prefix} · 이어쓰기 {args.n_cont}",
          flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_util, max_model_len=4864, enforce_eager=True)

    def chat(u):
        msgs = [{"role": "system", "content": PROMPT_VARIANTS["new"]},
                {"role": "user", "content": u}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    heads, golds = [], []
    for _, r in df.iterrows():
        p = r["prompt"]
        heads.append(chat(p[-1]["content"] if isinstance(p, (list, np.ndarray)) else str(p)))
        golds.append((list(int(v) for v in r["nums"]), int(r["target"])))

    # ── 프리픽스 만들기: 문제당 n_prefix 개, 답을 쓰기 전에서 자른다 ─────────
    first = llm.generate(heads, SamplingParams(temperature=1.0, top_p=1.0,
                                               max_tokens=args.cut_tokens,
                                               n=args.n_prefix, seed=args.seed))
    stems, keys = [], []
    for i, o in enumerate(first):
        nums, tgt = golds[i]
        for k, g in enumerate(o.outputs):
            txt = g.text
            if "\\boxed" in txt:
                if grade(txt, nums, tgt):
                    continue                       # 이미 푼 프리픽스는 제외
                txt = txt[:txt.index("\\boxed")]   # 답 선언 직전까지만
            if len(txt) < 50:
                continue
            stems.append(heads[i] + txt)
            keys.append((i, k))
    print(f"[steer] 프리픽스 {len(stems)}", flush=True)

    outs = llm.generate(stems, SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1400,
                                              n=args.n_cont, seed=args.seed + 7))
    rows = []
    for (i, k), o in zip(keys, outs):
        nums, tgt = golds[i]
        succ = sum(int(bool(grade(g.text, nums, tgt))) for g in o.outputs)
        rows.append(dict(task=args.tag, prob=i, pre=k, succ=succ, n=args.n_cont))
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)

    st = icc_within(out, args.n_cont)
    print(f"[steer] wrote {args.out} ({len(out)}행)", flush=True)
    print(f"[steer] {args.tag}  평균성공률 {out.succ.sum()/ (len(out)*args.n_cont):.4f}", flush=True)
    print(f"[steer] {args.tag}  ICC(within-problem) = {st.get('icc', float('nan')):.4f}   "
          f"관측분산 {st.get('var_obs', 0):.5f} = 참 {st.get('var_true', 0):.5f} "
          f"+ 잡음 {st.get('var_noise', 0):.5f}   문제수 {st.get('n_prob', 0)}", flush=True)


if __name__ == "__main__":
    main()
