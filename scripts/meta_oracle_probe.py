"""메타 «선택»의 상한 — 좋은 혼잣말만 고를 수 있다면 얼마나 좋아지는가.

앞선 상한 실험의 결함: 고정된 «방향전환» 블록 하나를 끼워 넣고 «혼잣말이 도움이 되나»
를 물었다. 그러나 가설은 「아무 혼잣말」이 아니라 「**좋은** 혼잣말」이 돕는다는 것이다.

이 실험은 그 가설을 정면으로 잰다.
  같은 지점에서 모델이 «자기 혼잣말을 N개» 쓰게 하고, 각각을 끝까지 이어 풀린다.
    oracle  = N개 중 «하나라도» 맞히면 성공          ← 완벽한 자가 있을 때의 상한
    random  = N개의 평균                              ← 지금 정책 (아무거나 하나)
    none    = 같은 지점에서 혼잣말 없이 이어 풀기       ← 혼잣말 자체의 기여
  oracle − random = 「좋은 혼잣말을 고를 수 있다면」 벌 수 있는 최대치 (= 자의 상한)
  random − none   = 「혼잣말을 쓰는 것 자체」의 효과

주: oracle 은 N개 중 최댓값이라 N 이 커질수록 올라간다(best-of-N 편향). 그래서
    같은 N 으로 «혼잣말 없이» N번 이어 쓴 것(none_oracle)도 함께 재서, 상한의 몫 중
    «혼잣말 덕분»과 «그냥 여러 번 시도한 덕분»을 분리한다. 이것이 이 실험의 핵심 통제다.
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

META_CUE = "\n<meta>\n"          # 모델이 «자기» 혼잣말을 쓰도록 여는 단서만 준다


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--n_traj", type=int, default=4, help="문제당 1차 궤적 수")
    ap.add_argument("--n_meta", type=int, default=4, help="절단점에서 뽑을 혼잣말 수")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    args = ap.parse_args()

    df = pd.read_parquet(args.data).head(args.limit)
    print(f"[oracle] 문제 {len(df)} · 궤적 {args.n_traj} · 혼잣말 {args.n_meta} · 씨앗 {args.seed}",
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

    users, golds = [], []
    for _, r in df.iterrows():
        p = r["prompt"]
        users.append(p[-1]["content"] if isinstance(p, (list, np.ndarray)) else str(p))
        golds.append((list(int(v) for v in r["nums"]), int(r["target"])))
    heads = [chat(u) for u in users]

    # ── 1차 궤적: 900토큰만 ────────────────────────────────────────────────
    first = llm.generate(heads, SamplingParams(temperature=1.0, top_p=1.0, max_tokens=900,
                                               n=args.n_traj, seed=args.seed))
    rng = np.random.RandomState(args.seed)
    stems, keys = [], []
    for i, o in enumerate(first):
        nums, tgt = golds[i]
        for k, g in enumerate(o.outputs):
            txt = g.text
            if "\\boxed" in txt and grade(txt, nums, tgt):
                continue                      # 이미 맞힌 궤적은 개입 대상 아님
            cut = txt.index("\\boxed") if "\\boxed" in txt else len(txt)
            pos = int(rng.randint(0, max(1, cut)))   # 앞선 실험에서 «이른 절단»이 나았다
            stems.append(heads[i] + txt[:pos])
            keys.append((i, k))
    print(f"[oracle] 개입 대상 {len(stems)} 궤적", flush=True)

    rows = []
    # ── 조건 META: 절단점에서 «모델 자신의 혼잣말» N개 → 각각 이어 풀기 ──────
    outs = llm.generate([s + META_CUE for s in stems],
                        SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1400,
                                       n=args.n_meta, seed=args.seed))
    for (i, k), o in zip(keys, outs):
        nums, tgt = golds[i]
        for j, g in enumerate(o.outputs):
            rows.append(dict(cond="meta", prob=i, traj=k, cand=j,
                             correct=int(bool(grade(g.text, nums, tgt))),
                             meta_text=g.text[:400]))
    print("[oracle] META 완료", flush=True)

    # ── 조건 NONE: 같은 절단점에서 혼잣말 «없이» 같은 수만큼 이어 풀기 ────────
    outs = llm.generate(stems, SamplingParams(temperature=1.0, top_p=1.0, max_tokens=1400,
                                              n=args.n_meta, seed=args.seed + 1000))
    for (i, k), o in zip(keys, outs):
        nums, tgt = golds[i]
        for j, g in enumerate(o.outputs):
            rows.append(dict(cond="none", prob=i, traj=k, cand=j,
                             correct=int(bool(grade(g.text, nums, tgt))), meta_text=""))
    print("[oracle] NONE 완료", flush=True)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    g = out.groupby(["cond", "prob", "traj"]).correct
    summ = pd.DataFrame({"oracle": g.max(), "random": g.mean()}).groupby("cond").mean()
    print(f"[oracle] wrote {args.out} ({len(out)}행)", flush=True)
    print(summ.round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
