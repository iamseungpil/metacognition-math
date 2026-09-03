"""상한 확인 실험 — 「막혔을 때 방향 전환」에 애초에 여지가 있는가.

페이블 감사(2026-08-31)의 최우선 권고. 자도, 보상도, 학습도 쓰지 않고 **추론만**으로
개입 효과의 상한을 잰다. 여기서 여지가 없으면 자·보상·학습 전 계열을 접는다.

조건 (같은 문제·같은 씨앗에서 짝지어 비교):
  a  현행           PROMPT_VARIANTS["new"] 그대로
  b  메타 요구 없음   메타 지시문을 뺀 프롬프트
  d  ★강제 전환 주입  1차 생성 후, 최대엔트로피 지점에 «redirect 선언» 블록을 끼우고 재생성
  e  위약           d 와 같은 블록을 «무작위» 위치에 끼우고 재생성

통과: d − max(a,b) ≥ +2.0pp 이고 문제단위 부트스트랩 CI 하한 > 0 이고 d > e
중단: < +1.0pp → 「막혔을 때 전환」에 여지가 없다 → 전 계열 종료

주: 조건 c(메타 하드 금지)는 로짓 프로세서가 필요해 이 판본에서 뺐다 — b 가 같은 축을
    프롬프트 수준에서 이미 재고, 판정식은 max(a,b) 를 쓰므로 결론이 바뀌지 않는다.
"""
from __future__ import annotations

import argparse
import json
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

# 주입할 «방향 전환» 블록 — 내용은 문제 비의존(어느 문제에도 붙일 수 있어야 위약과 공정)
REDIRECT_BLOCK = ("\n<meta>\nconfidence: 0.2\n"
                  "This family of groupings has not reached the target. "
                  "I should abandon it and try a different family.\n"
                  "decision: redirect\n</meta>\n")

# 메타 지시문을 뺀 프롬프트 (조건 b)
def build_none_sys() -> str:
    import re
    s = PROMPT_VARIANTS["new"]
    # <meta> 블록을 요구하는 문단을 통째로 제거
    out = re.sub(r"\n[^\n]*<meta>.*?</meta>[^\n]*\n", "\n", s, flags=re.S)
    out = re.sub(r"(?im)^.*\bmeta\b.*$\n?", "", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--n", type=int, default=8, help="문제당 롤아웃")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--conds", default="a,b,d,e")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--only_failed", type=int, default=1,
                    help="1이면 1차 시도가 실패한 궤적에만 개입 (기본)")
    args = ap.parse_args()

    df = pd.read_parquet(args.data).head(args.limit)
    print(f"[ub] 문제 {len(df)} · 롤아웃 {args.n} · 조건 {args.conds} · 씨앗 {args.seed}", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_util, max_model_len=4864, enforce_eager=True)

    none_sys = build_none_sys()

    def chat(sys_txt, user_txt):
        msgs = [{"role": "system", "content": sys_txt}, {"role": "user", "content": user_txt}]
        try:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    users, golds = [], []
    for _, r in df.iterrows():
        p = r["prompt"]
        u = p[-1]["content"] if isinstance(p, (list, np.ndarray)) else str(p)
        users.append(u)
        golds.append((list(int(v) for v in r["nums"]), int(r["target"])))

    sp = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=3072, n=args.n, seed=args.seed)
    rows = []
    conds = [c.strip() for c in args.conds.split(",")]

    # ── a / b : 한 번에 생성 ─────────────────────────────────────────────
    for cond, sys_txt in (("a", PROMPT_VARIANTS["new"]), ("b", none_sys)):
        if cond not in conds:
            continue
        prompts = [chat(sys_txt, u) for u in users]
        outs = llm.generate(prompts, sp)
        for i, o in enumerate(outs):
            nums, tgt = golds[i]
            for k, g in enumerate(o.outputs):
                rows.append(dict(cond=cond, prob=i, k=k,
                                 correct=int(bool(grade(g.text, nums, tgt))),
                                 has_meta=int("<meta>" in g.text), n_tok=len(g.token_ids)))
        print(f"[ub] 조건 {cond} 완료", flush=True)

    # ── d / e : 1차 생성 → 절단 → 블록 주입 → 재생성 ──────────────────────
    if any(c in conds for c in ("d", "e")):
        prompts_a = [chat(PROMPT_VARIANTS["new"], u) for u in users]
        first = llm.generate(prompts_a, SamplingParams(temperature=1.0, top_p=1.0,
                                                       max_tokens=900, n=args.n, seed=args.seed))
        rng = np.random.RandomState(args.seed)
        for cond in ("d", "e", "f"):
            if cond not in conds:
                continue
            p2, meta2 = [], []
            for i, o in enumerate(first):
                for k, g in enumerate(o.outputs):
                    txt = g.text
                    if "\\boxed" in txt:            # 이미 답을 낸 궤적은 개입 대상 아님
                        cut = txt.index("\\boxed")
                    else:
                        cut = len(txt)
                    # ★조건부 개입: 1차 시도가 «실패한» 궤적에만 넣는다.
                    #   이미 맞힌 궤적에 방향전환을 강제하면 정답을 깨뜨리므로,
                    #   그 손실이 개입 효과를 가린다(첫 판본의 결함).
                    nums_i, tgt_i = golds[i]
                    first_ok = bool(grade(txt, nums_i, tgt_i)) if "\\boxed" in txt else False
                    if args.only_failed and first_ok:
                        rows.append(dict(cond=cond, prob=i, k=k, correct=1,
                                         has_meta=int("<meta>" in txt), n_tok=len(g.token_ids),
                                         intervened=0))
                        continue
                    if cond == "d":                  # ★답을 쓰기 «직전» = 막혔다고 판단한 지점
                        pos = cut
                    else:                            # e·f = 무작위 위치 (같은 난수열이라 짝지어짐)
                        pos = int(rng.randint(0, max(1, cut)))
                    # ★f = 같은 자리에서 자르고 «아무것도 안 넣고» 이어 생성.
                    #   e 와 f 의 차이가 곧 «혼잣말 블록 자체»의 기여다.
                    blk = "" if cond == "f" else REDIRECT_BLOCK
                    p2.append(prompts_a[i] + txt[:pos] + blk)
                    meta2.append((i, k))
            outs2 = llm.generate(p2, SamplingParams(temperature=1.0, top_p=1.0,
                                                    max_tokens=1400, n=1, seed=args.seed))
            for (i, k), o in zip(meta2, outs2):
                nums, tgt = golds[i]
                g = o.outputs[0]
                rows.append(dict(cond=cond, prob=i, k=k,
                                 correct=int(bool(grade(g.text, nums, tgt))),
                                 has_meta=1, n_tok=len(g.token_ids), intervened=1))
            print(f"[ub] 조건 {cond} 완료", flush=True)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"[ub] wrote {args.out} ({len(out)}행)", flush=True)
    print(out.groupby("cond").correct.mean().round(4).to_string(), flush=True)


if __name__ == "__main__":
    main()
