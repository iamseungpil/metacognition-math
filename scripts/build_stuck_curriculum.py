"""막힌 프리픽스 커리큘럼 만들기 — 상태 게이팅 GRPO 의 학습 분포.

사전등록: cd6_work/PREREG_stuck_gated.md
근거: 강제 개입의 부호를 가르는 것은 «혼잣말 품질»이 아니라 «지금 개입해도 되는 자리인가»다.
      막힘 층에서만 +1.72pp [+0.67,+2.83], 중간 −10.4pp, 건강 −22.9pp (worthv_s41).

  각 문제에서 궤적을 만들고 중간에서 자른다.
  그 프리픽스에서 «메타 없이» K번 재시작해 «전부 실패»하면 → 막힘 프리픽스.
  ★막힘 판정은 혼잣말을 한 글자도 읽지 않는다 (메타 텍스트 라벨은 자기충족이라 금지).
  ★split-half: 가지 0..K/2−1 로 판정하고, 남은 절반은 검증용으로 따로 기록한다.

산출: verl 학습셋 형식(prompt·nums·target·witness…)에 프리픽스를 user 메시지 뒤에
      «이어쓰기 지시»로 붙인 parquet. 보상·프롬프트 서식은 건드리지 않는다.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
REPO = os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math")
sys.path.insert(0, REPO)
from src.training.countdown_task import PROMPT_VARIANTS, grade  # noqa: E402

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

# 프리픽스를 «이미 해 본 시도»로 제시하는 이어쓰기 지시 — 새 서식을 만들지 않고
# 기존 user 메시지에 덧붙이기만 한다(보상·파서가 그대로 작동해야 한다).
CONT = ("\n\nYou already made these attempts and none of them reached the target:\n"
        "---\n{prefix}\n---\n"
        "Continue from here. Do not repeat those attempts.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=4000)
    ap.add_argument("--k", type=int, default=8, help="막힘 판정용 재시작 수 (절반씩 split)")
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--max_prefix_chars", type=int, default=1200)
    args = ap.parse_args()
    assert args.k % 2 == 0 and args.k >= 4, "split-half 를 위해 K 는 4 이상 짝수"

    df = pd.read_parquet(args.data).head(args.limit).reset_index(drop=True)
    print(f"[stuck] 문제 {len(df)} · K={args.k} · 씨앗 {args.seed}", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_util, max_model_len=4864, enforce_eager=True)

    def sp(count, base, **kw):     # 'n' 은 SamplingParams 인자라 이름이 겹친다
        return [SamplingParams(seed=base + i, **kw) for i in range(count)]

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

    # ── 1) 프리픽스 후보: 답을 쓰기 «전»까지 ────────────────────────────────
    first = llm.generate(heads, sp(len(heads), args.seed * 100000,
                                   temperature=1.0, top_p=1.0, max_tokens=1000, n=1))
    stems, keys = [], []
    for i, o in enumerate(first):
        t = o.outputs[0].text
        body = t[:t.index("\\boxed")] if "\\boxed" in t else t
        body = body.strip()
        if len(body) < 120:
            continue
        stems.append(body[:args.max_prefix_chars])
        keys.append(i)
    print(f"[stuck] 프리픽스 후보 {len(stems)}", flush=True)

    # ── 2) 막힘 판정: 그 프리픽스에서 «메타 없이» K번 재시작 ─────────────────
    #     ★혼잣말을 읽지 않는다. 프리픽스 텍스트만 이어 붙여 다시 생성한다.
    cont = llm.generate([heads[k] + s for k, s in zip(keys, stems)],
                        sp(len(stems), args.seed * 100000 + 7,
                           temperature=1.0, top_p=1.0, max_tokens=1200, n=args.k))
    half = args.k // 2
    rows = []
    for (i, stem, o) in zip(keys, stems, cont):
        nums, tgt = golds[i]
        ok = [int(bool(grade(g.text, nums, tgt))) for g in o.outputs]
        rows.append(dict(prob=i, stem=stem, judge=sum(ok[:half]), verify=sum(ok[half:]),
                         half=half))
    st = pd.DataFrame(rows)
    stuck = st[st.judge == 0]
    print(f"[stuck] 막힘 판정(가지 0..{half-1} 전부 실패): {len(stuck)}/{len(st)} "
          f"= {len(stuck)/max(1,len(st)):.1%}", flush=True)
    if len(stuck):
        print(f"[stuck] 라벨 재현성 — 판정 절반이 0 일 때 검증 절반도 0 인 비율: "
              f"{(stuck.verify == 0).mean():.3f}   (사전등록 하한 0.80)", flush=True)

    # ── 3) verl 학습셋으로 쓰기 (프롬프트 서식·보상은 그대로) ─────────────────
    out = []
    for _, r in stuck.iterrows():
        src = df.iloc[int(r["prob"])]
        p = src["prompt"]
        u = p[-1]["content"] if isinstance(p, (list, np.ndarray)) else str(p)
        newp = [{"role": "system", "content": PROMPT_VARIANTS["new"]},
                {"role": "user", "content": u + CONT.format(prefix=r["stem"])}]
        d = src.to_dict()
        d["prompt"] = newp
        if isinstance(d.get("extra_info"), dict):
            d["extra_info"] = {**d["extra_info"], "stuck_prefix": 1}
        out.append(d)
    o = pd.DataFrame(out)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    o.to_parquet(args.out)
    st.to_parquet(str(Path(args.out).with_name(Path(args.out).stem + "_labels.parquet")))
    print(f"[stuck] wrote {args.out} ({len(o)}행) · 라벨 {len(st)}행", flush=True)


if __name__ == "__main__":
    main()
