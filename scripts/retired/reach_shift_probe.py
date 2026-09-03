"""reach-shift — 탐색 축의 PMI-shift. 「메타가 다음 «수»를 더 좋게 만드나」.

배경(2026-09-01):
  · PMI 계열 자 41개가 전멸했다. 원인은 공식이 아니라 «재는 대상»이었다 —
    메타는 정답 믿음을 잘 올린다(평균 +0.30 nats, SD 0.43). 그런데 그게 성공과 무관하다
    (ρ=0.081, p=0.35). 이 과제의 병목은 «정답을 알아보기»가 아니라 «찾아내기»다.
  · 같은 공식이 CPMI(arXiv:2604.10660)로 발표돼 있고, «정답이 유일한» 수학에서는 작동한다.
    그 논문의 유도가 델타 분포(정답 유일)를 요구하는데 Countdown 은 해가 문제당 중앙 6개다.
    그리고 정답만 보는 판은 그 논문에서도 AUC 0.419(우연 이하) — 우리 osd_gold 가 그것이다.

★구조는 PMI-shift 그대로 두고 «표적»만 바꾼다:

    reach_shift = P(다음 수가 해를 살림 | 메타 있음) − P(다음 수가 해를 살림 | 메타 지움)
                  └─ 같은 자리에서 K번 재생성 ─┘   └─ 메타만 빼고 K번 재생성 ─┘

  · 반사실이 «들어 있다» → 「메타가 바꾸는가」에 답한다
    (직전 판본은 메타 있는 궤적만 재서 귀속이 불가능했다 — 그 실수를 여기서 고친다)
  · 표적이 «다음 한 수»다 → 「탐색을 강화하는가」에 답한다
  · «해를 살림»은 완전 열거로 참/거짓이 나온다. 확률 추정도 대리변수도 아니다
  · 미끼가 저절로 생긴다: 25−8(살림) vs 25×3(죽임)은 같은 두 수, 연산만 다른 완벽한 짝

★동어반복 방지(실측으로 확인된 함정): 메타 뒤 시도가 몇 개 없으면 «첫 수»가 곧 «답»이라
  이 지표가 정답 여부를 다르게 물은 것이 된다. 실측 — 답 바로 앞 ρ=+0.853, 아주 멂 ρ=−0.030.
  → 이어쓰기에서 \\boxed 전에 최소 MIN_ATT 개의 조합이 나온 것만 센다.

★함께 기록하는 통제 변수: 이어쓰기 시도 수 · 메타 위치 · 문제 난이도 · 사이트 id
  (p_abl 은 디스크의 stageB 에서 사후 결합)
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
REPO = os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math")
sys.path.insert(0, REPO)
from src.training.countdown_task import grade  # noqa: E402

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

PAIR = re.compile(r"(\d{1,4})\s*([+\-*/×÷])\s*(\d{1,4})")


def combos(text: str, nums) -> list[tuple[int, int, str]]:
    """텍스트에서 «주어진 수 두 개의 결합»을 순서대로. 중복 수도 허용(다중집합 소모)."""
    want = Counter(int(v) for v in nums)
    out = []
    for a, o, b in PAIR.findall(str(text)):
        a, b = int(a), int(b)
        if all(want.get(k, 0) >= v for k, v in Counter((a, b)).items()):
            out.append((min(a, b), max(a, b), {"×": "*", "÷": "/"}.get(o, o)))
    return out


def solvable(vals, target) -> bool:
    """남은 값들로 목표에 도달 가능한가 — 완전 열거. Countdown 규칙(양의 정수 중간값)."""
    def go(v):
        if len(v) == 1:
            return v[0] == Fraction(target)
        for i, j in itertools.combinations(range(len(v)), 2):
            a, b = v[i], v[j]
            rest = [v[k] for k in range(len(v)) if k not in (i, j)]
            cands = [a + b, a * b, a - b, b - a]
            if b: cands.append(a / b)
            if a: cands.append(b / a)
            for x in cands:
                if x > 0 and x.denominator == 1 and go(rest + [x]):
                    return True
        return False
    return go([Fraction(int(x)) for x in vals])


def first_move_keeps_solution(text: str, nums, target) -> tuple[int | None, int]:
    """(첫 수가 해를 살렸나 0/1, \\boxed 전 조합 개수). 첫 수가 없으면 (None, 0)."""
    body = text[:text.index("\\boxed")] if "\\boxed" in text else text
    cs = combos(body, nums)
    if not cs:
        return None, 0
    lo, hi, op = cs[0]
    rest = [int(v) for v in nums]
    try:
        rest.remove(lo); rest.remove(hi)
    except ValueError:
        return None, len(cs)
    val = {"+": Fraction(lo + hi), "*": Fraction(lo * hi), "-": Fraction(hi - lo),
           "/": (Fraction(hi, lo) if lo else None)}.get(op)
    if val is None or val <= 0 or val.denominator != 1:
        return 0, len(cs)            # 규칙 위반 수 = 해를 죽인 것으로 센다
    return int(solvable(rest + [int(val)], int(target))), len(cs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_glob", default="cd6_work/probe/sites_shard*.parquet")
    ap.add_argument("--sites_file", default=None)
    ap.add_argument("--k", type=int, default=16, help="조건당 이어쓰기 수")
    ap.add_argument("--min_att", type=int, default=3,
                    help="★동어반복 방지 — \\boxed 전에 이만큼 조합이 나온 것만 센다")
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=91)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    args = ap.parse_args()

    sites = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(args.sites_glob))],
                      ignore_index=True).drop_duplicates("site_id").reset_index(drop=True)
    if args.sites_file:
        keep = {l.strip() for l in open(args.sites_file) if l.strip()}
        sites = sites[sites.site_id.isin(keep)].reset_index(drop=True)
    print(f"[reach] 사이트 {len(sites)} · K={args.k} · 최소시도 {args.min_att} · 씨앗 {args.seed}",
          flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_util, max_model_len=4864, enforce_eager=True)

    prompts, keys = [], []
    for _, r in sites.iterrows():
        msgs = json.loads(r["prompt_json"]) if isinstance(r["prompt_json"], str) else r["prompt_json"]
        try:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = str(r["response_text"])
        # with = 프리픽스 + 메타 / without = 프리픽스만 (메타를 지운 반사실)
        prompts.append(head + text[:int(r["meta_end"])]);   keys.append((r["site_id"], "with"))
        prompts.append(head + text[:int(r["meta_start"])]); keys.append((r["site_id"], "without"))
    print(f"[reach] 요청 {len(prompts)} × K{args.k}", flush=True)

    # 프롬프트마다 다른 씨앗 — vLLM 은 요청별 salt 를 넣지 않는다
    sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens,
                          n=args.k, seed=args.seed * 100000 + i) for i in range(len(prompts))]
    outs = llm.generate(prompts, sps)

    meta = sites.set_index("site_id")
    rows = []
    for (sid, cond), o in zip(keys, outs):
        r = meta.loc[sid]
        nums = [int(v) for v in r["nums"]]; tgt = int(r["target"])
        for j, g in enumerate(o.outputs):
            keep, natt = first_move_keeps_solution(g.text, nums, tgt)
            rows.append(dict(site_id=sid, cond=cond, k=j,
                             keeps=(-1 if keep is None else keep), n_att=natt,
                             correct=int(bool(grade(g.text, nums, tgt))),
                             n_tok=len(g.token_ids),
                             pos=float(r["pos"]), ncor=int(r["n_correct_of8"]),
                             decision=r["decision"], conf=float(r["confidence"])))
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"[reach] wrote {args.out} ({len(out)}행)\n", flush=True)

    # ── 요약 (판정은 감사 후. 여기 숫자는 원자료다) ────────────────────────
    ok = out[(out.keeps >= 0) & (out.n_att >= args.min_att)]
    print(f"동어반복 필터: {len(out)} → {len(ok)} 행 (\\boxed 전 조합 ≥{args.min_att})")
    g = ok.groupby(["site_id", "cond"]).keeps.mean().unstack()
    if {"with", "without"}.issubset(g.columns):
        d = (g["with"] - g["without"]).dropna()
        bs = [np.random.RandomState(s).choice(d.values, len(d)).mean() for s in range(3000)]
        print(f"\n★reach-shift = {d.mean():+.4f}  "
              f"CI[{np.percentile(bs,2.5):+.4f},{np.percentile(bs,97.5):+.4f}]  n={len(d)} 사이트")
        print(f"   메타 있음 {g['with'].mean():.4f}   메타 지움 {g['without'].mean():.4f}")
    print("\n조건별 이어쓰기 성공률(참고):")
    print(ok.groupby("cond").agg(n=("correct", "size"), 정답=("correct", "mean"),
                                 해살림=("keeps", "mean"), 시도=("n_att", "mean")).round(4).to_string())
    print("\n※ G3(고를 여지)·G4(동어반복)·G6(예산 대조)는 별도 분석에서 판정한다.", flush=True)


if __name__ == "__main__":
    main()
