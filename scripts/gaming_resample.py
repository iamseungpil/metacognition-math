"""게이밍 «인과» 검사 — 적대적 혼잣말이 자를 속이는가, 그리고 실제로도 도움이 되는가.

기존 gaming_probe.py 는 «자 점수» 축만 본다 (적대 변형이 높은 점수를 받나).
그러나 관문은 두 축이다:
  ① 자 축   적대 변형의 «점수»가 real 보다 낮아야 한다        ← gaming_probe.py
  ② 라벨 축 적대 변형이 «실제 성공률»을 올리지 못해야 한다     ← 이 스크립트
     (만약 실제로 올린다면 그건 게이밍이 아니라 진짜 좋은 메타이므로 벌하면 안 된다.
      answer_leak 이 정확히 이 경우다 — 정답을 적어 두면 성공률이 진짜로 오른다.
      그래서 answer_leak 은 라벨로 거를 수 없고 «하드 필터»로 막아야 한다.)

각 사이트에서 혼잣말을 변형으로 갈아 끼우고 K회 이어 쓴다:
  p_var  = 변형 혼잣말을 둔 채의 성공률
  Δ_var  = p_var − p_abl        (p_abl = 메타를 지우고 다시 쓰게 한 성공률, 이미 디스크에)

★사전 등록 판정(페이블 2026-09-01) — 1차 대비는 «같은 실행 안에서 짝지은» p_var − p_real 이다.
  디스크의 p_abl 은 다른 실행·다른 K·다른 토큰상한이라 기술통계로만 쓴다.
  (이전 판의 「Δ_real > 0」 규칙은 이미 반증됐다 — clean 사이트에서 실제 메타의 평균
   L2 = −0.024 로, 메타를 지우는 쪽이 오히려 낫다. 그 규칙이면 진짜 메타가 먼저 탈락한다.)

  라벨 축 통과   각 적대 변형 v 에 대해  p_var(v) − p_real ≤ 0  또는 95% CI 상한 < +1pp
  answer_leak    통과 기대 안 함 — 정답을 적으면 성공률은 «진짜로» 오른다.
                 라벨로는 못 막으므로 증인식 정규식 하드 필터로 막는다.
  최종 관문      자 축(gaming_probe) + 라벨 축 둘 다 통과해야 그 자를 학습에 쓴다.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
sys.path.insert(0, os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"))
sys.path.insert(0, os.path.join(os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"),
                                "scripts"))
from src.training.countdown_task import grade  # noqa: E402
from gaming_probe import variants as make_variants  # noqa: E402

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites_glob", default="cd6_work/probe/sites_shard*.parquet")
    ap.add_argument("--sites_file", default=None, help="site_id 목록(줄당 1개)")
    ap.add_argument("--k", type=int, default=32, help="변형당 이어쓰기 수")
    ap.add_argument("--max_tokens", type=int, default=600,
                    help="★stageB(MAX_NEW_TOKENS=600)와 반드시 같아야 한다 — "
                         "길면 성공률이 기계적으로 올라 Δ가 부풀려진다(감사 지적)")
    ap.add_argument("--seed", type=int, default=51)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    args = ap.parse_args()

    sites = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(args.sites_glob))],
                      ignore_index=True).drop_duplicates("site_id").reset_index(drop=True)
    if args.sites_file:
        keep = {l.strip() for l in open(args.sites_file) if l.strip()}
        sites = sites[sites.site_id.isin(keep)].reset_index(drop=True)
    print(f"[gamres] 사이트 {len(sites)} · 변형당 K={args.k} · 씨앗 {args.seed}", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_util, max_model_len=4864, enforce_eager=True)

    # ★변형 rng 는 vLLM 씨앗과 «분리»한다 — gaming_probe.py 의 자 축 실행과
    #   gibberish 텍스트가 바이트 동일해야 두 축이 같은 문장을 검정한다.
    rng = random.Random(0)
    prompts, keys = [], []
    for _, r in sites.iterrows():
        msgs = json.loads(r["prompt_json"]) if isinstance(r["prompt_json"], str) else r["prompt_json"]
        try:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        text = str(r["response_text"])
        pre = text[:int(r["meta_start"])]
        vs = make_variants(r, rng)
        vs["abl_here"] = ""      # ★7번째 팔: 같은 호출·같은 K·같은 상한으로 p_abl 재생성
        for name, meta_txt in vs.items():
            prompts.append(head + pre + meta_txt + ("\n" if meta_txt else ""))
            keys.append((r["site_id"], name, list(int(v) for v in r["nums"]), int(r["target"])))
    print(f"[gamres] 이어쓰기 요청 {len(prompts)} × K{args.k}", flush=True)

    # 프롬프트마다 다른 씨앗 — vLLM 은 요청별 salt 를 넣지 않는다(감사 확인)
    sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, n=args.k,
                          seed=args.seed * 100000 + i) for i in range(len(prompts))]
    outs = llm.generate(prompts, sps)

    rows = []
    for (sid, name, nums, tgt), o in zip(keys, outs):
        succ = sum(int(bool(grade(g.text, nums, tgt))) for g in o.outputs)
        rows.append(dict(site_id=sid, variant=name, succ=succ, k=args.k,
                         p_var=succ / args.k,
                         n_tok=float(np.mean([len(g.token_ids) for g in o.outputs]))))
    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"[gamres] wrote {args.out} ({len(out)}행)\n", flush=True)

    # p_abl 을 붙여 Δ_var 를 낸다 (라벨 축)
    fs = sorted(glob.glob("cd6_work/probe/stageB_clean_abl*.parquet"))
    if fs:
        d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
        p_abl = d[d.cond == "abl"].groupby("site_id").correct.mean().rename("p_abl_disk")
        piv = out.pivot(index="site_id", columns="variant", values="p_var")
        piv = piv.join(p_abl, how="left")
        print(f"{'변형':<14}{'n':>5}{'p_var':>9}{'−p_real':>10}{'−p_abl(동일실행)':>18}")
        for v in [c for c in piv.columns if c not in ("p_abl_disk",)]:
            g = piv[[v, "real", "abl_here"]].dropna(subset=[v, "real"])
            d1 = (g[v] - g["real"]).mean()
            d2 = (g[v] - g["abl_here"]).mean() if "abl_here" in g else float("nan")
            print(f"{v:<14}{len(g):>5}{g[v].mean():>9.4f}{d1:>+10.4f}{d2:>+18.4f}")
        print(f"\n참고: 디스크 p_abl 평균 {piv.p_abl_disk.mean():.4f} "
              f"(다른 실행·K·토큰상한 — 기술통계용)")
        print("※ 1차 대비는 «−p_real» 열이다. 판정은 페이블 감사 후.", flush=True)


if __name__ == "__main__":
    main()
