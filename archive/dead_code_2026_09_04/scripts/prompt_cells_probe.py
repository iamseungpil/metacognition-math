"""프롬프트 5칸 비교 — 학습 없이, 추론만으로 「선언 → 행동」 결합을 겨냥한다.

칸(steer_prompts.VARIANTS): P0 현행 · P0e 현행+예시 · P1 금지완화 · P2 구조 · P3 구조+예시
  분해:  금지 P1−P0 · 구조 P2−P1 · 예시 P0e−P0 및 P3−P2 · 헤드라인 P3−P0e

재는 것(전부 기계 채점, 정답 라벨을 쓰지 않는 항목은 순환 검증이 없다):
  정답률          프롬프트만으로 오르는가
  발화율          메타를 쓰기는 하는가
  필드율          ruled_out/next 를 형식대로 채우는가
  실재율          ruled_out 이 앞 텍스트에 실제로 있는가         ← 순응적 공허 차단
  신규율          next 가 «앞 5줄에 없던» 진부분집합 결합인가    ← 순응적 공허 + 쪼갠 유출 차단
  준수율          next 가 지목한 신규 결합을 바로 뒤에서 실제로 하는가  ★핵심
  거짓·유출률     ruled_out 거짓 주장 / 완성식 유출

사전 등록 관문(페이블):
  필드율 ≥ 0.60 (P2 가 0.30 미만이고 P3 가 0.60 이상이면 구조에 예시가 필요한 것)
  준수율 ≥ 0.30 (위약 바닥 0.106 대비)
  정답률 (P2−P1) 또는 (P3−P0e) ≥ +2pp 이고 합산 95% CI 가 0 을 배제
  중단   두 정답률 대비의 CI 상한이 모두 +1pp 미만 · 거짓률 > 5% · 유출 실격률 > 3%
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
REPO = os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math")
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
from src.training.countdown_task import grade  # noqa: E402
import steer_prompts as SP  # noqa: E402

MODEL = os.environ.get(
    "PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

_META = re.compile(r"<meta>(.*?)</meta>", re.S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--cells", default="P0,P0e,P1,P2,P3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    args = ap.parse_args()

    df = pd.read_parquet(args.data).head(args.limit)
    cells = [c.strip() for c in args.cells.split(",")]
    print(f"[cells] 문제 {len(df)} × 롤아웃 {args.n} × 칸 {cells} · 씨앗 {args.seed}", flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(model=MODEL, dtype="bfloat16", seed=args.seed, tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_util, max_model_len=4864, enforce_eager=True)

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
        golds.append((list(int(v) for v in r["nums"]), int(r["target"])))

    rows = []
    for ci, cell in enumerate(cells):
        prompts = [chat(SP.VARIANTS[cell], u) for u in users]
        # 프롬프트마다 다른 씨앗 (vLLM 은 요청별 salt 를 넣지 않는다)
        sps = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=3072, n=args.n,
                              seed=args.seed * 100000 + ci * 10000 + i)
               for i in range(len(prompts))]
        outs = llm.generate(prompts, sps)
        for i, o in enumerate(outs):
            nums, tgt = golds[i]
            for k, g in enumerate(o.outputs):
                t = g.text
                m = _META.search(t)
                rec = dict(cell=cell, prob=i, k=k,
                           correct=int(bool(grade(t, nums, tgt))),
                           has_meta=int(m is not None),
                           has_boxed=int("\\boxed" in t),
                           meta_pos=(m.start() if m else -1),
                           n_tok=len(g.token_ids),
                           capped=int(g.finish_reason == "length"),
                           text=t[:4000])          # ★원문 저장(감사 조건)
                if m:
                    body = m.group(1)
                    scorer = SP.score_meta_family if cell.startswith("P4") else SP.score_meta
                    rec.update(scorer(body, t[:m.start()], t[m.end():], nums, tgt))
                    rec["meta_text"] = body[:500]
                rows.append(rec)
        print(f"[cells] 칸 {cell} 완료", flush=True)

    out = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out)
    print(f"[cells] wrote {args.out} ({len(out)}행)\n", flush=True)

    def rate(g, col):
        v = g[col].dropna() if col in g else pd.Series(dtype=float)
        return float(pd.to_numeric(v, errors="coerce").mean()) if len(v) else float("nan")
    hdr = f"{'칸':<6}{'정답률':>9}{'발화율':>9}{'필드율':>9}{'실재율':>9}{'신규율':>9}{'준수율':>9}{'거짓':>7}{'유출':>7}"
    print(hdr); print("─" * len(hdr.encode()) // 2 * "─")
    for cell in cells:
        g = out[out.cell == cell]
        print(f"{cell:<6}{g.correct.mean():>9.4f}{g.has_meta.mean():>9.4f}"
              f"{rate(g,'has_fields'):>9.4f}{rate(g,'grounded'):>9.4f}{rate(g,'next_ok'):>9.4f}"
              f"{rate(g,'followed'):>9.4f}{rate(g,'false_claim'):>7.3f}{rate(g,'leak'):>7.3f}")


if __name__ == "__main__":
    main()
