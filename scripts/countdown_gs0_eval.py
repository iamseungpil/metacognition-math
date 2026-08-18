"""원점 사진(C-035) — **학습 전** 모델의 Countdown 텔레메트리 전부.

왜 이게 필요한가. 이 저장소가 반복해서 산 실수는 *"기준선 없는 수를 헤드라인에 쓴 것"*이다
(원장 0731: 결론이 세 번 바뀌었고 셋 다 '있던 두 수를 나란히 놓기'가 전부였다). 학습이
끝난 뒤 발화율 0.6 을 보고 "메타를 배웠다"고 읽으려면, **학습 전에 이미 0.6 이었는지**를
알아야 한다. 이 스크립트가 그 네 번째 칸을 만든다.

학습 팔과 **같은 함수로 잰다** — `countdown_rewards.telemetry_report`. 복제하면 두 곳이
갈리고, 그러면 "학습 전후 차이"가 측정 방식 차이와 구별되지 않는다.

형식 두 벌(`new` 블록형 / `old` 한 줄형)을 각각 재는 이유는 이번 판이 A~G 는 새 형식으로
돌지만 형식 자체가 후속 실험의 처치라서다. 두 기준선을 지금 같이 찍어 두면 그때 다시
GPU 를 잡지 않아도 된다.

출력: `<out_dir>/telemetry.json`(대장이 읽는 수) · `samples.jsonl`(눈으로 볼 표본).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/scratch/metacognition")

from src.training import countdown_rewards as cdr          # noqa: E402
from src.training.countdown_task import (                  # noqa: E402
    build_prompt, extract_expr, grade,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", required=True, help="countdown_val.parquet")
    ap.add_argument("--meta_format", default="new", choices=["new", "old"])
    ap.add_argument("--num_samples", type=int, default=16,
                    help="문제당 롤아웃 수 = p̂ 의 그룹 크기")
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tp_size", type=int, default=1)
    ap.add_argument("--gpu_util", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="0 이면 전부")
    ap.add_argument("--out_dir", required=True)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.data)
    if args.limit:
        df = df.head(args.limit)
    print(f"[gs0] {len(df)} 문제 x {args.num_samples} 롤아웃 · 형식={args.meta_format}",
          flush=True)

    llm = LLM(model=args.model_path, dtype="bfloat16", seed=args.seed,
              tensor_parallel_size=args.tp_size,
              gpu_memory_utilization=args.gpu_util,
              max_model_len=args.max_tokens + 1024)
    tok = llm.get_tokenizer()

    def chat(msgs) -> str:
        try:
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:                       # enable_thinking 없는 토크나이저
            return tok.apply_chat_template(msgs, tokenize=False,
                                           add_generation_prompt=True)

    prompts, insts = [], []
    for _, row in df.iterrows():
        inst = {"nums": [int(v) for v in row["nums"]], "target": int(row["target"]),
                "witness": row.get("witness", ""), "decoy": row.get("decoy", "")}
        prompts.append(chat(build_prompt(inst, variant=args.meta_format)))
        insts.append(inst)

    outs = llm.generate(prompts, SamplingParams(
        n=args.num_samples, temperature=args.temperature, top_p=args.top_p,
        top_k=args.top_k, max_tokens=args.max_tokens, seed=args.seed))

    # ── 행을 만든다. 키 이름은 학습 팔의 행 규약과 **같아야 한다** ────────────────
    #    (countdown_rewards 의 텔레메트리가 그 이름으로 읽는다).
    def parse_ok(t: str) -> int:
        return int(extract_expr(t) is not None)

    groups, samples = [], []
    n_trunc = 0
    for gi, (inst, o) in enumerate(zip(insts, outs)):
        grp = []
        for x in o.outputs:
            text = x.text
            truncated = x.finish_reason == "length"
            n_trunc += int(truncated)
            r = {
                "text": text,
                "r_corr": int(grade(text, inst["nums"], inst["target"])),
                "format_ok": cdr.format_ok_row(text, "A" if args.meta_format == "new"
                                               else "H", parse_expr_ok=parse_ok),
                # ⚠`or ""` 필수 — answer_leak 은 None 에 예외를 던진다(의도적).
                #   \boxed 없는 행이 하나만 있어도 텔레메트리 전체가 터진다.
                "final_expr": extract_expr(text) or "",
                "meta_n_tok": 0,
                "group_id": f"g{gi}",
                "n_tok": len(x.token_ids),
                "truncated": int(truncated),
            }
            m = cdr.parse_meta(text, form=args.meta_format)
            r["meta"] = m
            r["emitted"] = int(m.get("emitted", 0))
            if r["emitted"]:
                inner = m.get("body", "") or ""
                r["meta_n_tok"] = len(tok.encode(inner, add_special_tokens=False))
            grp.append(r)
        # p̂ 는 gold 를 안 쓴다 — 목표수가 프롬프트에 있어 식을 평가하면 자가검증된다.
        phat = cdr.compute_phat(grp)
        mean_corr = sum(x["r_corr"] for x in grp) / len(grp)
        for x in grp:
            x["phat"] = phat
            x["adv_corr"] = float(x["r_corr"]) - mean_corr
        groups.append(grp)
        if len(samples) < 40 and grp[0]["emitted"]:
            samples.append({"target": inst["target"], "nums": inst["nums"],
                            "phat": phat, "r_corr": grp[0]["r_corr"],
                            "meta": (grp[0]["meta"] or {}).get("body", "")[:400],
                            "tail": grp[0]["text"][-300:]})

    rep = cdr.telemetry_report(groups, form=args.meta_format)
    rep["meta_format"] = args.meta_format
    rep["n_problems"] = len(groups)
    rep["n_rollouts_per_problem"] = args.num_samples
    rep["trunc_rate_measured"] = n_trunc / max(1, sum(len(g) for g in groups))
    rep["spec_version"] = cdr.SPEC_VERSION
    rep["model_path"] = args.model_path

    (out / "telemetry.json").write_text(json.dumps(rep, indent=1, default=str))
    with (out / "samples.jsonl").open("w") as fh:
        for s in samples:
            fh.write(json.dumps(s, ensure_ascii=False, default=str) + "\n")

    # ── 사람이 읽는 요약. 학습 후 표와 **같은 칸**이라 나란히 놓을 수 있다. ───────
    def g(d, k):
        v = rep.get(d, {})
        return v.get(k) if isinstance(v, dict) else None

    print(f"\n===== 원점 사진 · 형식={args.meta_format} =====", flush=True)
    for label, val in [
        ("발화율", rep.get("emit_rate")),
        ("정형문 비율", g("boilerplate", "boilerplate_rate")),
        ("답 누출률", rep.get("answer_leak_rate")),
        ("메타먼저 비율", g("meta_position", "frac_meta_first")),
        ("선택성 지수", g("selectivity", "selectivity")),
        ("p̂ 평균", g("phat", "mean")),
        ("p̂=0 비율", g("phat", "frac_zero")),
        ("p̂=1 비율", g("phat", "frac_one")),
        ("정답률", rep.get("acc")),
        ("길이 p95", g("length", "len_p95")),
        ("절단율", rep["trunc_rate_measured"]),
        ("redirect 비율", g("decision", "redirect")),
        ("confidence 종수", g("confidence", "n_unique")),
        ("메타 내 산수", rep.get("arith_in_meta_rate")),
    ]:
        print(f"  {label:16s} {val}")
    hits = cdr.check_abort(rep)
    print(f"  중단 조건: {hits or '이상 없음'}")
    print(f"[gs0] wrote {out}/telemetry.json", flush=True)


if __name__ == "__main__":
    main()
