#!/usr/bin/env python
"""math_rollout — 수학 벤치마크 롤아웃 + 채점 + (기존) 메타 텔레메트리.

사용자 지시(2026-09-12): "예전 자를 기준으로 math 로 옮겨서 다시 테스트". 그래서
출력 스키마와 메타 블록 문법을 Countdown 쪽과 **같게** 맞춘다 — `countdown_rewards`
의 자·텔레메트리(emit_rate / meta_position / boilerplate / decision / confidence /
group_emit)와 `src/eval/pmi_shift_signal.py` 가 그대로 붙는다.

행 스키마(텍스트 한 줄 = 롤아웃 하나):
    group_id, problem_id, problem, gold, r_corr, text, final_answer, truncated, n_tok

채점은 `math_verify`(parse+verify). ⚠이 저장소에는 math_verify 의 timeout 래퍼가
워커 스레드에서 **정답을 조용히 오답으로 만드는** 알려진 함정이 있다
(`scripts/patch_math_verify.py`). 그래서 시작할 때 자가검사를 돌리고, 깨져 있으면
조용히 0 을 흘리지 않고 **즉사**한다.

사용법:
  math_rollout.py --dataset math500 --model_path $WORK/models/Qwen3.5-4B \
      --variant math_opt --num_samples 8 --out_dir $WORK/eval/math500_qwen35_opt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS, render_generation_prompt  # noqa: E402
from src.training import countdown_rewards as cdr  # noqa: E402

# (hf id, config, split, 문제 컬럼, 정답 컬럼)
DATASETS = {
    "math500":  ("HuggingFaceH4/MATH-500", None, "test", "problem", "answer"),
    "aime24":   ("HuggingFaceH4/aime_2024", None, "train", "problem", "answer"),
    "aime25":   ("math-ai/aime25", None, "test", "problem", "answer"),
    # ★LiveMathBench 는 gated(인증 필요) — 같은 «최신·비오염» 역할을 하는 공개 대안으로
    #   HMMT 2025-02 를 쓴다(MathArena, 30문제). livemath 라는 이름은 남기지 않는다:
    #   실제로 무엇을 돌렸는지가 파일명에 남아야 한다.
    "hmmt25":   ("MathArena/hmmt_feb_2025", None, "train", "problem", "answer"),
}


# ★0914: 채점·자가검사·\boxed 추출은 RL 트레이너(src/training/math_meta.py)와 **같은
#   함수**다 — 롤아웃 평가와 학습 보상이 다른 채점기를 쓰면 팔 간 비교가 무의미해진다.
from src.training.math_meta import (  # noqa: E402
    grade_math as _grade,
    last_boxed as _final_answer,
    selftest_math_verify as _selftest_math_verify,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    help=f"{sorted(DATASETS)} 또는 parquet:<path>[:<level>]")
    ap.add_argument("--model_path", required=True)
    # ★0913 수리: 선택지를 하드코딩했다가 math_new 추가 후 CLI 가 거부해 롤아웃 2건이
    #   죽었다. 단일 진실 원천(MATH_PROMPT_VARIANTS)에서 끌어와 다시 어긋나지 않게 한다.
    ap.add_argument("--variant", default="math_opt", choices=sorted(MATH_PROMPT_VARIANTS))
    ap.add_argument("--num_samples", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=3072)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--gpu_util", type=float, default=0.4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shuffle_seed", type=int, default=0, help="0 이 아니면 --limit 전에 섞는다")
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()

    if not a.dataset.startswith("parquet:") and a.dataset not in DATASETS:
        raise SystemExit(f"unknown dataset {a.dataset!r}")
    _selftest_math_verify()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from datasets import load_dataset
    from vllm import LLM, SamplingParams

    if a.dataset.startswith("parquet:"):
        # ★0914 수정 2: 학습 파케이(hendrycks MATH train)에서 난이도별 문제를 뽑는다.
        #   형식 parquet:<path>[:Level 5]. MATH-500 은 이 정책에 너무 쉬워(52% 문제 8/8) 자리 밀도가 없다.
        import pandas as pd
        parts = a.dataset.split(":", 2)
        df = pd.read_parquet(parts[1])
        lvl = parts[2] if len(parts) > 2 else None
        rows_in = [{"problem": r["problem"], "gold": str(r["gold"])}
                   for _, r in df.iterrows() if not lvl or (r["extra_info"] or {}).get("level") == lvl]
        print(f"[math] parquet {parts[1]} level={lvl!r}: {len(rows_in)} 문제", flush=True)
    else:
        ds_id, cfg, split, qcol, acol = DATASETS[a.dataset]
        ds = load_dataset(ds_id, cfg, split=split) if cfg else load_dataset(ds_id, split=split)
        rows_in = [{"problem": r[qcol], "gold": str(r[acol])} for r in ds]
    if a.shuffle_seed:
        import random as _rnd
        _rnd.Random(a.shuffle_seed).shuffle(rows_in)
    if a.limit:
        rows_in = rows_in[: a.limit]
    print(f"[math] {a.dataset}: {len(rows_in)} 문제 x {a.num_samples} 롤아웃 "
          f"· variant={a.variant}", flush=True)

    llm = LLM(model=a.model_path, dtype="bfloat16", seed=a.seed,
              gpu_memory_utilization=a.gpu_util,
              max_model_len=a.max_tokens + 2048, enforce_eager=True)
    tok = llm.get_tokenizer()

    # ★중복 제거(0914): math_sites/math_ruler_pivot/math_retry_eval 과 공유하는 단일 진실 원천.
    prompts = [render_generation_prompt(tok, a.variant, r["problem"]) for r in rows_in]
    outs = llm.generate(prompts, SamplingParams(
        n=a.num_samples, temperature=a.temperature, top_p=a.top_p,
        max_tokens=a.max_tokens, seed=a.seed))

    rows, n_trunc = [], 0
    for gi, (src, o) in enumerate(zip(rows_in, outs)):
        for x in o.outputs:
            trunc = int(x.finish_reason == "length")
            n_trunc += trunc
            rows.append({
                "group_id": f"g{gi}",
                "problem_id": gi,
                "problem": src["problem"],
                "gold": src["gold"],
                "text": x.text,
                "r_corr": _grade(x.text, src["gold"]),
                "final_answer": _final_answer(x.text),
                "truncated": trunc,
                "n_tok": len(x.token_ids),
            })

    with (out / "texts.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ── 기존(Countdown) 메타 텔레메트리를 그대로 재사용한다 ────────────────────
    n = len(rows)
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["group_id"], []).append(r)
    tel = {
        "dataset": a.dataset, "model_path": a.model_path, "variant": a.variant,
        "n_rows": n, "n_groups": len(groups),
        "acc": sum(r["r_corr"] for r in rows) / max(1, n),
        "pass_at_n": sum(int(any(x["r_corr"] for x in g)) for g in groups.values()) / max(1, len(groups)),
        "trunc_rate": n_trunc / max(1, n),
        "no_answer_rate": sum(1 for r in rows if not r["final_answer"]) / max(1, n),
        "len_mean": sum(r["n_tok"] for r in rows) / max(1, n),
        "emit_rate": cdr.emit_rate(rows, form="math"),
        "meta_position": cdr.meta_position_stats(rows, form="math"),
        "boilerplate": cdr.boilerplate_rate(rows, form="math"),
    }
    # 오답 분포(디코이 후보) — PMI-shift 는 «모델이 실제로 낸 오답»을 디코이로 써야 한다
    # (규칙기반 오답은 구세대 수학 판에서 AUC 0.539 로 포화됐다 — pmi_shift.py 주석).
    dec = {}
    for gid, g in groups.items():
        wrong = Counter(x["final_answer"] for x in g
                        if x["final_answer"] and not x["r_corr"])
        if wrong:
            dec[gid] = wrong.most_common(1)[0][0]
    tel["decoy_available_rate"] = len(dec) / max(1, len(groups))
    (out / "decoys.json").write_text(json.dumps(dec, ensure_ascii=False, indent=2))
    (out / "telemetry.json").write_text(json.dumps(tel, ensure_ascii=False, indent=2,
                                                   default=float))
    for k, v in tel.items():
        if k not in ("model_path", "meta_position", "boilerplate"):
            print(f"  {k:22s} {v}")
    print(f"  meta_position.p50      {tel['meta_position'].get('p50')}")
    print(f"[out] {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
