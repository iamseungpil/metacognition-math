#!/usr/bin/env python
"""build_math_parquet — MATH_META 학습/검증 parquet 빌더.

★데이터 선택 이유(0914). MATH-500(HuggingFaceH4/MATH-500)은 **시험 세트**다 — 거기서
학습하면 held-out 판정이 오염된다(Countdown E-133 과 같은 사고). 그래서 학습은
`EleutherAI/hendrycks_math` **train** split(7 config 전부, 7,500문제)에서 뽑고, MATH-500
문제는 **문제 텍스트 완전 일치**로 제외한다(MATH-500 은 hendrycks test 의 부분집합이라
원리상 train 과 겹치지 않지만, 겹침을 «가정»하지 않고 확인해 제거한다). 검증은 train 에서
떼어 낸 200문제(학습에서 제외).

정답(gold)은 `solution` 의 **마지막** \\boxed{...} — `math_meta.last_boxed`(트레이너·롤아웃과
같은 함수). 빈 gold 행은 버린다(math_verify 가 채점할 수 없으므로 조용한 0 이 된다).

행 스키마(verl `data.prompt_key=prompt`):
    prompt        chat messages (build_math_prompt(problem, variant))
    problem, gold flat 컬럼
    extra_info    {problem, gold}   ★async 경로는 flat 을 pop 하므로 여기서도 읽는다
    reward_model  {style: rule, ground_truth: gold}
    data_source   "math_meta"

출력:  /hdd_data/seungpil/scratch/data/math_train_<variant>.parquet
       /hdd_data/seungpil/scratch/data/math_val_<variant>.parquet
HF 캐시: /hdd_data/seungpil/scratch/hf_home (루트 디스크 금지 규약).
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/hdd_data/seungpil/scratch/hf_home")

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS, build_math_prompt  # noqa: E402
from src.training.math_meta import last_boxed, norm_problem  # noqa: E402

HENDRYCKS = "EleutherAI/hendrycks_math"
HENDRYCKS_CONFIGS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                     "number_theory", "prealgebra", "precalculus")
MATH500 = "HuggingFaceH4/MATH-500"
OUT_DIR = Path("/hdd_data/seungpil/scratch/data")


def make_record(problem: str, gold: str, variant: str, *, level: str = "", subject: str = "") -> dict:
    return {
        "data_source": "math_meta",
        "prompt": build_math_prompt(problem, variant),
        "problem": problem,
        "gold": gold,
        "reward_model": {"style": "rule", "ground_truth": gold},
        "extra_info": {"problem": problem, "gold": gold, "level": level, "subject": subject,
                       "prompt_variant": variant},
    }


def split_records(train_rows: list[dict], heldout_problems: set[str], *, val_n: int, seed: int,
                  variant: str) -> tuple[list[dict], list[dict], dict]:
    """순수 함수(테스트 가능). train_rows: [{problem, solution, level?, type?}]."""
    seen: set[str] = set()
    kept, n_dup, n_heldout, n_nogold = [], 0, 0, 0
    for r in train_rows:
        p = str(r["problem"])
        k = norm_problem(p)
        if k in heldout_problems:
            n_heldout += 1
            continue
        if k in seen:
            n_dup += 1
            continue
        g = last_boxed(str(r.get("solution", "")))
        if not g:
            n_nogold += 1
            continue
        seen.add(k)
        kept.append(make_record(p, g, variant, level=str(r.get("level", "")),
                                subject=str(r.get("type", ""))))
    rng = random.Random(seed)
    rng.shuffle(kept)
    val, train = kept[:val_n], kept[val_n:]
    stats = {"n_in": len(train_rows), "n_heldout_removed": n_heldout, "n_dup": n_dup,
             "n_nogold": n_nogold, "n_train": len(train), "n_val": len(val)}
    return train, val, stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="math_opt", choices=sorted(MATH_PROMPT_VARIANTS))
    ap.add_argument("--val_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out_dir", default=str(OUT_DIR))
    a = ap.parse_args()

    import pandas as pd
    from datasets import load_dataset

    m500 = load_dataset(MATH500, split="test")
    heldout = {norm_problem(r["problem"]) for r in m500}
    rows = []
    for cfg in HENDRYCKS_CONFIGS:
        ds = load_dataset(HENDRYCKS, cfg, split="train")
        rows.extend({"problem": r["problem"], "solution": r["solution"],
                     "level": r.get("level", ""), "type": r.get("type", cfg)} for r in ds)
    train, val, stats = split_records(rows, heldout, val_n=a.val_n, seed=a.seed, variant=a.variant)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tp, vp = out / f"math_train_{a.variant}.parquet", out / f"math_val_{a.variant}.parquet"
    pd.DataFrame(train).to_parquet(tp, index=False)
    pd.DataFrame(val).to_parquet(vp, index=False)
    print(f"[build_math_parquet] variant={a.variant} {stats}")
    print(f"[build_math_parquet] {tp} ({len(train)} rows)\n[build_math_parquet] {vp} ({len(val)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
