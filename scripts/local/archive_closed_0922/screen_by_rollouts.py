#!/usr/bin/env python
"""screen_by_rollouts — 스크리닝 롤아웃으로 학습 parquet 을 "자리 밀도 있는" 문제로 좁힌다.

★목적(0914). AIME-급 소스(deepscaler/dapo17k)로 옮겨도 8샘플 스크리닝 롤아웃에서
전부 맞거나 전부 틀리는 문제는 GRPO 신호가 없다(그룹 std=0). `--max_mean_tok`/
`--max_trunc_rate` 는 6–8k 예산 안에 들어오는 문제만 남긴다 — 평균 응답이 예산을
넘거나 잘림(truncation)이 잦은 문제는 두 번째 시도(재시도 팔)를 쓸 여지가 없다.

입력: math_rollout.py 가 쓴 texts.jsonl(group_id/problem/r_corr/n_tok/truncated) +
     build_math_parquet.py 가 만든 train parquet(problem/gold/prompt/extra_info).
출력: <parquet 이름>_mixed.parquet — problem 이 스크리닝 group_pass_rate
     0<rate<1 **AND** mean_tok <= --max_mean_tok **AND** trunc_rate <= --max_trunc_rate
     인 행. `--include_allwrong_states` 를 주면 pass_rate==0 인 문제도 그 문제의(gold 없는)
     agreement_state 가 지정된 집합에 있을 때 같은 길이/절단 필터를 적용해 추가로 남는다
     (기본값 빈 문자열 = 옛 동작, mixed 만). extra_info.group_pass_rate 가 붙고(mixed 는
     0<rate<1, all-wrong 은 0.0), extra_info.agree_state 도 kept 행 전부에 붙는다.

forced_redirect: --variant math_retry|math_agree 일 때만 --forced_frac 비율을 그 팔의
forced 변형(math_retry_forced / math_agree_forced, --forced_variant 로 덮어쓸 수 있다)으로
다시 쓴다(마스크 안 문제는 이미 전원 mixed — build_math_parquet.py 의 mixed-only 로직을
다시 적용할 필요가 없다, 셋 전체가 후보 풀이다). 다른 --variant 에서 forced_frac>0 은
no-op 경고만 찍는다(재시도 판단 항이 없는 변형에 forced_redirect 를 붙이는 것은 무의미하다).
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from math_activation_gate import agreement_state  # noqa: E402
from src.metacot.math_meta_prompt import MATH_PROMPT_VARIANTS, build_math_prompt  # noqa: E402
from src.training.math_meta import norm_problem  # noqa: E402

# ★0914b: build_math_parquet.py 의 _FORCED_VARIANT_OF 와 같은 매핑(의도적 중복 — 별개 스크립트,
#   결합시키지 않는다). --variant 가 math_retry/math_agree 일 때 --forced_variant 기본값을 정한다.
_FORCED_VARIANT_OF = {"math_retry": "math_retry_forced", "math_agree": "math_agree_forced"}


def load_group_stats(rollouts_path: str) -> dict[str, dict]:
    """texts.jsonl → {norm_problem: {pass_rate, mean_tok, trunc_rate, n}}.

    ★같은 group_id 가 problem 을 넘어 섞이면 즉사(load_group_pass_rates 와 같은 관례,
    조용한 평균 오염 방지)."""
    by_problem: dict[str, list[dict]] = {}
    seen_group: dict[str, set] = {}
    with open(rollouts_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            k = norm_problem(r["problem"])
            by_problem.setdefault(k, []).append(r)
            seen_group.setdefault(k, set()).add(str(r.get("group_id", "")))
    bad = {k: g for k, g in seen_group.items() if len(g) > 1}
    if bad:
        raise RuntimeError(f"[screen_by_rollouts] 같은 문제에 group_id 가 둘 이상: {list(bad)[:5]}")
    stats = {}
    for k, rows in by_problem.items():
        n = len(rows)
        answers = [str(r.get("final_answer", "") or "") for r in rows]
        stats[k] = {
            "pass_rate": sum(int(r["r_corr"]) for r in rows) / n,
            "mean_tok": sum(int(r.get("n_tok", 0)) for r in rows) / n,
            "trunc_rate": sum(int(r.get("truncated", 0)) for r in rows) / n,
            "n": n,
            "agree_state": agreement_state(answers, k=n)["state"],
        }
    return stats


def screen_rows(rows: list[dict], stats: dict[str, dict], *, max_mean_tok: float,
                max_trunc_rate: float, variant: str,
                include_allwrong_states: Sequence[str] = ()) -> tuple[list[dict], dict]:
    """순수 함수(테스트 가능). rows: parquet.to_dict('records'). 반환: (kept_rows, counters).

    ★버그 수리(0914): 예전엔 kept 행의 prompt 를 입력 parquet 그대로 남겨서
    (--variant math_retry 로 걸러도 build_math_parquet.py 가 만든 math_opt prompt 가
    그대로 붙어 있었다), M_RETRY 로 학습하면 forced 아닌 대다수 행이 실제로는
    math_opt 시스템 프롬프트로 롤아웃되던 것과 다른 프롬프트로 학습됐다.
    이제 kept 행 전부 --variant 로 prompt 를 재생성한다 — math_opt(디폴트)면
    입력과 동일한 결과(멱등)이고, math_retry 면 kept 전체가 실제로 math_retry
    시스템 프롬프트를 받는다. forced 행은 이후 apply_forced_redirect 가 덮어쓴다.

    `include_allwrong_states`: pass_rate==0 인 문제도, 그 문제의 (gold 없는) agree_state 가
    이 집합에 있으면 같은 길이/절단 필터를 적용해 추가로 남긴다(기본 빈 튜플 = 옛 동작)."""
    allow_allwrong = set(include_allwrong_states)
    kept = []
    n_no_stats = n_not_mixed = n_too_long = n_too_trunc = 0
    n_kept_mixed = n_kept_allwrong = 0
    for r in rows:
        k = norm_problem(str(r["problem"]))
        st = stats.get(k)
        if st is None:
            n_no_stats += 1
            continue
        pr = st["pass_rate"]
        mixed = 0.0 < pr < 1.0
        allwrong_ok = pr == 0.0 and st.get("agree_state") in allow_allwrong
        if not (mixed or allwrong_ok):
            n_not_mixed += 1
            continue
        if st["mean_tok"] > max_mean_tok:
            n_too_long += 1
            continue
        if st["trunc_rate"] > max_trunc_rate:
            n_too_trunc += 1
            continue
        r = dict(r)
        r["prompt"] = build_math_prompt(str(r["problem"]), variant)
        ei = dict(r.get("extra_info") or {})
        ei["group_pass_rate"] = float(pr)
        ei["prompt_variant"] = variant
        ei["agree_state"] = st.get("agree_state")
        r["extra_info"] = ei
        kept.append(r)
        if mixed:
            n_kept_mixed += 1
        else:
            n_kept_allwrong += 1
    counters = {"n_in": len(rows), "n_kept": len(kept), "n_no_stats": n_no_stats,
                "n_not_mixed": n_not_mixed, "n_too_long": n_too_long, "n_too_trunc": n_too_trunc,
                "n_kept_mixed": n_kept_mixed, "n_kept_allwrong": n_kept_allwrong}
    return kept, counters


def apply_forced_redirect(kept: list[dict], *, forced_frac: float, forced_variant: str,
                          seed: int) -> tuple[list[dict], int]:
    """kept 전체가 이미 mixed(0<pass_rate<1)이므로 build_math_parquet.py 의 mixed-only
    필터를 다시 적용할 필요 없이 kept 전체가 후보 풀이다 — forced_frac 은 이 풀에 대한
    비율이다."""
    if forced_frac <= 0:
        return kept, 0
    rng = random.Random(seed + 97)  # ★build_math_parquet.py 와 같은 오프셋 관례
    idx = list(range(len(kept)))
    rng.shuffle(idx)
    n_forced = round(forced_frac * len(kept))
    forced_idx = set(idx[:n_forced])
    out = []
    for i, r in enumerate(kept):
        if i in forced_idx:
            r = dict(r)
            r["prompt"] = build_math_prompt(str(r["problem"]), forced_variant)
            ei = dict(r["extra_info"])
            ei["forced_redirect"] = 1
            ei["prompt_variant"] = forced_variant
            r["extra_info"] = ei
        out.append(r)
    return out, n_forced


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollouts", required=True, help="math_rollout.py 가 쓴 texts.jsonl")
    ap.add_argument("--parquet", required=True, help="build_math_parquet.py 가 만든 train parquet")
    ap.add_argument("--max_mean_tok", type=float, default=5000.0)
    ap.add_argument("--max_trunc_rate", type=float, default=0.2)
    ap.add_argument("--include_allwrong_states", default="",
                    help="콤마 목록(예: DOMINANT,SPLIT,SCATTER,NOANS). 기본 빈 문자열 = "
                         "옛 동작(mixed 만). 지정한 상태의 all-wrong(pass_rate==0) 문제도 "
                         "같은 길이/절단 필터로 추가로 남긴다.")
    ap.add_argument("--variant", default="math_opt", choices=sorted(MATH_PROMPT_VARIANTS),
                    help="forced_redirect 재작성에 쓰는 원 변형 — math_retry/math_agree 일 때만 의미가 있다")
    ap.add_argument("--forced_frac", type=float, default=0.0)
    ap.add_argument("--forced_variant", default=None,
                    help="기본: --variant 에서 유도(math_retry→math_retry_forced, "
                         "math_agree→math_agree_forced)")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default=None, help="기본: <parquet>_mixed.parquet")
    a = ap.parse_args()

    import pandas as pd

    stats = load_group_stats(a.rollouts)
    df = pd.read_parquet(a.parquet)
    rows = df.to_dict("records")
    include_allwrong_states = [s.strip().upper() for s in a.include_allwrong_states.split(",")
                               if s.strip()]
    kept, counters = screen_rows(rows, stats, max_mean_tok=a.max_mean_tok,
                                 max_trunc_rate=a.max_trunc_rate, variant=a.variant,
                                 include_allwrong_states=include_allwrong_states)
    print(f"[screen_by_rollouts] {counters}")

    if a.forced_frac > 0:
        if a.variant in _FORCED_VARIANT_OF:
            forced_variant = a.forced_variant or _FORCED_VARIANT_OF[a.variant]
            kept, n_forced = apply_forced_redirect(kept, forced_frac=a.forced_frac,
                                                    forced_variant=forced_variant, seed=a.seed)
            print(f"[screen_by_rollouts] forced_redirect: n_forced={n_forced} forced_variant={forced_variant} "
                  f"(target={round(a.forced_frac * counters['n_kept'])})")
        else:
            print(f"[screen_by_rollouts] --forced_frac={a.forced_frac} 무시됨 — "
                  f"--variant={a.variant!r} 는 {sorted(_FORCED_VARIANT_OF)} 에 없다 (no-op)")

    out_path = Path(a.out) if a.out else Path(a.parquet).with_name(
        Path(a.parquet).stem + "_mixed.parquet")
    pd.DataFrame(kept).to_parquet(out_path, index=False)
    print(f"[screen_by_rollouts] {out_path} ({len(kept)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
