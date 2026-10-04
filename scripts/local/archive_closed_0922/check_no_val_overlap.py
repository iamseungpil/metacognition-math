#!/usr/bin/env python
r"""CLI — (nums,target) 문제 겹침 검사: 학습/판정용 parquet 이 held-out val 셋과
겹치지 않는지 확인한다 (E-133 오염 재발 방지 하드 불변).

    python scripts/local/check_no_val_overlap.py \
        --val $WORK/data/countdown_val_4num_opt.parquet \
        $WORK/data/sites_v4/sites_train.parquet \
        $WORK/data/sites_v4/sites_judge.parquet \
        $WORK/data/sites_v4/mixed_train_v4.parquet

각 입력 parquet 의 (nums,target) 집합과 --val 의 (nums,target) 집합의 교집합을
구해, 하나라도 비어 있지 않으면 파일별 겹침 개수를 stderr 에 찍고 **exit 1**
한다. 전부 비어 있으면 "OK: no overlap" 을 찍고 exit 0.

CPU 전용.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _problem_set(df) -> set[tuple]:
    out = set()
    for nums, target in zip(df["nums"], df["target"]):
        # ★숫자 «순서» 는 문제 정체성이 아니다 — [5,19,25,3] 과 [3,5,19,25] 는 같은 문제다.
        #   정렬해서 키를 만들지 않으면 순열만 다른 오염을 놓친다(0907 지적).
        out.add((tuple(sorted(int(v) for v in nums)), int(target)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val", required=True, help="held-out val parquet (겹치면 안 되는 기준 집합)")
    ap.add_argument("files", nargs="+", help="검사할 parquet 파일들")
    args = ap.parse_args()

    import pandas as pd

    val_df = pd.read_parquet(args.val)
    val_problems = _problem_set(val_df)
    print(f"[check_no_val_overlap] val={args.val} 문제 {len(val_problems)}개")

    any_overlap = False
    for f in args.files:
        p = Path(f)
        if not p.exists():
            print(f"[check_no_val_overlap] SKIP (없음): {f}", file=sys.stderr)
            continue
        df = pd.read_parquet(p)
        problems = _problem_set(df)
        overlap = problems & val_problems
        if overlap:
            any_overlap = True
            sample = list(overlap)[:5]
            print(f"[check_no_val_overlap] FAIL: {f} 문제 {len(problems)}개 중 "
                  f"{len(overlap)}개가 val 과 겹친다. 예: {sample}", file=sys.stderr)
        else:
            print(f"[check_no_val_overlap] OK: {f} 문제 {len(problems)}개, val 겹침 0")

    if any_overlap:
        print("[check_no_val_overlap] FAIL: val 겹침 발견 (E-133 재발)", file=sys.stderr)
        return 1
    print("[check_no_val_overlap] OK: no overlap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
