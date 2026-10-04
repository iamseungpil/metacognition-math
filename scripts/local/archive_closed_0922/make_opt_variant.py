#!/usr/bin/env python
r"""CLI — mixed_train(v4 형) parquet 의 시스템 메시지를 다른 프롬프트 변형으로
바꾼 쌍둥이 parquet 을 만든다 (예: `mixed_train_v4.parquet`(new, 메타 강제)
→ `mixed_train_v4_opt.parquet`(opt, 메타 허용만)).

    python scripts/local/make_opt_variant.py \
        --in  $WORK/data/sites_v4/mixed_train_v4.parquet \
        --out $WORK/data/sites_v4/mixed_train_v4_opt.parquet \
        --variant opt

왜 이렇게 해도 되는가: `data/countdown_train_4num_new.parquet` 과
`_opt.parquet` 을 실측 비교하면 두 판은 시스템 메시지만 다르고 user 메시지·
nums·target 순서까지 바이트 동일하다(0907 확인). site 행의 프롬프트도
`countdown_sites.render_prefix_prompt` → `countdown_task.build_prompt(inst,
variant)` + 프리픽스 접합이라 구조가 같다 — 즉 "제대로 된 opt 판을 새로
빌드"하는 것과 "new 판에서 시스템 메시지만 opt 로 바꿔치기"하는 것이 결과적으로
바이트 동일하다. `scripts/local/build_cf_twins.py::_twin_prompt` 가 이미 같은
가정으로 twin 행을 만들고 있다(그 파일의 docstring 참조) — 이 스크립트는 그
가정을 "행 하나"가 아니라 "parquet 전체"에 적용한 것뿐이다.

CPU 전용.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.training import countdown_task as ct  # noqa: E402


def swap_system_prompt(prompt, variant: str):
    """`prompt`(system/user/assistant 메시지 리스트)에서 system 메시지만
    `PROMPT_VARIANTS[variant]` 로 바꾼 **새** 리스트를 돌려준다. build_cf_twins.py
    의 `_twin_prompt` 와 동일한 가정/계약.
    """
    msgs = [dict(m) for m in prompt]
    if not msgs or msgs[0].get("role") != "system":
        raise ValueError("prompt[0] 이 system 메시지가 아니다 — 스키마 가정이 깨졌다.")
    msgs[0] = {"role": "system", "content": ct.PROMPT_VARIANTS[variant]}
    return msgs


def make_opt_variant(df, variant: str):
    import pandas as pd

    rows = []
    for _, row in df.iterrows():
        r = row.to_dict()
        r["prompt"] = swap_system_prompt(list(r["prompt"]), variant)
        ei = dict(r.get("extra_info") or {})
        if "prompt_variant" in ei:
            ei["prompt_variant"] = variant
            r["extra_info"] = ei
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--variant", default="opt", choices=sorted(ct.PROMPT_VARIANTS))
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.inp)
    out_df = make_opt_variant(df, args.variant)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"[make_opt_variant] {args.inp} ({len(df)}행) -> {args.out} "
          f"(variant={args.variant})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
