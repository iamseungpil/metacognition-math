#!/usr/bin/env python
r"""CLI — sites + continuations parquet 위에서 모든 자를 채점해 markdown+JSON 표를 낸다.

    python scripts/local/ruler_table.py --sites data/sites_v1/sites_judge.parquet \
        --conts data/continuations_v1.parquet --model_path models/Qwen3-4B \
        --out_dir results/ruler_table_v1

    # CPU 전용, 모델 forward 없이(테스트/파이프라인 점검용):
    python scripts/local/ruler_table.py --sites ... --conts ... --out_dir ... --no-model

`--conts`의 스키마는 (아직 없는) `scripts/local/gen_continuations.py`가 낼 것으로
명시된 것을 그대로 가정한다: site_id, mode, policy_tag, k_index, continuation,
full_text, r_corr, emitted, meta_raw, decision, confidence, novel, followed,
checked, donor_meta_raw. 그 파일이 실제로 생기면 컬럼명이 갈릴 수 있으니 로드
직후 필수 컬럼을 검사한다(fail-loud).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.rulers.table import run_table  # noqa: E402

REQUIRED_CONT_COLS = ("site_id", "mode", "continuation", "meta_raw", "decision",
                     "confidence", "r_corr", "meta_start", "meta_end")

MODEL_FREE_ONLY_MSG = (
    "--no-model 이 켜져 있어 needs_model=True 인 자는 전부 NaN 으로 스킵된다 "
    "(baselines/oracle/simulation 은 정상 계산된다).")


def _all_rulers():
    from src.rulers.pmi_shift import PmiShiftLast, PmiShiftMean, PmiShiftSum
    from src.rulers.osd import OsdSigned, OsdUnsigned
    from src.rulers.inv import InvMean, InvMin
    from src.rulers.dcont import DCont
    from src.rulers.move_kl import MoveKl, MoveKlSigned
    return [PmiShiftSum(), PmiShiftLast(), PmiShiftMean(), OsdUnsigned(), OsdSigned(),
           InvMin(), InvMean(), DCont(), MoveKl(), MoveKlSigned()]


def _select_rulers(names: str):
    all_r = {r.name: r for r in _all_rulers()}
    if names == "all":
        return list(all_r.values())
    wanted = [n.strip() for n in names.split(",") if n.strip()]
    missing = [n for n in wanted if n not in all_r]
    if missing:
        raise SystemExit(f"ruler_table.py: 모르는 자 {missing} — 가능한 이름: {sorted(all_r)}")
    return [all_r[n] for n in wanted]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", required=True)
    ap.add_argument("--conts", required=True)
    ap.add_argument("--model_path", default=None)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--rulers", default="all")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no-model", dest="no_model", action="store_true",
                    help="모델 forward 없이 model-free 자/기준선/오라클/시뮬레이션만 돈다")
    args = ap.parse_args()

    sites_df = pd.read_parquet(args.sites)
    conts_df = pd.read_parquet(args.conts)
    missing = [c for c in REQUIRED_CONT_COLS if c not in conts_df.columns]
    if missing:
        raise SystemExit(
            f"ruler_table.py: continuations parquet 에 필수 컬럼이 없다: {missing}. "
            "scripts/local/gen_continuations.py 의 스키마가 바뀌었는지 확인하라.")

    rulers = _select_rulers(args.rulers)

    ctx = None
    if not args.no_model:
        if not args.model_path:
            raise SystemExit("ruler_table.py: --no-model 이 아니면 --model_path 가 필요하다.")
        from src.rulers.hf_ctx import HfCtx
        ctx = HfCtx(model_path=args.model_path)
    else:
        print(f"[ruler_table] {MODEL_FREE_ONLY_MSG}", file=sys.stderr)

    result = run_table(sites_df, conts_df, rulers, ctx=ctx, limit=args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ruler_table.md").write_text(result["markdown"])
    with open(out_dir / "ruler_table.json", "w") as fh:
        json.dump(result["json"], fh, indent=2, default=str)
    result["scored"].to_parquet(out_dir / "scored_rows.parquet")
    print(f"[ruler_table] wrote {out_dir}/ruler_table.md, ruler_table.json, scored_rows.parquet "
         f"({result['json']['n_rows']} rows)")


if __name__ == "__main__":
    main()
