#!/usr/bin/env python
r"""CLI — sites + continuations parquet 위에서 모든 자를 채점해 markdown+JSON 표를 낸다.

    python scripts/local/ruler_table.py --sites data/sites_v1/sites_judge.parquet \
        --conts data/continuations_v1.parquet --model_path models/Qwen3-4B \
        --out_dir results/ruler_table_v1

    # CPU 전용, 모델 forward 없이(테스트/파이프라인 점검용):
    python scripts/local/ruler_table.py --sites ... --conts ... --out_dir ... --no-model

`--conts`의 스키마는 `scripts/local/gen_continuations.py:build_record`(2026-09-04
확인)가 실제로 내는 컬럼을 기준으로 검사한다: site_id, mode, policy_tag, k_index,
continuation, full_text, r_corr, emitted, meta_raw, decision, confidence, novel,
followed, checked, donor_meta_raw, 그리고 `meta_start`나 `meta_start_in_cont`
둘 중 하나(그 파일은 후자만 낸다 — `meta_end`/`next_move`는 없고
`src/rulers/table.py:sample_from_row`가 `meta_raw`에서 복원한다). 과제 지시문이
원래 가정한 대안 스키마(`meta_start`/`meta_end`/`next_move` 컬럼이 직접 있는 것)도
계속 받는다.
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
                     "confidence", "r_corr")
# meta_start/meta_end 아니면 meta_start_in_cont(실제 gen_continuations.py 컬럼) 중
# 하나는 있어야 한다 — 정확히 이 두 이름 중 하나를 요구하므로 별도로 검사한다.
META_START_COLS = ("meta_start", "meta_start_in_cont")

MODEL_FREE_ONLY_MSG = (
    "--no-model 이 켜져 있어 needs_model=True 인 자는 전부 NaN 으로 스킵된다 "
    "(baselines/oracle/simulation 은 정상 계산된다).")


def _all_rulers():
    from src.rulers.pmi_shift import PmiShiftLast, PmiShiftMean, PmiShiftSum
    from src.rulers.osd import OsdSigned, OsdUnsigned
    from src.rulers.inv import InvMean, InvMin
    from src.rulers.dcont import DCont
    from src.rulers.move_kl import MoveKl, MoveKlSigned, MoveNovelShift, MoveNovelShiftStuck
    return [PmiShiftSum(), PmiShiftLast(), PmiShiftMean(), OsdUnsigned(), OsdSigned(),
           InvMin(), InvMean(), DCont(), MoveKl(), MoveKlSigned(),
           MoveNovelShift(), MoveNovelShiftStuck()]


def select_sites_stratified(sites_df: pd.DataFrame, conts_df: pd.DataFrame, n_sites: int,
                            seed: int = 0):
    r"""`n_sites` 개 사이트를 family_dead(1 vs 0/None) 50/50 층화로 뽑고, `conts_df`도
    그 사이트들로만 거른다. `n_sites >= len(sites_df)`이면 원본을 그대로 돌려준다.

    ⚠️구 `--limit`(행 개수)은 사이트당 행이 여럿(mode×k_index)이라 임의로 잘린 몇
    사이트만 남기기 쉽다(과제 배경: 200행 제한이 5사이트짜리 쓸모없는 표를 냈다).
    이 함수는 그 대신 **사이트를 먼저** 고른 뒤 그 사이트의 행을 전부 남긴다.
    """
    if n_sites <= 0 or n_sites >= len(sites_df):
        return sites_df, conts_df
    fam = pd.to_numeric(sites_df.get("family_dead"), errors="coerce")
    dead = sites_df[fam == 1]
    alive = sites_df[fam != 1]  # family_dead==0 그리고 NaN(메타 앞 시도 없음) 둘 다 포함
    n_dead = n_sites // 2
    n_alive = n_sites - n_dead
    picked = []
    if len(dead):
        picked.append(dead.sample(n=min(n_dead, len(dead)), random_state=seed))
    if len(alive):
        picked.append(alive.sample(n=min(n_alive, len(alive)), random_state=seed))
    out = pd.concat(picked) if picked else sites_df.iloc[0:0]
    if len(out) < n_sites:  # 한쪽 풀이 모자라면 남은 풀에서 채운다
        rest = sites_df[~sites_df["site_id"].isin(out["site_id"])]
        extra = rest.sample(n=min(n_sites - len(out), len(rest)), random_state=seed)
        out = pd.concat([out, extra])
    out = out.reset_index(drop=True)
    keep_ids = set(out["site_id"].astype(str))
    filtered_conts = conts_df[conts_df["site_id"].astype(str).isin(keep_ids)].reset_index(drop=True)
    return out, filtered_conts


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
    ap.add_argument("--limit", type=int, default=0,
                    help="⚠️행(row) 개수 상한이다 — 사이트 개수가 아니다. 한 사이트당 "
                    "행이 여럿(mode×k_index)이라 --limit 은 «임의로 잘린 몇 사이트만» "
                    "뽑는 결과를 내기 쉽다. 사이트 단위로 뽑으려면 --limit_sites 를 써라.")
    ap.add_argument("--limit_sites", type=int, default=0,
                    help="이 개수만큼 «사이트»를 family_dead(1/0) 50/50 층화 표집해 "
                    "고르고, conts 도 그 사이트들로만 거른다. 0=전부.")
    ap.add_argument("--attack_limit", type=int, default=100,
                    help="공격 배터리(정직+4공격=행당 5배 forward)를 새로 보는 사이트 "
                    "몇 개까지만 돌릴지(누적 관측 사이트 수 상한). 0=무제한.")
    ap.add_argument("--device", default=None, help="cuda|cpu (기본: cuda 가 보이면 cuda)")
    ap.add_argument("--no-model", dest="no_model", action="store_true",
                    help="모델 forward 없이 model-free 자/기준선/오라클/시뮬레이션만 돈다")
    args = ap.parse_args()

    if args.limit:
        print(f"[ruler_table] ⚠️ --limit={args.limit} 은 행(row) 개수다, 사이트 개수가 "
             "아니다 — 사이트 몇 개만 뽑고 싶으면 --limit_sites 를 써라.", file=sys.stderr)

    sites_df = pd.read_parquet(args.sites)
    conts_df = pd.read_parquet(args.conts)
    if args.limit_sites:
        sites_df, conts_df = select_sites_stratified(sites_df, conts_df, args.limit_sites)
        print(f"[ruler_table] --limit_sites={args.limit_sites} → sites={len(sites_df)}, "
             f"conts rows={len(conts_df)}", flush=True)
    missing = [c for c in REQUIRED_CONT_COLS if c not in conts_df.columns]
    if missing:
        raise SystemExit(
            f"ruler_table.py: continuations parquet 에 필수 컬럼이 없다: {missing}. "
            "scripts/local/gen_continuations.py 의 스키마가 바뀌었는지 확인하라.")
    if not any(c in conts_df.columns for c in META_START_COLS):
        raise SystemExit(
            f"ruler_table.py: {META_START_COLS} 중 하나도 없다 — 메타 시작 오프셋을 "
            "모르면 어떤 자도 못 돈다.")

    rulers = _select_rulers(args.rulers)

    ctx = None
    if not args.no_model:
        if not args.model_path:
            raise SystemExit("ruler_table.py: --no-model 이 아니면 --model_path 가 필요하다.")
        from src.rulers.hf_ctx import HfCtx
        # 0904: 기본값 cpu 로 돌린 첫 판이 48,000 이어쓰기에 며칠이 걸렸다. GPU 가 보이면 GPU·bf16.
        import torch  # noqa: PLC0415
        _dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        ctx = HfCtx(model_path=args.model_path, device=_dev, dtype=("bfloat16" if _dev == "cuda" else None))
        print(f"[ruler_table] device={_dev}", flush=True)
    else:
        print(f"[ruler_table] {MODEL_FREE_ONLY_MSG}", file=sys.stderr)

    result = run_table(sites_df, conts_df, rulers, ctx=ctx, limit=args.limit,
                      attack_limit=args.attack_limit)

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
