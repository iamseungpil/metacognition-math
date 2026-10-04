#!/usr/bin/env python
r"""CLI — «교사 신뢰도 게이트»를 자리(site)별로 계산해 mixed_train parquet 의
각 행 `extra_info` 에 `opd_gate`/`hint_rate`/`nometa_rate` 를 얹는다 (OPT_OPDG 팔,
docs/PREREGISTRATION_countdown_anchored_2x2.md §11).

    python scripts/local/build_gate_sites.py \
        --mixed $WORK/data/sites_v4/mixed_train_v4_opt.parquet \
        --hint  $WORK/conts_v4/hint_train.parquet \
        --base  $WORK/conts_v4/base_train.parquet \
        --out   $WORK/data/sites_v4/mixed_train_v4_gate_opt.parquet \
        --tau 0.10 --min_hint 0.25

게이트의 정의(자리 하나당):
    hint_rate   = mean(r_corr)  over 힌트 모드 롤아웃 행 (mode=="hint" 가 있으면 그것만)
    nometa_rate = mean(r_corr)  over base 롤아웃 행 중 mode=="nometa"
    gate = 1  iff  (hint_rate − nometa_rate) >= tau  AND  hint_rate >= min_hint

왜: OPD/OPDC 는 «힌트 교사와 가까운가»만 재고 교사가 그 자리에서 실제로 도움이
되는지는 안 본다. 교사가 아무것도 안 하는 것보다 못한 자리에서까지 증류하면 잡음을
배운다 — 게이트는 교사가 실증적으로 이긴 자리에서만 opd_meta_c 를 켜고, 나머지 행은
순수 결과 GRPO 로 남긴다(항 기여 0).

관례: 자리(site_id)가 없는 **일반 행**과 게이트 표에 없는 자리는
`opd_gate=0, hint_rate=0.0, nometa_rate=0.0` 으로 채운다(NaN 대신 0.0 —
`_col` 폴백이 나르는 값이 항상 유한수여야 배선 사고를 무음으로 만들지 않는다).
게이트가 0 인 행은 verl_sdc 가 opd_meta_c 를 0 으로 강제하므로 rate 값 자체는
진단용이다.

CPU 전용.
"""
from __future__ import annotations

import argparse
import sys

# ★correctness 보상 컬럼의 단일 정의처. conts_*/hint_* parquet(gen_continuations.py
#   산출)은 `r_corr`(int 0/1)를 쓴다 — 다른 이름이 오면 fail-loud 로 죽는다.
REWARD_COL_CANDIDATES = ("r_corr", "reward", "rm_scores")

DEFAULT_TAU = 0.10
DEFAULT_MIN_HINT = 0.25


def _reward_col(df, what: str) -> str:
    for c in REWARD_COL_CANDIDATES:
        if c in df.columns:
            return c
    raise ValueError(
        f"{what} parquet 에 정답 보상 컬럼이 없다 — 후보 {REWARD_COL_CANDIDATES}, "
        f"실제 컬럼 {sorted(df.columns)}")


def _rate_by_site(df, *, mode=None, what="") -> dict:
    """site_id → mean(정답 보상). `mode` 가 주어지고 df 에 `mode` 컬럼이 있으면
    그 모드 행만 쓴다(없으면 전 행 — 힌트 전용 parquet 이 그렇다)."""
    col = _reward_col(df, what)
    d = df
    if mode is not None and "mode" in d.columns:
        d = d[d["mode"].astype(str) == mode]
    if "site_id" not in d.columns:
        raise ValueError(f"{what} parquet 에 site_id 컬럼이 없다 — 자리 단위 집계 불가.")
    d = d[d["site_id"].astype(str) != ""]
    g = d.groupby(d["site_id"].astype(str))[col].mean()
    return {str(k): float(v) for k, v in g.items()}


def compute_site_gates(hint_df, base_df, tau: float = DEFAULT_TAU,
                       min_hint: float = DEFAULT_MIN_HINT) -> dict:
    """자리별 게이트 표: `{site_id: {"hint_rate", "nometa_rate", "delta", "gate"}}`.

    힌트 행이 있는 자리만 표에 들어간다(교사가 그 자리에서 뭘 했는지 모르면 판정
    불가). base 에 nometa 행이 없는 자리는 `nometa_rate=0.0` 으로 본다.
    """
    hint = _rate_by_site(hint_df, mode="hint", what="hint")
    nometa = _rate_by_site(base_df, mode="nometa", what="base")
    out = {}
    for sid, hr in hint.items():
        nr = float(nometa.get(sid, 0.0))
        delta = hr - nr
        out[sid] = {
            "hint_rate": float(hr),
            "nometa_rate": nr,
            "delta": float(delta),
            "gate": int(delta >= float(tau) and hr >= float(min_hint)),
        }
    return out


def augment_mixed_with_gate(mixed_df, site_gates: dict):
    """mixed parquet 의 모든 행 `extra_info` 에 opd_gate/hint_rate/nometa_rate 를 얹는다.

    `extra_info` 안에만 넣는다(평평한 컬럼 X) — `verl_sdc._col` 의 extra_info 폴백이
    읽는 경로이고, cf_role/cf_key(build_cf_twins.py)가 이미 쓰는 관례다.
    """
    df = mixed_df.copy()
    new_ei = []
    for ei_raw, sid_raw in zip(df["extra_info"], df.get("site_id", [""] * len(df))):
        ei = dict(ei_raw or {})
        sid = str(ei.get("site_id", sid_raw) or "")
        st = site_gates.get(sid) if sid else None
        if st is None:
            ei["opd_gate"] = 0
            ei["hint_rate"] = 0.0
            ei["nometa_rate"] = 0.0
        else:
            ei["opd_gate"] = int(st["gate"])
            ei["hint_rate"] = float(st["hint_rate"])
            ei["nometa_rate"] = float(st["nometa_rate"])
        new_ei.append(ei)
    df["extra_info"] = new_ei
    return df


def summarize(site_gates: dict, mixed_df=None, tau: float = DEFAULT_TAU,
              min_hint: float = DEFAULT_MIN_HINT) -> str:
    n = len(site_gates)
    gated = sum(1 for v in site_gates.values() if v["gate"])
    lines = [
        f"[gate] tau={tau:g} min_hint={min_hint:g}",
        f"[gate] gated sites / total sites = {gated}/{n} "
        f"({(gated / n * 100 if n else 0):.1f}%)",
        "[gate] decile of (hint_rate - nometa_rate) across sites:",
    ]
    deltas = sorted(v["delta"] for v in site_gates.values())
    if deltas:
        for p in range(0, 101, 10):
            i = min(len(deltas) - 1, int(round(p / 100 * (len(deltas) - 1))))
            lines.append(f"    p{p:>3d}  {deltas[i]:+.4f}")
    if mixed_df is not None:
        ei = list(mixed_df["extra_info"])
        rows_gated = sum(1 for e in ei if int((e or {}).get("opd_gate", 0)))
        lines.append(f"[gate] mixed rows with opd_gate==1: {rows_gated}/{len(ei)}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mixed", required=True)
    ap.add_argument("--hint", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--min_hint", type=float, default=DEFAULT_MIN_HINT)
    a = ap.parse_args(argv)

    import pandas as pd

    mixed = pd.read_parquet(a.mixed)
    hint = pd.read_parquet(a.hint)
    base = pd.read_parquet(a.base)

    gates = compute_site_gates(hint, base, tau=a.tau, min_hint=a.min_hint)
    out_df = augment_mixed_with_gate(mixed, gates)
    print(summarize(gates, out_df, tau=a.tau, min_hint=a.min_hint), flush=True)
    out_df.to_parquet(a.out, index=False)
    print(f"[gate] wrote {a.out}  rows={len(out_df)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
