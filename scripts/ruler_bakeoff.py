"""자 총평가 — OSD · PMI · 도치 · doom 등 모든 «자» 후보를 «수정된 라벨»로 한 판에 놓는다.

사용자 지시(2026-09-01): "Osd, pmi 등 메타 보상 종류와, 도치나 정본 등 log p 계산 방식을
올바르게 비교하고, 자가 잘 구분하는지 → 학습이 실제로 효용이 있는지를 단계적으로 체크".
이 스크립트가 그 «자가 잘 구분하는지» 단계다. GPU 를 쓰지 않는다 (전부 디스크에 있음).

★라벨을 감사 판정대로 «둘로 쪼갠다». 둘 다 메타를 읽지 않고 계산 → 메타가 자기 점수를
  스스로 만들 수 없다(순환·게이밍 원천 차단).
    p_abl  = 그 자리에서 «메타를 지우고 다시 쓰게» 했을 때의 성공률  (stageB abl)
    p_orig = 원래 메타를 둔 채의 성공률                              (stageB orig)
    L1 지각 = −|confidence − p_abl|     자기 확신이 «실제» 상태와 맞나
    L2 유도 = p_orig − p_abl            이 메타가 있고 없고가 성공률을 바꾸나

★이전 판정이 뒤집힌 이유를 반복하지 않기 위해: 최종 정답 라벨로는 `post` 컬럼을 절대
  쓰지 않는다 — 그것은 «찾았다고 말했나» 라는 사후보고 플래그다(2026-09-01 사고).

검정: 사이트 자체가 선별된 표본이므로 문제 단위 군집 부트스트랩으로 CI 를 낸다.
      라벨 신뢰도(Spearman–Brown 0.77)로 감쇠 보정한 상한도 같이 보고한다.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

P = "cd6_work/probe"


def _cond_rate(glob_pat: str, cond: str) -> pd.Series:
    fs = sorted(glob.glob(f"{P}/{glob_pat}.parquet"))
    if not fs:
        return pd.Series(dtype=float)
    d = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    d = d[d.cond == cond]
    return d.groupby("site_id").correct.mean()


def build_labels() -> pd.DataFrame:
    p_abl = _cond_rate("stageB_clean_abl*", "abl")
    p_orig = _cond_rate("stageB_clean_o32*", "orig")
    if p_orig.empty:
        p_orig = _cond_rate("stageB_clean_abl*", "orig")
    sites = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{P}/sites_shard*.parquet"))],
                      ignore_index=True).drop_duplicates("site_id")
    lab = pd.DataFrame({"p_abl": p_abl, "p_orig": p_orig}).dropna().reset_index()
    lab = lab.merge(sites[["site_id", "confidence", "decision", "prob_idx", "pos",
                           "n_correct_of8"]], on="site_id", how="left")
    hy = pd.read_parquet(f"{P}/hygiene.parquet")
    lab = lab.merge(hy[["site_id", "terminal", "post_any", "echo_meta"]], on="site_id", how="left")
    lab["L1"] = -(lab.confidence - lab.p_abl).abs()
    lab["L2"] = lab.p_orig - lab.p_abl
    return lab


def collect_rulers() -> pd.DataFrame:
    """자 후보를 site_id 기준으로 한 표에 모은다. 계열 이름을 붙여 둔다."""
    out, fam = None, {}

    def add(df, cols, family, prefix=""):
        nonlocal out
        keep = ["site_id"] + [c for c in cols if c in df.columns]
        d = df[keep].copy()
        ren = {c: f"{prefix}{c}" for c in keep if c != "site_id"}
        d = d.rename(columns=ren)
        for c in ren.values():
            fam[c] = family
        out = d if out is None else out.merge(d, on="site_id", how="outer")

    try:
        d = pd.read_parquet(f"{P}/decoy_var.parquet")
        add(d, ["osd_gold", "osd_self"], "OSD")
        add(d, ["shift_near", "shift_family", "shift_other", "shift_shuffle"], "PMI(미끼종류)")
    except Exception:
        pass
    try:
        d = pd.read_parquet(f"{P}/pmi_agg.parquet")
        add(d, [c for c in d.columns if c.startswith("R2_")], "PMI(집계)")
    except Exception:
        pass
    try:
        d = pd.read_parquet(f"{P}/reverse_scores.parquet")
        add(d, [c for c in d.columns if c.startswith(("V1_", "V2_"))], "도치")
    except Exception:
        pass
    try:
        d = pd.read_parquet(f"{P}/inv_unified.parquet")
        d = d[d.variant == "real"] if "variant" in d.columns else d
        add(d, [c for c in d.columns if c.startswith(("A2D_", "D2A_"))], "도치(통일)")
    except Exception:
        pass
    try:
        d = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{P}/prefix_state_full*.parquet"))],
                      ignore_index=True)
        add(d, ["doom", "doom_gold", "doom_fam"], "프리픽스상태")
    except Exception:
        pass
    try:
        d = pd.read_parquet(f"{P}/ruler_scores.parquet")
        # ★C_conf / C_len 은 «카나리아»다 (ruler_battery.py:23). C_conf 는 confidence 그 자체라
        #   L1 = −|conf − p_abl| 와 기계적으로 상관 −1 이 된다. 자 후보가 아니므로 제외.
        add(d, [c for c in d.columns
                if c.startswith(("R1", "R2", "R3", "R4", "R5", "R6")) and c not in ("C_conf", "C_len")],
            "R배터리", prefix="rb_")
    except Exception:
        pass
    out.attrs["family"] = fam
    return out


def cluster_boot(x, y, groups, n=3000, seed=0):
    """문제 단위 군집 부트스트랩 스피어만 CI."""
    g = pd.DataFrame({"x": x, "y": y, "g": groups}).dropna()
    if len(g) < 12 or g.g.nunique() < 6:
        return (np.nan, np.nan, np.nan, len(g))
    rho = stats.spearmanr(g.x, g.y).correlation
    keys = g.g.unique()
    idx = {k: g.index[g.g == k].values for k in keys}
    rs = []
    rng = np.random.RandomState(seed)
    for _ in range(n):
        pick = np.concatenate([idx[k] for k in rng.choice(keys, len(keys))])
        s = g.loc[pick]
        if s.x.nunique() < 3 or s.y.nunique() < 3:
            continue
        rs.append(stats.spearmanr(s.x, s.y).correlation)
    if not rs:
        return (rho, np.nan, np.nan, len(g))
    return (rho, float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)), len(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{P}/bakeoff.parquet")
    ap.add_argument("--drop_terminal", type=int, default=1,
                    help="1이면 terminal(메타 직후 boxed) 사이트 제외 — Δ가 기계적으로 무력")
    ap.add_argument("--rel", type=float, default=0.77,
                    help="라벨 신뢰도(Spearman-Brown). 감쇠 보정 상한 계산용")
    args = ap.parse_args()

    lab = build_labels()
    print(f"[bake] 라벨 사이트 {len(lab)}  ·  p_abl {lab.p_abl.mean():.3f}  "
          f"p_orig {lab.p_orig.mean():.3f}  L2 평균 {lab.L2.mean():+.4f}", flush=True)
    if args.drop_terminal:
        n0 = len(lab)
        lab = lab[lab.terminal != 1]
        print(f"[bake] terminal 제외 {n0} → {len(lab)}", flush=True)

    rul = collect_rulers()
    fam = rul.attrs["family"]
    m = lab.merge(rul, on="site_id", how="inner")
    cands = [c for c in rul.columns if c != "site_id"]
    print(f"[bake] 자 후보 {len(cands)}개 · 합류 사이트 {len(m)}", flush=True)

    rows = []
    for c in cands:
        if m[c].notna().sum() < 20 or m[c].nunique() < 5:
            continue
        for lname in ("L2", "L1"):
            rho, lo, hi, n = cluster_boot(m[c], m[lname], m.prob_idx)
            rows.append(dict(ruler=c, family=fam.get(c, "?"), label=lname,
                             rho=rho, lo=lo, hi=hi, n=n,
                             rho_corrected=rho / np.sqrt(args.rel) if rho == rho else np.nan))
    res = pd.DataFrame(rows)
    res.to_parquet(args.out)
    print(f"[bake] wrote {args.out} ({len(res)}행)\n", flush=True)

    for lname in ("L2", "L1"):
        sub = res[res.label == lname].copy()
        sub["absrho"] = sub.rho.abs()
        sub = sub.sort_values("absrho", ascending=False).head(14)
        title = "L2 유도가치 (p_orig − p_abl)" if lname == "L2" else "L1 지각정확도 (−|conf − p_abl|)"
        print(f"══ {title} ══")
        print(f"{'자':<22}{'계열':<14}{'ρ':>8}{'95% CI':>20}{'n':>6}  {'0배제'}")
        for _, r in sub.iterrows():
            ex = "★" if (r.lo == r.lo and (r.lo > 0 or r.hi < 0)) else " "
            print(f"{r.ruler:<22}{r.family:<14}{r.rho:>8.3f}"
                  f"{f'[{r.lo:+.3f},{r.hi:+.3f}]':>20}{int(r.n):>6}  {ex}")
        print()


if __name__ == "__main__":
    main()
