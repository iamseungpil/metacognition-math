"""자 최종 비교 — 교란(위치·바닥효과)을 통제한 뒤 어떤 자가 가장 잘 맞히는가.

검토관 지적과 정량 검증으로 확인된 세 가지를 반영한다:
  · 바닥효과: clean 사이트의 58.5%가 orig·abl 양팔 모두 ≈0 → 측정 자체가 불가능
  · 위치 교란: ρ(pos, Δ_abl) = −0.185 (후반 메타일수록 Δ 낮음)
  · 개별 사이트 라벨은 대부분 잡음 (K=64에서 유의 13%)
따라서 ① 측정가능 대역만 사용 ② 위치를 층으로 통제 ③ 연속 Δ 를 쓴다.

사용: final_ruler_compare.py --sets base,x   (base=cd6_work/probe, x=cd6_work/probe_x)
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

W = "/home/jovyan/beomi/splee"


def auc(y, s):
    y = np.asarray(y, float); s = np.asarray(s, float)
    ok = ~np.isnan(s) & ~np.isnan(y)
    y, s = y[ok], s[ok]
    if len(set(y.tolist())) < 2:
        return float("nan")
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_auc(y, s, n=4000, seed=0):
    rng = np.random.RandomState(seed); y = np.asarray(y); s = np.asarray(s); v = []
    for _ in range(n):
        i = rng.randint(0, len(y), len(y))
        if len(set(y[i].tolist())) == 2:
            v.append(auc(y[i], s[i]))
    return (np.percentile(v, 2.5), np.percentile(v, 97.5)) if v else (np.nan, np.nan)


def partial_spearman(x, y, z):
    """z 를 통제한 x-y 부분 스피어만 (순위 잔차 상관)."""
    def resid(a, b):
        A = rankdata(a).astype(float); B = rankdata(b).astype(float)
        B = np.c_[np.ones(len(B)), B]
        beta = np.linalg.lstsq(B, A, rcond=None)[0]
        return A - B @ beta
    return float(np.corrcoef(resid(x, z), resid(y, z))[0, 1])


def load_set(tag):
    """tag 별 (Δ_abl 표, sites) 반환."""
    if tag == "base":
        oc = pd.concat([pd.read_parquet(f'{W}/cd6_work/probe/stageB_clean_shard{i}.parquet')
                        for i in range(2)]
                       + [pd.read_parquet(f'{W}/cd6_work/probe/stageB_clean_o32_shard0.parquet')])
        ab = pd.concat([pd.read_parquet(p) for p in
                        glob.glob(f'{W}/cd6_work/probe/stageB_clean_abl*shard0.parquet')])
        K = sorted(ab.k.unique())
        o = oc[(oc.cond == 'orig') & (oc.k.isin(K))].groupby('site_id').correct.mean()
        a = ab.groupby('site_id').correct.mean()
        sites = pd.concat([pd.read_parquet(p) for p in
                           glob.glob(f'{W}/cd6_work/probe/sites_shard*.parquet')])
    else:
        sp = pd.concat([pd.read_parquet(p) for p in
                        glob.glob(f'{W}/cd6_work/probe_x/stageB_clean_origabl_shard*.parquet')
                        + glob.glob(f'{W}/cd6_work/probe_x/stageB_clean_shard*.parquet')])
        o = sp[sp.cond == 'orig'].groupby('site_id').correct.mean()
        a = sp[sp.cond == 'abl'].groupby('site_id').correct.mean()
        sites = pd.concat([pd.read_parquet(p) for p in
                           glob.glob(f'{W}/cd6_work/probe_x/sites_shard*.parquet')])
    d = pd.DataFrame({'p_orig': o, 'p_abl': a}).dropna().reset_index()
    d['d_abl'] = d.p_orig - d.p_abl
    d['set'] = tag
    cols = ['site_id', 'pos', 'confidence', 'decision', 'meta_len', 'n_correct_of8', 'prob_idx']
    return d.merge(sites[cols].drop_duplicates('site_id'), on='site_id')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="base,x")
    ap.add_argument("--lo", type=float, default=0.05, help="측정가능 대역 하한 (p_abl)")
    ap.add_argument("--hi", type=float, default=0.95)
    args = ap.parse_args()

    d = pd.concat([load_set(t) for t in args.sets.split(",")], ignore_index=True)
    d['redirect'] = (d.decision == 'redirect').astype(int)

    # 자 점수 병합 (있는 것만)
    parts = []
    rs = [pd.read_parquet(p)[['site_id', 'R1_pmi1tok', 'R2_full', 'R3_family',
                              'R4_osdgold', 'R5_osdself']]
          for p in glob.glob(f'{W}/cd6_work/probe*/ruler_scores*.parquet')]
    if rs:
        parts.append(pd.concat(rs, ignore_index=True).drop_duplicates('site_id'))
    ps = [pd.read_parquet(p)[['site_id', 'doom', 'doom_gold', 'doom_fam']]
          for p in glob.glob(f'{W}/cd6_work/probe*/prefix_state.parquet')]
    if ps:                      # 여러 파일은 concat (개별 merge 하면 doom_x/doom_y 로 갈라진다)
        parts.append(pd.concat(ps, ignore_index=True).drop_duplicates('site_id'))
    for pt in parts:
        d = d.merge(pt, on='site_id', how='left')

    print(f"전체 {len(d)} 사이트 ({d.groupby('set').size().to_dict()})")
    floor = (d.p_orig <= args.lo) & (d.p_abl <= args.lo)
    ceil_ = (d.p_orig >= args.hi) & (d.p_abl >= args.hi)
    m = d[~floor & ~ceil_].copy()
    print(f"바닥/천장 제외 → 측정가능 {len(m)} 사이트 (제외 {floor.sum()}+{ceil_.sum()})")
    print(f"측정가능 대역 평균 Δ_abl = {m.d_abl.mean():+.4f} · 표준편차 {m.d_abl.std():.3f}")

    m['harm'] = (m.d_abl < -0.05).astype(int)
    cands = [
        ('decision=verify (메타 자기신고)', (1 - m.redirect).astype(float)),
        ('confidence (높을수록 해로움)', m.confidence.astype(float)),
        ('conf+verify 결합', ((m.confidence - m.confidence.mean()) / m.confidence.std()
                              + (1 - m.redirect)).astype(float)),
        ('위치 pos (후반일수록 해로움)', m.pos.astype(float)),
    ]
    for c, nm in [('R2_full', 'R2_full (고친 PMI)'), ('R3_family', 'R3_family'),
                  ('R1_pmi1tok', 'R1 (현행 PMI)'), ('R5_osdself', 'R5 (현행 OSD)'),
                  ('doom', 'doom (프리픽스 막힘도)')]:
        if c in m.columns:
            cands.append((nm, -m[c].astype(float)))
    if 'doom' in m.columns:
        z = lambda s: (s - s.mean()) / s.std()
        cands.append(('★막힘×결정 정합 (doom 낮음 & verify)', (-z(m.doom) + (1 - m.redirect)).astype(float)))

    print("\n══ 해로움 탐지 (Δ_abl < −0.05), 측정가능 대역 ══")
    print(f"{'신호':38s} {'AUC':>6s}  {'95% CI':>16s}  {'n':>5s}")
    rows = []
    for nm, s in cands:
        dd = pd.DataFrame({'y': m.harm, 's': s}).dropna()
        if len(dd) < 30:
            continue
        a_ = auc(dd.y, dd.s); lo, hi = boot_auc(dd.y.values, dd.s.values)
        rows.append((nm, a_, lo, hi, len(dd)))
    for nm, a_, lo, hi, n in sorted(rows, key=lambda r: -r[1]):
        star = '★' if lo > 0.5 else ''
        print(f"{nm:38s} {a_:6.3f}  [{lo:5.3f}, {hi:5.3f}]  {n:5d} {star}")

    print("\n══ 연속 Δ_abl 상관 · 위치(pos) 통제 전/후 ══")
    print(f"{'신호':38s} {'ρ':>7s} {'ρ|pos':>8s} {'p':>8s}")
    for nm, s in cands:
        dd = pd.DataFrame({'y': m.d_abl, 's': s, 'p': m.pos}).dropna()
        if len(dd) < 30:
            continue
        rho, pv = spearmanr(dd.s, dd.y)
        pr = partial_spearman(dd.s.values, dd.y.values, dd.p.values)
        print(f"{nm:38s} {-rho:7.3f} {-pr:8.3f} {pv:8.4f}")
    print("\n(부호는 «해로움 예측» 방향으로 정렬 — 값이 클수록 해로운 메타를 잘 집어낸다)")


if __name__ == "__main__":
    main()
