"""Rebuttal checks on the selfcons report. CPU only. No repo edits."""
import json, random, re, math
import numpy as np
from collections import Counter

BASE = '/tmp/claude-587327809/-home-v-seungplee/41a99d3b-d246-48cd-b893-68375dc4e059/scratchpad/mini_abl/'
d = json.load(open(BASE + 'splice2_4b.json'))
PS = d['per_sample']; ARMS = d['conds']; LEV = d['levels']; N = d['n_problems']


def norm(p):
    if p is None:
        return None
    s = p.strip()
    s = s.replace('\\left', '').replace('\\right', '')
    s = s.replace('\\!', '').replace('\\,', '').replace('\\;', '').replace('\\ ', '')
    s = s.replace('dfrac', 'frac').replace('tfrac', 'frac')
    s = s.replace('^\\circ', '').replace('\\%', '').replace('%', '')
    s = s.replace('\\$', '').replace('$', '')
    s = re.sub(r'\s+', '', s)
    s = s.rstrip('.')
    s = re.sub(r'^\\text\{(.*)\}$', r'\1', s)
    s = s.replace('{', '').replace('}', '')
    if s.startswith('\\text'):
        s = s[5:]
    return s if s else None


CLS = {a: [[norm(s['pred']) for s in PS[a][i]] for i in range(N)] for a in ARMS}
GRD = {a: [[s['graded'] for s in PS[a][i]] for i in range(N)] for a in ARMS}
RAW = {a: [[s['pred'] for s in PS[a][i]] for i in range(N)] for a in ARMS}
acc = {a: np.array([np.mean(GRD[a][i]) for i in range(N)]) for a in ARMS}


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    a = np.concatenate([pos, neg])
    order = a.argsort(kind='mergesort')
    r = np.empty(len(a)); r[order] = np.arange(1, len(a) + 1)
    s = np.sort(a); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[np.isin(a, [s[i]])] = (i + 1 + j + 1) / 2
        i = j + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def mkpool(arms):
    out = []
    for i in range(N):
        lst = []; u = 0
        for a in arms:
            for j in range(6):
                c = CLS[a][i][j]
                if c is None:
                    c = f'__NONE{u}__'; u += 1
                lst.append((c, GRD[a][i][j]))
        out.append(lst)
    return out


POOL = {'N6': mkpool(['N']), 'BBp12': mkpool(['B', 'Bp']), 'all36': mkpool(ARMS),
        'B6': mkpool(['B']), 'Bp6': mkpool(['Bp'])}


def label_from(arms):
    lab = []
    for i in range(N):
        g = [GRD[a][i][j] for a in arms for j in range(6)]
        lab.append(1 if all(x == 1 for x in g) else (0 if all(x == 0 for x in g) else None))
    return lab


LAB_N = label_from(['N'])


def consensus(sub):
    cnt = {}
    for c, g in sub:
        cnt.setdefault(c, [0, 0]); cnt[c][0] += 1; cnt[c][1] += g
    best = max(cnt.items(), key=lambda kv: kv[1][0])
    return best[1][0] / len(sub), (1 if best[1][1] * 2 >= best[1][0] else 0)


def draw(p, k, rng):
    return rng.sample(p, k) if k <= len(p) else [p[rng.randrange(len(p))] for _ in range(k)]


def sig_avg(poolname, k, seeds=30):
    pool = POOL[poolname]; out = np.zeros(N)
    for s in range(seeds):
        rng = random.Random(5000 + s * 104729)
        for i in range(N):
            out[i] += consensus(draw(pool[i], k, rng))[0]
    return out / seeds


def sig_single(poolname, k, seed):
    pool = POOL[poolname]; rng = random.Random(seed)
    return np.array([consensus(draw(pool[i], k, rng))[0] for i in range(N)])


print('=' * 78)
print('R1. AUC CI type: report gives SEED spread, original 0.9071 gave PROBLEM bootstrap')
print('=' * 78)
for pn, k in (('BBp12', 8), ('N6', 6), ('all36', 8), ('all36', 36)):
    # per-seed AUC (report style) and problem-bootstrap AUC (correct style)
    seed_aucs = []
    per_seed_sig = []
    for s in range(30):
        rng = random.Random(1000 + s * 7919)
        ag = np.array([consensus(draw(POOL[pn][i], k, rng))[0] for i in range(N)])
        per_seed_sig.append(ag)
        seed_aucs.append(auc([ag[i] for i in range(N) if LAB_N[i] == 1],
                             [ag[i] for i in range(N) if LAB_N[i] == 0]))
    seed_aucs = np.array(seed_aucs)
    # bootstrap over problems, resampling seeds too
    pos_idx = [i for i in range(N) if LAB_N[i] == 1]
    neg_idx = [i for i in range(N) if LAB_N[i] == 0]
    rb = np.random.default_rng(7)
    bs = []
    for b in range(2000):
        ag = per_seed_sig[b % 30]
        p = ag[rb.choice(pos_idx, len(pos_idx), replace=True)]
        q = ag[rb.choice(neg_idx, len(neg_idx), replace=True)]
        bs.append(auc(p, q))
    bs = np.array(bs)
    print(f'  pool={pn:6s} k={k:2d}  mean AUC {seed_aucs.mean():.4f} | '
          f'SEED-only [{np.percentile(seed_aucs,2.5):.4f},{np.percentile(seed_aucs,97.5):.4f}] | '
          f'PROBLEM-boot [{np.percentile(bs,2.5):.4f},{np.percentile(bs,97.5):.4f}]  '
          f'(n+={len(pos_idx)}, n-={len(neg_idx)})')

print()
print('=' * 78)
print('R2. 0.9071 vs 0.9228: is it the normalizer, or the agreement DENOMINATOR?')
print('=' * 78)
nrm_sp = lambda s: (s or '').replace(' ', '').strip()
ag_sp = []      # signal_probe definition: denominator = non-empty preds only, all 36
ag_den36 = []   # denominator = 36, empties as unique classes (report definition)
for i in range(N):
    c = Counter(nrm_sp(r['pred']) for cd in ARMS for r in PS[cd][i] if r['pred'])
    c.pop('', None)
    tot = sum(c.values())
    ag_sp.append((c.most_common(1)[0][1] / tot) if tot else 0.0)
    ag_den36.append(consensus(POOL['all36'][i])[0])
ag_sp = np.array(ag_sp); ag_den36 = np.array(ag_den36)
for nm, v in (('signal_probe def (empty preds dropped from denominator)', ag_sp),
              ('report def (empty preds = unique class, denom=36)', ag_den36)):
    print(f'  {nm:58s} AUC={auc([v[i] for i in range(N) if LAB_N[i]==1],[v[i] for i in range(N) if LAB_N[i]==0]):.4f}')
# also: same normalizer, but signal_probe raw-string normalizer inside report machinery
ag_sp_norm = []
for i in range(N):
    c = Counter(norm(r['pred']) for cd in ARMS for r in PS[cd][i] if norm(r['pred']))
    tot = sum(c.values())
    ag_sp_norm.append((c.most_common(1)[0][1] / tot) if tot else 0.0)
ag_sp_norm = np.array(ag_sp_norm)
print(f"  {'signal_probe denominator + REPORT normalizer':58s} "
      f"AUC={auc([ag_sp_norm[i] for i in range(N) if LAB_N[i]==1],[ag_sp_norm[i] for i in range(N) if LAB_N[i]==0]):.4f}")
nempty = sum(1 for i in range(N) for cd in ARMS for r in PS[cd][i] if not r['pred'])
print(f'  samples with empty pred: {nempty}/{36*N} ({100*nempty/(36*N):.2f}%)')

print()
print('=' * 78)
print('R3. seed coupling: B/N/E/S/R all generated with seed=0; only Bp used seed=977')
print('=' * 78)
def idxmatch(a, b):
    m = t = 0
    for i in range(N):
        for j in range(6):
            x, y = CLS[a][i][j], CLS[b][i][j]
            t += 1
            if x is not None and x == y:
                m += 1
    return m / t
for pair in (('N', 'B'), ('N', 'Bp'), ('B', 'Bp'), ('N', 'R'), ('N', 'S'), ('N', 'E')):
    print(f'  index-aligned pred match {pair[0]}[i][j]=={pair[1]}[i][j]: {idxmatch(*pair):.4f}')
# shuffled-index control
rngc = random.Random(3)
m = t = 0
for i in range(N):
    perm = list(range(6)); rngc.shuffle(perm)
    for j in range(6):
        x, y = CLS['N'][i][j], CLS['B'][i][perm[j]]
        t += 1
        if x is not None and x == y:
            m += 1
print(f'  control (B index shuffled within problem): {m/t:.4f}')
print('  --> AUC with B-only pool (seed-coupled to N labels) vs Bp-only pool (independent seed):')
for pn in ('B6', 'Bp6'):
    a6 = np.array([consensus(POOL[pn][i])[0] for i in range(N)])
    print(f'     {pn}: AUC={auc([a6[i] for i in range(N) if LAB_N[i]==1],[a6[i] for i in range(N) if LAB_N[i]==0]):.4f}')

print()
print('=' * 78)
print('R4. section 3c "internal separation" is destroyed by averaging over 30 reps')
print('=' * 78)
accN_half = np.zeros(N); REP = 30
for s in range(REP):
    r = random.Random(900 + s)
    for i in range(N):
        idxs = list(range(6)); r.shuffle(idxs)
        accN_half[i] += np.mean([GRD['N'][i][j] for j in idxs[3:]])
accN_half /= REP
print(f'  corr(accN_half, accN) = {np.corrcoef(accN_half, acc["N"])[0,1]:.4f}   '
      f'max|diff| = {np.abs(accN_half-acc["N"]).max():.4f}  mean|diff| = {np.abs(accN_half-acc["N"]).mean():.4f}')

print()
print('=' * 78)
print('R5. gate gains vs the project delta_eq = 2.23pp, with paired bootstrap CI')
print('=' * 78)
print(f'  arm means: N {acc["N"].mean():.4f}  B {acc["B"].mean():.4f}  Bp {acc["Bp"].mean():.4f}  R {acc["R"].mean():.4f}')
print(f'  delta_eq (2 x halfwidth of Bp-B) = {100*d["delta_eq"]:.2f}pp')


def boot_gain(gain_vec, nb=4000):
    rb = np.random.default_rng(11)
    v = np.array(gain_vec)
    bs = [v[rb.integers(0, N, N)].mean() for _ in range(nb)]
    return 100 * v.mean(), 100 * np.percentile(bs, 2.5), 100 * np.percentile(bs, 97.5)


s_bbp8 = sig_avg('BBp12', 8)
for base_arm in ('N', 'B'):
    ab = acc[base_arm]
    print(f'  --- fallback arm = {base_arm} ---')
    m, lo, hi = boot_gain(acc['R'] - ab)
    print(f'    always fire (R everywhere) : {m:+.3f}pp [{lo:+.3f},{hi:+.3f}]')
    for t in (0.251, 0.501, 0.751, 0.876):
        g = s_bbp8 < t
        m, lo, hi = boot_gain(np.where(g, acc['R'], ab) - ab)
        print(f'    t={t:.3f} fire {100*g.mean():5.1f}%      : {m:+.3f}pp [{lo:+.3f},{hi:+.3f}]')

print()
print('=' * 78)
print('R6. section-5 bucket table with CIs, and with the preregistered baseline B')
print('=' * 78)
def bucket(sig, ref):
    for lo, hi in [(0.0, .30), (.30, .55), (.55, .80), (.80, .999), (.999, 1.01)]:
        m = (sig >= lo) & (sig < hi)
        if m.sum() == 0:
            continue
        v = (acc['R'][m] - acc[ref][m])
        rb = np.random.default_rng(5)
        bs = [v[rb.integers(0, len(v), len(v))].mean() for _ in range(4000)]
        print(f'   [{lo:.2f},{hi:.2f}) n={m.sum():4d}  R-{ref} = {100*v.mean():+6.2f}pp '
              f'[{100*np.percentile(bs,2.5):+6.2f},{100*np.percentile(bs,97.5):+6.2f}]')
print('  by BBp k=8 agreement, vs N:'); bucket(s_bbp8, 'N')
print('  by BBp k=8 agreement, vs B (preregistered comparator):'); bucket(s_bbp8, 'B')
print('  NOISE FLOOR control: same buckets, Bp-B (identical prompt, pure decoding noise):')
for lo, hi in [(0.0, .30), (.30, .55), (.55, .80), (.80, .999), (.999, 1.01)]:
    m = (s_bbp8 >= lo) & (s_bbp8 < hi)
    if m.sum() == 0:
        continue
    v = acc['Bp'][m] - acc['B'][m]
    rb = np.random.default_rng(5)
    bs = [v[rb.integers(0, len(v), len(v))].mean() for _ in range(4000)]
    print(f'   [{lo:.2f},{hi:.2f}) n={m.sum():4d}  Bp-B  = {100*v.mean():+6.2f}pp '
          f'[{100*np.percentile(bs,2.5):+6.2f},{100*np.percentile(bs,97.5):+6.2f}]')
print('  by level, R-N vs R-B vs noise Bp-B:')
for L in (1, 2, 3, 4, 5):
    m = np.array([LEV[i] == L for i in range(N)])
    print(f'   L{L} n={m.sum():3d}  R-N {100*(acc["R"][m]-acc["N"][m]).mean():+6.2f}   '
          f'R-B {100*(acc["R"][m]-acc["B"][m]).mean():+6.2f}   Bp-B {100*(acc["Bp"][m]-acc["B"][m]).mean():+6.2f}')

print()
print('=' * 78)
print('R7. the label excludes 71/500 mixed problems -- what happens on the full set?')
print('=' * 78)
sp = np.array(ag_den36)
for nm, sig in (('BBp k=8 (avg)', s_bbp8), ('all36', sp)):
    # a) easy(6/6) vs hard(0/6), report label
    p = [sig[i] for i in range(N) if LAB_N[i] == 1]; q = [sig[i] for i in range(N) if LAB_N[i] == 0]
    a1 = auc(p, q)
    # b) realistic gate label: any correct vs none correct, all 500
    p2 = [sig[i] for i in range(N) if acc['N'][i] > 0]; q2 = [sig[i] for i in range(N) if acc['N'][i] == 0]
    a2 = auc(p2, q2)
    # c) 6/6 vs everything else (the split a gate actually has to make)
    p3 = [sig[i] for i in range(N) if acc['N'][i] >= .999]; q3 = [sig[i] for i in range(N) if acc['N'][i] < .999]
    a3 = auc(p3, q3)
    # d) restricted to L4+L5 only
    hard = [i for i in range(N) if LEV[i] in (4, 5)]
    p4 = [sig[i] for i in hard if LAB_N[i] == 1]; q4 = [sig[i] for i in hard if LAB_N[i] == 0]
    a4 = auc(p4, q4)
    # e) among the MIXED problems only: does it rank partial accuracy?
    mix = [i for i in range(N) if LAB_N[i] is None]
    rho = np.corrcoef(sig[mix], acc['N'][mix])[0, 1]
    print(f'  {nm}:  6/6>0/6 (n={len(p)}/{len(q)}) {a1:.4f} | anycorrect>allwrong (n={len(p2)}/{len(q2)}) {a2:.4f} | '
          f'6/6>rest (n={len(p3)}/{len(q3)}) {a3:.4f} | L4L5-only (n={len(p4)}/{len(q4)}) {a4:.4f} | '
          f'within-mixed pearson vs accN (n={len(mix)}) {rho:+.3f}')

print()
print('=' * 78)
print('R8. pseudo-label accuracy needs its baseline')
print('=' * 78)
single = np.mean([np.mean([GRD[a][i][j] for j in range(6)]) for a in ARMS for i in range(N)])
print(f'  single-sample accuracy (k=1, mean over 36 samples) = {single:.4f}')
print(f'  acc_N single = {acc["N"].mean():.4f}   acc_B single = {acc["B"].mean():.4f}')
print('  => report table: all36 k=8 pseudo-label .8644, BBp k=8 .8568, N k=6 .8721')
print('     lift over single-sample: all36 +%.2fpp, BBp k=8 over acc_B +%.2fpp, N k=6 over acc_N +%.2fpp'
      % (100 * (0.8644 - single), 100 * (0.8568 - acc['B'].mean()), 100 * (0.8721 - acc['N'].mean())))
# L5 label noise
m5 = np.array([LEV[i] == 5 for i in range(N)])
print(f'  L5: n={m5.sum()}  acc_N {acc["N"][m5].mean():.3f}  pseudo-label acc .68-.72 -> '
      f'label noise ~28-32% exactly where a reward would be used')

print()
print('=' * 78)
print('R9. single-draw vs 30-draw-averaged gate signal (training sees ONE draw of 8)')
print('=' * 78)
for s in range(5):
    ss = sig_single('BBp12', 8, 4242 + s)
    print(f'  seed {s}: unique values {len(np.unique(ss))}, corr with 30-avg {np.corrcoef(ss,s_bbp8)[0,1]:.3f}, '
          f'AUC {auc([ss[i] for i in range(N) if LAB_N[i]==1],[ss[i] for i in range(N) if LAB_N[i]==0]):.4f}')
print(f'  30-avg signal: unique values {len(np.unique(np.round(s_bbp8,6)))}')
