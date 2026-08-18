"""Rebuttal part 2: prefix confound, prereg comparator, label-set effects. CPU only."""
import json, random, re
import numpy as np
from collections import Counter

BASE = '/tmp/claude-587327809/-home-v-seungplee/41a99d3b-d246-48cd-b893-68375dc4e059/scratchpad/mini_abl/'
d = json.load(open(BASE + 'splice2_4b.json'))
PS = d['per_sample']; ARMS = d['conds']; LEV = d['levels']; N = d['n_problems']; IT = d['items']


def norm(p):
    if p is None:
        return None
    s = p.strip()
    for a, b in (('\\left', ''), ('\\right', ''), ('\\!', ''), ('\\,', ''), ('\\;', ''),
                 ('\\ ', ''), ('dfrac', 'frac'), ('tfrac', 'frac'), ('^\\circ', ''),
                 ('\\%', ''), ('%', ''), ('\\$', ''), ('$', '')):
        s = s.replace(a, b)
    s = re.sub(r'\s+', '', s).rstrip('.')
    s = re.sub(r'^\\text\{(.*)\}$', r'\1', s).replace('{', '').replace('}', '')
    if s.startswith('\\text'):
        s = s[5:]
    return s or None


CLS = {a: [[norm(s['pred']) for s in PS[a][i]] for i in range(N)] for a in ARMS}
GRD = {a: [[s['graded'] for s in PS[a][i]] for i in range(N)] for a in ARMS}
acc = {a: np.array([np.mean(GRD[a][i]) for i in range(N)]) for a in ARMS}


def auc(pos, neg):
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    a = np.concatenate([pos, neg]); o = a.argsort(kind='mergesort')
    r = np.empty(len(a)); r[o] = np.arange(1, len(a) + 1)
    s = np.sort(a); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            r[np.isin(a, [s[i]])] = (i + 1 + j + 1) / 2
        i = j + 1
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def bootci(f, npos, nneg, pos, neg, nb=2000, seed=3):
    rb = np.random.default_rng(seed); pos = np.asarray(pos); neg = np.asarray(neg)
    v = [auc(pos[rb.integers(0, len(pos), len(pos))], neg[rb.integers(0, len(neg), len(neg))]) for _ in range(nb)]
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


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


POOL = {'N6': mkpool(['N']), 'BBp12': mkpool(['B', 'Bp']), 'all36': mkpool(ARMS)}


def label_from(arms):
    lab = []
    for i in range(N):
        g = [GRD[a][i][j] for a in arms for j in range(6)]
        lab.append(1 if all(x == 1 for x in g) else (0 if all(x == 0 for x in g) else None))
    return lab


LAB_N = label_from(['N']); LAB_BBP = label_from(['B', 'Bp'])


def consensus(sub):
    cnt = {}
    for c, g in sub:
        cnt.setdefault(c, [0, 0]); cnt[c][0] += 1; cnt[c][1] += g
    b = max(cnt.items(), key=lambda kv: kv[1][0])
    return b[1][0] / len(sub), (1 if b[1][1] * 2 >= b[1][0] else 0)


print('=' * 78)
print('R10. report claim: "N arm k=6 AUC 0.8436 also clears 0.80" -- problem bootstrap')
print('=' * 78)
ag_n6 = np.array([consensus(POOL['N6'][i])[0] for i in range(N)])
p = [ag_n6[i] for i in range(N) if LAB_BBP[i] == 1]; q = [ag_n6[i] for i in range(N) if LAB_BBP[i] == 0]
lo, hi = bootci(None, None, None, p, q)
print(f'  N6 vs LAB_BBp: AUC {auc(p,q):.4f}  PROBLEM-boot [{lo:.4f},{hi:.4f}]  n={len(p)}/{len(q)}   '
      f'-> lower bound {"CLEARS" if lo>0.8 else "DOES NOT CLEAR"} 0.80')

print()
print('=' * 78)
print('R11. empty/no-boxed preds are DIFFICULTY-CARRYING; the agreement denominator choice')
print('    turns "no answer" into "disagreement" and that is what moves the AUC')
print('=' * 78)
empt = np.array([sum(1 for a in ARMS for j in range(6) if CLS[a][i][j] is None) for i in range(N)])
print(f'  mean empty-pred count per problem: easy(6/6) {empt[[i for i in range(N) if LAB_N[i]==1]].mean():.3f}  '
      f'hard(0/6) {empt[[i for i in range(N) if LAB_N[i]==0]].mean():.3f}  '
      f'mixed {empt[[i for i in range(N) if LAB_N[i] is None]].mean():.3f}')
print(f'  AUC of "-empty count" alone (no agreement at all): '
      f'{auc([-empt[i] for i in range(N) if LAB_N[i]==1],[-empt[i] for i in range(N) if LAB_N[i]==0]):.4f}')
trunc = np.array([sum(1 for a in ARMS for j in range(6) if PS[a][i][j]['finish'] == 'length') for i in range(N)])
print(f'  AUC of "-truncation count" alone: '
      f'{auc([-trunc[i] for i in range(N) if LAB_N[i]==1],[-trunc[i] for i in range(N) if LAB_N[i]==0]):.4f}')
ntok = np.array([np.mean([PS[a][i][j]['ntok'] for a in ARMS for j in range(6)]) for i in range(N)])
print(f'  AUC of "-mean continuation length" alone: '
      f'{auc([-ntok[i] for i in range(N) if LAB_N[i]==1],[-ntok[i] for i in range(N) if LAB_N[i]==0]):.4f}')

print()
print('=' * 78)
print('R12. PREFIX CONFOUND: all 36 samples continue ONE shared 60%-complete solution')
print('=' * 78)
plen = np.array([len(IT[i]['prefix']) for i in range(N)])
print(f'  prefix chars: median {np.median(plen):.0f}  mean {plen.mean():.0f}')
print(f'  mean continuation tokens: {ntok.mean():.0f}')
ag36 = np.array([consensus(POOL['all36'][i])[0] for i in range(N)])
print(f'  unanimity rate (agreement==1.0) in splice pool: all36 {np.mean(ag36>=0.999):.3f}  '
      f'BBp12 {np.mean(np.array([consensus(POOL["BBp12"][i])[0] for i in range(N)])>=0.999):.3f}')
# independent full rollouts, no shared prefix: pilot 200 problems x 4 samples
pil = json.load(open(BASE + 'pilot_Qwen_Qwen3-4B.json'))
by = {}
for r in pil['records']:
    by.setdefault(r['problem'], []).append(r)
probs = [k for k, v in by.items() if len(v) == 4]
gr = {}; cl = {}
for k in probs:
    g = norm(by[k][0]['gold'])
    cl[k] = [norm(r['answer']) for r in by[k]]
    gr[k] = [1 if (c is not None and c == g) else 0 for c in cl[k]]
accp = np.array([np.mean(gr[k]) for k in probs])
print(f'  [independent rollouts, pilot k=4, n={len(probs)}] exact-match acc {accp.mean():.4f}')
lab_p = [1 if all(x == 1 for x in gr[k]) else (0 if all(x == 0 for x in gr[k]) else None) for k in probs]
agp = []
for k in probs:
    c = Counter([x if x is not None else f'__E{j}__' for j, x in enumerate(cl[k])])
    agp.append(c.most_common(1)[0][1] / 4)
agp = np.array(agp)
print(f'  unanimity rate at k=4, INDEPENDENT rollouts: {np.mean(agp>=0.999):.3f}   '
      f'vs splice all36 k=4 / BBp k=4 below')
for pn in ('all36', 'BBp12', 'N6'):
    rng = random.Random(11)
    u = np.mean([consensus(rng.sample(POOL[pn][i], 4))[0] >= 0.999 for i in range(N)])
    print(f'     splice {pn} k=4 unanimity {u:.3f}')
pp = [agp[i] for i in range(len(probs)) if lab_p[i] == 1]
qq = [agp[i] for i in range(len(probs)) if lab_p[i] == 0]
lo, hi = bootci(None, None, None, pp, qq)
print(f'  AUC(4/4correct > 4/4wrong) on INDEPENDENT rollouts: {auc(pp,qq):.4f} [{lo:.4f},{hi:.4f}]  n={len(pp)}/{len(qq)}')
print('  (caveat: different prompt (meta-instruction), 200 problems, exact-match grading only -- '
      'not a matched control, but it is the only non-spliced k>1 data in this directory)')

print()
print('=' * 78)
print('R13. held-out CV gate with the PREREGISTERED comparator B instead of N')
print('=' * 78)


def sig_avg(poolname, k, seeds=30):
    pool = POOL[poolname]; out = np.zeros(N)
    for s in range(seeds):
        rng = random.Random(5000 + s * 104729)
        for i in range(N):
            out[i] += consensus(rng.sample(pool[i], k) if k <= len(pool[i])
                                else [pool[i][rng.randrange(len(pool[i]))] for _ in range(k)])[0]
    return out / seeds


s8 = sig_avg('BBp12', 8)


def gate_cv(sig, fallback, treat='R'):
    ts = np.unique(np.round(sig, 6)); cand = sorted(set(list(ts) + [t + 1e-9 for t in ts]))
    idx = np.arange(N); res = []
    fb = acc[fallback]; tr = acc[treat]
    for fit, ev in ((idx[::2], idx[1::2]), (idx[1::2], idx[::2])):
        bt, bv = None, -9
        for t in cand:
            g = sig[fit] < t
            v = np.where(g, tr[fit], fb[fit]).mean() - fb[fit].mean()
            if v > bv:
                bv, bt = v, t
        g = sig[ev] < bt
        res.append((bt, g.mean(), bv, np.where(g, tr[ev], fb[ev]).mean() - fb[ev].mean()))
    return res


for fb in ('N', 'B'):
    r = gate_cv(s8, fb)
    print(f'  fallback={fb}: t*={r[0][0]:.3f}/{r[1][0]:.3f} fire={100*(r[0][1]+r[1][1])/2:.1f}% '
          f'in-sample {100*(r[0][2]+r[1][2])/2:+.3f}pp  HELD-OUT {100*(r[0][3]+r[1][3])/2:+.3f}pp')
print('  PLACEBO: same machinery, treat=Bp (identical prompt to B) -- any "gain" here is pure noise')
r = gate_cv(s8, 'B', treat='Bp')
print(f'  fallback=B treat=Bp: t*={r[0][0]:.3f}/{r[1][0]:.3f} fire={100*(r[0][1]+r[1][1])/2:.1f}% '
      f'in-sample {100*(r[0][2]+r[1][2])/2:+.3f}pp  HELD-OUT {100*(r[0][3]+r[1][3])/2:+.3f}pp')
r = gate_cv(s8, 'N', treat='S')
print(f'  PLACEBO fallback=N treat=S (word-shuffled meta): HELD-OUT {100*(r[0][3]+r[1][3])/2:+.3f}pp')
r = gate_cv(s8, 'N', treat='E')
print(f'  PLACEBO fallback=N treat=E (empty meta tag):     HELD-OUT {100*(r[0][3]+r[1][3])/2:+.3f}pp')
r = gate_cv(s8, 'N', treat='B')
print(f'  PLACEBO fallback=N treat=B (boilerplate meta):   HELD-OUT {100*(r[0][3]+r[1][3])/2:+.3f}pp')

print()
print('=' * 78)
print('R14. section-4 oracle gate is the v1 contamination the prereg banned')
print('    (select on acc_N, then evaluate acc_N on the selected subset)')
print('=' * 78)
for t in (0.001, 0.334, 0.667):
    m = acc['N'] <= t
    if m.sum() == 0:
        continue
    print(f'  acc_N<={t:.3f}: n={m.sum():3d}  acc_N {acc["N"][m].mean():.4f}  acc_R {acc["R"][m].mean():.4f} '
          f'(R-N {100*(acc["R"][m]-acc["N"][m]).mean():+.2f}pp)  acc_S {acc["S"][m].mean():.4f} '
          f'(S-N {100*(acc["S"][m]-acc["N"][m]).mean():+.2f}pp)  acc_B {acc["B"][m].mean():.4f} '
          f'(B-N {100*(acc["B"][m]-acc["N"][m]).mean():+.2f}pp)  acc_E {acc["E"][m].mean():.4f} '
          f'(E-N {100*(acc["E"][m]-acc["N"][m]).mean():+.2f}pp)')
print('  if S/B/E rise as much as R on the acc_N-selected subset, the "gain" is regression, not meta.')
