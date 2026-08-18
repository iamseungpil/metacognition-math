"""Self-consistency at small k: does majority agreement know difficulty without gold?
CPU only. numpy/statistics/random only."""
import json, random, re, statistics
import numpy as np

P = '/tmp/claude-587327809/-home-v-seungplee/41a99d3b-d246-48cd-b893-68375dc4e059/scratchpad/mini_abl/splice2_4b.json'
d = json.load(open(P))
PS = d['per_sample']
ARMS = d['conds']
LEV = d['levels']
N = d['n_problems']


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


# flatten samples: cls (answer class string or None), graded
CLS = {a: [[norm(s['pred']) for s in PS[a][i]] for i in range(N)] for a in ARMS}
GRD = {a: [[s['graded'] for s in PS[a][i]] for i in range(N)] for a in ARMS}

# normalizer sanity: do two correct samples in the same problem always share a class?
ok = bad = 0
for i in range(N):
    cor = [CLS[a][i][j] for a in ARMS for j in range(6) if GRD[a][i][j] == 1]
    for x in cor[1:]:
        if x == cor[0] and x is not None:
            ok += 1
        else:
            bad += 1
print(f'[normalizer] correct-pairs same class {ok}/{ok+bad} ({ok/max(1,ok+bad):.4f})')

# ---- pools -------------------------------------------------------------
# pool = list per problem of (cls, graded); NONE preds get a unique class (never agree)
def mkpool(arms):
    out = []
    for i in range(N):
        lst = []
        u = 0
        for a in arms:
            for j in range(6):
                c = CLS[a][i][j]
                if c is None:
                    c = f'__NONE{u}__'
                    u += 1
                lst.append((c, GRD[a][i][j]))
        out.append(lst)
    return out


POOL = {
    'N-only(6)': mkpool(['N']),
    'BBp-single-cond(12)': mkpool(['B', 'Bp']),
    'all36': mkpool(ARMS),
    'nonN(30)': mkpool(['E', 'B', 'Bp', 'S', 'R']),
}

# ---- labels ------------------------------------------------------------
def label_from(arms):
    """1 = all-correct (easy), 0 = all-wrong (hard), None = mixed"""
    lab = []
    for i in range(N):
        g = [GRD[a][i][j] for a in arms for j in range(6)]
        lab.append(1 if all(x == 1 for x in g) else (0 if all(x == 0 for x in g) else None))
    return lab


LAB_N = label_from(['N'])          # 382 / 47  (headline)
LAB_BBP = label_from(['B', 'Bp'])  # disjoint from N
print('[labels] LAB_N easy/hard =', sum(1 for x in LAB_N if x == 1), '/', sum(1 for x in LAB_N if x == 0))
print('[labels] LAB_BBp easy/hard =', sum(1 for x in LAB_BBP if x == 1), '/', sum(1 for x in LAB_BBP if x == 0))

acc = {a: np.array([np.mean(GRD[a][i]) for i in range(N)]) for a in ARMS}


def draw(pool_i, k, rng):
    n = len(pool_i)
    if k <= n:
        return rng.sample(pool_i, k)
    return [pool_i[rng.randrange(n)] for _ in range(k)]  # with replacement (flagged)


def consensus(sub):
    cnt = {}
    for c, g in sub:
        cnt.setdefault(c, [0, 0])
        cnt[c][0] += 1
        cnt[c][1] += g
    best = max(cnt.items(), key=lambda kv: kv[1][0])
    n = len(sub)
    agree = best[1][0] / n
    maj_correct = 1 if best[1][1] * 2 >= best[1][0] else 0  # samples in a class share an answer
    return agree, maj_correct


def auc(pos, neg):
    """P(score_pos > score_neg) with ties = 0.5"""
    pos = np.asarray(pos); neg = np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float('nan')
    a = np.concatenate([pos, neg])
    order = a.argsort(kind='mergesort')
    ranks = np.empty(len(a)); ranks[order] = np.arange(1, len(a) + 1)
    # average ranks for ties
    s = np.sort(a)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            avg = (i + 1 + j + 1) / 2
            ranks[np.isin(a, [s[i]])] = avg
        i = j + 1
    r1 = ranks[:len(pos)].sum()
    return (r1 - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


KS = [4, 6, 8, 12, 16, 24, 36]
SEEDS = 30


def run(pool_name, lab, ks):
    pool = POOL[pool_name]
    psz = len(pool[0])
    rows = []
    for k in ks:
        wr = k > psz
        aucs, pacc, plev = [], [], []
        for s in range(SEEDS):
            rng = random.Random(1000 + s * 7919)
            ag = np.empty(N); mc = np.empty(N)
            for i in range(N):
                a, m = consensus(draw(pool[i], k, rng))
                ag[i] = a; mc[i] = m
            pos = [ag[i] for i in range(N) if lab[i] == 1]
            neg = [ag[i] for i in range(N) if lab[i] == 0]
            aucs.append(auc(pos, neg))
            pacc.append(mc.mean())
            plev.append([np.mean([mc[i] for i in range(N) if LEV[i] == L]) for L in (1, 2, 3, 4, 5)])
        aucs = np.array(aucs); pacc = np.array(pacc); plev = np.array(plev)
        rows.append(dict(k=k, wr=wr,
                         auc=aucs.mean(), lo=np.percentile(aucs, 2.5), hi=np.percentile(aucs, 97.5),
                         pacc=pacc.mean(), placc_lo=np.percentile(pacc, 2.5), placc_hi=np.percentile(pacc, 97.5),
                         lev=plev.mean(0)))
    print(f'\n=== pool={pool_name} (pool size {psz}) ===')
    print(' k   repl   AUC(easy>hard)  [2.5,97.5]        pseudo-label acc  L1     L2     L3     L4     L5')
    for r in rows:
        print(f" {r['k']:<3} {'W' if r['wr'] else '-':<5}  {r['auc']:.4f}          [{r['lo']:.4f},{r['hi']:.4f}]   "
              f"{r['pacc']:.4f} [{r['placc_lo']:.3f},{r['placc_hi']:.3f}]  " +
              '  '.join(f'{x:.3f}' for x in r['lev']))
    return rows


print('\n########## 1-2. sample-size curve + pseudo-label accuracy ##########')
print('\n--- headline reproduction: label=N 6/6, score pool=all 36 arms ---')
run('all36', LAB_N, KS)
print('\n--- LEAKAGE-FREE single-condition pool: B/Bp (identical prompt, 12 rollouts), label=N 6/6 ---')
run('BBp-single-cond(12)', LAB_N, [4, 6, 8, 12])
print('\n--- N arm only (closest to training policy), label from B/Bp 12/12 (disjoint) ---')
run('N-only(6)', LAB_BBP, [4, 6])
print('   (k=8,12 from N with replacement -- NOT a real 8-rollout draw, shown only as a bound)')
run('N-only(6)', LAB_BBP, [8, 12])
print('\n--- N arm only, label = own 6/6 (CIRCULAR, for comparability only) ---')
run('N-only(6)', LAB_N, [4, 6])

# ---------------- 3. gate simulation -----------------------------------
print('\n########## 3. gate simulation:  agreement < t  =>  emit meta (use R) ##########')
accN, accR = acc['N'], acc['R']
base = accN.mean()
print(f'baseline mean acc_N = {base:.4f} ; mean acc_R = {accR.mean():.4f} ; oracle-free ceiling below')


def gate_eval(sig, tag):
    """odd/even split CV, both directions, averaged"""
    ts = np.unique(np.round(sig, 6))
    cand = sorted(set(list(ts) + [t + 1e-9 for t in ts]))
    idx = np.arange(N)
    res = []
    for fit, ev in ((idx[::2], idx[1::2]), (idx[1::2], idx[::2])):
        best_t, best_v = None, -9
        for t in cand:
            g = sig[fit] < t
            v = np.where(g, accR[fit], accN[fit]).mean() - accN[fit].mean()
            if v > best_v:
                best_v, best_t = v, t
        g = sig[ev] < best_t
        gain = np.where(g, accR[ev], accN[ev]).mean() - accN[ev].mean()
        res.append((best_t, g.mean(), best_v, gain))
    t1, f1, i1, o1 = res[0]
    t2, f2, i2, o2 = res[1]
    print(f'  {tag}: t*={t1:.3f}/{t2:.3f}  fire={100*(f1+f2)/2:.1f}%  '
          f'in-sample gain {100*(i1+i2)/2:+.3f}pp  HELD-OUT gain {100*(o1+o2)/2:+.3f}pp')
    return (o1 + o2) / 2


def sweep(sig, tag):
    print(f'  -- full-data sweep {tag} (in-sample, biased) --')
    print('     t      fire%    gain(pp)')
    for t in [0.0, 0.126, 0.251, 0.376, 0.501, 0.626, 0.751, 0.876, 1.001]:
        g = sig < t
        v = 100 * (np.where(g, accR, accN).mean() - base)
        print(f'    {t:.3f}  {100*g.mean():6.1f}   {v:+.3f}')


def sig_from(pool_name, k, seeds=30):
    pool = POOL[pool_name]
    out = np.zeros(N)
    for s in range(seeds):
        rng = random.Random(5000 + s * 104729)
        for i in range(N):
            out[i] += consensus(draw(pool[i], k, rng))[0]
    return out / seeds


print('\n[3a] signal = agreement over k=8 drawn from B/Bp (12 rollouts, single identical condition;')
print('     disjoint from both N and R, so acc_N/acc_R evaluation is unbiased)')
s_bbp8 = sig_from('BBp-single-cond(12)', 8)
sweep(s_bbp8, 'BBp k=8')
gate_eval(s_bbp8, 'CV')

print('\n[3b] signal = agreement over all 6 N samples (same samples as acc_N -> optimistic)')
s_n6 = np.array([consensus(POOL['N-only(6)'][i])[0] for i in range(N)])
sweep(s_n6, 'N k=6')
gate_eval(s_n6, 'CV')

print('\n[3c] signal = 3 of N, acc_N evaluated on the other 3 (disjoint within N)')
rng = random.Random(77)
sig3 = np.zeros(N); accN_half = np.zeros(N)
REP = 30
for s in range(REP):
    r = random.Random(900 + s)
    for i in range(N):
        idxs = list(range(6)); r.shuffle(idxs)
        a = [(CLS['N'][i][j] if CLS['N'][i][j] else f'__X{j}__', GRD['N'][i][j]) for j in idxs[:3]]
        sig3[i] += consensus(a)[0]
        accN_half[i] += np.mean([GRD['N'][i][j] for j in idxs[3:]])
sig3 /= REP; accN_half /= REP
_saveN = accN
accN = accN_half; base = accN.mean()
sweep(sig3, 'N k=3 (held-out acc_N)')
gate_eval(sig3, 'CV')
accN = _saveN; base = accN.mean()

# ---------------- 4. oracle gate ---------------------------------------
print('\n########## 4. ORACLE gates (gold known) ##########')
print(f'  per-problem argmax oracle (uses the very samples being scored): '
      f'{100*(np.maximum(accN, accR).mean()-base):+.3f}pp  (fire {100*(accR>accN).mean():.1f}%)')
gate_eval(accN + 1e-12, 'oracle: fire when acc_N low, threshold CV')
# debiased oracle: pick gate on odd half by TRUE accN, evaluate on even
print('  (the line above is the gold-difficulty oracle gate, held-out)')
best = -9; bt = None
for t in np.unique(accN):
    g = accN <= t
    v = np.where(g, accR, accN).mean() - base
    if v > best:
        best, bt = v, t
print(f'  full-data best gold-difficulty threshold acc_N<={bt:.3f}: {100*best:+.3f}pp (in-sample)')

print('\n########## 5. why the gate cannot pay: conditional meta value ##########')
print('  bucket by BBp-k=8 agreement:  n   mean acc_N  mean acc_R   R-N (pp)')
for lo, hi in [(0.0, 0.30), (0.30, 0.55), (0.55, 0.80), (0.80, 0.999), (0.999, 1.01)]:
    m = (s_bbp8 >= lo) & (s_bbp8 < hi)
    if m.sum() == 0:
        continue
    print(f'   [{lo:.2f},{hi:.2f})  {m.sum():4d}   {accN[m].mean():.4f}     {accR[m].mean():.4f}    {100*(accR[m]-accN[m]).mean():+.2f}')
print('  same, by N-k=6 agreement:')
for lo, hi in [(0.0, 0.30), (0.30, 0.55), (0.55, 0.80), (0.80, 0.999), (0.999, 1.01)]:
    m = (s_n6 >= lo) & (s_n6 < hi)
    if m.sum() == 0:
        continue
    print(f'   [{lo:.2f},{hi:.2f})  {m.sum():4d}   {accN[m].mean():.4f}     {accR[m].mean():.4f}    {100*(accR[m]-accN[m]).mean():+.2f}')
print('  by gold level:  L  n  acc_N  acc_R  R-N(pp)')
for L in (1, 2, 3, 4, 5):
    m = np.array([LEV[i] == L for i in range(N)])
    print(f'   L{L}  {m.sum():3d}  {accN[m].mean():.4f}  {accR[m].mean():.4f}  {100*(accR[m]-accN[m]).mean():+.2f}')
