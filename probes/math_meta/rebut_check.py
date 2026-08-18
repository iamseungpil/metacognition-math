"""반박 검증 — CPU only, 읽기 전용. 정본/스크립트 수정 없음."""
import json, math, random, statistics as st
from collections import Counter

D = "/tmp/claude-587327809/-home-v-seungplee/41a99d3b-d246-48cd-b893-68375dc4e059/scratchpad/mini_abl/"
b = json.load(open(D + "splice2_4b.json"))
CONDS = b["conds"]; ps = b["per_sample"]; acc = b["per_problem"]; lv = b["levels"]; items = b["items"]
N = len(items)
HARD = [i for i, L in enumerate(lv) if L in (4, 5)]

def boot(pairs, n=10000, seed=0):
    rng = random.Random(seed); M = len(pairs)
    obs = st.mean(a - c for a, c in pairs); ds = []
    for _ in range(n):
        s = [pairs[rng.randrange(M)] for _ in range(M)]
        ds.append(st.mean(a - c for a, c in s))
    ds.sort(); return obs, ds[int(.025*n)], ds[int(.975*n)]

def con(a, c, idx): return boot([(acc[a][i], acc[c][i]) for i in idx])

print("=== 1. 팔 평균 + delta_eq + 주지표 ===")
for c in CONDS: print(f"  {c:3s} {st.mean(acc[c]):.4f}")
o,l,h = con("Bp","B",range(N)); print(f"  B'-B ALL  {o*100:+.2f} [{l*100:+.2f},{h*100:+.2f}] width {(h-l)*100:.4f}")
o2,l2,h2 = con("Bp","B",HARD);  print(f"  B'-B HARD {o2*100:+.2f} [{l2*100:+.2f},{h2*100:+.2f}] width {(h2-l2)*100:.4f}")
o3,l3,h3 = con("R","B",HARD);   print(f"  R-B  HARD {o3*100:+.2f} [{l3*100:+.2f},{h3*100:+.2f}] width {(h3-l3)*100:.4f} n={len(HARD)}")

print("\n=== 2. HARD 대비 전체 ===")
for a,c in (("N","B"),("R","N"),("R","E"),("B","E"),("E","N"),("R","S"),("N","E"),("Bp","N")):
    o,l,h = con(a,c,HARD); print(f"  {a}-{c} HARD {o*100:+6.2f} [{l*100:+6.2f},{h*100:+6.2f}]")

print("\n=== 3. 팔쌍별 Var(diff)  — 'B'만 별도 호출이라 부풀었다' 검정 ===")
def vard(a,c):
    d=[acc[a][i]-acc[c][i] for i in range(N)]; return st.pvariance(d)
pairs=[(x,y) for ix,x in enumerate(CONDS) for y in CONDS[ix+1:]]
for x,y in pairs:
    print(f"  Var({x}-{y}) = {vard(x,y):.5f}")

print("\n=== 4. 해석적 이항 바닥 ===")
K=6
def binfloor(a,c):
    s=0.0
    for i in range(N):
        pa,pc=acc[a][i],acc[c][i]
        s += pa*(1-pa)/(K-1) + pc*(1-pc)/(K-1)
    return s/N
for x,y in (("Bp","B"),("R","B"),("S","B"),("N","B"),("E","B")):
    print(f"  analytic({x},{y}) = {binfloor(x,y):.5f}   empirical Var = {vard(x,y):.5f}")

print("\n=== 5. 초과분산 + 음성대조군 (문제단위 부트) ===")
def excess_boot(a, base="B", ref="Bp", n=4000, seed=0):
    rng=random.Random(seed)
    d1=[acc[a][i]-acc[base][i] for i in range(N)]
    d0=[acc[ref][i]-acc[base][i] for i in range(N)]
    obs=st.pvariance(d1)-st.pvariance(d0); out=[]
    for _ in range(n):
        idx=[rng.randrange(N) for _ in range(N)]
        out.append(st.pvariance([d1[i] for i in idx])-st.pvariance([d0[i] for i in idx]))
    out.sort(); return obs,out[int(.025*n)],out[int(.975*n)]
for a in ("R","N","E","S"):
    o,l,h=excess_boot(a); print(f"  excess({a}-B) {o:+.5f} [{l:+.5f},{h:+.5f}]")
def diff_excess(a1,a2,n=4000,seed=0):
    rng=random.Random(seed)
    dA=[acc[a1][i]-acc["B"][i] for i in range(N)]
    dB=[acc[a2][i]-acc["B"][i] for i in range(N)]
    obs=st.pvariance(dA)-st.pvariance(dB); out=[]
    for _ in range(n):
        idx=[rng.randrange(N) for _ in range(N)]
        out.append(st.pvariance([dA[i] for i in idx])-st.pvariance([dB[i] for i in idx]))
    out.sort(); return obs,out[int(.025*n)],out[int(.975*n)]
for a2 in ("N","E","S"):
    o,l,h=diff_excess("R",a2); print(f"  excess(R-B)-excess({a2}-B) {o:+.5f} [{l:+.5f},{h:+.5f}]")

print("\n=== 6. 자기일관성 agree 정의 대조 ===")
nrm = lambda s: (s or "").replace(" ","").strip()
def agree_signal(exclude=None, arms=None):
    A = arms if arms else [c for c in CONDS if c != exclude]
    out=[]
    for i in range(N):
        c=Counter(nrm(r["pred"]) for cd in A for r in ps[cd][i] if r["pred"])
        c.pop("",None); tot=sum(c.values())
        out.append(c.most_common(1)[0][1]/tot if tot else 0.0)
    return out
def agree_denom36(arms=None):
    A = arms if arms else CONDS
    out=[]
    for i in range(N):
        allp=[nrm(r["pred"]) for cd in A for r in ps[cd][i]]
        c=Counter(p for p in allp if p); tot=len(allp)
        out.append(c.most_common(1)[0][1]/tot if c else 0.0)
    return out
def auc(p,q): return sum(1. if a>x else .5 if a==x else 0. for a in p for x in q)/(len(p)*len(q))
HI=[i for i in range(N) if acc["N"][i]>=.999]; LO=[i for i in range(N) if acc["N"][i]<=.001]
print(f"  HI={len(HI)} LO={len(LO)}")
for nm,f in (("signal_probe 정의(비어있는 pred 분모제외)", agree_signal()),
             ("분모=36 (None 포함)", agree_denom36()),
             ("N제외 30샘플 (signal 정의)", agree_signal(exclude="N")),
             ("R팔 6샘플만", agree_signal(arms=["R"])),
             ("N팔 6샘플만", agree_signal(arms=["N"]))):
    print(f"  {nm:42s} AUC {auc([f[i] for i in HI],[f[i] for i in LO]):.4f}")

print("\n=== 7. 교차팔 분리 AUC ===")
aR=agree_signal(arms=["R"]); aN=agree_signal(arms=["N"])
HIr=[i for i in range(N) if acc["R"][i]>=.999]; LOr=[i for i in range(N) if acc["R"][i]<=.001]
print(f"  pred=R일치율, label=N정오  AUC {auc([aN and aR[i] for i in HI],[aR[i] for i in LO]):.4f}  (n {len(HI)}/{len(LO)})")
print(f"  pred=N일치율, label=R정오  AUC {auc([aN[i] for i in HIr],[aN[i] for i in LOr]):.4f}  (n {len(HIr)}/{len(LOr)})")

print("\n=== 8. 절단 시뮬 ===")
def acc_cap(c, cap):
    return [st.mean(0 if r["ntok"]>cap else r["graded"] for r in ps[c][i]) for i in range(N)]
L5=[i for i,L in enumerate(lv) if L==5]
for cap in (700,900,1200,2048):
    aR_=acc_cap("R",cap); aB_=acc_cap("B",cap); aN_=acc_cap("N",cap)
    o,l,h=boot([(aR_[i],aB_[i]) for i in L5])
    print(f"  cap{cap:5d} R-B|L5 {o*100:+6.2f} [{l*100:+6.2f},{h*100:+6.2f}]   meanN {st.mean(aN_):.4f}")

print("\n=== 9. ntok / finish ===")
for c in ("R","B","N"):
    allr=[r for i in range(N) for r in ps[c][i]]
    print(f"  {c}: mean ntok {st.mean(r['ntok'] for r in allr):.1f}  len% {100*sum(1 for r in allr if r['finish']=='length')/len(allr):.2f}")
for c in ("R","B"):
    hr=[r for i in HARD for r in ps[c][i]]
    l5=[r for i in L5 for r in ps[c][i]]
    print(f"  {c}: HARD len% {100*sum(1 for r in hr if r['finish']=='length')/len(hr):.2f}  L5 len% {100*sum(1 for r in l5 if r['finish']=='length')/len(l5):.2f}  L5 boxed% {100*sum(1 for r in l5 if r['has_boxed'])/len(l5):.2f}")
