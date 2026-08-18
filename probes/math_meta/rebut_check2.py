import json, math, random, re, statistics as st
from collections import Counter
D = "/tmp/claude-587327809/-home-v-seungplee/41a99d3b-d246-48cd-b893-68375dc4e059/scratchpad/mini_abl/"
b = json.load(open(D+"splice2_4b.json"))
CONDS=b["conds"]; ps=b["per_sample"]; acc=b["per_problem"]; lv=b["levels"]; items=b["items"]; N=len(items)
HARD=[i for i,L in enumerate(lv) if L in (4,5)]
nrm=lambda s:(s or "").replace(" ","").strip()
def auc(p,q):
    return sum(1. if a>x else .5 if a==x else 0. for a in p for x in q)/(len(p)*len(q)) if p and q else None
HI=[i for i in range(N) if acc["N"][i]>=.999]; LO=[i for i in range(N) if acc["N"][i]<=.001]

print("=== A. 단일팔 일치율 — 분모 정의 두 가지 ===")
def ag(arms, denom36):
    out=[]
    for i in range(N):
        allp=[nrm(r["pred"]) for cd in arms for r in ps[cd][i]]
        c=Counter(p for p in allp if p)
        tot=len(allp) if denom36 else sum(c.values())
        out.append(c.most_common(1)[0][1]/tot if (c and tot) else 0.0)
    return out
for arms in (["R"],["N"],["B"],CONDS):
    for d36 in (False,True):
        f=ag(arms,d36)
        print(f"  arms={'+'.join(arms):20s} denom36={d36!s:5s} AUC {auc([f[i] for i in HI],[f[i] for i in LO]):.4f}")

print("\n=== B. v1 (splice_4b.json) ===")
v1=json.load(open(D+"splice_4b.json"))
print("  n_problems:",v1["n_problems"],"arms:",list(v1["per_problem"].keys()))
print("  mean_acc:",{k:round(v,4) for k,v in v1["mean_acc"].items()})
# 문제 집합 겹침
s1={ " ".join(it["problem"].split()) for it in v1["items"]}
s2={ " ".join(it["problem"].split()) for it in items}
print(f"  v1 items {len(s1)} / v2 items {len(s2)} / 교집합 {len(s1&s2)}")
# 같은 문제만으로 v2 cap700 N
common=[i for i,it in enumerate(items) if " ".join(it["problem"].split()) in s1]
def acc_cap(c,cap): return [st.mean(0 if r["ntok"]>cap else r["graded"] for r in ps[c][i]) for i in range(N)]
for cap in (700,2048):
    a=acc_cap("N",cap)
    print(f"  v2 cap{cap} N: 전체500 {st.mean(a):.4f} | v1과 공통 {len(common)}문제 {st.mean(a[i] for i in common):.4f}")
# prefix 동일성
pm1={" ".join(it["problem"].split()):it["prefix"] for it in v1["items"]}
same=sum(1 for it in items if pm1.get(" ".join(it["problem"].split()))==it["prefix"])
print(f"  prefix 문자열 동일한 문제: {same}/{len(common)}  ← v1/v2 가 같은 프리픽스인가")

print("\n=== C. split-half 정직한 오라클 (P5-a) ===")
def half(c,idx3): return [st.mean(ps[c][i][j]["graded"] for j in idx3) for i in range(N)]
def boot(pairs,n=8000,seed=0):
    rng=random.Random(seed); M=len(pairs); obs=st.mean(a-c for a,c in pairs); ds=[]
    for _ in range(n):
        s=[pairs[rng.randrange(M)] for _ in range(M)]; ds.append(st.mean(a-c for a,c in s))
    ds.sort(); return obs,ds[int(.025*n)],ds[int(.975*n)]
A=(0,1,2); Bh=(3,4,5)
for treat,base in (("R","N"),("R","B")):
    for lab,idx in (("HARD",HARD),("ALL",list(range(N)))):
        pr=[]
        for gsel,esel in ((A,Bh),(Bh,A)):
            gt,gb=half(treat,gsel),half(base,gsel); et,eb=half(treat,esel),half(base,esel)
            pr.append([( et[i] if gt[i]>gb[i] else eb[i], eb[i]) for i in idx])
        pairs=[((x[0]+y[0])/2,(x[1]+y[1])/2) for x,y in zip(*pr)]
        o,l,h=boot(pairs); print(f"  honest max({treat},{base})-{base} | {lab:4s} {o*100:+.2f} [{l*100:+.2f},{h*100:+.2f}]")
# naive
for treat,base in (("R","N"),):
    pairs=[(max(acc[treat][i],acc[base][i]),acc[base][i]) for i in HARD]
    o,l,h=boot(pairs); print(f"  naive  max({treat},{base})-{base} | HARD {o*100:+.2f} [{l*100:+.2f},{h*100:+.2f}]")

print("\n=== D. -conf / meta_len 을 v2 에서 (P3-c/P3-d 재현 주장) ===")
def get_conf(s):
    m=re.search(r"confidence:\s*([0-9.]+)",s or "");
    return float(m.group(1)) if m else None
def claims_error(s):
    m=re.search(r"decision:\s*(\w+)",s or "")
    return 1 if m else 0
conf=[get_conf(it["R"]) for it in items]
mlen=[len(it["R"].split()) for it in items]
print(f"  conf 파싱 성공 {sum(1 for c in conf if c is not None)}/{N}")
for cap in (700,900,1200,2048):
    aN=acc_cap("N",cap)
    hi=[i for i in range(N) if aN[i]>=.999 and conf[i] is not None]
    lo=[i for i in range(N) if aN[i]<=.001 and conf[i] is not None]
    print(f"  cap{cap:5d}  AUC(-conf → 틀림) {auc([-conf[i] for i in lo],[-conf[i] for i in hi]):.4f}  "
          f"AUC(meta_len → 틀림) {auc([mlen[i] for i in lo],[mlen[i] for i in hi]):.4f}  n {len(lo)}/{len(hi)}")
# alarm_probe2 와 같은 모집단(주장O·claim==1) 은 v2 에선 전부 claim=1 인가
print(f"  decision 필드 있는 행 {sum(claims_error(it['R']) for it in items)}/{N}")

print("\n=== E. 층별 (보너스 검증) split-half ===")
def strat():
    rows={}
    for gsel,esel in (((0,1,2),(3,4,5)),((3,4,5),(0,1,2))):
        gN=half("N",gsel)
        for s in range(4):
            idx=[i for i in range(N) if round(gN[i]*3)==s]
            for pair in (("R","B"),("R","N"),("R","E")):
                t,bb=half(pair[0],esel),half(pair[1],esel)
                rows.setdefault((s,pair),[]).append([(t[i],bb[i]) for i in idx])
    return rows
R=strat()
for (s,pair),lst in sorted(R.items(), key=lambda kv:(kv[0][0],str(kv[0][1]))):
    merged=lst[0]+lst[1]
    o,l,h=boot(merged,n=4000)
    print(f"  N {s}/3  {pair[0]}-{pair[1]}  n={len(lst[0])}+{len(lst[1])}  {o*100:+6.2f} [{l*100:+6.2f},{h*100:+6.2f}]")

print("\n=== F. meta_len 상관 ===")
def spear(x,y):
    def rk(v):
        o=sorted(range(len(v)),key=lambda i:v[i]); r=[0.]*len(v); i=0
        while i<len(o):
            j=i
            while j+1<len(o) and v[o[j+1]]==v[o[i]]: j+=1
            for k in range(i,j+1): r[o[k]]=(i+j)/2+1
            i=j+1
        return r
    a,bq=rk(x),rk(y); ma,mb=st.mean(a),st.mean(bq)
    num=sum((p-ma)*(q-mb) for p,q in zip(a,bq))
    den=math.sqrt(sum((p-ma)**2 for p in a)*sum((q-mb)**2 for q in bq))
    return num/den if den else 0.
Rntok=[st.mean(r["ntok"] for r in ps["R"][i]) for i in range(N)]
Rtr=[st.mean(1 if r["finish"]=="length" else 0 for r in ps["R"][i]) for i in range(N)]
print(f"  rho(meta_len, R ntok) {spear(mlen,Rntok):+.4f}")
print(f"  rho(meta_len, R 절단율) {spear(mlen,Rtr):+.4f}")
print(f"  rho(meta_len, level)  {spear(mlen,[l or 0 for l in lv]):+.4f}")
print(f"  rho(conf, level) {spear([c if c is not None else .5 for c in conf],[l or 0 for l in lv]):+.4f}")

print("\n=== G. 부트스트랩 양측 p (Holm 용) ===")
def bp(a,c,idx,n=20000,seed=1):
    rng=random.Random(seed); pr=[(acc[a][i],acc[c][i]) for i in idx]; M=len(pr)
    obs=st.mean(x-y for x,y in pr); cnt=0
    ds=[]
    for _ in range(n):
        s=[pr[rng.randrange(M)] for _ in range(M)]; ds.append(st.mean(x-y for x,y in s))
    # centered
    m=st.mean(ds); cnt=sum(1 for d in ds if abs(d-m)>=abs(obs))
    return obs,(cnt+1)/(n+1)
for a,c in (("N","B"),("B","E"),("E","N"),("R","S"),("R","B"),("R","N"),("R","E")):
    o,p=bp(a,c,HARD); print(f"  {a}-{c} HARD  {o*100:+6.2f}pp  boot p={p:.4f}")
