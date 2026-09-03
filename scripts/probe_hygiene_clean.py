"""sites parquet → 위생 통과 site_id 목록. 사용: probe_hygiene_clean.py IN.parquet OUT.txt"""
import pandas as pd, re, sys
REACH = re.compile(r'(match(es)? the target|this (gives|equals|is) (us )?the target|\(success\)|'
                   r'uses? (all|each)( of)?( the)? (four|4|five|5|numbers)|exactly once.{0,40}valid|'
                   r'we have (found|reached)|which is the target|equals? the target)', re.I)
ECHO = re.compile(r'(one or two sentences|metacognitive block|meta block|do not solve the puzzle|'
                  r'do not do arithmetic|just report your|judging YOUR OWN APPROACH)', re.I)
BOXED = re.compile(r'\\boxed\{')
def tgt_reach(p,t):
    for m in re.finditer(rf'=\s*{int(t)}\b', p[-600:]):
        if re.search(r'(target|success|match|!|valid|correct)', p[-600:][m.end():m.end()+80], re.I): return 1
    return 0
df = pd.read_parquet(sys.argv[1]); keep=[]
for _,r in df.iterrows():
    pre=r.response_text[:int(r.meta_start)]; suf=r.response_text[int(r.meta_end):]
    if not (r.post==1 or REACH.search(pre[-800:]) or tgt_reach(pre,r.target)
            or BOXED.search(suf[:120]) or ECHO.search(str(r.meta_body)) or ECHO.search(str(r.donor_raw))):
        keep.append(r.site_id)
pd.Series(keep).to_csv(sys.argv[2], index=False, header=False)
print(f"{sys.argv[1]}: {len(df)} → clean {len(keep)} ({len(keep)/len(df):.0%})")
