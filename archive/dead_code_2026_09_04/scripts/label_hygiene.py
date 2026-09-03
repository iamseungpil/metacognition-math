"""검토관(품질검문) 조건 반영: 사이트별 위생 플래그 → hygiene.parquet
  post2     : prefix에 «도달 선언» 문자열 (식 매칭이 놓친 사후상태 위음성 보정)
  terminal  : 원본 롤아웃에서 meta 직후 120자 내 \boxed (이어생성이 사실상 결정됨)
  echo_meta : 자기 메타가 지시문 에코/placeholder
  echo_donor: ctrl 공여 메타가 에코 (노이즈 주입 오염)
stageA Δ̂ 랭킹(top/bot100) 안의 오염 수도 보고."""
import pandas as pd, numpy as np, re, glob, sys

W='cd6_work/probe'
sites = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{W}/sites_shard*.parquet'))], ignore_index=True)
sa = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f'{W}/stageA_shard*.parquet'))], ignore_index=True)

REACH = re.compile(r'(match(es)? the target|this (gives|equals|is) (us )?the target|\(success\)|'
                   r'uses? (all|each)( of)?( the)? (four|4|five|5|numbers)|exactly once.{0,40}valid|'
                   r'we have (found|reached)|which is the target|equals? the target)', re.I)
ECHO = re.compile(r'(one or two sentences|metacognitive block|meta block|do not solve the puzzle|'
                  r'do not do arithmetic|just report your|judging YOUR OWN APPROACH)', re.I)
BOXED = re.compile(r'\\boxed\{')

def tgt_reach(prefix, target):
    # "= 333" 뒤 성공 어감 근접 (target 숫자 직접 도달 선언)
    for m in re.finditer(rf'=\s*{int(target)}\b', prefix[-600:]):
        tail = prefix[-600:][m.end():m.end()+80]
        if re.search(r'(target|success|match|!|valid|correct)', tail, re.I): return 1
    return 0

rows=[]
for _, r in sites.iterrows():
    pre = r.response_text[:int(r.meta_start)]
    suf = r.response_text[int(r.meta_end):]
    rows.append(dict(site_id=r.site_id, post=int(r.post),
        post2=int(bool(REACH.search(pre[-800:])) or tgt_reach(pre, r.target)),
        terminal=int(bool(BOXED.search(suf[:120]))),
        echo_meta=int(bool(ECHO.search(str(r.meta_body)))),
        echo_donor=int(bool(ECHO.search(str(r.donor_raw))))))
hy = pd.DataFrame(rows)
hy['post_any'] = ((hy.post==1)|(hy.post2==1)).astype(int)
hy.to_parquet(f'{W}/hygiene.parquet')

# stageA Δ̂ 랭킹 (stageB 선별 로직 재현: post=0만, Δ̂=p_orig−p_ctrl)
piv = sa.pivot_table(index='site_id', columns='cond', values='correct', aggfunc='mean')
m = piv.join(hy.set_index('site_id'))
np_ = m[m.post==0].copy(); np_['dhat'] = np_['orig']-np_['ctrl']
np_ = np_.sort_values('dhat', ascending=False)
top, bot = np_.head(100), np_.tail(100)
def rep(name, d):
    print(f"{name}: post2(위음성)={int(d.post2.sum())} terminal={int(d.terminal.sum())} "
          f"echo_meta={int(d.echo_meta.sum())} echo_donor={int(d.echo_donor.sum())} "
          f"오염합계(중복제거)={int(((d.post2==1)|(d.terminal==1)|(d.echo_meta==1)|(d.echo_donor==1)).sum())}")
print(f"전체 455: post(기존)=180 post2(추가검출)={int(((hy.post==0)&(hy.post2==1)).sum())} "
      f"terminal={int(hy.terminal.sum())} echo_meta={int(hy.echo_meta.sum())} echo_donor={int(hy.echo_donor.sum())}")
rep("top100(=good 후보)", top); rep("bot100(=harm 후보)", bot)
clean = np_[(np_.post2==0)&(np_.terminal==0)&(np_.echo_meta==0)&(np_.echo_donor==0)]
print(f"비오염 non-post 사이트: {len(clean)}/{len(np_)} → clean top/bot 확보 가능량: {min(100,len(clean)//2)}")
print(f"clean 기준 top100 Δ̂평균={clean.head(100).dhat.mean():+.3f} bot100 Δ̂평균={clean.tail(100).dhat.mean():+.3f}")
