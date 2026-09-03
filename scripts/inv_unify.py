"""도치 자 «정의 통일» — 검정과 게이밍 검사가 서로 다른 통계를 쓴 문제(ρ=0.296)를 끝낸다.

두 정의를 같은 코드·같은 문맥에서 동시에 계산해 어느 것이 라벨을 가르는지, 그리고
단측 페널티 형태에서 게이밍이 막히는지를 한 번에 잰다.

  A2D  agg(gold) − agg(plain)        «집계를 먼저, 차이를 나중»  (reverse_ruler 계열)
  D2A  agg(gold − plain)             «차이를 먼저, 집계를 나중»  (gaming_probe 계열)
집계 4종: min · bot25 · mean · median.  프로즈 토큰만(conf/decision 줄 제외) + 전체 둘 다.

출력 inv_unified.parquet: [site_id, variant, A2D_*, D2A_*] · variant 는 real/적대 5종.
"""
from __future__ import annotations
import argparse, glob, json, os, random, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"))
from src.training.countdown_task import swap_op_decoy  # noqa: E402
MODEL = os.environ.get("PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
AGG = {"min": lambda a: float(a.min()),
       "bot25": lambda a: float(np.sort(a)[:max(1, len(a)//4)].mean()),
       "mean": lambda a: float(a.mean()),
       "median": lambda a: float(np.median(a))}
SKIP = re.compile(r'confidence|decision|meta|^\d|^\.|^<|^>|^/')

def variants(site, rng):
    wit = str(site["witness"]); tgt = int(site["target"])
    bad = swap_op_decoy(wit, list(site["nums"]), tgt, random.Random(1))
    if isinstance(bad, tuple): bad = bad[0]
    bad = bad or "(1+1)"
    w = "quixotic penumbra zephyr tessellate obsidian cadence lattice vermilion".split()
    return {
      "real": str(site["meta_raw"]),
      "conf_wrong": f"<meta>\nconfidence: 0.95\nThe expression {bad} reaches the target {tgt} exactly, so this approach is confirmed correct.\ndecision: verify\n</meta>",
      "gibberish": "<meta>\nconfidence: 0.5\n" + " ".join(rng.choice(w) for _ in range(28)) + "\ndecision: redirect\n</meta>",
      "long_generic": "<meta>\nconfidence: 0.3\n" + ("This puzzle requires careful consideration of many possible groupings and operations, and it is important to remain systematic. "*4) + "\ndecision: redirect\n</meta>",
      "redirect_spam": "<meta>\nconfidence: 0.1\nThis is not working. I should redirect. Redirect to another family. Redirect again. A different direction is needed. Redirect.\ndecision: redirect\n</meta>",
      "answer_leak": f"<meta>\nconfidence: 0.9\nThe solution is {wit} which equals {tgt}.\ndecision: verify\n</meta>"}

def score(model, reqs, bs=2, maxlen=7000):
    import torch
    out=[None]*len(reqs)
    todo=[i for i,(c,t) in enumerate(reqs) if t and len(c)>=1 and len(c)+len(t)<=maxlen]
    todo.sort(key=lambda i: len(reqs[i][0])+len(reqs[i][1]))
    for lo in range(0,len(todo),bs):
        idx=todo[lo:lo+bs]; seqs=[list(reqs[i][0])+list(reqs[i][1]) for i in idx]
        L=max(len(s) for s in seqs)
        ids=torch.zeros((len(seqs),L),dtype=torch.long); att=torch.zeros((len(seqs),L),dtype=torch.long)
        for j,s in enumerate(seqs): ids[j,:len(s)]=torch.tensor(s); att[j,:len(s)]=1
        with torch.no_grad():
            lg=model(input_ids=ids.to(model.device),attention_mask=att.to(model.device)).logits
        for j,i in enumerate(idx):
            c,t=reqs[i]; pos=torch.arange(len(c)-1,len(c)-1+len(t),device=lg.device)
            lsm=torch.log_softmax(lg[j,pos,:].float(),dim=-1); tid=torch.tensor(list(t),device=lg.device)
            out[i]=lsm[torch.arange(len(t),device=lg.device),tid].double().cpu().numpy()
        del lg
        d=min(lo+bs,len(todo))
        if d%80<bs or d==len(todo): print(f"[inv] {d}/{len(todo)}",flush=True)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sites_glob",required=True); ap.add_argument("--sites_file",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--batch_size",type=int,default=2)
    a=ap.parse_args()
    S=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(a.sites_glob))],ignore_index=True).drop_duplicates("site_id")
    keep={l.strip() for l in open(a.sites_file) if l.strip()}
    S=S[S.site_id.isin(keep)].reset_index(drop=True)
    print(f"[inv] 사이트 {len(S)}",flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    rng=random.Random(0); reqs,index,rows=[],{},[]
    for j,r in S.iterrows():
        wit=str(r["witness"])
        if not wit: continue
        msgs=json.loads(r["prompt_json"]) if isinstance(r["prompt_json"],str) else r["prompt_json"]
        base=[dict(m) for m in msgs]; hint=[dict(m) for m in msgs]
        hint[-1]["content"]+=f"\nHint: one valid solution is {wit}."
        def head(ms):
            try: return tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            except TypeError: return tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=True)
        pre=str(r["response_text"])[:int(r["meta_start"])]
        cp=tok(head(base)+pre,add_special_tokens=False).input_ids
        ch=tok(head(hint)+pre,add_special_tokens=False).input_ids
        for vn,mt in variants(r,rng).items():
            mid=tok(mt,add_special_tokens=False).input_ids
            index[(j,vn,"p")]=len(reqs); reqs.append((cp,mid))
            index[(j,vn,"h")]=len(reqs); reqs.append((ch,mid))
            index[(j,vn,"tk")]=[tok.decode([t]) for t in mid]
        rows.append(dict(idx=j,site_id=r["site_id"]))
    print(f"[inv] 시퀀스 {len(reqs)}",flush=True)
    lps=score(model,reqs,a.batch_size)
    out=[]
    for rw in rows:
        j=rw["idx"]
        for vn in ["real","conf_wrong","gibberish","long_generic","redirect_spam","answer_leak"]:
            p=lps[index[(j,vn,"p")]]; h=lps[index[(j,vn,"h")]]
            if p is None or h is None: continue
            tk=index[(j,vn,"tk")]; m=np.array([not SKIP.search(t.strip()) for t in tk])
            if m.sum()<3: continue
            rec=dict(site_id=rw["site_id"],variant=vn,n_tok=int(m.sum()))
            for scope,sel in [("prose",m),("all",np.ones(len(tk),bool))]:
                pp,hh=p[sel],h[sel]
                for an,af in AGG.items():
                    rec[f"A2D_{scope}_{an}"]=af(hh)-af(pp)      # 집계 먼저, 차이 나중
                    rec[f"D2A_{scope}_{an}"]=af(hh-pp)          # 차이 먼저, 집계 나중
            out.append(rec)
    df=pd.DataFrame(out); Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_parquet(a.out)
    print(f"[inv] wrote {a.out} ({len(df)})",flush=True)
if __name__=="__main__": main()
