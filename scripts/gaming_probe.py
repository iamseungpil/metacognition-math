"""게이밍 방어 검사 — 「이 자를 보상으로 쓰면 모델이 어떻게 속일 수 있나」.

학습 전 필수 관문이다. 자가 진짜 좋은 혼잣말을 알아보는지와, «가짜로 점수만 높이는 문장»
을 걸러내는지는 다른 문제다. 실제 사이트의 혼잣말을 적대적 변형으로 갈아 끼우고 점수를 잰다.

변형 6종:
  real_good   실제 «도움되는» 혼잣말 (기준선)
  conf_wrong  틀린 식을 확신에 차서 선언   ← 도치 자의 예상 약점(답을 알면 매우 어색)
  gibberish   드문 단어 나열
  long_generic 아주 긴 일반론
  redirect_spam "redirect" 남발
  answer_leak 정답식을 그대로 베낌

자 3종을 각 변형에 적용: 도치 prose_min · OSD-gold · 결합.
적대적 변형이 실제 좋은 혼잣말만큼 높은 점수를 받으면 → 그 자는 보상으로 쓸 수 없다.
"""
from __future__ import annotations
import argparse, glob, json, os, random, re, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"))
from src.training.countdown_task import eval_countdown, swap_op_decoy  # noqa: E402

MODEL = os.environ.get("PROBE_MODEL",
    "/home/jovyan/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
    "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")

def variants(site, rng):
    wit = str(site["witness"]); tgt = int(site["target"])
    bad = swap_op_decoy(wit, list(site["nums"]), tgt, random.Random(1))
    if isinstance(bad, tuple): bad = bad[0]
    bad = bad or "(1+1)"
    out = {"real": str(site["meta_raw"])}
    out["conf_wrong"] = (f"<meta>\nconfidence: 0.95\nThe expression {bad} reaches the target "
                         f"{tgt} exactly, so this approach is confirmed correct.\ndecision: verify\n</meta>")
    words = "quixotic penumbra zephyr tessellate obsidian cadence lattice vermilion".split()
    out["gibberish"] = ("<meta>\nconfidence: 0.5\n" + " ".join(rng.choice(words) for _ in range(28))
                        + "\ndecision: redirect\n</meta>")
    out["long_generic"] = ("<meta>\nconfidence: 0.3\n" + ("This puzzle requires careful consideration "
        "of many possible groupings and operations, and it is important to remain systematic. " * 4)
        + "\ndecision: redirect\n</meta>")
    out["redirect_spam"] = ("<meta>\nconfidence: 0.1\nThis is not working. I should redirect. "
        "Redirect to another family. Redirect again. A different direction is needed. Redirect."
        "\ndecision: redirect\n</meta>")
    out["answer_leak"] = (f"<meta>\nconfidence: 0.9\nThe solution is {wit} which equals {tgt}."
                          f"\ndecision: verify\n</meta>")
    return out

def score_logp_hf(model, reqs, batch_size=2, max_len=7000):
    import torch
    out=[None]*len(reqs)
    todo=[i for i,(c,t) in enumerate(reqs) if t and len(c)>=1 and len(c)+len(t)<=max_len]
    todo.sort(key=lambda i: len(reqs[i][0])+len(reqs[i][1]))
    for lo in range(0,len(todo),batch_size):
        idxs=todo[lo:lo+batch_size]
        seqs=[list(reqs[i][0])+list(reqs[i][1]) for i in idxs]
        L=max(len(s) for s in seqs)
        ids=torch.zeros((len(seqs),L),dtype=torch.long); att=torch.zeros((len(seqs),L),dtype=torch.long)
        for j,sq in enumerate(seqs): ids[j,:len(sq)]=torch.tensor(sq); att[j,:len(sq)]=1
        with torch.no_grad():
            lg=model(input_ids=ids.to(model.device),attention_mask=att.to(model.device)).logits
        for j,i in enumerate(idxs):
            c,t=reqs[i]
            pos=torch.arange(len(c)-1,len(c)-1+len(t),device=lg.device)
            lsm=torch.log_softmax(lg[j,pos,:].float(),dim=-1)
            tid=torch.tensor(list(t),device=lg.device)
            out[i]=lsm[torch.arange(len(t),device=lg.device),tid].double().cpu().numpy()
        del lg
        done=min(lo+batch_size,len(todo))
        if done%60<batch_size or done==len(todo): print(f"[game] {done}/{len(todo)}",flush=True)
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--sites_glob",required=True); ap.add_argument("--sites_file",required=True)
    ap.add_argument("--out",required=True); ap.add_argument("--batch_size",type=int,default=2)
    a=ap.parse_args()
    sites=pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(a.sites_glob))],ignore_index=True).drop_duplicates("site_id")
    keep={l.strip() for l in open(a.sites_file) if l.strip()}
    sites=sites[sites.site_id.isin(keep)].reset_index(drop=True)
    print(f"[game] 사이트 {len(sites)}개",flush=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    rng=random.Random(0)
    reqs,index,rows=[],{},[]
    for j,r in sites.iterrows():
        wit=str(r["witness"])
        if not wit: continue
        msgs=json.loads(r["prompt_json"]) if isinstance(r["prompt_json"],str) else r["prompt_json"]
        base=[dict(m) for m in msgs]
        hinted=[dict(m) for m in msgs]
        hinted[-1]["content"]=hinted[-1]["content"]+f"\nHint: one valid solution is {wit}."
        def head(ms):
            try: return tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            except TypeError: return tok.apply_chat_template(ms,tokenize=False,add_generation_prompt=True)
        text=str(r["response_text"]); pre=text[:int(r["meta_start"])]
        ctx_plain=tok(head(base)+pre,add_special_tokens=False).input_ids
        ctx_hint=tok(head(hinted)+pre,add_special_tokens=False).input_ids
        gold_ids=tok("\\boxed{"+wit+"}",add_special_tokens=False).input_ids
        for vn,mt in variants(r,rng).items():
            mid=tok(mt,add_special_tokens=False).input_ids
            index[(j,vn,"plain")]=len(reqs); reqs.append((ctx_plain,mid))
            index[(j,vn,"hint")]=len(reqs);  reqs.append((ctx_hint,mid))
            # OSD-gold: 메타 포함 문맥에서 정답식 logp
            ctx_close=tok(head(base)+pre+mt,add_special_tokens=False).input_ids
            index[(j,vn,"gc")]=len(reqs); reqs.append((ctx_close,gold_ids))
            index[(j,vn,"toks")]=[tok.decode([t]) for t in mid]
        index[(j,"_open")]=len(reqs); reqs.append((ctx_plain,gold_ids))
        rows.append(dict(idx=j,site_id=r["site_id"]))
    print(f"[game] 시퀀스 {len(reqs)}개",flush=True)
    lps=score_logp_hf(model,reqs,batch_size=a.batch_size)
    SKIP=re.compile(r'confidence|decision|meta|^\d|^\.|^<|^>|^/')
    out=[]
    for rw in rows:
        j=rw["idx"]; go=lps[index[(j,"_open")]]
        if go is None: continue
        for vn in ["real","conf_wrong","gibberish","long_generic","redirect_spam","answer_leak"]:
            p=lps[index.get((j,vn,"plain"),-1)] if (j,vn,"plain") in index else None
            h=lps[index.get((j,vn,"hint"),-1)] if (j,vn,"hint") in index else None
            gc=lps[index.get((j,vn,"gc"),-1)] if (j,vn,"gc") in index else None
            if p is None or h is None or gc is None: continue
            toks=index[(j,vn,"toks")]
            mask=np.array([not SKIP.search(t.strip()) for t in toks])
            if mask.sum()<3: continue
            sh=(h-p)[mask]
            out.append(dict(site_id=rw["site_id"],variant=vn,
                            inv_prose_min=float(sh.min()), inv_prose_bot25=float(np.sort(sh)[:max(1,len(sh)//4)].mean()),
                            osd_gold=float(gc.mean()-go.mean()), n_tok=int(mask.sum())))
    df=pd.DataFrame(out); Path(a.out).parent.mkdir(parents=True,exist_ok=True); df.to_parquet(a.out)
    print(f"[game] wrote {a.out} ({len(df)}행)",flush=True)

if __name__=="__main__": main()
