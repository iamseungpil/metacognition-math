"""읽기 전용 진단: PMI 채점기의 «자 굵기»가 AUC 를 바꾸는가.

학습·보상 코드를 일절 건드리지 않는다. 원본 모델로 롤아웃을 만들고,
같은 롤아웃에 대해 두 종류의 오답(decoy)으로 PMI-shift 를 각각 계산해
정답 여부와의 AUC 를 비교한다.

  현행 오답 : 연산자 1개 교체        → 실측 300/300 이 1토큰 차이
  대안 오답 : 숫자 두 개 자리바꿈    → 실측 평균 4~5토큰 차이

PMI_open  = logp(gold | 프롬프트)            − logp(decoy | 프롬프트)
PMI_close = logp(gold | 프롬프트+메타까지)   − logp(decoy | 같은 문맥)
shift     = PMI_close − PMI_open
"""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training import countdown_rewards as cdr
from src.training.countdown_task import build_prompt, grade


def wide_decoy(witness, nums, rng):
    """숫자 두 개를 자리바꿈한 오답. 길이를 보존하고 여러 토큰이 달라진다."""
    ns = [str(int(v)) for v in nums]
    for _ in range(40):
        a, b = rng.sample(ns, 2)
        if a != b and a in witness and b in witness:
            cand = witness.replace(a, "\x00", 1).replace(b, a, 1).replace("\x00", b, 1)
            if cand != witness:
                return cand
    return None


@torch.no_grad()
def span_logp(model, tok, ctx_ids, ans_text, device):
    """ctx 뒤에 ans 를 붙였을 때 ans 토큰들의 per-token logprob."""
    ans_ids = tok.encode(ans_text, add_special_tokens=False)
    if not ans_ids:
        return None
    ids = torch.tensor([ctx_ids + ans_ids], device=device)
    out = model(ids).logits[0, :-1].float().log_softmax(-1)
    tgt = torch.tensor(ans_ids, device=device)
    lp = out[len(ctx_ids) - 1:len(ctx_ids) - 1 + len(ans_ids)].gather(1, tgt[:, None])[:, 0]
    return lp.cpu().numpy(), ans_ids


def pmi(model, tok, ctx_ids, gold, decoy, device):
    g = span_logp(model, tok, ctx_ids, "\\boxed{%s}" % gold, device)
    d = span_logp(model, tok, ctx_ids, "\\boxed{%s}" % decoy, device)
    if g is None or d is None or len(g[1]) != len(d[1]):
        return float("nan")
    mask = np.array([a != b for a, b in zip(g[1], d[1])])
    if not mask.any():
        return float("nan")
    return float((g[0][mask] - d[0][mask]).sum())


def auc(scores, labels):
    pairs = [(s, l) for s, l in zip(scores, labels) if s == s]
    pos = [s for s, l in pairs if l], 
    P = [s for s, l in pairs if l]; N = [s for s, l in pairs if not l]
    if not P or not N:
        return float("nan"), len(P), len(N)
    wins = sum((p > n) + 0.5 * (p == n) for p in P for n in N)
    return wins / (len(P) * len(N)), len(P), len(N)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--n", type=int, default=4, help="문제당 롤아웃")
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoModelForCausalLM

    df = pd.read_parquet(a.data).head(a.limit)
    tok = AutoTokenizer.from_pretrained(a.model_path)
    rng = random.Random(0)

    prompts, insts = [], []
    for _, r in df.iterrows():
        inst = {"nums": [int(v) for v in r["nums"]], "target": int(r["target"]),
                "witness": r["witness"], "decoy": r["decoy"]}
        msgs = build_prompt(inst, variant="shot")
        try:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)
        except TypeError:
            p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        prompts.append(p); insts.append(inst)

    llm = LLM(model=a.model_path, dtype="bfloat16", seed=0, gpu_memory_utilization=0.42,
              max_model_len=a.max_tokens + 1024, enforce_eager=True)
    outs = llm.generate(prompts, SamplingParams(n=a.n, temperature=1.0,
                                                max_tokens=a.max_tokens, seed=0))
    rows = []
    for inst, p, o in zip(insts, prompts, outs):
        for x in o.outputs:
            m = cdr.parse_meta(x.text, "new")
            if not m["emitted"]:
                continue
            rows.append({"prompt": p, "pre": x.text[: m["end"] or 0],
                         "corr": int(grade(x.text, inst["nums"], inst["target"])),
                         "gold": inst["witness"], "narrow": inst["decoy"],
                         "wide": wide_decoy(inst["witness"], inst["nums"], rng)})
    del llm
    import gc, contextlib
    gc.collect(); torch.cuda.empty_cache()
    print(f"[probe] 메타 발화 롤아웃 {len(rows)}개", flush=True)

    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.bfloat16,
                                                 device_map=dev).eval()
    res = {"narrow": [], "wide": [], "corr": []}
    for i, r in enumerate(rows):
        if r["wide"] is None:
            continue
        open_ids = tok.encode(r["prompt"], add_special_tokens=False)
        close_ids = tok.encode(r["prompt"] + r["pre"], add_special_tokens=False)
        rec = {}
        for kind, dec in (("narrow", r["narrow"]), ("wide", r["wide"])):
            po = pmi(model, tok, open_ids, r["gold"], dec, dev)
            pc = pmi(model, tok, close_ids, r["gold"], dec, dev)
            rec[kind] = pc - po
        res["narrow"].append(rec["narrow"]); res["wide"].append(rec["wide"])
        res["corr"].append(r["corr"])
        if (i + 1) % 50 == 0:
            print(f"[probe] {i+1}/{len(rows)}", flush=True)

    out = {}
    for kind in ("narrow", "wide"):
        A, npos, nneg = auc(res[kind], res["corr"])
        v = [x for x in res[kind] if x == x]
        out[kind] = {"auc": A, "n_pos": npos, "n_neg": nneg, "n_scored": len(v),
                     "n_total": len(res[kind]),
                     "std": float(np.std(v)) if v else float("nan")}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
