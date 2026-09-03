"""진단: 같은 롤아웃에 대해 Δcert 를 «두 방식»으로 재고 대조한다.

0826 관측: 같은 데이터·같은 모델인데 관문 Δ p90=1.069, 학습 Δ p90=0.386 (2.8배).
후보는 «W 를 어떻게 토큰화하느냐» 다.

  A. 관문 방식 : W 문자열을 잘라 **다시 인코딩**, 문맥도 문자열을 다시 인코딩
  B. 학습 방식 : 응답을 **한 번만** 인코딩하고 토큰 인덱스로 잘라 쓴다

두 방식이 같은 값을 내면 원인은 다른 데 있다. 다르면 원인 확정이다.
학습·보상 코드는 건드리지 않는다.
"""
import argparse, json, sys, statistics as st
from pathlib import Path
import pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.training import countdown_rewards as cdr
from src.training.countdown_task import build_prompt, extract_expr, grade


def boxed_end(text):
    i = text.rfind("\\boxed{")
    if i < 0: return None
    d = 0
    for j in range(i + 6, len(text)):
        if text[j] == "{": d += 1
        elif text[j] == "}":
            d -= 1
            if d == 0: return j + 1
    return None


@torch.no_grad()
def mean_logp(model, ctx_ids, w_ids, dev):
    if not w_ids: return None
    ids = torch.tensor([list(ctx_ids) + list(w_ids)], device=dev)
    lg = model(ids).logits[0]
    s = len(ctx_ids) - 1
    part = lg[s:s + len(w_ids)].float().log_softmax(-1)
    tgt = torch.tensor(list(w_ids), device=dev)
    return float(part.gather(1, tgt[:, None])[:, 0].mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--w_max", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(a.model_path)
    df = pd.read_parquet(a.data).head(a.limit)
    prompts, insts = [], []
    for _, r in df.iterrows():
        inst = {"nums": [int(v) for v in r["nums"]], "target": int(r["target"]),
                "witness": r.get("witness", ""), "decoy": r.get("decoy", "")}
        msgs = build_prompt(inst, variant="new")
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
            if not m["emitted"]: continue
            rows.append({"prompt": p, "text": x.text, "t0": m["start"], "t1": m["end"],
                         "corr": int(grade(x.text, inst["nums"], inst["target"]))})
    del llm
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"[ab] 발화 롤아웃 {len(rows)}개", flush=True)

    dev = "cuda"
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=torch.bfloat16,
                                                 device_map=dev).eval()
    A, B, C = [], [], []
    for k, r in enumerate(rows):
        be = boxed_end(r["text"])
        if be is None or be <= r["t1"]: continue

        # ── A. 관문 방식: 문자열을 잘라 다시 인코딩 ────────────────────────
        w_txt = r["text"][r["t1"]:be]
        wA = tok.encode(w_txt, add_special_tokens=False)[:a.w_max]
        cA_close = tok.encode(r["prompt"] + r["text"][:r["t1"]], add_special_tokens=False)
        cA_open = tok.encode(r["prompt"] + r["text"][:r["t0"]], add_special_tokens=False)
        lc, lo = mean_logp(model, cA_close, wA, dev), mean_logp(model, cA_open, wA, dev)
        dA = None if (lc is None or lo is None) else lc - lo

        # ── B. 학습 방식: 응답을 한 번만 인코딩하고 인덱스로 자른다 ─────────
        enc = tok(r["text"], add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = list(enc["input_ids"]), list(enc["offset_mapping"])
        t0i = next((i for i, (s0, e0) in enumerate(offs) if e0 > r["t0"]), None)
        t1i = next((i for i, (s0, e0) in enumerate(offs) if s0 >= r["t1"]), None)
        bei = next((i + 1 for i, (s0, e0) in enumerate(offs) if e0 >= be), len(ids))
        if t0i is None or t1i is None or bei <= t1i: continue
        wB = ids[t1i:min(bei, t1i + a.w_max)]
        pid = tok.encode(r["prompt"], add_special_tokens=False)
        cB_close, cB_open = pid + ids[:t1i], pid + ids[:t0i]
        lc2, lo2 = mean_logp(model, cB_close, wB, dev), mean_logp(model, cB_open, wB, dev)
        dB = None if (lc2 is None or lo2 is None) else lc2 - lo2

        if dA is None or dB is None: continue
        A.append(dA); B.append(dB); C.append(r["corr"])
        if (k + 1) % 40 == 0: print(f"[ab] {k+1}/{len(rows)}", flush=True)

    def stats(v): return {"n": len(v), "mean": st.mean(v), "std": st.stdev(v) if len(v) > 1 else 0.0,
                          "p90": sorted(map(abs, v))[int(0.9 * len(v))] if v else 0.0}
    def auc(sc, lb):
        P = [s for s, l in zip(sc, lb) if l]; N = [s for s, l in zip(sc, lb) if not l]
        if not P or not N: return float("nan")
        return sum((p > n) + 0.5 * (p == n) for p in P for n in N) / (len(P) * len(N))

    out = {"gate_style": stats(A), "train_style": stats(B),
           "auc_gate_style": auc(A, C), "auc_train_style": auc(B, C),
           "n_pos": sum(C), "n_neg": len(C) - sum(C),
           "ratio_p90": (stats(A)["p90"] / stats(B)["p90"]) if stats(B)["p90"] else None}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
