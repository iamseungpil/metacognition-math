"""Math port of the Countdown meta-splice probe (scripts/meta_splice_probe.py).

stageA: rollouts with the <meta> prompt → harvest every <meta>…</meta> block as a
        site (problems with 0 < pass_n < n unless --keep_all) → sites.parquet,
        rollouts.parquet.
stageB: per site continue from prefix+meta (orig) and prefix with meta deleted
        (abl), k samples each with det_seed → stageB.parquet.
        Downstream: p_orig, p_abl, L2 = p_orig − p_abl.
check : offline self-checks (prompt, grader, seed disjointness). No GPU.

GPU is chosen only via CUDA_VISIBLE_DEVICES. Pure vLLM 0.10.2 offline.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
import numpy as np  # noqa: E402

W = "/home/jovyan/beomi/splee"
REPO = f"{W}/metacognition-math"
MODEL = (f"{os.path.expanduser('~')}/.cache/huggingface/hub/models--Qwen--Qwen3-4B/"
         "snapshots/1cfa9a7208912126459214e8b04321603b3df60c")
MATH_GLOBS = [f"{REPO}/hf_data/metacot-sdc-data/datasets--EleutherAI--hendrycks_math/"
              "snapshots/*/*/train-*.parquet",
              f"{os.path.expanduser('~')}/.cache/huggingface/hub/datasets--EleutherAI--"
              "hendrycks_math/snapshots/*/*/train-*.parquet"]
DAPO_GLOB = (f"{REPO}/hf_data/metacot-sdc-data/datasets--BytedTsinghua-SIA--DAPO-Math-17k/"
             "snapshots/*/data/*.parquet")

TEMPERATURE, TOP_P, TOP_K = 1.0, 1.0, -1
MAX_MODEL_LEN = 12288
ENABLE_THINKING = False

RULES = "Solve the problem step by step. Put the final answer in \\boxed{}.\n\n"
META_INSTR = (
    "At least once DURING your solution — after you have started working but "
    "BEFORE you write the final \\boxed{} answer — stop and write a metacognitive "
    "block in EXACTLY this form, on its own lines. Write it right after your first "
    "attempt or first key step, in the MIDDLE of the solution; a block written after "
    "the answer does not count and must not appear:\n\n"
    "<meta>\n"
    "confidence: <a single number between 0 and 1>\n"
    "<One or two sentences judging YOUR OWN APPROACH so far: which method you are "
    "using and whether it is worth continuing. ★Do NOT do arithmetic in here — no "
    "expressions, no equalities, no candidate answer. Assess the approach; do not "
    "solve the problem.>\n"
    "decision: verify\n"
    "</meta>\n\n"
    "Write `decision: verify` when the confidence you just wrote is high and the "
    "current approach deserves to be pushed through and checked. Write "
    "`decision: redirect` when that confidence is low and the current approach "
    "should be abandoned for a different method. The decision must follow from the "
    "confidence. Then continue solving in the way that decision commits you to. "
    "A response with no <meta> block, or with the block only after the answer, is "
    "incomplete.\n\n"
    "Your response MUST end with the final answer in \\boxed{...}. Never end "
    "without a \\boxed{...}."
)
SYS = {"meta": RULES + META_INSTR, "plain": RULES.strip()}

_META_BLOCK = re.compile(r"<meta>(.*?)</meta>", re.IGNORECASE | re.DOTALL)
_CONF = re.compile(r"confidence\s*:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
_DECISION = re.compile(r"decision\s*:\s*(verify|redirect)", re.IGNORECASE)


# ══ deterministic seeds / parsing / grading ══════════════════════════════════
def det_seed(*parts) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31 - 1)


def rollout_seeds(base_seed: int, n_prompts: int, n: int):
    """Per-prompt seeds with stride n: vLLM salts child i as seed+i, so the sets
    {s, …, s+n-1} are pairwise disjoint across prompts."""
    base = det_seed("rollout", base_seed) % (2**30)
    return [base + i * n for i in range(n_prompts)]


def assert_seeds_disjoint(seeds, n):
    seen = set()
    for s in seeds:
        blk = set(range(s, s + n))
        assert not (blk & seen), f"seed overlap at {s}"
        seen |= blk


def last_boxed(text: str):
    key = "\\boxed{"
    idx = text.rfind(key)
    while idx != -1:
        i = idx + len(key)
        depth, j = 1, i
        while j < len(text) and depth:
            depth += {"{": 1, "}": -1}.get(text[j], 0)
            j += 1
        if depth == 0:
            return text[i:j - 1].strip()
        idx = text.rfind(key, 0, idx)
    return None


def grade(text: str, gold: str) -> bool:
    try:
        from math_verify import parse, verify
        pred = last_boxed(text or "")
        if pred is None:
            return False
        g = parse(f"\\boxed{{{gold}}}") or parse(str(gold))
        p = parse(f"\\boxed{{{pred}}}")
        return bool(g and p and verify(g, p))
    except Exception:
        return False


def find_metas(text: str):
    """All <meta> blocks: (start, end, raw, confidence, decision)."""
    out = []
    for m in _META_BLOCK.finditer(text):
        c, d = _CONF.search(m.group(1)), _DECISION.search(m.group(1))
        out.append((m.start(), m.end(), m.group(0),
                    float(c.group(1)) if c else None, d.group(1).lower() if d else None))
    return out


# ══ data / model ═════════════════════════════════════════════════════════════
def load_problems(args):
    import pandas as pd
    if args.data == "math":
        files = next((sorted(glob.glob(g)) for g in MATH_GLOBS if glob.glob(g)), None)
        assert files, "hendrycks_math parquet not found"
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        lv = {f"Level {l}" for l in args.levels.split(",")}
        df = df[df["level"].isin(lv)].copy()
        df["gold"] = df["solution"].map(last_boxed)
        df = df[df["gold"].notna()]
        df = df.rename(columns={"problem": "question"})[["question", "gold"]]
    else:
        f = sorted(glob.glob(DAPO_GLOB))
        assert f, "DAPO parquet not found"
        df = pd.read_parquet(f[0], columns=["prompt", "reward_model"])
        df["question"] = df["prompt"].map(lambda p: re.sub(
            r"\n\nRemember to put your answer.*$", "", re.sub(
                r"^Solve the following math problem.*?\n\n", "", p[0]["content"],
                flags=re.S), flags=re.S))
        df["gold"] = df["reward_model"].map(lambda r: str(r["ground_truth"]))
        df = df.drop_duplicates("question")[["question", "gold"]]
    rng = np.random.RandomState(args.seed)
    sel = rng.choice(len(df), size=min(args.n_problems, len(df)), replace=False)
    return [{"prob_idx": int(df.index[i]), "question": df.iloc[i]["question"],
             "gold": df.iloc[i]["gold"]} for i in sel]


def build_llm(args):
    from vllm import LLM
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed,
              gpu_memory_utilization=float(os.environ.get("PROBE_GPU_UTIL", "0.85")), max_model_len=MAX_MODEL_LEN,
              # torch.compile/cudagraph capture dies silently on this shared box
              # (2026-09-02 smoke); eager unless --compile.
              enforce_eager=not args.compile)
    return llm, llm.get_tokenizer()


def messages(variant: str, question: str):
    return [{"role": "system", "content": SYS[variant]},
            {"role": "user", "content": question}]


def chat_text(tok, msgs) -> str:
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=ENABLE_THINKING)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ══ stageA ═══════════════════════════════════════════════════════════════════
def stage_a(args):
    import pandas as pd
    from vllm import SamplingParams
    probs = load_problems(args)
    seeds = rollout_seeds(args.seed, len(probs), args.n)
    assert_seeds_disjoint(seeds, args.n)
    print(f"[stageA] {len(probs)} problems × n={args.n} · variant={args.variant} "
          f"· data={args.data}", flush=True)
    llm, tok = build_llm(args)
    prompts = [chat_text(tok, messages(args.variant, p["question"])) for p in probs]
    params = [SamplingParams(n=args.n, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                             max_tokens=args.max_tokens, seed=s) for s in seeds]
    t0 = time.time()
    outs = llm.generate(prompts, params)
    gen_tok = sum(len(x.token_ids) for o in outs for x in o.outputs)
    print(f"[stageA] generated {gen_tok} tokens in {time.time()-t0:.0f}s "
          f"({gen_tok/(time.time()-t0):.0f} tok/s)", flush=True)

    rollouts, sites = [], []
    for p, o in zip(probs, outs):
        texts = [x.text for x in o.outputs]
        corr = [int(grade(t, p["gold"])) for t in texts]
        nc = sum(corr)
        for ri, (t, x) in enumerate(zip(texts, o.outputs)):
            metas = find_metas(t)
            rollouts.append({"prob_idx": p["prob_idx"], "rollout_idx": ri, "correct": corr[ri],
                             "n_correct": nc, "n_tok": len(x.token_ids),
                             "has_meta": int(bool(metas)), "text": t[:6000]})
            if not args.keep_all and not (0 < nc < args.n):
                continue
            for mi, (s, e, raw, conf, dec) in enumerate(metas):
                sites.append({
                    "site_id": f"p{p['prob_idx']}_r{ri}_m{mi}", "prob_idx": p["prob_idx"],
                    "rollout_idx": ri, "meta_idx": mi, "gold": p["gold"], "n_correct": nc,
                    "meta_raw": raw, "confidence": conf, "decision": dec,
                    "meta_start": s, "meta_end": e, "pos": e / max(1, len(t)),
                    "boxed_in_prefix": int(last_boxed(t[:s]) is not None),
                    "response_text": t,
                    "prompt_json": json.dumps(messages(args.variant, p["question"]),
                                              ensure_ascii=False),
                    "correct": corr[ri]})
    tag = "" if args.variant == "meta" else f"_{args.variant}"
    pd.DataFrame(rollouts).to_parquet(args.out / f"rollouts{tag}.parquet")
    pd.DataFrame(sites).to_parquet(args.out / f"sites{tag}.parquet")
    ncs = [r["n_correct"] for r in rollouts[::args.n]]
    print(f"[stageA] pass counts per problem: {ncs}", flush=True)
    print(f"[stageA] rollouts with meta: {sum(r['has_meta'] for r in rollouts)}/{len(rollouts)} "
          f"· sites harvested: {len(sites)} → {args.out}/sites{tag}.parquet", flush=True)
    if sites:
        print(f"[stageA] sample meta_raw:\n{sites[0]['meta_raw']}", flush=True)


# ══ stageB ═══════════════════════════════════════════════════════════════════
def prefix_of(site, cond: str) -> str:
    t, s, e = site["response_text"], int(site["meta_start"]), int(site["meta_end"])
    return t[:e] if cond == "orig" else t[:s]


def stage_b(args):
    import pandas as pd
    from vllm import SamplingParams, TokensPrompt
    sites = pd.read_parquet(args.out / "sites.parquet")
    # ★감사(0902): 답 뒤 메타(boxed_in_prefix) 제외 + 위치 밴드 0.05–0.85 (참조 구현 L308-323 과 동형)
    n0 = len(sites)
    sites = sites[(sites.boxed_in_prefix == 0) & (sites.pos >= 0.05) & (sites.pos <= 0.85)]
    print(f"[stageB] 사이트 선별 {n0} → {len(sites)} (post 제외, pos 0.05–0.85)", flush=True)
    if args.max_sites:
        sites = sites.head(args.max_sites)
    conds = [c for c in args.conds.split(",") if c]
    print(f"[stageB] {len(sites)} sites × {conds} × k={args.k}", flush=True)
    llm, tok = build_llm(args)
    prompts, params, keep, cache, skipped = [], [], [], {}, 0
    for _, row in sites.iterrows():
        site = row.to_dict()
        key = site["prompt_json"]
        if key not in cache:
            cache[key] = tok.encode(chat_text(tok, json.loads(key)), add_special_tokens=False)
        for cond in conds:
            ids = cache[key] + tok.encode(prefix_of(site, cond), add_special_tokens=False)
            if len(ids) + args.max_new >= MAX_MODEL_LEN:
                skipped += args.k
                continue
            for k in range(args.k):
                # paired CRN: same (site, k) seed across conds, as in the Countdown probe.
                prompts.append(TokensPrompt(prompt_token_ids=ids))
                params.append(SamplingParams(n=1, temperature=TEMPERATURE, top_p=TOP_P,
                                             top_k=TOP_K, max_tokens=args.max_new,
                                             seed=det_seed(site["site_id"], "B", k)))
                keep.append((site, cond, k))
    print(f"[stageB] {len(keep)} requests ({skipped} skipped: context overflow)", flush=True)
    t0 = time.time()
    outs = llm.generate(prompts, params) if keep else []
    rows = [{"site_id": site["site_id"], "cond": cond, "k": k,
             "correct": int(grade(o.outputs[0].text, site["gold"])),
             "gen_len": len(o.outputs[0].token_ids)}
            for (site, cond, k), o in zip(keep, outs)]
    df = pd.DataFrame(rows)
    df.to_parquet(args.out / "stageB.parquet")
    print(f"[stageB] {time.time()-t0:.0f}s · wrote {args.out}/stageB.parquet ({len(df)} rows)",
          flush=True)
    if len(df):
        p = df.groupby(["site_id", "cond"])["correct"].mean().unstack("cond")
        print(p.to_string(), flush=True)


# ══ check ════════════════════════════════════════════════════════════════════
def check(args):
    assert "<meta>" not in SYS["plain"] and "<meta>" in SYS["meta"]
    assert grade("\\boxed{\\frac{1}{2}}", "1/2") is True
    assert grade("\\boxed{3}", "4") is False
    assert grade("no answer", "4") is False
    seeds = rollout_seeds(args.seed, 300, 8)
    assert_seeds_disjoint(seeds, 8)
    assert det_seed("x", "B", 0) != det_seed("x", "B", 1)
    m = find_metas("a<meta>\nconfidence: 0.7\nok\ndecision: redirect\n</meta>b<META></META>")
    assert len(m) == 2 and m[0][3] == 0.7 and m[0][4] == "redirect"
    print("[check] all self-checks passed", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["A", "B", "check"], required=True)
    ap.add_argument("--data", choices=["math", "dapo"], default="math")
    ap.add_argument("--levels", default="3,4,5")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--variant", choices=["meta", "plain"], default="meta")
    ap.add_argument("--out", type=Path, default=Path(f"{W}/cd6_work/probe_math"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true", help="disable enforce_eager")
    ap.add_argument("--n_problems", type=int, default=300)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max_tokens", type=int, default=6144)
    ap.add_argument("--keep_all", action="store_true")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--conds", default="orig,abl")
    ap.add_argument("--max_new", type=int, default=4096)
    ap.add_argument("--max_sites", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    {"A": stage_a, "B": stage_b, "check": check}[args.stage](args)


if __name__ == "__main__":
    main()
