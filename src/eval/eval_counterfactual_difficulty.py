"""Difficulty-stratified counterfactual eval: does the meta channel CAUSE accuracy?

For one checkpoint we decode each validation problem TWICE under identical
weights/prompt/decoding, differing only in whether the meta channel is allowed:

  arm A (with-meta):     normal greedy decode (model may emit <|meta|>...</|meta|>)
  arm B (without-meta):  the <|meta|> OPEN token is driven to -inf logit bias, so
                         no meta block can start -> pure answer path.

The causal contribution of metacognition is acc_with - acc_without, reported
STRATIFIED by difficulty (easy/medium/hard) and scenario (verify/redirect),
because meta is expected to help only where the first attempt would otherwise
fail (hard / redirect); aggregate Δ can be diluted to ~0 by easy problems.

Greedy (temperature 0) makes the two arms a paired, noise-free ablation. vLLM
batches all prompts per arm so the full 594-problem val set finishes in minutes.

Usage:
  python -m src.eval.eval_counterfactual_difficulty \
      --model_path /scratch/eval_results/merged_v2_gs50 \
      --val_parquet /scratch/metacognition/data/verl_val_meta_mix.parquet \
      --out /scratch/eval_results/cf_difficulty.jsonl \
      --tp_size 4 --max_new_tokens 2048
"""
import argparse
import json
import os

import pandas as pd
from transformers import AutoTokenizer

from src.training.rewards import _check_correctness
# Prose-leak detector (v3m). Banning only the <|meta|> TAG lets the model leak the
# reflection as plain text ("confidence: ..."), so `emitted_meta_without` (a tag
# substring test) can read 0.0 while arm B still carries meta CONTENT. Our own
# signature_suppression_ids docstring records that v3l discarded ~3/4 of CFs for
# exactly this. Imported (not re-implemented) so the regex cannot drift.
from src.training.dcpo_region import _has_meta_signature


def _messages(prom):
    return [dict(m) for m in (prom.tolist() if hasattr(prom, "tolist") else prom)]


def _load_problems(path, max_per_stratum):
    df = pd.read_parquet(path)
    rows = []
    for _, r in df.iterrows():
        t = r["split_tags"]
        rows.append({
            "messages": _messages(r["prompt"]),
            "gold": str(r["reward_model"]["ground_truth"]),
            "difficulty": t.get("difficulty", "unknown"),
            "scenario": t.get("scenario", "unknown"),
            "data_source": r["data_source"],
            "row_index": int(t.get("row_index", -1)),
        })
    if max_per_stratum and max_per_stratum > 0:
        seen = {}
        capped = []
        for row in sorted(rows, key=lambda x: (x["difficulty"], x["row_index"])):
            d = row["difficulty"]
            seen[d] = seen.get(d, 0)
            if seen[d] < max_per_stratum:
                capped.append(row)
                seen[d] += 1
        rows = capped
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--val_parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--max_per_stratum", type=int, default=0)
    # A-vs-A noise floor: decode arm A a SECOND time with identical params. Greedy
    # should be deterministic, but vLLM batching/kernel selection is not guaranteed
    # to be, and the paired Δ this eval reports is the same size as that jitter
    # (prior runs: |Δ| median 0.012). Emits correct_with2 -> |U_AA| must be < 0.005
    # for the run to be readable at all.
    ap.add_argument("--repeat_arm_a", action="store_true")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    meta_open_id = tok.convert_tokens_to_ids("<|meta|>")
    assert isinstance(meta_open_id, int) and meta_open_id >= 0, "no <|meta|> token in tokenizer"
    print(f"[cf] <|meta|> id = {meta_open_id}")

    problems = _load_problems(args.val_parquet, args.max_per_stratum)
    prompts = [
        tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True)
        for p in problems
    ]
    print(f"[cf] {len(problems)} problems")

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp_size,
              dtype="bfloat16", trust_remote_code=True, gpu_memory_utilization=0.85)

    sp_with = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)
    # arm B: ban the meta-open token by driving its logit to -inf.
    sp_without = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens,
                                logit_bias={meta_open_id: -100.0})

    print("[cf] generating arm A (with-meta) ...")
    out_with = llm.generate(prompts, sp_with)
    print("[cf] generating arm B (without-meta, meta-open banned) ...")
    out_without = llm.generate(prompts, sp_without)
    out_a2 = None
    if args.repeat_arm_a:
        print("[cf] generating arm A AGAIN (A-vs-A noise floor) ...")
        out_a2 = llm.generate(prompts, sp_with)

    n_leak_tag = n_leak_prose = 0
    with open(args.out, "w") as f:
        for i, (p, ow, oo) in enumerate(zip(problems, out_with, out_without)):
            gw = ow.outputs[0].text
            go = oo.outputs[0].text
            leak_tag = "<|meta|>" in go
            leak_prose = _has_meta_signature(go)
            n_leak_tag += int(leak_tag)
            n_leak_prose += int(leak_prose)
            rec = {
                "row_index": p["row_index"],
                "difficulty": p["difficulty"],
                "scenario": p["scenario"],
                "data_source": p["data_source"],
                "correct_with": bool(_check_correctness(gw, p["gold"])),
                "correct_without": bool(_check_correctness(go, p["gold"])),
                "emitted_meta_with": "<|meta|>" in gw,
                "emitted_meta_without": leak_tag,
                # --- added 2026-08-10 (E-190 wiring checks W-1b / W-4 / W-3) ---
                # W-1b: tag-free but meta-CONTENT-carrying arm-B rows. The 9 prior
                # runs on HF report emitted_meta_without 0/5346 but that is the TAG
                # test only; this field is the one that can falsify arm B.
                "meta_signature_with": _has_meta_signature(gw),
                "meta_signature_without": leak_prose,
                # W-4: truncation. A run where either arm hits the token cap is not
                # a clean paired ablation.
                "finish_with": ow.outputs[0].finish_reason,
                "finish_without": oo.outputs[0].finish_reason,
                "ntok_with": len(ow.outputs[0].token_ids),
                "ntok_without": len(oo.outputs[0].token_ids),
                # W-3 / re-gradability: the prior runs stored 8 scalar keys and NO
                # text, so their leak rate and grader version can never be rechecked.
                # Keep the generations with the verdict they produced.
                "gen_with": gw,
                "gen_without": go,
            }
            if out_a2 is not None:
                g2 = out_a2[i].outputs[0].text
                rec["correct_with2"] = bool(_check_correctness(g2, p["gold"]))
                rec["identical_a2"] = (g2 == gw)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n = len(problems)
    print(f"[cf] arm-B leak: tag {n_leak_tag}/{n} = {n_leak_tag/n:.4f} | "
          f"PROSE {n_leak_prose}/{n} = {n_leak_prose/n:.4f}  (gate: prose < 0.05)")
    print(f"[cf] wrote {args.out}; summarize with eval_counterfactual_difficulty_summarize.py")


if __name__ == "__main__":
    main()
