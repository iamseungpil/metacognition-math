"""pass@k headroom eval — how much accuracy is RECOVERABLE by selecting among the
model's own samples (the ceiling any verification/self-consistency could capture).

For each problem we draw k samples (temperature>0) and report, stratified by
difficulty / data_source:
  pass@1   = mean per-sample correctness (single-shot accuracy)
  pass@k   = fraction of problems with >=1 correct sample (PERFECT-verifier ceiling)
  maj@k    = majority-vote (boxed) answer correct (naive self-consistency)
The gap (pass@k - pass@1) is the metacognition HEADROOM: large => verification can
help (worth a data redesign); small => the model knows-it-or-not (task-fit limit).

Reuses _load_problems + _check_correctness from the cf eval / rewards. vLLM.
"""
import argparse
import json
import re
from collections import Counter, defaultdict

from src.eval.eval_counterfactual_difficulty import _load_problems
from src.training.rewards import _check_correctness

_BOXED = re.compile(r"\\boxed\{([^{}]+)\}")
_ANS_IS = re.compile(r"answer is\s*\$?\\?\(?\s*([^\$\n.]+)")


def _extract(text):
    m = list(_BOXED.finditer(text or ""))
    if m:
        return m[-1].group(1).strip()
    m = list(_ANS_IS.finditer(text or ""))
    if m:
        return m[-1].group(1).strip().rstrip("$).")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--val_parquet", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max_new_tokens", type=int, default=16384)
    ap.add_argument("--tp_size", type=int, default=4)
    ap.add_argument("--max_per_stratum", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    problems = _load_problems(args.val_parquet, args.max_per_stratum)
    print(f"[passk] {len(problems)} problems, k={args.k} T={args.temperature}")

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts = [
        tok.apply_chat_template(p["messages"], tokenize=False, add_generation_prompt=True)
        for p in problems
    ]

    llm = LLM(model=args.model_path, tensor_parallel_size=args.tp_size,
              dtype="bfloat16", trust_remote_code=True, gpu_memory_utilization=0.85)
    sp = SamplingParams(n=args.k, temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_new_tokens)
    print("[passk] generating ...")
    outs = llm.generate(prompts, sp)

    recs = []
    for p, o in zip(problems, outs):
        texts = [c.text for c in o.outputs]
        corr = [bool(_check_correctness(t, p["gold"])) for t in texts]
        # majority vote on extracted boxed answers
        ans = [a for a in (_extract(t) for t in texts) if a]
        maj_correct = False
        if ans:
            maj = Counter(ans).most_common(1)[0][0]
            maj_correct = bool(_check_correctness(f"\\boxed{{{maj}}}", p["gold"]))
        recs.append({
            "difficulty": p["difficulty"], "data_source": p["data_source"],
            "n": len(corr), "n_correct": sum(corr),
            "pass1": sum(corr) / max(1, len(corr)),
            "passk": 1.0 if any(corr) else 0.0,
            "majk": 1.0 if maj_correct else 0.0,
        })
    with open(args.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def agg(rows, label):
        n = len(rows)
        if not n:
            return
        p1 = sum(r["pass1"] for r in rows) / n
        pk = sum(r["passk"] for r in rows) / n
        mk = sum(r["majk"] for r in rows) / n
        print(f"{label:28s} n={n:4d}  pass@1={p1:.3f}  pass@{args.k}={pk:.3f}  "
              f"maj@{args.k}={mk:.3f}  | HEADROOM(passk-pass1)={pk-p1:+.3f}  "
              f"selfconsist(majk-pass1)={mk-p1:+.3f}")

    print("\n================ pass@k HEADROOM ================")
    agg(recs, "OVERALL")
    print("--- by difficulty ---")
    by = defaultdict(list)
    for r in recs:
        by[r["difficulty"]].append(r)
    for d in sorted(by):
        agg(by[d], d)
    print("--- by data_source ---")
    by = defaultdict(list)
    for r in recs:
        by[r["data_source"]].append(r)
    for d in sorted(by):
        agg(by[d], d)
    print("=================================================")


if __name__ == "__main__":
    main()
