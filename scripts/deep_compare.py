"""Deep qualitative comparison: Base SFT vs Meta SFT vs E3 vs E5.
Focus on: same problem, different models — what changed and why."""
import json
import os
import glob
import re

results_dir = "/scratch/metacognition/results"

# Load all results
models = {}
for f in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
    name = os.path.basename(f).replace("eval_", "").replace(".json", "")
    with open(f) as fh:
        models[name] = json.load(fh)["results"]

print(f"Models: {list(models.keys())}")

# Find problems where models disagree
# Group by question (first 80 chars as key)
from collections import defaultdict
problems = defaultdict(dict)
for name, results in models.items():
    for r in results:
        key = r["question"][:80]
        problems[key][name] = r

# Analysis 1: Problems that Base SFT gets right but Meta models get wrong
print("\n" + "="*70)
print("CASE 1: Base SFT CORRECT, Meta/GRPO WRONG (meta hurts)")
print("="*70)
count = 0
for q, preds in problems.items():
    base = preds.get("qwen3_base_sft")
    meta = preds.get("qwen3_meta_sft")
    if not base or not meta:
        continue
    if base["is_correct"] and not meta["is_correct"]:
        count += 1
        if count <= 3:
            print(f"\nQ: {q}")
            print(f"  Base SFT: CORRECT, meta={base['num_meta_blocks']}")
            print(f"    {base['completion'][:200]}...")
            print(f"  Meta SFT: WRONG, conf={meta.get('avg_confidence','?')}, meta={meta['num_meta_blocks']}")
            print(f"    {meta['completion'][:200]}...")
            # Check if meta completion is truncated (no boxed)
            has_boxed = "boxed" in meta["completion"]
            print(f"    has_boxed={has_boxed}")
print(f"\nTotal: {count} problems where base correct, meta wrong")

# Analysis 2: Problems that Meta/GRPO gets right but Base gets wrong (meta helps)
print("\n" + "="*70)
print("CASE 2: Meta SFT CORRECT, Base SFT WRONG (meta helps)")
print("="*70)
count2 = 0
for q, preds in problems.items():
    base = preds.get("qwen3_base_sft")
    meta = preds.get("qwen3_meta_sft")
    if not base or not meta:
        continue
    if not base["is_correct"] and meta["is_correct"]:
        count2 += 1
        if count2 <= 3:
            print(f"\nQ: {q}")
            print(f"  Base SFT: WRONG")
            print(f"    {base['completion'][:200]}...")
            print(f"  Meta SFT: CORRECT, conf={meta.get('avg_confidence','?')}, meta={meta['num_meta_blocks']}")
            print(f"    {meta['completion'][:200]}...")
print(f"\nTotal: {count2} problems where meta correct, base wrong")

# Analysis 3: E3 vs E5 — where do they differ?
print("\n" + "="*70)
print("CASE 3: E3 vs E5 differences")
print("="*70)
e3_key = [k for k in models.keys() if "checkpoint" in k or "e3" in k.lower()]
e5_key = [k for k in models.keys() if "e5" in k.lower()]
if e3_key and e5_key:
    e3_name, e5_name = e3_key[0], e5_key[0]
    e3_right_e5_wrong = 0
    e5_right_e3_wrong = 0
    for q, preds in problems.items():
        e3 = preds.get(e3_name)
        e5 = preds.get(e5_name)
        if not e3 or not e5:
            continue
        if e3["is_correct"] and not e5["is_correct"]:
            e3_right_e5_wrong += 1
        if not e3["is_correct"] and e5["is_correct"]:
            e5_right_e3_wrong += 1
    print(f"  E3 right, E5 wrong: {e3_right_e5_wrong}")
    print(f"  E5 right, E3 wrong: {e5_right_e3_wrong}")

# Analysis 4: Confidence vs correctness correlation
print("\n" + "="*70)
print("CASE 4: Confidence accuracy by confidence bucket")
print("="*70)
for name, results in models.items():
    confs = [(r.get("avg_confidence"), r["is_correct"]) for r in results if r.get("avg_confidence")]
    if not confs:
        continue
    buckets = {"<0.7": [0, 0], "0.7-0.9": [0, 0], ">0.9": [0, 0]}
    for c, correct in confs:
        if c < 0.7:
            bucket = "<0.7"
        elif c < 0.9:
            bucket = "0.7-0.9"
        else:
            bucket = ">0.9"
        buckets[bucket][1] += 1
        if correct:
            buckets[bucket][0] += 1
    print(f"\n  {name}:")
    for b, (corr, total) in buckets.items():
        acc = corr / total * 100 if total > 0 else 0
        print(f"    conf {b}: {corr}/{total} correct ({acc:.0f}%)")
