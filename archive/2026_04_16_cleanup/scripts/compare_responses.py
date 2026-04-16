"""Compare Meta SFT vs E3 responses for qualitative analysis."""
import json
import os
import glob

results_dir = "/scratch/metacognition/results"
files = glob.glob(os.path.join(results_dir, "eval_*.json"))
print(f"Result files: {[os.path.basename(f) for f in files]}")

# Load all results
data = {}
for f in files:
    name = os.path.basename(f).replace("eval_", "").replace(".json", "")
    with open(f) as fh:
        data[name] = json.load(fh)

for name, d in data.items():
    results = d["results"]
    print(f"\n=== {name}: {len(results)} problems ===")

    # Per-benchmark stats
    from collections import defaultdict
    stats = defaultdict(lambda: {"correct": 0, "total": 0, "meta_sum": 0, "conf_sum": 0, "conf_n": 0})
    for r in results:
        b = r["benchmark"]
        stats[b]["total"] += 1
        if r["is_correct"]:
            stats[b]["correct"] += 1
        stats[b]["meta_sum"] += r["num_meta_blocks"]
        if r["avg_confidence"] is not None:
            stats[b]["conf_sum"] += r["avg_confidence"]
            stats[b]["conf_n"] += 1

    for b, s in sorted(stats.items()):
        acc = s["correct"] / s["total"] if s["total"] else 0
        meta = s["meta_sum"] / s["total"] if s["total"] else 0
        conf = s["conf_sum"] / s["conf_n"] if s["conf_n"] else 0
        print(f"  {b}: acc={acc:.1%}, meta={meta:.1f}, conf={conf:.2f}")

    # Show 2 examples: 1 correct, 1 wrong
    correct_ex = next((r for r in results if r["is_correct"] and r["benchmark"] == "math500"), None)
    wrong_ex = next((r for r in results if not r["is_correct"] and r["benchmark"] == "math500"), None)

    if correct_ex:
        print(f"\n  [CORRECT] Q: {correct_ex['question'][:80]}")
        print(f"  meta={correct_ex['num_meta_blocks']}, conf={correct_ex.get('avg_confidence', '?')}")
        print(f"  completion: {correct_ex['completion'][:200]}...")

    if wrong_ex:
        print(f"\n  [WRONG] Q: {wrong_ex['question'][:80]}")
        print(f"  meta={wrong_ex['num_meta_blocks']}, conf={wrong_ex.get('avg_confidence', '?')}")
        print(f"  completion: {wrong_ex['completion'][:200]}...")
