"""Final analysis: V2 accuracy drop + AIME ECE improvement."""
import json, os, glob, re

results_dir = "/scratch/metacognition/results"
models = {}
for f in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
    name = os.path.basename(f).replace("eval_", "").replace(".json", "")
    with open(f) as fh:
        models[name] = json.load(fh)["results"]

print("=== MATH-500 DETAILED ===")
for name, data in models.items():
    math = [r for r in data if r["benchmark"] == "math500"]
    if not math:
        continue
    correct = [r for r in math if r["is_correct"]]
    wrong = [r for r in math if not r["is_correct"]]
    c_conf = [r["avg_confidence"] for r in correct if r.get("avg_confidence")]
    w_conf = [r["avg_confidence"] for r in wrong if r.get("avg_confidence")]
    c_len = sum(len(r.get("completion", "")) for r in correct) / max(len(correct), 1)
    w_len = sum(len(r.get("completion", "")) for r in wrong) / max(len(wrong), 1)
    c_boxed = sum(1 for r in correct if "boxed" in r.get("completion", ""))
    w_boxed = sum(1 for r in wrong if "boxed" in r.get("completion", ""))
    cc = sum(c_conf) / len(c_conf) if c_conf else 0
    wc = sum(w_conf) / len(w_conf) if w_conf else 0
    print(f"\n{name}: {len(correct)}/{len(math)} correct")
    print(f"  boxed: correct={c_boxed}/{len(correct)}, wrong={w_boxed}/{len(wrong)}")
    print(f"  length: correct={c_len:.0f}, wrong={w_len:.0f}")
    print(f"  conf: correct={cc:.3f}, wrong={wc:.3f}, gap={cc-wc:.3f}")

print("\n=== AIME ECE PATH ===")
for name, data in models.items():
    aime = [r for r in data if r["benchmark"] == "aime2024"]
    if not aime:
        continue
    confs = [r["avg_confidence"] for r in aime if r.get("avg_confidence")]
    correct = sum(1 for r in aime if r["is_correct"])
    if confs:
        c_conf = [r["avg_confidence"] for r in aime if r["is_correct"] and r.get("avg_confidence")]
        w_conf = [r["avg_confidence"] for r in aime if not r["is_correct"] and r.get("avg_confidence")]
        print(f"{name}: acc={correct}/30, conf_mean={sum(confs)/len(confs):.3f}, "
              f"wrong_conf={sum(w_conf)/len(w_conf):.3f}" if w_conf else "")
    else:
        print(f"{name}: acc={correct}/30, no confidence")

print("\n=== CONFIDENCE BUCKETS (ALL BENCHMARKS) ===")
for name, data in models.items():
    confs = [(r.get("avg_confidence"), r["is_correct"]) for r in data if r.get("avg_confidence")]
    if not confs:
        continue
    print(f"\n{name}:")
    for lo, hi, label in [(0, 0.5, "<0.5"), (0.5, 0.7, "0.5-0.7"), (0.7, 0.9, "0.7-0.9"), (0.9, 1.01, ">0.9")]:
        bucket = [(c, corr) for c, corr in confs if lo <= c < hi]
        if bucket:
            acc = sum(1 for _, corr in bucket if corr) / len(bucket)
            print(f"  {label}: {len(bucket)} items, acc={acc:.1%}")
