"""Selective abstention: accuracy when only answering high-confidence problems."""
import json, os, glob

results_dir = "/scratch/metacognition/results"
models = {}
for f in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
    name = os.path.basename(f).replace("eval_", "").replace(".json", "")
    with open(f) as fh:
        models[name] = json.load(fh)["results"]

print("=== SELECTIVE ABSTENTION ANALYSIS ===")
print("(Answer only when confidence >= threshold)\n")

for name, data in models.items():
    confs = [(r.get("avg_confidence"), r["is_correct"]) for r in data if r.get("avg_confidence")]
    if not confs:
        # Base SFT has no confidence — always answers
        total = len(data)
        correct = sum(1 for r in data if r["is_correct"])
        print(f"{name}: always answers → {correct}/{total} = {correct/total:.1%} (no abstention possible)")
        continue

    print(f"{name}:")
    for threshold in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
        answered = [(c, corr) for c, corr in confs if c >= threshold]
        if not answered:
            print(f"  conf>={threshold}: 0 answered")
            continue
        correct = sum(1 for _, corr in answered if corr)
        coverage = len(answered) / len(confs)
        accuracy = correct / len(answered)
        print(f"  conf>={threshold}: {correct}/{len(answered)} = {accuracy:.1%} accuracy, coverage={coverage:.1%}")
    print()
