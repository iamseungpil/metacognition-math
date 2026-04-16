"""Analyze WHY the model gets wrong answers — categorize error types."""
import json
import os
import glob
import re

results_dir = "/scratch/metacognition/results"

for f in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
    name = os.path.basename(f).replace("eval_", "").replace(".json", "")
    with open(f) as fh:
        data = json.load(fh)

    results = data["results"]
    wrong = [r for r in results if not r["is_correct"]]
    if not wrong:
        continue

    print(f"\n{'='*60}")
    print(f"MODEL: {name} — {len(wrong)} wrong out of {len(results)}")
    print(f"{'='*60}")

    # Categorize errors
    categories = {
        "no_boxed": 0,        # No \boxed{} → parsing failure
        "overconfident": 0,    # conf > 0.9 but wrong
        "short_completion": 0, # completion too short (truncated?)
        "no_meta": 0,          # no meta blocks
        "wrong_approach": 0,   # has meta but wrong answer
    }

    for r in wrong:
        comp = r["completion"]
        conf = r.get("avg_confidence")
        meta = r["num_meta_blocks"]

        if "\\boxed" not in comp and "boxed" not in comp:
            categories["no_boxed"] += 1
        if conf and conf > 0.9:
            categories["overconfident"] += 1
        if len(comp) < 200:
            categories["short_completion"] += 1
        if meta == 0:
            categories["no_meta"] += 1
        if meta > 0 and "boxed" in comp:
            categories["wrong_approach"] += 1

    print(f"\nError categories:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        pct = count / len(wrong) * 100
        print(f"  {cat}: {count}/{len(wrong)} ({pct:.0f}%)")

    # Show 5 detailed wrong examples per benchmark
    for bench in ["gsm8k", "math500", "aime2024"]:
        bwrong = [r for r in wrong if r["benchmark"] == bench]
        if not bwrong:
            continue
        print(f"\n--- {bench}: {len(bwrong)} wrong ---")
        for r in bwrong[:3]:
            comp = r["completion"]
            conf = r.get("avg_confidence", "?")
            meta = r["num_meta_blocks"]
            gold = r.get("gold_answer", "?")[:50]

            # Extract what model answered
            boxed = re.findall(r'\\boxed\{([^}]+)\}', comp)
            model_ans = boxed[-1] if boxed else "NO_BOXED"

            # Check if meta mentioned difficulty
            has_difficulty = bool(re.search(r'difficult|hard|tricky|careful|risk|watch out', comp, re.I))

            print(f"  conf={conf}, meta={meta}, model_ans={model_ans[:30]}, gold={gold}")
            print(f"  difficulty_mentioned={has_difficulty}")
            # First 100 chars of completion
            first_line = comp[:120].replace("\n", " ")
            print(f"  start: {first_line}...")
            print()
