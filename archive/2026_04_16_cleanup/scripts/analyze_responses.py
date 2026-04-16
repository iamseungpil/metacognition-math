"""Analyze 30+ responses from each model for confidence patterns."""
import json
import os
import glob
import re
from collections import defaultdict

results_dir = "/scratch/metacognition/results"

for f in sorted(glob.glob(os.path.join(results_dir, "eval_*.json"))):
    name = os.path.basename(f).replace("eval_", "").replace(".json", "")
    with open(f) as fh:
        data = json.load(fh)

    results = data["results"]
    print(f"\n{'='*60}")
    print(f"MODEL: {name} ({len(results)} problems)")
    print(f"{'='*60}")

    # Confidence distribution
    confs = [r["avg_confidence"] for r in results if r["avg_confidence"] is not None]
    correct_confs = [r["avg_confidence"] for r in results if r["avg_confidence"] is not None and r["is_correct"]]
    wrong_confs = [r["avg_confidence"] for r in results if r["avg_confidence"] is not None and not r["is_correct"]]

    if confs:
        print(f"\nConfidence Distribution:")
        print(f"  All:     mean={sum(confs)/len(confs):.3f}, min={min(confs):.3f}, max={max(confs):.3f}")
        if correct_confs:
            print(f"  Correct: mean={sum(correct_confs)/len(correct_confs):.3f} (n={len(correct_confs)})")
        if wrong_confs:
            print(f"  Wrong:   mean={sum(wrong_confs)/len(wrong_confs):.3f} (n={len(wrong_confs)})")
        print(f"  Conf>0.95: {sum(1 for c in confs if c > 0.95)}/{len(confs)} ({sum(1 for c in confs if c > 0.95)/len(confs)*100:.0f}%)")
        print(f"  Conf<0.50: {sum(1 for c in confs if c < 0.50)}/{len(confs)} ({sum(1 for c in confs if c < 0.50)/len(confs)*100:.0f}%)")

    # Per-benchmark analysis
    for bench in ["gsm8k", "math500", "aime2024"]:
        br = [r for r in results if r["benchmark"] == bench]
        if not br:
            continue
        acc = sum(1 for r in br if r["is_correct"]) / len(br)
        meta_avg = sum(r["num_meta_blocks"] for r in br) / len(br)
        bc = [r["avg_confidence"] for r in br if r["avg_confidence"] is not None]
        conf_avg = sum(bc) / len(bc) if bc else 0

        # Key: does confidence match accuracy?
        calibration_gap = abs(conf_avg - acc)
        print(f"\n  {bench}: acc={acc:.1%}, conf={conf_avg:.3f}, gap={calibration_gap:.3f}, meta={meta_avg:.1f}")

        # Show 3 wrong examples with high confidence (overconfident)
        overconfident = [r for r in br if not r["is_correct"] and r.get("avg_confidence", 0) and r["avg_confidence"] > 0.8]
        if overconfident:
            print(f"  Overconfident wrong ({len(overconfident)} cases):")
            for r in overconfident[:2]:
                comp = r["completion"][:150].replace("\n", " ")
                print(f"    conf={r['avg_confidence']:.2f}: {comp}...")

    # Completion length analysis
    lengths = [len(r["completion"]) for r in results]
    print(f"\n  Completion length: mean={sum(lengths)/len(lengths):.0f}, max={max(lengths)}")
