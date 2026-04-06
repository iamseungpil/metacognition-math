"""Rescore eval results with fixed answer extraction.

The v1 _extract_answer_fallback missed currency formats like \\$70,000 or $460,
causing 40+ false negatives in SFT models. This script re-judges all results
using the fixed _check_correctness and reports flips.

Usage:
  python scripts/rescore_eval.py --results_dir results/eval_1030_v5
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")

from src.training.rewards import _check_correctness, _extract_answer_fallback


def rescore(results_dir):
    json_files = sorted(
        f
        for f in os.listdir(results_dir)
        if f.startswith("eval_") and f.endswith(".json") and "metadata" not in f
    )

    if not json_files:
        print(f"No eval_*.json files found in {results_dir}")
        return

    total_flips_correct = 0
    total_flips_wrong = 0
    total_samples = 0

    for jf in json_files:
        path = os.path.join(results_dir, jf)
        with open(path) as f:
            data = json.load(f)

        model_name = data.get("model", jf)
        results = data["results"]

        flips_to_correct = 0
        flips_to_wrong = 0

        for r in results:
            completion = r.get("completion", "")
            gold = r.get("full_gold_answer", r.get("gold_answer", ""))

            new_correct = _check_correctness(completion, gold)
            old_correct = r["is_correct"]

            r["is_correct_v2"] = new_correct
            r["answer_extracted_v2"] = _extract_answer_fallback(completion)

            if not old_correct and new_correct:
                flips_to_correct += 1
            elif old_correct and not new_correct:
                flips_to_wrong += 1

        # Save updated JSON
        v2_path = path.replace(".json", "_v2.json")
        with open(v2_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        old_acc = sum(1 for r in results if r["is_correct"]) / len(results) * 100
        new_acc = sum(1 for r in results if r["is_correct_v2"]) / len(results) * 100

        print(
            f"{model_name}: {old_acc:.1f}% -> {new_acc:.1f}% "
            f"(D{new_acc - old_acc:+.1f}pp, +{flips_to_correct} correct, "
            f"-{flips_to_wrong} lost)"
        )

        total_flips_correct += flips_to_correct
        total_flips_wrong += flips_to_wrong
        total_samples += len(results)

    print(f"\n--- Total: {total_flips_correct} flipped correct, "
          f"{total_flips_wrong} flipped wrong across {total_samples} samples ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rescore eval results with fixed currency/comma answer extraction."
    )
    parser.add_argument("--results_dir", required=True, help="Directory with eval_*.json files")
    args = parser.parse_args()
    rescore(args.results_dir)
