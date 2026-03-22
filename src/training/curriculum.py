"""Self-directed curriculum learning — Phase 3 of Meta-CoT.

The model diagnoses its weaknesses from Meta-CoT chains, then selects
problems from the data pool to study. This is the key mechanism that
differentiates Meta-CoT from standard SFT/RL.

Flow:
1. Run model on eval set → collect Meta-CoT outputs with diagnoses
2. Extract weak categories from diagnoses (e.g., "modular inverse")
3. Match weak categories to problems in the training data pool
4. Create a curriculum of selected problems
5. Fine-tune on selected problems → re-evaluate
"""
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd


def extract_weaknesses_from_chains(chains: list[str]) -> dict:
    """Extract weakness categories from Meta-CoT diagnose/strategize stages."""
    category_counts = {}
    categories = [
        "algebra", "geometry", "number_theory", "combinatorics",
        "probability", "calculus", "arithmetic", "modular",
        "trigonometry", "sequences", "inequalities", "polynomials",
        "counting", "precalculus", "prealgebra",
    ]

    for chain in chains:
        chain_lower = chain.lower()
        # Look for weakness indicators
        is_weakness = any(kw in chain_lower for kw in [
            "weak", "struggle", "incorrect", "error", "mistake",
            "needs practice", "should study", "needs improvement",
            "confidence: 0." # low confidence
        ])
        if not is_weakness:
            continue

        for cat in categories:
            if cat in chain_lower:
                category_counts[cat] = category_counts.get(cat, 0) + 1

    # Sort by frequency (most common weakness first)
    return dict(sorted(category_counts.items(), key=lambda x: -x[1]))


def select_curriculum_problems(
    data_pool_path: str,
    weaknesses: dict,
    max_problems: int = 500,
    difficulty_preference: str = "medium",
) -> pd.DataFrame:
    """Select problems from data pool that target identified weaknesses."""
    df = pd.read_parquet(data_pool_path)

    selected = []
    remaining = max_problems

    for category, count in weaknesses.items():
        if remaining <= 0:
            break

        # Find problems matching this weakness category
        # Search in question text and category field
        mask = (
            df["question"].str.lower().str.contains(category, na=False) |
            df["category"].str.lower().str.contains(category, na=False)
        )

        # Prefer problems the model got wrong (more learning signal)
        wrong_mask = mask & (~df["is_correct"])
        correct_mask = mask & df["is_correct"]

        # Take more wrong problems than correct (3:1 ratio)
        n_wrong = min(int(remaining * 0.75), wrong_mask.sum())
        n_correct = min(remaining - n_wrong, correct_mask.sum())

        if n_wrong > 0:
            selected.append(df[wrong_mask].sample(n=min(n_wrong, wrong_mask.sum()), random_state=42))
        if n_correct > 0:
            selected.append(df[correct_mask].sample(n=min(n_correct, correct_mask.sum()), random_state=42))

        remaining -= (n_wrong + n_correct)

    if not selected:
        # Fallback: random sample from wrong answers
        wrong = df[~df["is_correct"]]
        selected.append(wrong.sample(n=min(max_problems, len(wrong)), random_state=42))

    result = pd.concat(selected).drop_duplicates("problem_id")
    print(f"Curriculum: {len(result)} problems selected for {len(weaknesses)} weak categories")
    for cat, count in list(weaknesses.items())[:5]:
        n = result["question"].str.lower().str.contains(cat, na=False).sum()
        print(f"  {cat}: {n} problems (diagnosed {count} times)")

    return result


def run_self_directed_cycle(
    model_path: str,
    data_pool_path: str,
    output_dir: str,
    n_cycles: int = 3,
    problems_per_cycle: int = 500,
):
    """Run N cycles of: diagnose → select → train → evaluate.

    This is the full Phase 3 self-directed learning loop.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    current_model = model_path
    cycle_results = []

    for cycle in range(n_cycles):
        print(f"\n=== Self-Directed Cycle {cycle + 1}/{n_cycles} ===")

        # Step 1: Generate Meta-CoT chains on eval problems
        print(f"  Step 1: Generating Meta-CoT diagnoses...")
        # (This would call the model to generate chains — simplified here)

        # Step 2: Extract weaknesses
        print(f"  Step 2: Extracting weaknesses from diagnoses...")

        # Step 3: Select curriculum problems
        print(f"  Step 3: Selecting curriculum problems...")
        # For now, use profile's weak_categories as proxy
        profile_path = output_path.parent / "profiles" / "capability_profile.json"
        if profile_path.exists():
            with open(profile_path) as f:
                profile = json.load(f)
            weaknesses = {cat: 10 for cat in profile.get("weak_categories", ["algebra"])}
        else:
            weaknesses = {"algebra": 10, "geometry": 8, "number_theory": 6}

        curriculum = select_curriculum_problems(
            data_pool_path, weaknesses,
            max_problems=problems_per_cycle,
        )

        # Save curriculum
        curriculum_path = output_path / f"curriculum_cycle{cycle}.parquet"
        curriculum.to_parquet(curriculum_path, index=False)

        # Step 4: Fine-tune on selected problems
        print(f"  Step 4: Fine-tuning on {len(curriculum)} selected problems...")
        # (Would call SFT here with the curriculum data)

        cycle_results.append({
            "cycle": cycle,
            "n_problems": len(curriculum),
            "weaknesses": weaknesses,
            "curriculum_path": str(curriculum_path),
        })

    # Save results
    with open(output_path / "self_directed_results.json", "w") as f:
        json.dump(cycle_results, f, indent=2)

    return cycle_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--data-pool", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-cycles", type=int, default=3)
    parser.add_argument("--problems-per-cycle", type=int, default=500)
    args = parser.parse_args()

    run_self_directed_cycle(
        model_path=args.model_path,
        data_pool_path=args.data_pool,
        output_dir=args.output_dir,
        n_cycles=args.n_cycles,
        problems_per_cycle=args.problems_per_cycle,
    )
