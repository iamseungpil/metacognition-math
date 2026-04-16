"""V6.2 Step 1: Generate switch + verification seed data via TRAPI GPT-5.4-mini.

3-phase pipeline:
  Phase A: Generate 2 independent solutions per problem (TRAPI)
           -> filter (wrong, correct) pairs with structurally different approaches
  Phase B: Stitch (wrong, correct) pairs into switch trajectories
           -> validate: meta conf < 0.4, structural difference, answer correct
  Phase C: Generate independent verification trajectories
           -> 2-call pattern: verify answer WITHOUT seeing original solution method

Usage:
  python scripts/gen_v6_switch_data.py \\
    --n_problems 500 \\
    --output_dir data/v6_switch_seed \\
    --concurrent 8 \\
    --model gpt-5.4-mini_2026-03-17
"""
import argparse
import json
import os
import random
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.rewards import _check_correctness
from scripts.gen_switch_trajectory import (
    SWITCH_SYSTEM_PROMPT,
    _build_user_prompt,
    _approaches_differ,
    _extract_meta_blocks,
    validate_trajectory,
)


# ---------------------------------------------------------------------------
# TRAPI client (matching gen_metacot_v2.py pattern)
# ---------------------------------------------------------------------------

def get_trapi_client():
    """Create TRAPI client with Azure CLI credential.

    Auth strategy:
      1. TRAPI_TOKEN env var (pre-authenticated environments)
      2. AzureCliCredential + get_bearer_token_provider (host VM)
    """
    from openai import AzureOpenAI

    endpoint = "https://trapi.research.microsoft.com/gcr/shared"
    api_version = "2025-04-01-preview"
    trapi_scope = "api://trapi/.default"

    token = os.environ.get("TRAPI_TOKEN")
    if token:
        print("Using TRAPI_TOKEN env var (pre-authenticated)")
        return AzureOpenAI(
            azure_endpoint=endpoint,
            api_key=token,
            api_version=api_version,
        )

    from azure.identity import AzureCliCredential, get_bearer_token_provider

    provider = get_bearer_token_provider(AzureCliCredential(), trapi_scope)
    # Use azure_ad_token_provider for auto-refresh (not one-shot api_key)
    return AzureOpenAI(
        azure_endpoint=endpoint,
        azure_ad_token_provider=provider,
        api_version=api_version,
    )


# ---------------------------------------------------------------------------
# Problem loading (hendrycks_math medium+hard, from gen_control_v5_trapi.py)
# ---------------------------------------------------------------------------

def _extract_boxed_answer(solution: str) -> str:
    """Extract the last boxed answer from a solution string."""
    matches = re.findall(
        r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}",
        solution or "",
    )
    return matches[-1].strip() if matches else ""


def load_math_problems(n_problems: int = 500, seed: int = 42) -> list[dict]:
    """Load medium+hard MATH train problems from hendrycks_math.

    Filters to Level 3-5 problems that have extractable gold answers.
    Returns up to n_problems rows, shuffled.

    Each row: {question, gold_answer, difficulty, source}
    """
    from datasets import load_dataset

    subjects = [
        "algebra",
        "counting_and_probability",
        "geometry",
        "intermediate_algebra",
        "number_theory",
        "prealgebra",
        "precalculus",
    ]

    rows = []
    for subject in subjects:
        ds = list(load_dataset("EleutherAI/hendrycks_math", subject, split="train"))
        for row in ds:
            level = row.get("level", "")
            # Keep medium (Level 3) and hard (Level 4, 5) only
            if level not in {"Level 3", "Level 4", "Level 5"}:
                continue
            gold = _extract_boxed_answer(str(row.get("solution", "")))
            if not gold:
                continue
            difficulty = "hard" if level in {"Level 4", "Level 5"} else "medium"
            rows.append({
                "question": row["problem"],
                "gold_answer": gold,
                "difficulty": difficulty,
                "source": f"hendrycks_math/{subject}",
            })

    random.seed(seed)
    random.shuffle(rows)
    selected = rows[:n_problems]
    print(f"Loaded {len(selected)} problems from hendrycks_math "
          f"(medium={sum(1 for r in selected if r['difficulty'] == 'medium')}, "
          f"hard={sum(1 for r in selected if r['difficulty'] == 'hard')})")
    return selected


# ---------------------------------------------------------------------------
# Phase A: Independent solution generation
# ---------------------------------------------------------------------------

SOLVE_SYSTEM = (
    "Solve this math problem step by step. "
    "Show your full reasoning and end with \\boxed{answer}."
)


def _trapi_call_with_retry(client, model, system, user, max_retries=5):
    """Generic TRAPI call with retry logic matching gen_metacot_v2.py.

    Returns response text or None on total failure.
    """
    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.7,
            )
            text = resp.output_text
            if text:
                return text
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(120, 5 * (2 ** attempt) + random.uniform(0, 3))
                time.sleep(wait)
            elif "500" in err or "502" in err or "503" in err:
                time.sleep(10 + random.uniform(0, 5))
            else:
                print(f"  Error (attempt {attempt + 1}): {err[:120]}")
                time.sleep(5)
    return None


def generate_solution(client, question: str, model: str, max_retries: int = 5) -> str | None:
    """Generate one independent solution via TRAPI.

    Returns solution text containing \\boxed{answer}, or None on failure.
    """
    text = _trapi_call_with_retry(
        client, model, SOLVE_SYSTEM, question, max_retries=max_retries,
    )
    if text and ("\\boxed" in text or "boxed{" in text):
        return text
    return None


def _process_problem_phase_a(
    client, problem: dict, model: str, max_retries: int,
) -> dict | None:
    """Generate 2 independent solutions for a problem and check correctness.

    Returns a candidate dict if we get a (wrong, correct) pair with
    structurally different approaches, else None.
    """
    question = problem["question"]
    gold = problem["gold_answer"]

    sol_a = generate_solution(client, question, model, max_retries)
    sol_b = generate_solution(client, question, model, max_retries)

    if sol_a is None or sol_b is None:
        return None

    correct_a = _check_correctness(sol_a, gold)
    correct_b = _check_correctness(sol_b, gold)

    # We need exactly one wrong and one correct
    if correct_a == correct_b:
        return None

    if correct_a:
        wrong_sol, correct_sol = sol_b, sol_a
    else:
        wrong_sol, correct_sol = sol_a, sol_b

    # Both must be non-trivial
    if len(wrong_sol) < 50 or len(correct_sol) < 50:
        return None

    # Approaches must be structurally different
    if not _approaches_differ(wrong_sol, correct_sol):
        return None

    return {
        "question": question,
        "gold_answer": gold,
        "difficulty": problem["difficulty"],
        "source": problem["source"],
        "wrong_solution": wrong_sol,
        "correct_solution": correct_sol,
    }


# ---------------------------------------------------------------------------
# Phase B: Stitching (reuses SWITCH_SYSTEM_PROMPT from gen_switch_trajectory)
# ---------------------------------------------------------------------------

def generate_stitch_trajectory(
    client,
    candidate: dict,
    model: str,
    max_retries: int = 10,
) -> dict | None:
    """Generate and validate a stitched switch trajectory for one candidate.

    Calls GPT to create "failed start -> meta diagnosis -> switch -> correct completion".
    Validates via validate_trajectory from gen_switch_trajectory.

    Returns a row dict ready for the output DataFrame, or None.
    """
    question = candidate["question"]
    gold = candidate["gold_answer"]

    user_prompt = _build_user_prompt(
        question=question,
        gold_answer=gold,
        wrong_completion=candidate["wrong_solution"],
        correct_completion=candidate["correct_solution"],
    )

    for attempt in range(max_retries):
        try:
            resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": SWITCH_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
            )
            text = resp.output_text
            if not text:
                continue

            # Quick sanity before full validation
            if "\\boxed" not in text and "boxed{" not in text:
                continue

            validation = validate_trajectory(text, gold)
            if not validation["valid"]:
                continue

            # Build SFT-compatible row
            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": text},
            ]
            return {
                "messages": json.dumps(messages, ensure_ascii=False),
                "question": question,
                "gold_answer": gold,
                "scenario": "redirect",
                "has_switch": True,
                "confidence_at_switch": validation["confidence_at_switch"],
                "source": candidate["source"],
                "difficulty": candidate["difficulty"],
            }

        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = min(120, 5 * (2 ** attempt) + random.uniform(0, 3))
                time.sleep(wait)
            elif "500" in err or "502" in err or "503" in err:
                time.sleep(10 + random.uniform(0, 5))
            else:
                print(f"  Stitch error (attempt {attempt + 1}): {err[:120]}")
                time.sleep(5)

    return None


# ---------------------------------------------------------------------------
# Phase C: Independent verification generation
# ---------------------------------------------------------------------------

VERIFY_SYSTEM = """\
You are verifying a math answer. You are given ONLY the problem and a proposed answer.
Do NOT repeat the original solution method. Instead, verify using ONE of:
- Substitution: plug the answer back into the original equation
- Reverse operation: work backwards from the answer
- Boundary/special case test: check extreme or simple cases
- Dimensional/unit analysis

Show your verification work step by step.
Conclude with: "The answer \\boxed{X} is CORRECT." or "The answer is INCORRECT because..."
"""


def generate_verification(
    client,
    question: str,
    proposed_answer: str,
    gold_answer: str,
    source: str,
    difficulty: str,
    model: str,
    max_retries: int = 5,
) -> dict | None:
    """Generate an independent verification trajectory.

    The model sees only the problem and proposed answer, NOT the solution method.
    Returns an SFT-compatible row dict, or None.
    """
    user = (
        f"Problem: {question}\n\n"
        f"Proposed answer: {proposed_answer}\n\n"
        f"Verify this answer using a method DIFFERENT from direct solving."
    )

    text = _trapi_call_with_retry(
        client, model, VERIFY_SYSTEM, user, max_retries=max_retries,
    )

    if text is None:
        return None

    # Verify that the verification itself reaches the correct conclusion
    # (the answer is correct, and the verification says so)
    mentions_correct = bool(re.search(
        r"\b(correct|verified|confirms?|checks?\s+out|consistent)\b",
        text[-300:],
        re.IGNORECASE,
    ))

    if not mentions_correct:
        return None

    # Build verification prompt as the user would see it
    verify_question = (
        f"{question}\n\n"
        f"The answer is {proposed_answer}. "
        f"Verify using substitution/reverse/boundary test."
    )

    messages = [
        {"role": "user", "content": verify_question},
        {"role": "assistant", "content": text},
    ]

    return {
        "messages": json.dumps(messages, ensure_ascii=False),
        "question": question,
        "gold_answer": gold_answer,
        "scenario": "verify",
        "has_switch": False,
        "confidence_at_switch": None,
        "source": source,
        "difficulty": difficulty,
    }


# ---------------------------------------------------------------------------
# Intermediate save helper
# ---------------------------------------------------------------------------

def _save_intermediate(rows: list[dict], output_dir: Path, filename: str) -> str:
    """Save rows as parquet. Returns the file path."""
    df = pd.DataFrame(rows)
    path = output_dir / filename
    df.to_parquet(path, index=False)
    return str(path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="V6.2 Step 1: Generate switch + verification seed data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n_problems", type=int, default=500,
        help="Number of MATH problems to load (default: 500)",
    )
    parser.add_argument(
        "--output_dir", default="data/v6_switch_seed",
        help="Output directory for parquet files (default: data/v6_switch_seed)",
    )
    parser.add_argument(
        "--concurrent", type=int, default=8,
        help="Number of concurrent TRAPI requests (default: 8)",
    )
    parser.add_argument(
        "--model", default="gpt-5.4-mini_2026-03-17",
        help="TRAPI model name (default: gpt-5.4-mini_2026-03-17)",
    )
    parser.add_argument(
        "--max_retries", type=int, default=5,
        help="Max retries per TRAPI call (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  V6.2 Switch + Verification Data Generator")
    print("=" * 60)
    print(f"  n_problems:  {args.n_problems}")
    print(f"  output_dir:  {args.output_dir}")
    print(f"  concurrent:  {args.concurrent}")
    print(f"  model:       {args.model}")
    print(f"  max_retries: {args.max_retries}")
    print(f"  seed:        {args.seed}")
    print()

    # ------------------------------------------------------------------
    # Load problems
    # ------------------------------------------------------------------
    problems = load_math_problems(n_problems=args.n_problems, seed=args.seed)
    if not problems:
        print("Error: no problems loaded.")
        sys.exit(1)

    client = get_trapi_client()
    print(f"TRAPI client ready. Using {args.concurrent} concurrent workers.\n")

    # ==================================================================
    # Phase A: Generate independent solutions
    # ==================================================================
    print("Phase A: Generating independent solutions ...")
    print(f"  Submitting {len(problems)} problems x 2 solutions each\n")

    candidates: list[dict] = []
    phase_a_failed = 0
    phase_a_no_pair = 0

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = {
            executor.submit(
                _process_problem_phase_a,
                client, prob, args.model, args.max_retries,
            ): prob
            for prob in problems
        }

        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                if result is not None:
                    candidates.append(result)
                else:
                    phase_a_no_pair += 1
            except Exception:
                phase_a_failed += 1
                traceback.print_exc()

            if i % 10 == 0:
                print(
                    f"  [{i}/{len(problems)}] "
                    f"candidates={len(candidates)} "
                    f"no_pair={phase_a_no_pair} "
                    f"failed={phase_a_failed}",
                    flush=True,
                )

    print(f"\nPhase A complete: {len(candidates)} valid (wrong, correct) pairs "
          f"from {len(problems)} problems")
    print(f"  no_pair={phase_a_no_pair}, failed={phase_a_failed}\n")

    if not candidates:
        print("Error: no switch candidates found. Exiting.")
        sys.exit(1)

    # Save Phase A candidates with full solutions (for Phase B/C reuse)
    _save_intermediate(
        [{"question": c["question"], "gold_answer": c["gold_answer"],
          "difficulty": c["difficulty"], "source": c["source"],
          "wrong_solution": c["wrong_solution"],
          "correct_solution": c["correct_solution"],
          "wrong_len": len(c["wrong_solution"]),
          "correct_len": len(c["correct_solution"])}
         for c in candidates],
        output_dir, "phase_a_candidates.parquet",
    )

    # ==================================================================
    # Phase B: Stitching
    # ==================================================================
    print(f"Phase B: Stitching {len(candidates)} pairs into switch trajectories ...")

    switch_rows: list[dict] = []
    phase_b_failed = 0

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = {
            executor.submit(
                generate_stitch_trajectory,
                client, cand, args.model, args.max_retries * 2,
            ): cand
            for cand in candidates
        }

        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                if result is not None:
                    switch_rows.append(result)
                else:
                    phase_b_failed += 1
            except Exception:
                phase_b_failed += 1
                traceback.print_exc()

            if i % 10 == 0:
                print(
                    f"  [{i}/{len(candidates)}] "
                    f"valid={len(switch_rows)} "
                    f"failed={phase_b_failed}",
                    flush=True,
                )

            # Intermediate save every 50 valid trajectories
            if len(switch_rows) > 0 and len(switch_rows) % 50 == 0:
                ckpt = _save_intermediate(
                    switch_rows, output_dir,
                    f"switch_checkpoint_{len(switch_rows)}.parquet",
                )
                print(f"  Checkpoint saved: {ckpt}", flush=True)

    print(f"\nPhase B complete: {len(switch_rows)} valid switch trajectories "
          f"(failed={phase_b_failed})\n")

    # ==================================================================
    # Phase C: Independent verification
    # ==================================================================
    # Use problems that had correct solutions (from Phase A candidates)
    verify_sources = candidates  # all have a correct_solution
    print(f"Phase C: Generating verifications for {len(verify_sources)} problems ...")

    verify_rows: list[dict] = []
    phase_c_failed = 0

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = {
            executor.submit(
                generate_verification,
                client,
                cand["question"],
                cand["gold_answer"],
                cand["gold_answer"],
                cand["source"],
                cand["difficulty"],
                args.model,
                args.max_retries,
            ): cand
            for cand in verify_sources
        }

        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                if result is not None:
                    verify_rows.append(result)
                else:
                    phase_c_failed += 1
            except Exception:
                phase_c_failed += 1
                traceback.print_exc()

            if i % 10 == 0:
                print(
                    f"  [{i}/{len(verify_sources)}] "
                    f"valid={len(verify_rows)} "
                    f"failed={phase_c_failed}",
                    flush=True,
                )

    print(f"\nPhase C complete: {len(verify_rows)} valid verifications "
          f"(failed={phase_c_failed})\n")

    # ==================================================================
    # Save final results
    # ==================================================================
    if switch_rows:
        switch_path = output_dir / "switch_trajectories.parquet"
        pd.DataFrame(switch_rows).to_parquet(switch_path, index=False)
        print(f"Switch trajectories saved: {switch_path} ({len(switch_rows)} rows)")
    else:
        print("Warning: no valid switch trajectories generated.")

    if verify_rows:
        verify_path = output_dir / "verify_trajectories.parquet"
        pd.DataFrame(verify_rows).to_parquet(verify_path, index=False)
        print(f"Verify trajectories saved: {verify_path} ({len(verify_rows)} rows)")
    else:
        print("Warning: no valid verification trajectories generated.")

    # ==================================================================
    # Summary
    # ==================================================================
    print()
    print("=" * 60)
    print("  V6.2 Generation Summary")
    print("=" * 60)
    print(f"  Problems loaded:           {len(problems)}")
    print(f"  Phase A candidates:        {len(candidates)}")
    print(f"  Phase B switch rows:       {len(switch_rows)}")
    print(f"  Phase C verify rows:       {len(verify_rows)}")
    print(f"  Total SFT rows:            {len(switch_rows) + len(verify_rows)}")

    if switch_rows:
        confs = [
            r["confidence_at_switch"]
            for r in switch_rows
            if r["confidence_at_switch"] is not None
        ]
        if confs:
            print(f"\n  Switch confidence stats:")
            print(f"    mean = {sum(confs) / len(confs):.3f}")
            print(f"    min  = {min(confs):.3f}")
            print(f"    max  = {max(confs):.3f}")

        # Per-difficulty breakdown
        for diff in ["medium", "hard"]:
            count = sum(1 for r in switch_rows if r["difficulty"] == diff)
            if count:
                print(f"    {diff}: {count}")

    if verify_rows:
        print(f"\n  Verification per-difficulty:")
        for diff in ["medium", "hard"]:
            count = sum(1 for r in verify_rows if r["difficulty"] == diff)
            if count:
                print(f"    {diff}: {count}")

    # Estimated API calls
    api_calls = len(problems) * 2 + len(candidates) + len(verify_sources)
    print(f"\n  Estimated API calls made:  ~{api_calls}")
    print(f"  Output directory:          {output_dir}")
    print()


if __name__ == "__main__":
    main()
