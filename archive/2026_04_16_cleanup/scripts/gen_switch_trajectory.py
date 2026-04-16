"""Generate route-switching Meta-CoT trajectories via TRAPI GPT-5.4.

Phase A: Extract switch candidates from pass@k data
  - For each problem, find (wrong_completion, correct_completion) pairs
    where the approaches are structurally different.

Phase B: GPT-5.4 stitching to create natural switch trajectories
  - Ask GPT-5.4 to write a continuous solution that starts with the
    failed approach, inserts a <|meta|> diagnosis block, then switches
    to the successful approach.

Each trajectory is validated:
  1. Has >= 1 <|meta|> block with confidence < 0.4
  2. Meta block contains switch language
  3. Final \\boxed{} answer matches gold (via _check_correctness)
  4. Total length < 8000 chars (~2048 tokens)

Usage:
  python scripts/gen_switch_trajectory.py \\
    --pass_at_k_dir results/pass_at_k/base_quick \\
    --output_dir data/switch_trajectories \\
    --max_trajectories 800 \\
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metacot.prompt_v2 import META_START, META_END
from src.training.rewards import _check_correctness


# ---------------------------------------------------------------------------
# TRAPI client (same pattern as gen_metacot_v2.py)
# ---------------------------------------------------------------------------

def get_trapi_client():
    """Create TRAPI client with Azure CLI credential.

    Auth strategy (matches gen_metacot_v2.py exactly):
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
    token = provider()
    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=token,
        api_version=api_version,
    )


# ---------------------------------------------------------------------------
# Phase A: Extract switch candidates from pass@k JSON
# ---------------------------------------------------------------------------

def load_pass_at_k_candidates(pass_at_k_dir: str) -> list[dict]:
    """Load pass@k results and extract (wrong, correct) candidate pairs.

    For each problem with both wrong and correct samples, pair the first
    wrong sample with the first correct sample.  Only includes problems
    where the two completions use detectably different approaches.

    Returns:
        List of dicts with keys:
            problem_id, question, gold_answer, benchmark,
            wrong_completion, correct_completion,
            wrong_sample_idx, correct_sample_idx
    """
    pass_at_k_dir = Path(pass_at_k_dir)
    json_files = sorted(pass_at_k_dir.glob("pass_at_k_*.json"))

    # Exclude summary-only files
    json_files = [f for f in json_files if "_summary" not in f.name]

    if not json_files:
        print(f"Error: no pass_at_k_*.json files found in {pass_at_k_dir}")
        return []

    print(f"Found {len(json_files)} pass@k result file(s) in {pass_at_k_dir}")

    candidates = []

    for jf in json_files:
        with open(jf) as f:
            data = json.load(f)

        raw_results = data.get("raw_results", [])
        if not raw_results:
            print(f"  Warning: {jf.name} has no raw_results, skipping")
            continue

        print(f"  {jf.name}: {len(raw_results)} problems")

        for prob in raw_results:
            samples = prob.get("samples", [])
            if not samples:
                continue

            wrong_samples = [
                s for s in samples
                if not s.get("is_correct", False)
            ]
            correct_samples = [
                s for s in samples
                if s.get("is_correct", False)
            ]

            # Need at least one wrong and one correct sample
            if not wrong_samples or not correct_samples:
                continue

            # Pick the first wrong and first correct
            wrong_s = wrong_samples[0]
            correct_s = correct_samples[0]

            wrong_comp = wrong_s.get("completion", "")
            correct_comp = correct_s.get("completion", "")

            # Filter: both completions must be non-trivial
            if len(wrong_comp) < 50 or len(correct_comp) < 50:
                continue

            # Filter: approaches should be structurally different
            if not _approaches_differ(wrong_comp, correct_comp):
                continue

            candidates.append({
                "problem_id": prob.get("problem_id", -1),
                "question": prob.get("question", ""),
                "gold_answer": prob.get("gold_answer", ""),
                "benchmark": prob.get("benchmark", "unknown"),
                "wrong_completion": wrong_comp,
                "correct_completion": correct_comp,
                "wrong_sample_idx": wrong_s.get("sample_idx", 0),
                "correct_sample_idx": correct_s.get("sample_idx", 0),
            })

    print(f"Total switch candidates: {len(candidates)}")
    return candidates


def _approaches_differ(wrong: str, correct: str) -> bool:
    """Heuristic check: do two completions use structurally different methods?

    Detects approach keywords and checks that the two completions emphasize
    different method families (e.g., algebra vs geometry, substitution vs
    factoring, etc.).
    """
    # Patterns use \b only at the start to anchor to word boundaries;
    # stems are followed by \w* to match any suffix (e.g. substitut\w*
    # matches "substitution", "substitute", "substituting", etc.)
    method_families = {
        "substitution": r"\b(substitut\w*|let\s+\w\s*=|plug\w*\s*in)",
        "factoring": r"\b(factor\w*|factoris\w*|factoring)",
        "quadratic_formula": r"\b(quadratic\s+formula|discriminant|b\^2\s*-\s*4ac)",
        "completing_square": r"\bcomplet\w*\s+the\s+square",
        "induction": r"\b(induction|base\s+case|inductive\s+step)",
        "contradiction": r"\b(contradiction|suppose\s+not|assume\s+the\s+contrary)",
        "coordinate": r"\b(coordinate\w*|x-axis|y-axis|slope|intercept)",
        "synthetic_geometry": r"\b(power\s+of\s+a\s+point|angle\s+bisector|circumscrib\w*|inscrib\w*|similar\s+triangle)",
        "trigonometry": r"\b(sin\b|cos\b|tan\b|trig\w*)",
        "combinatorics": r"\b(choose|binom\w*|C\(|permut\w*|combin\w*|inclusion.exclusion)",
        "modular": r"\b(mod\s+\d|modular|congru\w*|residue)",
        "casework": r"\b(case\s*\d|case\s+1|casework|consider\s+the\s+cases)",
        "generating_function": r"\b(generating\s+function|power\s+series)",
        "recursion": r"\b(recur\w*|recursive|recurrence)",
        "direct_computation": r"\b(comput\w*|calculat\w*|evaluat\w*|simplif\w*)",
        "algebraic_manipulation": r"\b(rearrang\w*|manipulat\w*|cross.multiply|expand\w*)",
    }

    wrong_lower = wrong.lower()
    correct_lower = correct.lower()

    wrong_methods = set()
    correct_methods = set()

    for family, pattern in method_families.items():
        if re.search(pattern, wrong_lower, re.IGNORECASE):
            wrong_methods.add(family)
        if re.search(pattern, correct_lower, re.IGNORECASE):
            correct_methods.add(family)

    # If BOTH have no detected methods, reject (no evidence of different approaches)
    if not wrong_methods and not correct_methods:
        return False
    # If exactly one side has methods and the other doesn't, allow (different by nature)
    if not wrong_methods or not correct_methods:
        return True

    # Require at least one method unique to each side (not just symmetric_diff >= 1)
    only_wrong = wrong_methods - correct_methods
    only_correct = correct_methods - wrong_methods
    # Exclude overly broad families from counting
    broad = {"direct_computation", "algebraic_manipulation"}
    only_wrong_specific = only_wrong - broad
    only_correct_specific = only_correct - broad
    return bool(only_wrong_specific) or bool(only_correct_specific)


# ---------------------------------------------------------------------------
# Phase B: GPT-5.4 stitching
# ---------------------------------------------------------------------------

SWITCH_SYSTEM_PROMPT = f"""\
You are writing a math solution that demonstrates metacognitive route switching.

Given:
- A math problem
- A FAILED first approach (the solver got stuck or made an error)
- A SUCCESSFUL alternative approach (different method, correct answer)

Write a SINGLE continuous solution that:
1. Starts with the failed approach (first 30-50% of it)
2. Inserts a {META_START} block where the solver RECOGNIZES the problem:
   - confidence: [0.15-0.35] (low, reflecting genuine uncertainty)
   - "The current route is weak because [specific reason from the failed approach]"
   - "I should switch to [specific method name from the successful approach]"
3. Continues with the successful approach to reach the correct \\boxed{{answer}}

Rules:
- The meta block MUST use exactly this format: {META_START}...{META_END}
- The meta block must contain confidence < 0.4
- The switch must be to a STRUCTURALLY DIFFERENT method
- The final answer must match the gold answer exactly in \\boxed{{}}
- Keep meta blocks SHORT (2-3 lines)
- Go straight into the math solution, no preamble

EXAMPLE of the exact format expected:

Let me try factoring the expression...
After expansion, $x^2 + 3x - 10 = (x+5)(x-2) = 0$, giving $x=-5$ or $x=2$.
But wait, substituting back: $(-5)^2 + 3(-5) - 10 = 25-15-10=0$ ✓
And $2^2 + 3(2) - 10 = 4+6-10=0$ ✓
Hmm, but the problem asks for $x>0$, so...

{META_START}confidence: 0.25
The current route is weak because factoring gives roots but does not address the constraint x > 0 combined with the second equation.
I should switch to substitution into the second equation directly.{META_END}

Using substitution instead: from $y = x+1$, we get...
[continues with different method]
$$\\boxed{{7}}$$
"""


def _build_user_prompt(
    question: str,
    gold_answer: str,
    wrong_completion: str,
    correct_completion: str,
) -> str:
    """Build the user prompt for GPT-5.4 stitching.

    Truncates the wrong completion to 500 chars (we only need the opening)
    and provides the full correct completion as reference.
    """
    # Truncate wrong completion at paragraph boundary (avoid mid-formula cut)
    max_trunc = 500
    wrong_truncated = wrong_completion[:max_trunc]
    if len(wrong_completion) > max_trunc:
        last_newline = wrong_truncated.rfind("\n")
        if last_newline > max_trunc // 2:
            wrong_truncated = wrong_truncated[:last_newline]
        wrong_truncated += "\n[... truncated, solver continued but got wrong answer]"

    return f"""\
Problem: {question}
Gold answer: {gold_answer}

Failed approach (DO NOT complete this -- use only the beginning):
{wrong_truncated}

Successful approach (use this method for the second half):
{correct_completion}

Write the complete solution with route switch."""


def generate_switch_trajectory(
    client,
    question: str,
    gold_answer: str,
    wrong_completion: str,
    correct_completion: str,
    model: str = "gpt-5.4-mini_2026-03-17",
    max_retries: int = 10,
) -> str | None:
    """Generate a single switch trajectory via TRAPI GPT-5.4.

    Returns the trajectory text, or None if all retries fail.
    Uses the same retry pattern as gen_metacot_v2.py:
      - 429 (rate limit): exponential backoff with jitter
      - 500/502/503 (server error): sleep 10-15s
      - Other errors: sleep 5s
    """
    system = SWITCH_SYSTEM_PROMPT
    user = _build_user_prompt(question, gold_answer, wrong_completion, correct_completion)

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
            if not text:
                continue

            # Quick sanity: must have a meta block and a boxed answer
            if META_START not in text:
                continue
            if "\\boxed" not in text and "boxed{" not in text:
                continue

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


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

_SWITCH_KEYWORDS = re.compile(
    r"\b(switch|instead|different\s+approach|alternative|change\s+method|"
    r"try\s+a\s+different|route\s+is\s+weak|this\s+approach\s+fails|"
    r"not\s+working|abandon|pivot|let\s+me\s+try)\b",
    re.IGNORECASE,
)


def _extract_meta_blocks(text: str) -> list[dict]:
    """Extract meta blocks from trajectory text.

    Returns list of dicts with keys: text, confidence.
    """
    pattern = re.compile(
        rf"{re.escape(META_START)}(.*?){re.escape(META_END)}",
        re.IGNORECASE | re.DOTALL,
    )
    blocks = []
    for match in pattern.finditer(text):
        block_text = match.group(1).strip()
        conf = _parse_confidence_from_block(block_text)
        blocks.append({"text": block_text, "confidence": conf})
    return blocks


def _parse_confidence_from_block(block_text: str) -> float | None:
    """Parse confidence value from a meta block's text content.

    Looks for patterns like 'confidence: 0.25', 'confidence 0.3',
    'probability 0.2', etc.
    """
    matches = re.findall(
        r"(?:probability|confidence)[:\s]*(\d+\.\d+|\d+)\s*%?",
        block_text,
        re.IGNORECASE,
    )
    if not matches:
        return None
    # Return the first confidence found in this block
    v = float(matches[0])
    if v > 1:
        v /= 100
    return max(0.0, min(1.0, v))


def validate_trajectory(text: str, gold_answer: str) -> dict:
    """Validate a generated switch trajectory.

    Checks:
      1. Has >= 1 meta block with confidence < 0.4
      2. At least one meta block contains switch language
      3. Final boxed answer matches gold (via _check_correctness)
      4. Total length < 8000 chars (~2048 tokens)

    Returns:
        Dict with keys: valid (bool), reason (str if invalid),
        confidence_at_switch (float|None), has_switch_language (bool),
        answer_correct (bool), n_meta_blocks (int)
    """
    result = {
        "valid": False,
        "reason": "",
        "confidence_at_switch": None,
        "has_switch_language": False,
        "answer_correct": False,
        "n_meta_blocks": 0,
    }

    # Check 4: length
    if len(text) > 8000:
        result["reason"] = f"too_long ({len(text)} chars)"
        return result

    # Check 1: meta blocks with low confidence
    blocks = _extract_meta_blocks(text)
    result["n_meta_blocks"] = len(blocks)

    if not blocks:
        result["reason"] = "no_meta_blocks"
        return result

    low_conf_blocks = [
        b for b in blocks
        if b["confidence"] is not None and b["confidence"] < 0.4
    ]
    if not low_conf_blocks:
        # Check if any block has confidence at all
        has_any_conf = any(b["confidence"] is not None for b in blocks)
        if has_any_conf:
            max_conf = max(
                b["confidence"] for b in blocks if b["confidence"] is not None
            )
            result["reason"] = f"no_low_confidence_block (min conf={max_conf:.2f})"
        else:
            result["reason"] = "no_confidence_in_meta_blocks"
        return result

    # Record the lowest confidence as the switch point
    result["confidence_at_switch"] = min(
        b["confidence"] for b in low_conf_blocks
    )

    # Check 2: switch language in any meta block
    for b in blocks:
        if _SWITCH_KEYWORDS.search(b["text"]):
            result["has_switch_language"] = True
            break

    if not result["has_switch_language"]:
        result["reason"] = "no_switch_language_in_meta"
        return result

    # Check 3: answer correctness — use LAST boxed answer (CRITICAL fix: avoid
    # first_match picking the wrong answer from the failed approach)
    last_meta_end = text.rfind(META_END)
    post_switch_text = text[last_meta_end:] if last_meta_end >= 0 else text
    result["answer_correct"] = _check_correctness(post_switch_text, gold_answer)
    if not result["answer_correct"]:
        # Fallback: try full text (in case answer is only in pre-switch)
        if not _check_correctness(text, gold_answer):
            result["reason"] = "wrong_answer"
            return result
        result["answer_correct"] = True

    # Check 5: structural method difference pre/post switch (CRITICAL fix:
    # prevent "fake switch" where model says "switch" but uses same method)
    if last_meta_end >= 0:
        pre_switch = text[:text.find(META_START)] if META_START in text else text[:last_meta_end]
        post_switch = text[last_meta_end + len(META_END):]
        if pre_switch and post_switch and not _approaches_differ(pre_switch, post_switch):
            result["reason"] = "same_method_pre_post_switch"
            return result

    # All checks passed
    result["valid"] = True
    result["reason"] = "ok"
    return result


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _save_results(rows: list[dict], output_dir: Path, tag: str) -> str:
    """Save results as parquet. Returns the file path."""
    df = pd.DataFrame(rows)
    path = output_dir / f"switch_trajectories_{tag}.parquet"
    df.to_parquet(path, index=False)
    return str(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate route-switching Meta-CoT trajectories via TRAPI GPT-5.4.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--pass_at_k_dir",
        default="results/pass_at_k/base_quick",
        help="Directory containing pass_at_k_*.json files (default: results/pass_at_k/base_quick)",
    )
    parser.add_argument(
        "--output_dir",
        default="data/switch_trajectories",
        help="Output directory for parquet files (default: data/switch_trajectories)",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=800,
        help="Maximum number of valid trajectories to generate (default: 800)",
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=8,
        help="Number of concurrent TRAPI requests (default: 8)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.4-mini_2026-03-17",
        help="TRAPI model name (default: gpt-5.4-mini_2026-03-17)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=10,
        help="Max retries per TRAPI call (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for candidate shuffling (default: 42)",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    print("=" * 60)
    print("  Route-Switching Trajectory Generator")
    print("=" * 60)
    print(f"  pass_at_k_dir:    {args.pass_at_k_dir}")
    print(f"  output_dir:       {args.output_dir}")
    print(f"  max_trajectories: {args.max_trajectories}")
    print(f"  concurrent:       {args.concurrent}")
    print(f"  model:            {args.model}")
    print(f"  max_retries:      {args.max_retries}")
    print(f"  seed:             {args.seed}")
    print()

    # ------------------------------------------------------------------
    # Phase A: Extract candidates
    # ------------------------------------------------------------------
    print("Phase A: Loading pass@k candidates ...")
    candidates = load_pass_at_k_candidates(args.pass_at_k_dir)

    if not candidates:
        print("Error: no switch candidates found. Ensure pass@k data exists.")
        print("Run compute_pass_at_k.py first to generate pass@k results.")
        sys.exit(1)

    # Shuffle and cap to avoid wasting API calls far beyond target
    random.shuffle(candidates)
    # Allow extra attempts since not all will validate
    attempt_cap = min(len(candidates), args.max_trajectories * 3)
    candidates = candidates[:attempt_cap]
    print(f"Will attempt up to {len(candidates)} candidates "
          f"(targeting {args.max_trajectories} valid trajectories)")
    print()

    # ------------------------------------------------------------------
    # Phase B: Generate trajectories
    # ------------------------------------------------------------------
    print("Phase B: Generating switch trajectories via TRAPI ...")
    client = get_trapi_client()
    print(f"TRAPI client ready. Using {args.concurrent} concurrent workers.\n")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    valid_rows: list[dict] = []
    failed_count = 0
    invalid_reasons: dict[str, int] = {}
    total_attempted = 0

    def _process_candidate(cand: dict) -> dict | None:
        """Process a single candidate: generate and validate."""
        text = generate_switch_trajectory(
            client=client,
            question=cand["question"],
            gold_answer=cand["gold_answer"],
            wrong_completion=cand["wrong_completion"],
            correct_completion=cand["correct_completion"],
            model=args.model,
            max_retries=args.max_retries,
        )

        if text is None:
            return {"status": "generation_failed", "cand": cand}

        validation = validate_trajectory(text, cand["gold_answer"])

        if not validation["valid"]:
            return {
                "status": "invalid",
                "reason": validation["reason"],
                "cand": cand,
            }

        return {
            "status": "valid",
            "row": {
                "problem_id": cand["problem_id"],
                "question": cand["question"],
                "gold_answer": cand["gold_answer"],
                "benchmark": cand["benchmark"],
                "completion": text,
                "has_switch": True,
                "confidence_at_switch": validation["confidence_at_switch"],
                "n_meta_blocks": validation["n_meta_blocks"],
                "source_wrong_idx": cand["wrong_sample_idx"],
                "source_correct_idx": cand["correct_sample_idx"],
            },
        }

    with ThreadPoolExecutor(max_workers=args.concurrent) as executor:
        futures = {}
        for cand in candidates:
            fut = executor.submit(_process_candidate, cand)
            futures[fut] = cand

        for future in as_completed(futures):
            total_attempted += 1

            try:
                result = future.result()
            except Exception:
                failed_count += 1
                traceback.print_exc()
                continue

            if result["status"] == "valid":
                valid_rows.append(result["row"])

            elif result["status"] == "generation_failed":
                failed_count += 1

            elif result["status"] == "invalid":
                reason = result["reason"]
                invalid_reasons[reason] = invalid_reasons.get(reason, 0) + 1

            # Progress: print every 10 trajectories
            if total_attempted % 10 == 0:
                n_invalid = sum(invalid_reasons.values())
                print(
                    f"  [{total_attempted}/{len(candidates)}] "
                    f"valid={len(valid_rows)} "
                    f"invalid={n_invalid} "
                    f"failed={failed_count}",
                    flush=True,
                )

            # Save intermediate results every 50 trajectories
            if len(valid_rows) > 0 and len(valid_rows) % 50 == 0:
                ckpt_path = _save_results(
                    valid_rows, output_dir, f"checkpoint_{len(valid_rows)}"
                )
                print(f"  Checkpoint saved: {ckpt_path}", flush=True)

            # Early stop: reached target
            if len(valid_rows) >= args.max_trajectories:
                print(
                    f"\n  Reached target of {args.max_trajectories} valid "
                    f"trajectories. Stopping early.",
                    flush=True,
                )
                # Cancel remaining futures (best-effort)
                for f in futures:
                    f.cancel()
                break

    # ------------------------------------------------------------------
    # Save final results
    # ------------------------------------------------------------------
    if valid_rows:
        final_path = _save_results(valid_rows, output_dir, "final")
        print(f"\nFinal results saved to {final_path}")
    else:
        print("\nWarning: no valid trajectories generated.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_invalid = sum(invalid_reasons.values())
    print()
    print("=" * 60)
    print("  Generation Summary")
    print("=" * 60)
    print(f"  Candidates attempted: {total_attempted}")
    print(f"  Valid trajectories:   {len(valid_rows)}")
    print(f"  Invalid (rejected):   {n_invalid}")
    print(f"  Failed (API error):   {failed_count}")
    print(f"  Yield rate:           {len(valid_rows) / total_attempted:.1%}"
          if total_attempted > 0 else "  Yield rate: N/A")

    if invalid_reasons:
        print(f"\n  Rejection reasons:")
        for reason, count in sorted(invalid_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count}")

    if valid_rows:
        confs = [
            r["confidence_at_switch"]
            for r in valid_rows
            if r["confidence_at_switch"] is not None
        ]
        if confs:
            print(f"\n  Confidence at switch point:")
            print(f"    mean = {sum(confs) / len(confs):.3f}")
            print(f"    min  = {min(confs):.3f}")
            print(f"    max  = {max(confs):.3f}")

        # Per-benchmark breakdown
        benchmarks = sorted(set(r["benchmark"] for r in valid_rows))
        if len(benchmarks) > 1:
            print(f"\n  Per-benchmark counts:")
            for bench in benchmarks:
                count = sum(1 for r in valid_rows if r["benchmark"] == bench)
                print(f"    {bench}: {count}")

    print()


if __name__ == "__main__":
    main()
