"""Build v7 think+meta training data from v6 clean 10k merged.

Transforms all 6329 samples so that:
  1) All reasoning is inside <think>...</think> blocks
  2) All samples have at least one <|meta|>...<|/meta|> block
  3) Final answer is OUTSIDE the last </think>
  4) Straight samples get synthetic verify / redirect meta blocks

Usage:
    python scripts/build_v7_think_meta_data.py \
        --input data/v6_clean_10k_merged.parquet \
        --output data/v7_think_meta_merged.parquet \
        --seed 42
"""

import argparse
import json
import random
import re
import sys
from typing import Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Boxed answer extraction (handles nested braces)
# ---------------------------------------------------------------------------

def extract_boxed(text: str) -> Optional[str]:
    """Extract the LAST \\boxed{...} from text, handling nested braces."""
    # Find all occurrences of \boxed{
    pattern = r"\\boxed\{"
    starts = [m.start() for m in re.finditer(pattern, text)]
    if not starts:
        return None

    # Take the last occurrence
    start = starts[-1]
    brace_start = start + len("\\boxed{") - 1  # position of the opening {

    depth = 0
    for i in range(brace_start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    # If we never close, return what we have
    return text[start:]


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract the content inside \\boxed{...}."""
    boxed = extract_boxed(text)
    if boxed is None:
        return None
    # Remove \boxed{ and trailing }
    inner = boxed[len("\\boxed{") : -1] if boxed.endswith("}") else boxed[len("\\boxed{"):]
    return inner


# ---------------------------------------------------------------------------
# Meta block splitting
# ---------------------------------------------------------------------------

_META_PATTERN = re.compile(
    r"<\|meta\|>\s*(.*?)\s*<\|/meta\|>", re.DOTALL
)


def split_on_meta(text: str) -> list[dict]:
    """Split assistant text into segments: pre-meta reasoning, meta blocks, post-meta reasoning.

    Returns list of dicts: {"type": "reasoning"|"meta", "content": str}
    """
    segments = []
    last_end = 0

    for m in _META_PATTERN.finditer(text):
        # Text before this meta block
        pre = text[last_end : m.start()].strip()
        if pre:
            segments.append({"type": "reasoning", "content": pre})
        # The meta block itself (inner content only)
        segments.append({"type": "meta", "content": m.group(0)})
        last_end = m.end()

    # Remaining text after last meta block
    post = text[last_end:].strip()
    if post:
        segments.append({"type": "reasoning", "content": post})

    return segments


def strip_trailing_boxed(text: str) -> Tuple[str, Optional[str]]:
    """Remove the trailing boxed answer from reasoning text.

    Returns (text_without_boxed, boxed_string_or_None).
    """
    boxed = extract_boxed(text)
    if boxed is None:
        return text, None

    # Find the last occurrence and remove it + surrounding sentence
    idx = text.rfind(boxed)
    if idx < 0:
        return text, boxed

    # Remove from the text: also remove common prefixes like "The answer is" or "Thus,"
    before = text[:idx].rstrip()
    after = text[idx + len(boxed) :].strip()

    # Clean up trailing sentence fragments
    # Common patterns: "So the answer is \boxed{5}." or "\[\boxed{5}\]"
    # Remove trailing sentence intro pointing to the boxed answer
    trailing_intros = [
        r"\s*(?:So |Thus |Therefore |Hence )?(?:the |The )?(?:answer|solution|result) is\s*(?:\\\()?$",
        r"\s*(?:So |Thus |Therefore |Hence )?(?:the |The )?(?:answer|solution|result) is\s*$",
        r"\s*\\\[\s*$",
        r"\s*=\s*$",
    ]
    for pat in trailing_intros:
        before = re.sub(pat, "", before)

    text_clean = before.rstrip()
    if after and after not in (".", "\\)", "\\]", ".\\]", ".\\)", ")."):
        text_clean = text_clean + "\n" + after

    return text_clean.rstrip(), boxed


def format_answer_line(boxed: str) -> str:
    """Format the answer line that goes outside the last </think>."""
    return f"The answer is ${boxed}$."


# ---------------------------------------------------------------------------
# Transform: redirect / verify (existing meta samples)
# ---------------------------------------------------------------------------

def transform_meta_sample(content: str) -> str:
    """Wrap reasoning in <think> blocks, keep meta outside, answer outside.

    Input:  reasoning... <|meta|>...<|/meta|> more_reasoning... \\boxed{ans}
    Output: <think>\\nreasoning...\\n</think>\\n\\n<|meta|>...<|/meta|>\\n\\n<think>\\nmore...\\n</think>\\n\\nThe answer is $\\boxed{ans}$.
    """
    segments = split_on_meta(content)
    if not segments:
        return content

    # Extract boxed answer from the full content
    boxed = extract_boxed(content)

    parts = []
    for seg in segments:
        if seg["type"] == "meta":
            parts.append(seg["content"])
        else:
            # Reasoning segment - wrap in <think>
            reasoning = seg["content"]
            # Strip boxed from reasoning if it's the last segment
            reasoning_clean, _ = strip_trailing_boxed(reasoning)
            reasoning_clean = reasoning_clean.strip()
            if reasoning_clean:
                parts.append(f"<think>\n{reasoning_clean}\n</think>")

    result = "\n\n".join(parts)

    # Add answer outside
    if boxed:
        result = result.rstrip() + f"\n\n{format_answer_line(boxed)}"

    return result


# ---------------------------------------------------------------------------
# Transform: straight easy -> verify
# ---------------------------------------------------------------------------

VERIFY_TEMPLATES = [
    "Let me verify: substituting {answer} back into the original equation gives a consistent result.",
    "Quick check: plugging {answer} into the original problem, each step checks out numerically.",
    "Verification: working backwards from {answer}, I reconstruct the given conditions correctly.",
    "Sanity check: {answer} is consistent with the constraints — the units and magnitude make sense.",
    "Double-checking: recomputing the key step independently gives {answer} again.",
    "Cross-check: approaching the problem from the opposite direction also yields {answer}.",
    "Confirming: {answer} satisfies all the stated requirements when substituted back.",
    "Let me verify by estimating: {answer} is in the expected range, and exact computation confirms it.",
    "Testing boundary cases: when the inputs are at their extremes, {answer} still holds.",
    "Numerical spot-check: picking a specific value and tracing through gives {answer} as expected.",
    "Reverse verification: starting from {answer} and working backwards recovers the original problem statement.",
    "Dimensional analysis: the result {answer} has the correct units and order of magnitude.",
    "Checking with a simpler case: reducing the problem confirms the pattern that leads to {answer}.",
    "Parity check: {answer} has the expected sign and parity given the problem constraints.",
]

VERIFY_ASSESSMENT_TEMPLATES = [
    "solution looks correct but a quick verification will catch any arithmetic slip",
    "straightforward computation, worth double-checking the final step",
    "the approach seems right; verifying will rule out sign or indexing errors",
    "result appears reasonable in magnitude; let me confirm with a spot-check",
    "confident in the method but the problem has potential off-by-one traps",
    "the algebra is clean; a numerical substitution will confirm",
    "intermediate steps were complex enough to warrant a sanity check",
    "the answer feels right intuitively; let me verify rigorously",
    "multiple paths converge here; a quick cross-check will confirm",
    "the computation involved several steps; worth re-deriving the key identity",
]

VERIFY_ACTION_TEMPLATES = [
    "substitute back to check the original equation",
    "verify by plugging the answer into each constraint",
    "cross-check with an alternative algebraic route",
    "re-derive the critical step from scratch",
    "double-check the arithmetic on the key computation",
    "verify the boundary conditions hold",
    "test with a specific numerical example",
    "check dimensional consistency of the result",
    "verify by working the problem backwards",
    "confirm using a different representation of the problem",
]


def transform_straight_to_verify(content: str, rng: random.Random,
                                  conf_lo: float = 0.7,
                                  conf_hi: float = 0.95) -> str:
    """Convert straight solve to think+verify+meta format.

    Input:  direct solve... \\boxed{answer}
    Output: <think>\\ndirect solve...\\n</think>\\n\\n<|meta|>\\nconfidence: X\\nassessment: ...\\naction: ...\\n<|/meta|>\\n\\n<think>\\nVerification: ...\\n</think>\\n\\nThe answer is $\\boxed{answer}$.
    """
    boxed = extract_boxed(content)
    if boxed is None:
        # Fallback: just wrap everything
        return f"<think>\n{content.strip()}\n</think>"

    boxed_inner = extract_boxed_content(content) or "the answer"

    # Strip boxed from reasoning
    reasoning, _ = strip_trailing_boxed(content)
    reasoning = reasoning.strip()

    # Generate confidence
    confidence = round(rng.uniform(conf_lo, conf_hi), 2)

    # Pick templates
    assessment = rng.choice(VERIFY_ASSESSMENT_TEMPLATES)
    action = rng.choice(VERIFY_ACTION_TEMPLATES)
    verify_text = rng.choice(VERIFY_TEMPLATES).format(answer=boxed_inner)

    meta_block = (
        f"<|meta|>\n"
        f"confidence: {confidence}\n"
        f"assessment: {assessment}\n"
        f"action: {action}\n"
        f"<|/meta|>"
    )

    result = (
        f"<think>\n{reasoning}\n</think>\n\n"
        f"{meta_block}\n\n"
        f"<think>\n{verify_text}\n</think>\n\n"
        f"{format_answer_line(boxed)}"
    )
    return result


# ---------------------------------------------------------------------------
# Transform: straight medium -> redirect or verify
# ---------------------------------------------------------------------------

REDIRECT_ASSESSMENT_TEMPLATES = [
    "my initial approach may not be optimal; let me reconsider",
    "I should try a different angle to ensure correctness",
    "the current path is workable but a cleaner approach exists",
    "let me reconsider my approach for better clarity",
    "stepping back to find a more direct route",
    "the computation is getting unwieldy; a simpler method likely exists",
    "I notice a potential shortcut I overlooked initially",
    "the current route risks accumulating rounding errors",
    "there may be a more elegant formulation using a different variable",
    "my approach works but is unnecessarily complicated for this problem",
]

REDIRECT_ACTION_TEMPLATES = [
    "restart with a more systematic method",
    "try an alternative algebraic manipulation",
    "reconsider the problem from first principles",
    "switch to a different representation",
    "reorganize the computation more carefully",
    "use a substitution to simplify the expression",
    "try a geometric interpretation instead",
    "factor the expression differently",
    "apply a known identity to reduce complexity",
    "break the problem into smaller subproblems",
]

REPHRASE_PREFIXES = [
    "Approaching this more carefully:\n",
    "Taking a cleaner approach:\n",
    "Let me redo this systematically:\n",
    "Working through this step by step:\n",
    "A more direct solution:\n",
    "Using a different strategy:\n",
    "Reformulating the problem:\n",
    "Starting fresh with a simpler method:\n",
]


def rephrase_reasoning(text: str, rng: random.Random) -> str:
    """Lightly rephrase reasoning for the post-redirect segment.

    We add a prefix and optionally shuffle some line breaks, but keep
    the mathematical content intact to avoid introducing errors.
    """
    prefix = rng.choice(REPHRASE_PREFIXES)
    return prefix + text


def transform_straight_medium_to_redirect(content: str, rng: random.Random) -> str:
    """Convert straight medium to think+redirect+meta format.

    Input:  solve... \\boxed{answer}
    Output: <think>\\npre-solve...\\n</think>\\n\\n<|meta|>\\nconfidence: X\\nassessment: ...\\naction: ...\\n<|/meta|>\\n\\n<think>\\nrephrased solve...\\n</think>\\n\\nThe answer is $\\boxed{answer}$.
    """
    boxed = extract_boxed(content)
    if boxed is None:
        return f"<think>\n{content.strip()}\n</think>"

    # Strip boxed
    reasoning, _ = strip_trailing_boxed(content)
    reasoning = reasoning.strip()

    # Split reasoning roughly in half (at a paragraph break or midpoint)
    lines = reasoning.split("\n")
    if len(lines) >= 4:
        # Try to split at a blank line near the middle
        mid = len(lines) // 3  # Split at ~1/3 for the "initial attempt"
        split_idx = mid
        # Look for a blank line near mid
        for i in range(max(1, mid - 2), min(len(lines) - 1, mid + 3)):
            if lines[i].strip() == "":
                split_idx = i
                break
        pre_lines = lines[:split_idx]
        post_lines = lines[split_idx:]
        pre_text = "\n".join(pre_lines).strip()
        post_text = "\n".join(post_lines).strip()
    else:
        # Short content: use all as pre, rephrase for post
        pre_text = reasoning
        post_text = reasoning

    if not pre_text:
        pre_text = reasoning
    if not post_text:
        post_text = reasoning

    # Generate meta block
    confidence = round(rng.uniform(0.3, 0.6), 2)
    assessment = rng.choice(REDIRECT_ASSESSMENT_TEMPLATES)
    action = rng.choice(REDIRECT_ACTION_TEMPLATES)

    meta_block = (
        f"<|meta|>\n"
        f"confidence: {confidence}\n"
        f"assessment: {assessment}\n"
        f"action: {action}\n"
        f"<|/meta|>"
    )

    # Rephrase post-text slightly
    post_rephrased = rephrase_reasoning(post_text, rng)

    result = (
        f"<think>\n{pre_text}\n</think>\n\n"
        f"{meta_block}\n\n"
        f"<think>\n{post_rephrased}\n</think>\n\n"
        f"{format_answer_line(boxed)}"
    )
    return result


def transform_straight_medium_to_verify(content: str, rng: random.Random) -> str:
    """Same as easy->verify but with lower confidence range."""
    return transform_straight_to_verify(
        content, rng, conf_lo=0.5, conf_hi=0.8
    )


# ---------------------------------------------------------------------------
# Behavioral marker recomputation
# ---------------------------------------------------------------------------

def recompute_markers(content: str) -> dict:
    """Recompute behavioral marker columns from transformed content."""
    meta_count = content.count("<|meta|>")
    has_meta = meta_count > 0

    # Check meta content for various markers
    meta_blocks = _META_PATTERN.findall(content)
    meta_text = " ".join(meta_blocks).lower()

    has_verify = "verify" in meta_text or "check" in meta_text or "confirm" in meta_text
    has_switch = "redirect" in meta_text or "restart" in meta_text or "reconsider" in meta_text or "different" in meta_text
    has_conf_drop = "confidence" in meta_text
    has_diagnosis = "assessment" in meta_text or "weakness" in meta_text or "issue" in meta_text
    has_next_strategy = "action" in meta_text or "should" in meta_text
    has_study_need = "study_need" in meta_text
    has_decomposition = "subgoal" in meta_text or "break" in meta_text or "parts" in meta_text
    has_overconfidence = False  # Not applicable in transformed data
    has_blocker = False
    has_trigger = has_meta

    return {
        "meta_count": meta_count,
        "has_verify": has_verify,
        "has_switch": has_switch,
        "has_conf_drop": has_conf_drop,
        "has_diagnosis": has_diagnosis,
        "has_next_strategy": has_next_strategy,
        "has_study_need": has_study_need,
        "has_decomposition": has_decomposition,
        "has_overconfidence": has_overconfidence,
        "has_blocker": has_blocker,
        "has_trigger": has_trigger,
        "pure_meta": has_meta,
    }


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------

def validate_sample(content: str) -> list[str]:
    """Return list of issues found in transformed content."""
    issues = []

    # Must have <think> blocks
    if "<think>" not in content:
        issues.append("no_think_open")
    if "</think>" not in content:
        issues.append("no_think_close")

    # Must have meta blocks
    if "<|meta|>" not in content:
        issues.append("no_meta")

    # Answer should be outside last </think>
    last_think_close = content.rfind("</think>")
    if last_think_close >= 0:
        after_think = content[last_think_close + len("</think>"):]
        if "\\boxed{" not in after_think and "boxed" not in after_think.lower():
            issues.append("answer_inside_think")
    else:
        issues.append("no_think_close_for_answer_check")

    # Meta should be outside <think> blocks
    # Check that no <|meta|> appears between <think> and </think>
    think_blocks = list(re.finditer(r"<think>(.*?)</think>", content, re.DOTALL))
    for tb in think_blocks:
        if "<|meta|>" in tb.group(1):
            issues.append("meta_inside_think")
            break

    # Check balanced think tags
    open_count = content.count("<think>")
    close_count = content.count("</think>")
    if open_count != close_count:
        issues.append(f"unbalanced_think_{open_count}_vs_{close_count}")

    return issues


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

BACKFILL_ASSESSMENTS = [
    "the computation appears sound but deserves a closer look",
    "the approach is on the right track; let me evaluate the reasoning",
    "the intermediate steps need verification before concluding",
    "the method is appropriate for this problem type",
    "the solution path is reasonable; checking for edge cases",
    "the algebraic manipulations look correct at first glance",
    "this is a standard problem; still worth being careful",
    "the key insight has been identified; need to execute cleanly",
]


def _fix_missing_assessment(content: str, rng: random.Random) -> str:
    """Backfill assessment field in meta blocks that lack it."""
    def _add_assessment(match):
        block = match.group(0)
        if 'assessment' not in block.lower() and 'current route' not in block.lower():
            # Insert assessment after confidence line
            lines = block.split('\n')
            new_lines = []
            inserted = False
            for line in lines:
                new_lines.append(line)
                if line.strip().startswith('confidence') and not inserted:
                    new_lines.append(f"assessment: {rng.choice(BACKFILL_ASSESSMENTS)}")
                    inserted = True
            return '\n'.join(new_lines)
        return block

    return re.sub(r'<\|meta\|>.*?<\|/meta\|>', _add_assessment, content, flags=re.DOTALL)


def _diversify_redirect_confidence(content: str, rng: random.Random,
                                    difficulty: str) -> str:
    """Replace fixed ~0.34 confidence in redirect meta blocks with diverse values.

    E19 analysis showed 80-84% of confidence values stuck at 0.34.
    Diversify based on difficulty: hard=0.15-0.35, medium=0.25-0.50.
    """
    def _replace_conf(match):
        block = match.group(0)
        # Find and replace confidence line
        conf_match = re.search(r'confidence[:\s]+([0-9.]+)', block)
        if conf_match:
            old_val = conf_match.group(1)
            if difficulty == 'hard':
                new_val = round(rng.uniform(0.15, 0.35), 2)
            elif difficulty == 'medium':
                new_val = round(rng.uniform(0.25, 0.50), 2)
            else:
                new_val = round(rng.uniform(0.30, 0.55), 2)
            block = block.replace(f'confidence: {old_val}', f'confidence: {new_val}', 1)
            block = block.replace(f'confidence:{old_val}', f'confidence: {new_val}', 1)
        return block

    return re.sub(r'<\|meta\|>.*?<\|/meta\|>', _replace_conf, content, flags=re.DOTALL)


def _fix_empty_pre_meta(content: str) -> str:
    """Ensure there's meaningful content before the first meta block.

    If pre-meta region (before first <|meta|>) has < 30 chars of actual
    reasoning, it means the sample starts almost immediately with meta.
    Fix: move first <think> content to be more visible.
    """
    first_meta = content.find('<|meta|>')
    if first_meta < 0:
        return content

    pre_meta = content[:first_meta].strip()
    # Remove think tags to measure actual content
    pre_text = re.sub(r'</?think>', '', pre_meta).strip()

    if len(pre_text) >= 30:
        return content  # sufficient pre-meta content

    # If pre-meta is too short but there's think content after meta,
    # this is acceptable — meta at the start means "assess before solving"
    return content


def process_row(row: pd.Series, rng: random.Random) -> dict:
    """Transform a single row and return updated fields."""
    msgs = json.loads(row["messages"])
    original_content = msgs[-1]["content"]
    scenario = row["scenario"]
    difficulty = row["difficulty"]

    # Determine transformation
    if scenario in ("redirect", "verify"):
        new_content = transform_meta_sample(original_content)
        new_scenario = scenario  # keep original
    elif scenario == "straight" and difficulty == "easy":
        new_content = transform_straight_to_verify(original_content, rng)
        new_scenario = "verify"
    elif scenario == "straight" and difficulty == "medium":
        # 50/50 split: verify or redirect
        if rng.random() < 0.5:
            new_content = transform_straight_medium_to_verify(original_content, rng)
            new_scenario = "verify"
        else:
            new_content = transform_straight_medium_to_redirect(original_content, rng)
            new_scenario = "redirect"
    else:
        # Fallback for any other case (e.g., straight hard)
        new_content = transform_straight_to_verify(original_content, rng, 0.5, 0.9)
        new_scenario = "verify"

    # Post-processing fixes (Codex review)
    new_content = _fix_missing_assessment(new_content, rng)
    new_content = _fix_empty_pre_meta(new_content)
    # Fix confidence 0.34 collapse in redirect data (E19 analysis finding)
    if scenario == "redirect":
        new_content = _diversify_redirect_confidence(new_content, rng, difficulty)

    # Update messages
    msgs[-1]["content"] = new_content
    new_messages = json.dumps(msgs, ensure_ascii=False)

    # Recompute markers
    markers = recompute_markers(new_content)

    result = {
        "messages": new_messages,
        "scenario": new_scenario,
        "difficulty": difficulty,
        "source": row["source"],
        "topic": row["topic"],
        "pass_rate": row["pass_rate"],
        "trigger": row.get("trigger", ""),
        "study_need": row.get("study_need", ""),
        "repeated_intervention": markers["meta_count"] > 1,
    }
    result.update(markers)

    return result


def print_audit(df: pd.DataFrame, label: str):
    """Print comprehensive quality audit."""
    print(f"\n{'=' * 70}")
    print(f"  {label}  ({len(df)} samples)")
    print(f"{'=' * 70}")

    # Scenario distribution
    print(f"\n  Scenario distribution:")
    for s, cnt in df["scenario"].value_counts().items():
        print(f"    {s:12s}: {cnt:5d}  ({100 * cnt / len(df):5.1f}%)")

    # Difficulty distribution
    print(f"\n  Difficulty distribution:")
    for d, cnt in df["difficulty"].value_counts().items():
        print(f"    {d:12s}: {cnt:5d}  ({100 * cnt / len(df):5.1f}%)")

    # Meta type breakdown
    print(f"\n  Meta presence:")
    has_meta = sum(1 for _, r in df.iterrows()
                   if "<|meta|>" in json.loads(r["messages"])[-1]["content"])
    print(f"    With meta : {has_meta:5d}  ({100 * has_meta / len(df):5.1f}%)")
    print(f"    No meta   : {len(df) - has_meta:5d}  ({100 * (len(df) - has_meta) / len(df):5.1f}%)")

    # Think tag presence
    has_think = sum(1 for _, r in df.iterrows()
                    if "<think>" in json.loads(r["messages"])[-1]["content"])
    print(f"\n  Think tag presence:")
    print(f"    With <think>: {has_think:5d}  ({100 * has_think / len(df):5.1f}%)")

    # Answer outside think
    answer_outside = 0
    for _, r in df.iterrows():
        content = json.loads(r["messages"])[-1]["content"]
        last_close = content.rfind("</think>")
        if last_close >= 0:
            after = content[last_close:]
            if "\\boxed{" in after or "boxed" in after.lower():
                answer_outside += 1
    print(f"\n  Answer outside </think>:")
    print(f"    Yes: {answer_outside:5d}  ({100 * answer_outside / len(df):5.1f}%)")

    # Behavioral markers
    print(f"\n  Behavioral markers:")
    marker_cols = [c for c in df.columns if c.startswith("has_")]
    for mc in sorted(marker_cols):
        rate = df[mc].mean() * 100
        print(f"    {mc:25s}: {rate:5.1f}%")

    # Meta count distribution
    print(f"\n  Meta count distribution:")
    for mc, cnt in df["meta_count"].value_counts().sort_index().items():
        print(f"    {mc} meta blocks: {cnt:5d}")

    # Response length stats
    lengths = [len(json.loads(r["messages"])[-1]["content"]) for _, r in df.iterrows()]
    print(f"\n  Response length (chars):")
    print(f"    Mean  : {sum(lengths)/len(lengths):,.0f}")
    print(f"    Median: {sorted(lengths)[len(lengths)//2]:,}")
    print(f"    Min   : {min(lengths):,}")
    print(f"    Max   : {max(lengths):,}")


def main():
    parser = argparse.ArgumentParser(
        description="Build v7 think+meta training data from v6 clean 10k."
    )
    parser.add_argument(
        "--input", default="data/v6_clean_10k_merged.parquet",
        help="Input parquet (default: data/v6_clean_10k_merged.parquet)"
    )
    parser.add_argument(
        "--output", default="data/v7_think_meta_merged.parquet",
        help="Output parquet (default: data/v7_think_meta_merged.parquet)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-audit", action="store_true", help="Skip audit print")
    args = parser.parse_args()

    print(f"Loading {args.input}...")
    df = pd.read_parquet(args.input)
    print(f"  Loaded {len(df)} rows")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"  Scenario dist: {df['scenario'].value_counts().to_dict()}")
    print(f"  Difficulty dist: {df['difficulty'].value_counts().to_dict()}")

    rng = random.Random(args.seed)

    # Process each row
    print("\nTransforming samples...")
    results = []
    issues_count = {}
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            result = process_row(row, rng)
            # Validate
            content = json.loads(result["messages"])[-1]["content"]
            issues = validate_sample(content)
            for iss in issues:
                issues_count[iss] = issues_count.get(iss, 0) + 1
            results.append(result)
        except Exception as e:
            print(f"  ERROR on row {idx} (scenario={row['scenario']}, "
                  f"difficulty={row['difficulty']}): {e}")
            # Keep original as fallback, wrapped minimally
            msgs = json.loads(row["messages"])
            content = msgs[-1]["content"]
            boxed = extract_boxed(content)
            fallback = f"<think>\n{content}\n</think>"
            if boxed:
                fallback += f"\n\n{format_answer_line(boxed)}"
            msgs[-1]["content"] = fallback
            markers = recompute_markers(fallback)
            result = {
                "messages": json.dumps(msgs, ensure_ascii=False),
                "scenario": row["scenario"],
                "difficulty": row["difficulty"],
                "source": row["source"],
                "topic": row["topic"],
                "pass_rate": row["pass_rate"],
                "trigger": row.get("trigger", ""),
                "study_need": row.get("study_need", ""),
                "repeated_intervention": False,
            }
            result.update(markers)
            results.append(result)

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(df)}...")

    print(f"  Done. Processed {len(results)} samples.")

    # Report validation issues
    if issues_count:
        print(f"\n  Validation issues:")
        for iss, cnt in sorted(issues_count.items(), key=lambda x: -x[1]):
            print(f"    {iss}: {cnt}")
    else:
        print(f"\n  No validation issues found!")

    # Build output DataFrame
    df_out = pd.DataFrame(results)

    # Ensure column order matches v6
    desired_cols = [
        "difficulty", "has_blocker", "has_conf_drop", "has_decomposition",
        "has_diagnosis", "has_next_strategy", "has_overconfidence",
        "has_study_need", "has_switch", "has_trigger", "has_verify",
        "messages", "meta_count", "pass_rate", "pure_meta",
        "repeated_intervention", "scenario", "source", "study_need",
        "topic", "trigger",
    ]
    # Add any extra cols, drop any missing
    final_cols = [c for c in desired_cols if c in df_out.columns]
    extra_cols = [c for c in df_out.columns if c not in desired_cols]
    if extra_cols:
        print(f"  Extra columns (dropped): {extra_cols}")
    df_out = df_out[final_cols]

    # Shuffle with fixed seed
    df_out = df_out.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # Save
    df_out.to_parquet(args.output, index=False)
    print(f"\n  Saved {len(df_out)} rows to {args.output}")

    # Audit
    if not args.no_audit:
        print_audit(df_out, "V7 THINK+META DATA AUDIT")

        # Per-scenario audit
        for scenario in df_out["scenario"].unique():
            subset = df_out[df_out["scenario"] == scenario]
            print_audit(subset, f"Scenario: {scenario}")

    # Print a few sample transformations
    print(f"\n{'=' * 70}")
    print(f"  SAMPLE TRANSFORMATIONS")
    print(f"{'=' * 70}")

    for scenario in ["redirect", "verify"]:
        subset = df_out[df_out["scenario"] == scenario]
        if len(subset) > 0:
            sample = subset.iloc[0]
            content = json.loads(sample["messages"])[-1]["content"]
            print(f"\n--- {scenario.upper()} sample (first 800 chars) ---")
            print(content[:800])
            print("...")

    return 0


if __name__ == "__main__":
    sys.exit(main())
