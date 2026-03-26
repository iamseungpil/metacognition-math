"""Meta-CoT prompt templates with <|meta|> special tokens."""

# Special tokens to add to tokenizer
META_START = "<|meta|>"
META_END = "<|/meta|>"

# In-context example for GPT-5.4 data generation
INCONTEXT_EXAMPLE = f"""\
Example of the desired output format:

{META_START}
Q: Can I solve this problem?
A: This is a number theory problem about modular arithmetic. I think my probability of solving it correctly is about 0.40. The key risk is making errors in cycle detection.

Q: What should I watch out for?
A: I need to find the pattern in powers mod 7. I should verify each step carefully since remainder calculations are error-prone.
{META_END}

2^1 ≡ 2 (mod 7)
2^2 ≡ 4 (mod 7)
2^3 ≡ 8 ≡ 1 (mod 7)

{META_START}
Q: Is this calculation correct?
A: 2^3 = 8, and 8 - 7 = 1, so 2^3 ≡ 1 (mod 7). Yes, this is correct. Confidence: 0.90.
{META_END}

Since the powers repeat every 3 steps:
100 = 33 × 3 + 1

{META_START}
Q: Is this division correct?
A: 33 × 3 = 99, 100 - 99 = 1. Yes, remainder is 1. Confidence: 0.95.
{META_END}

Therefore 2^100 ≡ 2^1 ≡ 2 (mod 7).

\\boxed{{2}}

{META_START}
Q: What did I learn from this problem?
A: Cycle detection in modular arithmetic was the key technique. I should practice Euler's theorem for cases where the modulus is not prime.
{META_END}
"""

META_COT_SYSTEM_PROMPT = f"""\
You are solving math problems with self-awareness. \
Wrap your self-reflective thinking in {META_START} and {META_END} tags.

Inside these tags, ask yourself questions and answer them:
- Before solving: "Can I solve this?", "What should I watch out for?"
- During solving: "Is this step correct?", "Am I confident about this?"
- After solving: "What did I learn?"

Include a numeric probability or confidence (0.0 to 1.0) in your pre-solve assessment.
Put your final answer in \\boxed{{}}.

{INCONTEXT_EXAMPLE}
"""


def build_metacot_user_prompt(
    profile: dict,
    question: str,
    model_answer: str,
    correct_answer: str,
    is_correct: bool,
    data_pool_summary: str = "",
    rollout_pass_rate: float = None,
) -> str:
    """Build user prompt for GPT-5.4 data generation."""
    profile_str = _format_profile(profile)

    hint = ""
    if rollout_pass_rate is not None:
        hint = f"\nNote: The student solves similar problems correctly about {rollout_pass_rate:.0%} of the time."

    return f"""\
Student performance profile:
{profile_str}
{hint}

Problem:
{question}

Reference answer: {correct_answer}

Generate a solution in the format shown in the system prompt, \
with {META_START}/{META_END} self-reflection blocks before, during, and after solving.
"""


def _format_profile(profile: dict) -> str:
    lines = []
    lines.append(f"Overall accuracy: {profile.get('overall_pass_at_1', 0):.1%}")
    cat_acc = profile.get("category_accuracy", {})
    if cat_acc:
        for cat, diffs in cat_acc.items():
            if isinstance(diffs, dict):
                for diff, acc in diffs.items():
                    lines.append(f"  {cat}/{diff}: {acc:.1%}")
            else:
                lines.append(f"  {cat}: {diffs:.1%}")
    weak = profile.get("weak_categories", [])
    if weak:
        lines.append(f"Weak areas: {', '.join(weak)}")
    return "\n".join(lines)


def parse_meta_blocks(text: str) -> dict:
    """Parse <|meta|> blocks from model output for RL reward computation."""
    import re

    blocks = re.findall(
        rf'{re.escape(META_START)}(.*?){re.escape(META_END)}',
        text, re.DOTALL
    )

    result = {
        "num_blocks": len(blocks),
        "confidences": [],
        "has_pre_assessment": False,
        "has_mid_check": False,
        "has_post_reflection": False,
        "has_boxed": "\\boxed" in text,
        "valid": False,
    }

    for i, block in enumerate(blocks):
        block_lower = block.lower()

        # Extract confidence values (0.XX or XX% after probability/confidence keyword)
        # Also matches Korean: 확률 (probability), 확신 (confidence)
        conf_matches = re.findall(
            r'(?:probability|confidence|확률|확신)[:\s\w]*?(\d+\.\d+|\d+)\s*%?',
            block, re.IGNORECASE
        )
        for m in conf_matches:
            val = float(m)
            if val > 1.0:
                val /= 100.0
            val = min(1.0, max(0.0, val))
            if val > 0.001:  # skip near-zero (likely parsing artifacts)
                result["confidences"].append(val)

        # Classify block position
        if any(kw in block_lower for kw in ["can i solve", "probability of solving", "watch out"]):
            result["has_pre_assessment"] = True
        if any(kw in block_lower for kw in ["is this correct", "is this right", "let me check", "confident"]):
            result["has_mid_check"] = True
        if any(kw in block_lower for kw in ["what did i learn", "practice", "improve"]):
            result["has_post_reflection"] = True

    result["valid"] = (
        result["num_blocks"] >= 2
        and result["has_boxed"]
        and len(result["confidences"]) >= 1
    )

    return result
