"""Meta-CoT prompt templates for GPT-5.4 data generation."""

META_COT_SYSTEM_PROMPT = """\
You are training an AI model to develop metacognitive reasoning. Given the \
model's capability profile, a math problem, the model's (possibly wrong) \
answer, and the correct answer, generate a complete 5-stage Meta-CoT chain.

The chain must be a single continuous reasoning process where each stage \
flows naturally into the next. Do NOT separate stages with headers or \
labels in the output — write them as one coherent text. However, ensure \
all five stages are present in order.

**Stage 1 — Solve**: Reproduce or summarize the model's solution attempt. \
Show the key reasoning steps.

**Stage 2 — Diagnose**: Analyze whether the solution is correct. State a \
numeric confidence score (0.0 to 1.0). Identify the specific error type \
and which math category/subcategory it falls under.

**Stage 3 — Strategize**: Based on the diagnosis, propose a concrete \
learning plan. State exactly how many similar problems to practice (5-10) \
and at what difficulty level. If correct, suggest harder problems to push \
the boundary.

**Stage 4 — Select**: Choose problems from the data pool. Justify each \
selection with three criteria:
  L1 (relevance): domain match
  L2 (curriculum): appropriate difficulty progression
  L3 (metacognition): targets the diagnosed weakness

**Stage 5 — Predict**: Make a numeric prediction of accuracy improvement \
after studying the selected problems. State expected category accuracy \
before and after.

Requirements:
- Stage 2 MUST include a numeric confidence (e.g., "confidence: 0.35")
- Stage 3 MUST include a concrete problem count
- Stage 4 MUST include L1/L2/L3 reasoning
- Stage 5 MUST include numeric predictions
"""


def build_metacot_user_prompt(
    profile: dict,
    question: str,
    model_answer: str,
    correct_answer: str,
    is_correct: bool,
    data_pool_summary: str = "",
) -> str:
    """Build the user prompt for Meta-CoT chain generation."""
    profile_str = _format_profile(profile)

    status = "CORRECT" if is_correct else "INCORRECT"

    return f"""\
=== MODEL CAPABILITY PROFILE ===
{profile_str}

=== PROBLEM ===
{question}

=== MODEL'S ANSWER ({status}) ===
{model_answer}

=== CORRECT ANSWER ===
{correct_answer}

=== DATA POOL SUMMARY ===
{data_pool_summary if data_pool_summary else "Full MATH + NuminaMath + Omni-MATH pool available with problems across all categories and difficulty levels."}

Generate the complete 5-stage Meta-CoT chain now.
"""


def _format_profile(profile: dict) -> str:
    lines = [
        f"Model: {profile.get('model', 'Qwen2.5-7B')}",
        f"Overall pass@1: {profile.get('overall_pass_at_1', 0):.3f}",
        f"Overall majority vote: {profile.get('overall_pass_at_majority', 0):.3f}",
        "",
        "Category accuracy (majority vote):",
    ]
    for cat, diffs in profile.get("category_accuracy", {}).items():
        parts = [f"{d}={v:.2f}" for d, v in diffs.items()]
        lines.append(f"  {cat}: {', '.join(parts)}")

    weak = profile.get("weak_categories", [])
    if weak:
        lines.append(f"\nWeak categories: {', '.join(weak)}")

    return "\n".join(lines)


def parse_metacot_stages(chain_text: str) -> dict:
    """Parse a Meta-CoT chain into its 5 stages.

    Returns dict with keys: solve, diagnose, strategize, select, predict,
    plus extracted fields: confidence, problem_count, predicted_accuracy.
    """
    import re

    result = {
        "raw": chain_text,
        "confidence": None,
        "problem_count": None,
        "predicted_accuracy": None,
        "has_l1l2l3": False,
        "valid": True,
    }

    # Extract confidence
    conf_match = re.search(r'confidence[:\s]+([0-9]+\.?[0-9]*)', chain_text, re.IGNORECASE)
    if conf_match:
        try:
            result["confidence"] = float(conf_match.group(1))
        except ValueError:
            pass

    # Extract problem count from strategize
    count_match = re.search(r'(\d+)\s*(?:similar\s+)?problems?\s+to\s+(?:practice|study|solve)', chain_text, re.IGNORECASE)
    if not count_match:
        count_match = re.search(r'practice\s+(\d+)\s+problems?', chain_text, re.IGNORECASE)
    if count_match:
        try:
            result["problem_count"] = int(count_match.group(1))
        except ValueError:
            pass

    # Check L1/L2/L3
    result["has_l1l2l3"] = all(
        re.search(rf'L{i}', chain_text) for i in [1, 2, 3]
    )

    # Extract predicted accuracy
    pred_match = re.search(r'(?:predict|expect|estimate).*?(\d+\.?\d*)\s*%', chain_text, re.IGNORECASE)
    if not pred_match:
        pred_match = re.search(r'accuracy.*?(\d+\.?\d*)\s*(?:%|$)', chain_text, re.IGNORECASE)
    if pred_match:
        try:
            val = float(pred_match.group(1))
            result["predicted_accuracy"] = val / 100 if val > 1 else val
        except ValueError:
            pass

    # Validate completeness
    result["valid"] = (
        result["confidence"] is not None
        and result["problem_count"] is not None
        and result["has_l1l2l3"]
    )

    return result
