"""Meta-CoT prompt templates for GPT-5.4 data generation."""

META_COT_SYSTEM_PROMPT = """\
You are generating training data for a math-solving AI that has metacognitive \
awareness. You will be given the model's capability profile and a math problem.

Generate a complete solution that demonstrates THREE phases of metacognitive reasoning:

**Phase 1 — Pre-solve Assessment** (BEFORE attempting the solution):
- Identify the problem category and key concepts needed
- State the model's estimated probability of solving correctly (use the profile)
- Flag specific risks: "This requires [concept], which I get right only [X]% of the time"
- Identify what information or reasoning approach is needed

**Phase 2 — Solve with Epistemic Awareness** (DURING the solution):
- Solve step by step, BUT explicitly mark uncertain steps
- Use phrases like "Let me verify this step", "Wait, is this correct?", \
"I'm not confident about this calculation, let me double-check"
- When uncertain, try an alternative approach and compare
- State confidence at key decision points
- Put final answer in \\boxed{}

**Phase 3 — Post-solve Reflection** (AFTER the solution):
- Verify the answer by substitution or alternative method
- If errors were found during solving, explain what went wrong and how it was fixed
- State what additional practice would help: specific topic, difficulty level
- Predict improvement after practice

Requirements:
- Phase 1 MUST appear BEFORE any calculations
- Phase 2 MUST contain at least 2 epistemic expressions (uncertainty markers)
- Phase 2 MUST end with \\boxed{answer}
- Phase 3 MUST include specific study recommendations
- Use the capability profile to make realistic probability estimates
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

    # Tell GPT-5.4 whether the model got it right, so it can generate
    # realistic epistemic verbalization (uncertain when wrong, confident when right)
    return f"""\
=== MODEL CAPABILITY PROFILE ===
{profile_str}

=== MATH PROBLEM ===
{question}

=== REFERENCE ANSWER ===
{correct_answer}

Generate the 3-phase metacognitive solution following the system instructions.
"""


def _format_profile(profile: dict) -> str:
    """Format capability profile for prompt."""
    lines = []
    lines.append(f"Overall pass rate: {profile.get('overall_pass_at_1', 0):.1%}")

    cat_acc = profile.get("category_accuracy", {})
    if cat_acc:
        lines.append("Category accuracy:")
        for cat, diffs in cat_acc.items():
            if isinstance(diffs, dict):
                for diff, acc in diffs.items():
                    lines.append(f"  {cat}/{diff}: {acc:.1%}")
            else:
                lines.append(f"  {cat}: {diffs:.1%}")

    weak = profile.get("weak_categories", [])
    if weak:
        lines.append(f"Weak categories: {', '.join(weak)}")

    return "\n".join(lines)


def parse_metacot_stages(chain_text: str) -> dict:
    """Parse Meta-CoT chain to extract key information."""
    import re

    result = {
        "raw": chain_text,
        "confidence": None,
        "has_pre_assessment": False,
        "has_epistemic": False,
        "has_boxed_answer": False,
        "has_reflection": False,
        "epistemic_count": 0,
        "valid": False,
    }

    chain_lower = chain_text.lower()

    # Extract confidence
    conf_match = re.search(r'(?:confidence|probability)[:\s]+([0-9]+\.?[0-9]*)', chain_text, re.IGNORECASE)
    if conf_match:
        try:
            result["confidence"] = float(conf_match.group(1))
        except ValueError:
            pass

    # Check for pre-solve assessment
    pre_indicators = ["problem category", "probability of solving", "this requires",
                      "key concepts", "estimated probability", "this is a",
                      "i get right only", "risk", "before solving"]
    result["has_pre_assessment"] = any(ind in chain_lower for ind in pre_indicators)

    # Count epistemic expressions
    epistemic_phrases = [
        "wait", "let me verify", "let me check", "is this correct",
        "i'm not sure", "not confident", "double-check", "let me reconsider",
        "hmm", "alternatively", "on second thought", "actually",
        "let me re-examine", "this doesn't seem right", "확인",
    ]
    result["epistemic_count"] = sum(1 for phrase in epistemic_phrases if phrase in chain_lower)
    result["has_epistemic"] = result["epistemic_count"] >= 2

    # Check for boxed answer
    result["has_boxed_answer"] = "\\boxed" in chain_text

    # Check for post-solve reflection
    reflection_indicators = ["practice", "study", "improvement", "additional",
                            "recommend", "next time", "reflection", "verify the answer"]
    result["has_reflection"] = any(ind in chain_lower for ind in reflection_indicators)

    # Valid = has all three phases
    result["valid"] = (
        result["has_pre_assessment"]
        and result["has_epistemic"]
        and (result["has_boxed_answer"] or len(chain_text) > 300)
        and result["has_reflection"]
    )

    return result
