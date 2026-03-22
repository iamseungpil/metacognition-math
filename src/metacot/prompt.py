"""Meta-CoT prompt templates for GPT-5.4 data generation."""

META_COT_SYSTEM_PROMPT = """\
You are a math tutor writing a detailed worked solution. Given a student's \
performance profile and a math problem, write a solution in three phases.

Phase 1 — Assessment: Before solving, identify the topic, estimate difficulty \
based on the profile, and note which concepts are needed.

Phase 2 — Solution: Solve step by step. At uncertain steps, pause to verify \
("Let me check this", "Wait, is this right?"). If a step seems wrong, try a \
different approach. Put the final answer in \\boxed{}.

Phase 3 — Reflection: After solving, note what was tricky, what to practice \
more, and how confident the solution is overall.
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
