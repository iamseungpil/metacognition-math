"""Prompt templates for behavior-first metacognition data generation."""

META_START = "<|meta|>"
META_END = "<|/meta|>"

BEHAVIOR_SYSTEM_PROMPT = f"""\
You are generating training demonstrations for a student model that must learn genuine metacognitive control, not decorative self-talk.

Always solve the math problem correctly, and use {META_START} ... {META_END} blocks only when they change or verify behavior.

Rules:
1. Final answer must be in \\boxed{{}}.
2. When confidence should stay high, keep it high and do not add fake doubt.
3. For verification examples, perform an independent check such as substitution, recomputation, or sanity check.
4. For redirection examples, show a concrete trigger that the current path is weak, lower confidence, and switch to a genuinely different method.
5. If confidence changes, state it explicitly using `confidence_before`, `confidence_after`, or `confidence`.
6. Avoid vague meta filler like "let me think again" unless it leads to a concrete action.
7. Never leave the final answer wrong.

Use one of these styles depending on the requested scenario:

Scenario `straight`:
- Solve directly.
- At most one short verification block.

Scenario `verify`:
- Solve the problem.
- Add a meta block that names the verification method.
- Keep confidence high if the check passes.

Scenario `redirect`:
- Begin with a plausible route or assumption.
- Add a meta block with a concrete trigger such as contradiction, failed substitution, unsupported assumption, or unit mismatch.
- Lower confidence.
- Switch methods and finish correctly.

Good redirect example:
{META_START}
trigger: substitution_failed
diagnosis: Plugging x=6 back into the condition does not satisfy the equation.
confidence_before: 0.78
decision: switch_method
confidence_after: 0.36
{META_END}

Good verification example:
{META_START}
verification: substitute the candidate value back into the original constraint
confidence: 0.84
{META_END}
"""


def build_behavior_prompt(question: str, scenario: str, difficulty: str, pass_rate: float) -> str:
    """Build a scenario-specific TRAPI prompt."""
    scenario_instructions = {
        "straight": (
            "Solve directly. Do not add unnecessary doubt. "
            "Only use a short meta verification if it is natural."
        ),
        "verify": (
            "After solving, add an explicit verification step using substitution, recomputation, "
            "or an independent sanity check. Keep confidence high if the check confirms the answer."
        ),
        "redirect": (
            "Show one real revision moment: identify a concrete issue with the current route, "
            "lower confidence, switch to a different method, and then solve correctly."
        ),
    }
    return (
        f"Scenario: {scenario}\n"
        f"Estimated difficulty: {difficulty}\n"
        f"Student pass rate on similar problems: {pass_rate:.0%}\n"
        f"Instruction: {scenario_instructions[scenario]}\n\n"
        f"Problem:\n{question}\n"
    )
