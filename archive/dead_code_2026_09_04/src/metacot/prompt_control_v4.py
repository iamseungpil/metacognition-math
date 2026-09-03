"""Prompt templates for control-oriented Meta-CoT v4 data generation."""

META_START = "<|meta|>"
META_END = "<|/meta|>"


CONTROL_V4_SYSTEM_PROMPT = f"""\
You are generating training demonstrations for a math model that must learn metacognitive control, not decorative self-talk.

The target policy is conditional:
1. If confidence stays high, do an independent verification before the final answer.
2. If something feels off, a contradiction appears, or confidence drops, intervene: notice the anomaly, explain why the current route is weak, and switch strategy.

Global rules:
1. Always solve the problem correctly and end with a final \\boxed{{answer}}.
2. Use {META_START} ... {META_END} only when it changes behavior, checks behavior, or prevents overconfidence.
3. Do not add fake doubt to easy problems.
4. Every meta block must contain an explicit `confidence: 0.xx` line.
5. Meta text must be natural language. Do not use rigid templates such as `trigger:`, `failure_diagnosis:`, `confidence_before:`, or `confidence_after:`.
6. If you redirect, the later reasoning must actually use a different method, not just rephrase the same one.
7. Multiple meta blocks are allowed on hard problems if the model notices a new anomaly or needs another verification pass.

Scenario `straight`:
- Solve directly.
- No fake uncertainty.
- Usually use zero meta blocks.
- A single short verification block is allowed only if it feels natural.

Scenario `verify`:
- Reach a candidate answer.
- Add a meta block that states high confidence and names an independent check.
- Perform the check in the solution, then finalize.

Scenario `redirect`:
- Begin with a plausible route.
- At some point, notice that something feels off, inconsistent, unsupported, or incomplete.
- Add a meta block with `confidence: 0.xx` and natural-language diagnosis of why the current route is weak.
- If helpful, name a blocker or missing piece and break the task into smaller goals.
- Switch to a genuinely different strategy and solve correctly.
- On hard problems, you may use more than one meta block if another anomaly appears later or if a final verification is still needed.
- A redirect answer is invalid unless it contains at least one low-confidence intervention and then a real strategy switch.

Good verify example:
{META_START}
confidence: 0.86
This looks mostly right, but I should not commit without an independent check.
I will substitute the candidate back into the original equation instead of trusting the simplification.
{META_END}

Good redirect example:
{META_START}
confidence: 0.41
Something feels off. The current substitution is fitting the transformed expression but not the original constraint.
I may be forcing an algebraic route too early.
I should step back, identify the invariant the valid solutions must satisfy, and switch to a parity-based case split.
{META_END}
"""


def build_control_v4_prompt(
    question: str,
    scenario: str,
    difficulty: str,
    pass_rate: float,
    source: str,
    topic: str,
) -> str:
    """Build a control-oriented TRAPI prompt."""
    scenario_instructions = {
        "straight": (
            "Solve directly. Keep the solution concise. Only add a very short verification if it is natural."
        ),
        "verify": (
            "Solve the problem, keep confidence high if justified, and add one independent verification step "
            "such as substitution, recomputation, or a sanity check."
        ),
        "redirect": (
            "You must include a real intervention moment. Notice an anomaly or confidence drop, write at least one meta block "
            "with confidence at or below 0.55, explain in natural language why the current route is weak, then switch to a different strategy and solve correctly. "
            "A direct solve without a low-confidence intervention is invalid for this scenario."
        ),
    }
    return (
        f"Scenario: {scenario}\n"
        f"Difficulty: {difficulty}\n"
        f"Student pass rate on similar problems: {pass_rate:.0%}\n"
        f"Source dataset: {source}\n"
        f"Topic hint: {topic}\n"
        f"Instruction: {scenario_instructions[scenario]}\n\n"
        f"Problem:\n{question}\n"
    )
