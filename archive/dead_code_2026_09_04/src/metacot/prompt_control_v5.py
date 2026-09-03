"""Prompt templates for control-oriented Meta-CoT v5 data generation."""

META_START = "<|meta|>"
META_END = "<|/meta|>"


CONTROL_V5_SYSTEM_PROMPT = f"""\
You are generating training demonstrations for a math model that must learn metacognitive control.

The key rule is strict separation:
- The actual chain-of-thought, calculations, proof steps, and verification steps belong outside meta blocks.
- Meta blocks must contain only thinking about the current reasoning state: confidence, anomaly detection, overconfidence checks, why the current route is weak, what skill or perspective is missing, and what control action should happen next.

Target policy:
1. If the current route seems sufficiently supported and confidence appears well-calibrated, use little or no meta.
2. Do not use difficulty alone as the reason to add meta or verification.
3. If internal signs suggest overconfidence or premature commitment, add a short meta block that calls for independent verification. The verification reasoning itself must happen outside the meta block.
4. If confidence drops, the route stalls, or something feels inconsistent, add a meta block that diagnoses why the current route is failing, identifies what is missing, optionally decomposes the failure into missing sub-skills or missing subgoals, and then changes control state.

Global rules:
1. Always solve the problem correctly and end with a final \\boxed{{answer}}.
2. Use {META_START} ... {META_END} only when it changes control behavior.
3. Do not add decorative self-talk.
4. Every meta block must contain an explicit `confidence: 0.xx` line.
5. A meta block must not contain detailed calculations, substitutions, algebraic manipulations, full proof steps, equations, or variable-level symbolic expressions.
6. A meta block may contain:
   - confidence self-assessment
   - anomaly / mismatch notice
   - brief diagnosis of why the current route is weak
   - what is missing or what should be studied / recalled
   - the control-level next action (`verify`, `redirect`, or `study before retrying`)
7. If the model appears confident for the wrong reason, it may trigger verification even on an easy problem.
8. If confidence is stable and there is no overconfidence signal, do not force verification merely because the problem is easy, medium, or hard.
9. Use `study_need:` only when the current failure reveals a missing skill or perspective that would help solve similar future problems or support retrieval.
10. `study_need:` must be short and parseable, such as:
    - `study_need: parity / modular reasoning`
    - `study_need: invariant identification`
    - `study_need: geometric interpretation over symbolic expansion`

Scenario `straight`:
- Solve directly.
- Usually use zero meta blocks.
- A short verification-trigger meta block is allowed only if there is a credible overconfidence signal.

Scenario `verify`:
- Reach a candidate answer.
- Only add meta if the model should verify because confidence is high enough that it may be exceeding the actual support behind the answer.
- Typical internal signals are: the answer came too quickly, the support is thinner than the confidence, only one brittle route has been checked, or the model is about to commit without an independent check.
- The meta block should say that verification is needed, but the actual verification logic must appear outside the meta block.
- Avoid ritualized verification on every problem.

Scenario `redirect`:
- Begin with a plausible route.
- Notice a real anomaly, mismatch, unsupported leap, or meaningful confidence drop.
- Add a meta block that explains why the current route is failing.
- If useful, identify a missing skill, perspective, or object of study using `study_need:`.
- If decomposition appears, it must decompose the failure or the missing requirement, not narrate the full solution plan.
- Then actually switch strategy outside the meta block and solve correctly.
- On hard problems, a second meta block is allowed if another anomaly or overconfidence signal appears later.

Good verify example:
{META_START}
confidence: 0.87
My answer feels plausible, but the confidence is high enough that I might be committing too quickly.
Before finalizing, I should independently verify rather than trust the first route at face value.
{META_END}

Good redirect example:
{META_START}
confidence: 0.39
The issue is not lack of algebraic work; the current route is weak because it keeps transforming expressions without exposing the constraint that actually controls the answer.
What is missing is the right perspective on the structure of the problem.
study_need: invariant identification / modular reasoning
I should stop extending this route and switch to a constraint-based analysis.
{META_END}
"""


def build_control_v5_prompt(
    question: str,
    scenario: str,
    difficulty: str,
    pass_rate: float,
    source: str,
    topic: str,
) -> str:
    """Build a control-oriented TRAPI prompt for v5."""
    scenario_instructions = {
        "straight": (
            "Solve directly. Use no meta unless there is a credible overconfidence signal that makes a short verification trigger worthwhile."
        ),
        "verify": (
            "Solve the problem and include a short verification-trigger meta block. In this scenario, assume the solver has reached a plausible answer and feels ready to commit, but there is a real near-commit calibration gap or overconfidence signal that makes an independent check appropriate. "
            "Do not put the verification reasoning itself inside the meta block."
        ),
        "redirect": (
            "You must include a real intervention moment. In this scenario, the initial route really is weak: notice a mismatch, confidence drop, stuckness, or route failure; write a low-confidence meta block that diagnoses why the current route is weak. "
            "If appropriate, include a short `study_need:` line naming what perspective or skill is missing. "
            "Then switch strategy outside the meta block and solve correctly."
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
