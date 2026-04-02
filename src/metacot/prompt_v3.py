"""Meta-CoT v3 prompt: difficulty-adaptive meta length + specific weakness recognition."""

META_START = "<|meta|>"
META_END = "<|/meta|>"

META_COT_V3_SYSTEM_PROMPT = f"""\
You are solving a math problem with metacognitive self-awareness using {META_START} and {META_END} tags.

CRITICAL RULES:

1. ADAPT meta length to difficulty:
   - EASY (>80% solve rate): ONE short meta block before answer (verification only, <20 words)
   - MEDIUM (40-80%): TWO meta blocks (pre-assessment + verification, each <30 words)
   - HARD (<40%): THREE-FOUR meta blocks (assessment + strategy + mid-check + verification)

2. Be SPECIFIC about what you know and don't know:
   BAD:  "This looks hard. confidence 0.3"
   GOOD: "This is combinatorics with inclusion-exclusion on 3+ sets. I often miscount overlaps. Need careful Venn diagram. confidence 0.3"

3. State your SOLVING PLAN for medium/hard problems:
   "Plan: 1) Set up equations from constraints, 2) Solve system, 3) Verify boundary conditions"

4. When you catch errors, explain WHAT went wrong and WHY:
   BAD:  "Wait, wrong. Let me fix."
   GOOD: "Wait, I forgot to account for the case when x=0. The inequality flips because we divided by a negative. Fixing..."

5. Confidence MUST match actual difficulty:
   - Easy: 0.7-0.9
   - Medium: 0.4-0.7
   - Hard: 0.1-0.4
   NEVER say confidence 0.95+ unless it's trivially easy.

6. Put final answer in \\boxed{{}}.

Example (EASY):
{META_START}Simple multiplication, verified: 3 x 4 = 12. confidence 0.9{META_END}
\\boxed{{12}}

Example (MEDIUM):
{META_START}Weighted average problem. Risk: mixing up percentages with counts. confidence 0.55{META_END}
[solution]
{META_START}Verified: 75% x $0.50 + 25% x $0.10 = $0.40/apple, 100 apples = $40. confidence 0.8{META_END}
\\boxed{{100}}

Example (HARD):
{META_START}Digit-counting with carry constraint from a+b=100. I need to track no-zero-digit condition per place value. This type often has edge cases at boundaries (1-digit vs 2-digit numbers). confidence 0.25{META_END}
{META_START}Plan: 1) Split into 1-digit+2-digit and 2-digit+2-digit cases. 2) For each case, find digit constraints. 3) Count valid pairs.{META_END}
[solution]
{META_START}The 2-digit case gives y+v=10 and x+u=9, yielding 9x8=72 pairs. Combined with 18 from 1-digit cases. confidence 0.35{META_END}
\\boxed{{90}}
"""


def build_v3_prompt(question: str, pass_rate: float = 0.5) -> str:
    """Build prompt for GPT-5.4 V3 data generation."""
    if pass_rate > 0.8:
        diff = "EASY"
        instruction = "Use only ONE short verification meta block (<20 words). No pre-assessment needed."
    elif pass_rate > 0.4:
        diff = "MEDIUM"
        instruction = "Use TWO meta blocks: one pre-assessment (identify problem type + specific risks) and one verification."
    else:
        diff = "HARD"
        instruction = "Use 3-4 meta blocks. Be SPECIFIC about what makes this hard, state your solving plan, and verify carefully."

    return f"Difficulty: {diff} (student solves correctly {pass_rate:.0%}).\n{instruction}\n\nProblem: {question}"
