"""Meta-CoT v2 prompt: diverse confidence, error→fix, final verification."""

META_START = "<|meta|>"
META_END = "<|/meta|>"

# Key changes from v1:
# 1. Confidence MUST match rollout_pass_rate (not always 0.95+)
# 2. Error→correction pattern required in 30%+ of chains
# 3. Final verification meta is mandatory
# 4. Each meta block should be SHORT (<50 tokens)

INCONTEXT_EXAMPLE_CORRECT = f"""\
{META_START}
Q: Can I solve this? A: Modular arithmetic. probability 0.6. Risk: cycle detection errors.
{META_END}

2^1 ≡ 2, 2^2 ≡ 4, 2^3 ≡ 1 (mod 7). Cycle length 3.

{META_START}
Q: Correct? A: 2^3=8, 8-7=1. Yes. confidence 0.8.
{META_END}

100 = 33×3 + 1, so 2^100 ≡ 2^1 ≡ 2 (mod 7).

{META_START}
Q: Final check. A: Verified cycle, verified division. Answer 2. confidence 0.9.
{META_END}

\\boxed{{2}}
"""

INCONTEXT_EXAMPLE_ERROR_FIX = f"""\
{META_START}
Q: Can I solve this? A: System of equations. probability 0.4. I often make sign errors here.
{META_END}

From x + y = 5 and x - y = 1:
Adding: 2x = 6, x = 3.
Substituting: 3 + y = 5, y = 2.

{META_START}
Q: Wait, let me check. A: x-y=1 → 3-2=1. Correct. But let me verify the original: x+y=3+2=5. Yes. confidence 0.7.
{META_END}

{META_START}
Q: Final verification. A: x=3, y=2 satisfies both equations. confidence 0.85.
{META_END}

\\boxed{{(3, 2)}}
"""

INCONTEXT_EXAMPLE_HARD = f"""\
{META_START}
Q: Can I solve this? A: Competition geometry. probability 0.2. This looks very hard.
{META_END}

Let me try coordinate geometry. Place the triangle...

{META_START}
Q: Is this approach working? A: Getting complicated. confidence 0.15. Maybe try synthetic geometry instead.
{META_END}

Actually, by power of a point theorem...

{META_START}
Q: Better approach? A: Yes, this is cleaner. confidence 0.4.
{META_END}

[solution continues]

{META_START}
Q: Final check. A: Verified with numerical example. Answer seems right. confidence 0.55.
{META_END}

\\boxed{{371}}
"""

META_COT_V2_SYSTEM_PROMPT = f"""\
You are solving a math problem with self-awareness. Use {META_START} and {META_END} tags for brief self-reflection.

RULES:
1. Keep each meta block SHORT (1-2 lines, under 50 tokens).
2. Your confidence MUST match actual difficulty:
   - Easy problems (you'd solve >80% of the time): confidence 0.7-0.9
   - Medium problems (40-80%): confidence 0.4-0.7
   - Hard problems (<40%): confidence 0.1-0.4
   NEVER say confidence 0.95+ unless it's trivially easy.
3. If you catch an error mid-solution, show the correction:
   "Wait, that's wrong because... Let me fix: ..."
4. MANDATORY: Include a final verification meta before \\boxed{{}}.
5. Put final answer in \\boxed{{}}.

{INCONTEXT_EXAMPLE_CORRECT}

Example with error correction:
{INCONTEXT_EXAMPLE_ERROR_FIX}

Example with hard problem:
{INCONTEXT_EXAMPLE_HARD}
"""


def build_metacot_v2_prompt(
    question: str,
    rollout_pass_rate: float = 0.5,
) -> str:
    """Build prompt for GPT-5.4 data generation.

    rollout_pass_rate: fraction of times the student model solves this correctly.
    This calibrates the target confidence level.
    """
    if rollout_pass_rate > 0.8:
        difficulty = "easy"
        target_conf = f"0.7-0.9"
    elif rollout_pass_rate > 0.4:
        difficulty = "medium"
        target_conf = f"0.4-0.7"
    else:
        difficulty = "hard"
        target_conf = f"0.1-0.4"

    return f"""\
Solve this math problem. Difficulty level: {difficulty} (student solves it correctly {rollout_pass_rate:.0%} of the time).

Your initial confidence should be around {target_conf}.
If you make an error during solving, show the correction process.
Include a final verification step before your \\boxed{{}} answer.

Problem:
{question}
"""
