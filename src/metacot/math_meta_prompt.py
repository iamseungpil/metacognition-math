r"""수학 무대용 프롬프트 — Countdown `plain`/`opt` 의 **직역판**.

왜 직역인가 (2026-09-12, 사용자 지시 "예전 자를 기준으로 math 로 옮겨서 다시 테스트"):
cd7/cd8 의 자(ruler)와 텔레메트리는 전부 `countdown_rewards.parse_meta(form="new")`
가 뽑는 `<meta>\nconfidence: x\n…\ndecision: verify|redirect\n</meta>` 블록을 읽는다.
수학에서 블록 문법을 새로 만들면 그 자들을 한 줄도 재사용할 수 없다. 그래서 **블록
문법은 글자 그대로 같게 두고** 과제 서술만 수학으로 바꾼다.

두 변형:
  `math_plain` — 메타 지시문 자체가 없다(N0 대응, 세금 0 기준선).
  `math_opt`   — 메타를 **허용하되 요구하지 않는다**(OPT 대응). Countdown `opt` 의
                 "You MAY … pause and write ONE metacognitive block" 문장을 그대로
                 쓰고, 블록 안 지시만 수학 문맥으로 바꾼다.

★Countdown 과 다른 점 하나(의도적): Countdown 은 메타 안 산술을 금지했다(답이 곧
식이라 메타에 식을 쓰면 답 누출이다). 수학은 최종 답이 값 하나라 «접근 서술»과 «답»이
분리되므로, 금지를 «최종 답을 여기 쓰지 마라» 로만 좁힌다.
"""
from __future__ import annotations

MATH_RULES = (
    "Solve the mathematics problem.\n"
    "Work through it step by step. Be careful with algebra and arithmetic.\n"
)

MATH_CLOSING = (
    "\nEnd your response with the final answer in \\boxed{...} — the answer itself, "
    "nothing else inside the braces. Example: \\boxed{42}\n"
)

# ★Countdown `opt` 의 허가 문장과 **같은 문구**. 이 문장이 갈리면 «허용» 조건이
#   두 무대에서 달라져 발화율 비교가 무의미해진다.
_MATH_PERMISSION = (
    "You MAY, when you judge it useful (for example when you feel stuck), pause and "
    "write ONE metacognitive block in exactly this format, on its own lines:\n\n"
)

_MATH_BLOCK = (
    "<meta>\n"
    "confidence: <a single number between 0 and 1>\n"
    "<One or two sentences judging YOUR OWN APPROACH so far: which method you are "
    "using, and whether that method is worth continuing. \u2605Do NOT write the final "
    "answer in here. Assess the approach; do not state the result.>\n"
    "decision: verify\n"
    "</meta>\n\n"
)

_MATH_DECISION = (
    "Write `decision: verify` when the confidence you just wrote is high and the "
    "current method deserves to be pushed through and checked. Write "
    "`decision: redirect` when that confidence is low and the current method should "
    "be abandoned for a different one. The decision must follow from the confidence. "
    "Then continue in the way that decision commits you to.\n"
)

SOLVE_MATH_PLAIN = MATH_RULES + MATH_CLOSING.lstrip("\n")
SOLVE_MATH_OPT = (MATH_RULES + "\n" + _MATH_PERMISSION + _MATH_BLOCK
                  + _MATH_DECISION + MATH_CLOSING)

MATH_PROMPT_VARIANTS = {
    "math_plain": SOLVE_MATH_PLAIN,
    "math_opt": SOLVE_MATH_OPT,
}

assert "<meta>" not in SOLVE_MATH_PLAIN
assert "<meta>" in SOLVE_MATH_OPT and "You MAY" in SOLVE_MATH_OPT
assert "confidence:" in SOLVE_MATH_OPT and "decision: verify" in SOLVE_MATH_OPT


def build_math_prompt(problem: str, variant: str = "math_opt") -> list[dict]:
    """chat 메시지 리스트 — countdown_task.build_prompt 와 같은 모양."""
    if variant not in MATH_PROMPT_VARIANTS:
        raise ValueError(f"unknown math prompt variant: {variant!r}")
    return [
        {"role": "system", "content": MATH_PROMPT_VARIANTS[variant]},
        {"role": "user", "content": str(problem).strip()},
    ]
