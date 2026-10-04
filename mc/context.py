r"""프롬프트 문맥 — 1턴 프롬프트(현행 `math_opt` 와 바이트 동일, `PROMPT_VARIANT`) · 형제 합의 상태 · 1턴 답.
2턴 NOTICE 셀(H1 요인설계)은 끝났고 생성 경로는 0924 에 뺐다 — 남은 알림 문구·함수도 수정 38 에서 삭제(백업
/hdd_data/seungpil/tmp/context.py.pre38)."""
from __future__ import annotations

import os

from mc.grade import answers_equivalent, boxed_answer

# ── 1턴 프롬프트(`math_opt` 변형의 **바이트 동일** 사본) ──────────────────────
# ★`src/metacot/math_meta_prompt.py` 의 SOLVE_MATH_OPT 를 글자 그대로 옮긴 것이다.
#   tests/mc/test_context.py 가 그쪽 조립기와 바이트 동일함을 고정한다 — 갈리면 이 정책의
#   «허가된 메타의 세금»이 옛 기준선(M_G0/M_G1)과 달라져 모든 비교가 무의미해진다.
MATH_RULES = (
    "Solve the mathematics problem.\n"
    "Work through it step by step. Be careful with algebra and arithmetic.\n"
)
MATH_CLOSING = (
    "\nEnd your response with the final answer in \\boxed{...} — the answer itself, "
    "nothing else inside the braces. Example: \\boxed{42}\n"
)
_PERMISSION = (
    "You MAY, when you judge it useful (for example when you feel stuck), pause and "
    "write ONE metacognitive block in exactly this format, on its own lines:\n\n"
)
_BLOCK = (
    "<meta>\n"
    "confidence: <a single number between 0 and 1>\n"
    "<One or two sentences judging YOUR OWN APPROACH so far: which method you are "
    "using, and whether that method is worth continuing. \u2605Do NOT write the final "
    "answer in here. Assess the approach; do not state the result.>\n"
    "decision: verify\n"
    "</meta>\n\n"
)
_DECISION = (
    "Write `decision: verify` when the confidence you just wrote is high and the "
    "current method deserves to be pushed through and checked. Write "
    "`decision: redirect` when that confidence is low and the current method should "
    "be abandoned for a different one. The decision must follow from the confidence. "
    "Then continue in the way that decision commits you to.\n"
)
SOLVE_MATH_OPT = MATH_RULES + "\n" + _PERMISSION + _BLOCK + _DECISION + MATH_CLOSING
#: ★v4(0922) `plain` — `math_opt` 에서 «메타 블록 허가 문단»(_PERMISSION+_BLOCK+_DECISION)
#:   **하나만** 통째로 뺀 것이고 그 밖은 한 글자도 다르지 않다(tests/mc/test_context.py 가 바이트로
#:   고정). 왜: 허가문이 «let me verify…» 상투구를 프롬프트로 직접 심는다 — 행위(실제 재계산)를
#:   보상하는 v4 는 문구를 주지 않고 정책이 스스로 하게 둔다(analysis/habit_0922).
SOLVE_MATH_PLAIN = MATH_RULES + "\n" + MATH_CLOSING
VARIANTS = {"math_opt": SOLVE_MATH_OPT, "plain": SOLVE_MATH_PLAIN}


def default_variant() -> str:
    """`PROMPT_VARIANT`(기본 `math_opt`) — 미설정이면 옛 동작과 바이트 동일."""
    v = (os.environ.get("PROMPT_VARIANT") or "math_opt").strip().lower()
    if v not in VARIANTS:
        raise ValueError(f"[MC] PROMPT_VARIANT={v!r} 은 {'|'.join(VARIANTS)} 중 하나여야 한다.")
    return v


def build_math_prompt(problem: str, variant: str | None = None) -> list[dict]:
    """chat 메시지 — [system(변형), user(문제)]. `variant=None` → `PROMPT_VARIANT`."""
    variant = variant or default_variant()
    if variant not in VARIANTS:
        raise ValueError(f"[MC] unknown prompt variant: {variant!r} "
                         f"(mc 는 {'|'.join(VARIANTS)} 뿐이다)")
    return [{"role": "system", "content": VARIANTS[variant]},
            {"role": "user", "content": str(problem).strip()}]


def render_chat_messages(tok, msgs: list[dict]) -> str:
    """`enable_thinking=False` 를 시도하고 템플릿이 그 kwarg 를 모르면 없이 재시도한다."""
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def render_generation_prompt(tok, variant: str | None, problem: str) -> str:
    return render_chat_messages(tok, build_math_prompt(problem, variant))

#: 합의 상태(`src/training/trial2.agreement_state` 와 같은 이름·임계값).
#: 수정 51 문맥 증류 선생님 머리(문제 뒤) — 자기 풀이 없이 «이전 답 X» 사실만(정오 없음). 0921 H1 fact/notx 원문.
FACT_TMPL = ("\n\nNote: a previous attempt at this problem answered {answer}. Solve it "
             "again from scratch and give the final answer in \\boxed{{}}.")

AGREE_STATES = ("ALL_SAME", "DOMINANT", "SPLIT", "SCATTER", "NOANS")


def agreement_state(answers, *, k: int | None = None) -> dict:
    r"""★gold 없는 **합의 상태** — 형제 답들의 수학 동치 군집만 본다(정오는 쓰지 않는다).

    최대 군집 크기 top: == K → ALL_SAME, ≥5 → DOMINANT, ≥3 → SPLIT, 그 밖 → SCATTER,
    답이 없으면 NOANS. **분모 K 는 롤아웃 수**이고 무응답도 K 에 센다.
    ★`src/training/trial2.agreement_state` 와 같은 정의(test_context.py 가 고정)."""
    xs = [(a or "").strip() for a in answers]
    kk = int(k if k is not None else len(xs))
    ans = [a for a in xs if a]
    if not ans or kk <= 0:
        return {"state": "NOANS", "top": 0, "n_clusters": 0, "dom_frac": 0.0,
                "dominant_answer": "", "n_answered": len(ans), "k": kk}
    clusters: list[list] = []                 # [대표, 개수, 첫 등장]
    for i, a in enumerate(ans):
        for cl in clusters:
            if answers_equivalent(cl[0], a):
                cl[1] += 1
                break
        else:
            clusters.append([a, 1, i])
    clusters.sort(key=lambda c: (-c[1], c[2]))
    top = clusters[0][1]
    state = ("ALL_SAME" if top >= kk else "DOMINANT" if top >= 5
             else "SPLIT" if top >= 3 else "SCATTER")
    return {"state": state, "top": top, "n_clusters": len(clusters), "dom_frac": top / kk,
            "dominant_answer": clusters[0][0], "n_answered": len(ans), "k": kk}


def turn1_prompt(tok, problem: str, variant: str | None = None) -> str:
    """1턴 생성 프롬프트 — 현행 `math_opt` 와 **바이트 동일**(같은 조립기를 부른다)."""
    return render_generation_prompt(tok, variant, problem)


def turn1_answer(text: str) -> str:
    """1턴 최종 답 X(없으면 "")."""
    return boxed_answer(text) or ""
