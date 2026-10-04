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

# ★0912 `math_new`(강제) — Countdown `new` 의 대응. **라벨을 캘 때만** 쓴다.
#   이유: 모델이 "막혔다고 느낄 때만" 메타를 내면 관찰되는 메타가 편향 표본이 된다
#   (Countdown 실측: 발화 행 성공률 0.17 vs 비발화 0.91 — 메타가 나빠서가 아니라
#   질 때만 쓰기 때문). 인과를 재려면 위치를 우리가 정해 개입해야 한다.
#   배포·세금 측정은 반대로 `math_opt`(허용)로 한다.
_MATH_MANDATE = (
    "At least once while solving, stop and write a metacognitive block in EXACTLY "
    "this format, on its own lines:\n\n"
)
SOLVE_MATH_NEW = (MATH_RULES + "\n" + _MATH_MANDATE + _MATH_BLOCK
                  + _MATH_DECISION + MATH_CLOSING)

# ★0914 `math_retry`(재시도 판단, M_RETRY 팔) — 사전등록 수정 3. cut/own 두 자리 설계가
#   모두 «자리 라벨» 밀도 미달로 끝난 뒤(RESULTS_cd9 s3b/s5b/s6b: 판단 tie 88~99%) 남은
#   길은 메타의 «위치»를 우리가 정하는 것이 아니라 **판단의 결과가 스스로 드러나는 구조**다:
#   첫 답을 내고 → 그 답을 판단하고(verify/redirect) → redirect 면 다른 방법으로 다시 푼다.
#   그러면 오프라인 라벨 없이도 «첫 답이 틀렸는데 verify 했다 / 맞았는데 redirect 했다»가
#   행 안에서 채점된다(src/training/math_meta.retry_judgment_term). 블록 문법은 그대로 —
#   cd7/cd8 의 자(parse_meta form="math")가 그대로 붙는다.
_MATH_RETRY_RULES = (
    "Solve the mathematics problem.\n"
    "Work through it step by step. Be careful with algebra and arithmetic.\n"
    "\n"
    "Follow this three-part structure:\n"
    "\n"
    "1. FIRST ATTEMPT. Solve the problem and give your first answer in \\boxed{...} — the "
    "answer itself, nothing else inside the braces. Example: \\boxed{42}\n"
    "\n"
    "2. JUDGMENT. Immediately after the first \\boxed{...}, write EXACTLY ONE metacognitive "
    "block in exactly this format, on its own lines:\n\n"
)
_MATH_RETRY_BLOCK = (
    "<meta>\n"
    "confidence: <a single number between 0 and 1>\n"
    "<One or two sentences judging YOUR OWN SOLUTION above: what could have gone wrong, "
    "and whether the answer deserves trust. \u2605Do NOT write any \\boxed{...} in here.>\n"
    "decision: verify\n"
    "</meta>\n\n"
)
_MATH_RETRY_DECISION = (
    "Write `decision: verify` when the confidence you just wrote is high and you trust "
    "the first answer. Write `decision: redirect` when that confidence is low and the "
    "first solution should not be trusted. The decision must follow from the confidence.\n"
    "\n"
    "3. ACT ON THE DECISION. If you wrote `decision: redirect`, you MUST write the line "
    "\"Second attempt:\" and solve the problem again with a GENUINELY DIFFERENT method "
    "(not a re-reading of the first one), ending with a new \\boxed{...}. If you wrote "
    "`decision: verify`, stop — write nothing after the block.\n"
    "\n"
    "Your final answer is the LAST \\boxed{...} in your response.\n"
)
SOLVE_MATH_RETRY = _MATH_RETRY_RULES + _MATH_RETRY_BLOCK + _MATH_RETRY_DECISION

# ★0914 `math_retry_forced` — 강제 탐색용(M_RETRY redirect_rate 붕괴 대응책). 이 문제에
#   대해서는 decision 이 반드시 redirect 여야 하고 두 번째 시도가 강제된다는 것만 다르다
#   (판단 신호가 아니므로 build_math_parquet.py 의 forced_frac 이 이 변형을 고른 행에서
#   forced_redirect=1 로 마킹 → math_meta.py 가 judgment 항을 undefined 처리한다).
#   블록 문법·머리/꼬리 문장은 math_retry 와 글자 그대로 같아야 자·파서가 그대로 붙는다.
_MATH_RETRY_FORCED_DECISION = (
    "For THIS problem, your decision MUST be `decision: redirect` — write it regardless "
    "of the confidence number, and treat the first solution as not to be trusted.\n"
    "\n"
    "3. ACT ON THE DECISION. Because the decision is `redirect`, you MUST write the line "
    "\"Second attempt:\" and solve the problem again with a GENUINELY DIFFERENT method "
    "(not a re-reading of the first one), ending with a new \\boxed{...}.\n"
    "\n"
    "Your final answer is the LAST \\boxed{...} in your response.\n"
)
SOLVE_MATH_RETRY_FORCED = _MATH_RETRY_RULES + _MATH_RETRY_BLOCK + _MATH_RETRY_FORCED_DECISION

# ★0914b `math_agree`/`math_agree_forced` — 형제 동의 예측 팔(M_AGREE, RESULTS_cd9 참조).
#   math_retry 와 **머리·꼬리 문장이 글자 그대로 같다** — 갈리는 것은 블록 안에 `agreement:`
#   줄 하나가 느는 것뿐이다(스펙 지시: "same as math_retry / math_retry_forced but the block
#   has an extra mandatory line BEFORE decision:"). 그래서 _MATH_RETRY_RULES/_MATH_RETRY_DECISION/
#   _MATH_RETRY_FORCED_DECISION 을 그대로 재사용한다 — 복제하면 두 팔의 머리/꼬리가 조용히
#   갈릴 수 있다.
_MATH_AGREE_BLOCK = (
    "<meta>\n"
    "confidence: <a single number between 0 and 1>\n"
    "<One or two sentences judging YOUR OWN SOLUTION above: what could have gone wrong, "
    "and whether the answer deserves trust. ★Do NOT write any \\boxed{...} in here.>\n"
    "agreement: <a number between 0 and 1: the fraction of independent attempts at this "
    "problem that you expect to reach the same final answer as yours>\n"
    "decision: verify\n"
    "</meta>\n\n"
)
SOLVE_MATH_AGREE = _MATH_RETRY_RULES + _MATH_AGREE_BLOCK + _MATH_RETRY_DECISION
SOLVE_MATH_AGREE_FORCED = _MATH_RETRY_RULES + _MATH_AGREE_BLOCK + _MATH_RETRY_FORCED_DECISION

# ★math_agree 는 math_retry 와 블록 문법(<meta>/confidence:/decision:/</meta>)·머리 두 줄·
#   꼬리 두 줄이 같아야 같은 파서(parse_meta form="math")가 그대로 붙는다. agreement: 줄만 는다.
assert "<meta>" in SOLVE_MATH_AGREE and "EXACTLY ONE" in SOLVE_MATH_AGREE
assert "confidence:" in SOLVE_MATH_AGREE and "decision: verify" in SOLVE_MATH_AGREE
assert "agreement:" in SOLVE_MATH_AGREE
assert "Second attempt:" in SOLVE_MATH_AGREE and "LAST \\boxed" in SOLVE_MATH_AGREE
assert SOLVE_MATH_AGREE.startswith(_MATH_RETRY_RULES)
assert _MATH_AGREE_BLOCK.splitlines()[0:2] == _MATH_BLOCK.splitlines()[0:2]   # 머리 두 줄 동일
assert _MATH_AGREE_BLOCK.splitlines()[-2:] == _MATH_BLOCK.splitlines()[-2:]   # 꼬리 두 줄 동일
# ★math_retry 와의 차이는 딱 agreement 줄 하나 — 그 줄을 지우면 math_retry 블록과 바이트 동일.
assert _MATH_AGREE_BLOCK.replace(
    "agreement: <a number between 0 and 1: the fraction of independent attempts at this "
    "problem that you expect to reach the same final answer as yours>\n", "") == _MATH_RETRY_BLOCK

assert SOLVE_MATH_AGREE_FORCED.startswith(_MATH_RETRY_RULES + _MATH_AGREE_BLOCK)
assert "agreement:" in SOLVE_MATH_AGREE_FORCED and "confidence:" in SOLVE_MATH_AGREE_FORCED
assert "decision: redirect` — write it regardless" in SOLVE_MATH_AGREE_FORCED
assert "Second attempt:" in SOLVE_MATH_AGREE_FORCED and "LAST \\boxed" in SOLVE_MATH_AGREE_FORCED
assert "decision: verify" not in SOLVE_MATH_AGREE_FORCED.split("</meta>", 1)[1]

# ★0914c `math_crit` — 비평 팔(M_CRIT). cd9 의 정박 진단(RESULTS_cd9 «수정 6 후보») 이후 설계:
#   같은 문맥 안의 **두 번째 시도는 값이 0**이었다(1,248행 전부 구제 0, 86% 가 첫 답을 글자
#   그대로 재기입 — SCoRe 의 행동 붕괴). 그래서 행동(재시도)을 문맥에서 **빼고** 판단·비평만
#   남긴다: 첫 답 → 한 블록(확신도 + 2~3문장 비평 + 결정) → 끝. 비평의 값은 «그 비평을 들고
#   답 없이 다시 풀면 오르는가»(math_critique_resolve_gate)와 «정답 형제 풀이의 로그확률을
#   얼마나 올리는가»(정보 이득 IG, src/training/critique_scorer.py)로 잰다.
#   ★답·숫자 금지가 이 변형의 핵심이다 — 비평이 답을 흘리면 «비평의 값»이 아니라 «답을 다시
#   보여 준 값»이 되고(math_meta.critique_leaks 가드), 재풀이는 원래 오답에 고정된다.
#   블록 문법(<meta>/confidence:/decision:/</meta>)은 math_retry 와 글자 그대로 같다 —
#   parse_meta(form="math")·자·텔레메트리가 그대로 붙는다.
_MATH_CRIT_RULES = (
    "Solve the mathematics problem.\n"
    "Work through it step by step. Be careful with algebra and arithmetic.\n"
    "\n"
    "Follow this two-part structure:\n"
    "\n"
    "1. SOLUTION. Solve the problem and give your answer in \\boxed{...} \u2014 the "
    "answer itself, nothing else inside the braces. Example: \\boxed{42}\n"
    "\n"
    "2. CRITIQUE. Immediately after the \\boxed{...}, write EXACTLY ONE metacognitive "
    "block in exactly this format, on its own lines:\n\n"
)
_MATH_CRIT_BLOCK = (
    "<meta>\n"
    "confidence: <a single number between 0 and 1>\n"
    "<Two or three sentences criticising YOUR OWN SOLUTION above: name the step or "
    "assumption most likely to be mistaken, and say what a correct approach must do "
    "differently. \u2605Do NOT state the final answer, any \\boxed{...}, or any number "
    "from your solution in here \u2014 criticise the reasoning, not the result.>\n"
    "decision: verify\n"
    "</meta>\n\n"
)
_MATH_CRIT_DECISION = (
    "Write `decision: verify` when the confidence you just wrote is high and you trust "
    "the answer. Write `decision: redirect` when that confidence is low and the solution "
    "should not be trusted. The decision must follow from the confidence.\n"
    "\n"
    "Then STOP. Write nothing after `</meta>` \u2014 do not solve the problem again.\n"
    "\n"
    "Your final answer is the \\boxed{...} in your solution.\n"
)
SOLVE_MATH_CRIT = _MATH_CRIT_RULES + _MATH_CRIT_BLOCK + _MATH_CRIT_DECISION

# ★0914d `math_dis` — 불일치 진단 팔(M_DIS, src/training/math_dis.py 참조). 시스템 프롬프트는
#   **math_opt 와 바이트 동일**하다(허가된 메타의 세금이 두 팔에서 같아야 한다). 갈리는 것은
#   사용자 턴뿐이다 — 문제 뒤에 이 정책 자신의 후보 풀이 마무리 4개(`[Candidate 1..4]`)와
#   진단 지시(math_dis.DIS_ASK)가 붙는다.
#   ★그래서 이 변형은 `build_math_prompt(problem, "math_dis")` 만으로는 만들 수 없다 — 후보는
#   **행마다 다르다**. 완성된 사용자 턴을 parquet 의 `prompt` 컬럼에 통째로 싣고(verl 이 거기서
#   채팅 메시지를 읽는다), 이 변형은 «사용자 턴이 이미 후보를 담고 있는가»만 확인하는 표식으로
#   둔다(build_math_prompt 의 어서션). 조용히 후보 없는 프롬프트로 학습하면 이 팔은 M_G1 과
#   바이트 동일한 무효 레버가 된다.
SOLVE_MATH_DIS = SOLVE_MATH_OPT
# math_dis.CANDIDATE_MARKER 와 같은 문자열(그 모듈을 import 하면 순환이 된다 — 여기서는 상수로
# 두고 tests/test_math_dis.py 가 둘이 같은지 확인한다).
MATH_DIS_MARKER = "[Candidate 1]"

# ★0914e `math_diff` — 난이도 판단 팔(M_DIFF, src/training/math_diff.py 참조). 시스템 프롬프트는
#   **math_opt 와 바이트 동일**하다(허가된 메타의 세금이 두 팔에서 같아야 한다). 갈리는 것은
#   사용자 턴뿐이다 — 문제 뒤에 «풀기 전에 난이도 한 줄 + 이유 한 문장» 지시(math_diff.DIFF_ASK)가
#   붙는다.
#   ★math_dis 와 달리 이 변형은 `build_math_prompt(problem, "math_diff")` **하나로 조립된다** —
#   행마다 다른 재료(후보 스케치)가 없기 때문이다. 그래서 아래 build_math_prompt 가 사용자 턴에
#   접미를 직접 붙이고, parquet 빌더(build_math_parquet.py --variant math_diff)는 다른 팔과
#   바이트 동일한 경로를 탄다.
# math_diff.DIFF_ASK 와 같은 문자열(그 모듈을 import 하면 순환이 된다 — MATH_DIS_MARKER 와 같은
# 규약으로 여기서는 상수로 두고 tests/test_math_diff.py 가 둘이 바이트 동일한지 확인한다).
MATH_DIFF_ASK = (
    "\n\nBefore solving, inside one <meta>...</meta> block write "
    "`difficulty: easy|medium|hard` (how likely YOU are to solve this correctly on one "
    "attempt: easy = almost surely, medium = uncertain, hard = probably not) and one "
    "sentence `why: ...`. Then solve and give the final answer in \\boxed{}."
)
SOLVE_MATH_DIFF = SOLVE_MATH_OPT

MATH_PROMPT_VARIANTS = {
    "math_plain": SOLVE_MATH_PLAIN,
    "math_opt": SOLVE_MATH_OPT,
    "math_new": SOLVE_MATH_NEW,
    "math_retry": SOLVE_MATH_RETRY,
    "math_retry_forced": SOLVE_MATH_RETRY_FORCED,
    "math_agree": SOLVE_MATH_AGREE,
    "math_agree_forced": SOLVE_MATH_AGREE_FORCED,
    "math_crit": SOLVE_MATH_CRIT,
    "math_dis": SOLVE_MATH_DIS,
    "math_diff": SOLVE_MATH_DIFF,
}

# ★math_dis 의 시스템 프롬프트는 math_opt 와 **바이트 동일**해야 한다 — 갈리면 «허가된 메타»의
#   세금이 두 팔에서 달라져 M_G1 대조가 무의미해진다(후보·진단 지시는 사용자 턴에만 있다).
assert MATH_PROMPT_VARIANTS["math_dis"] == MATH_PROMPT_VARIANTS["math_opt"]

# ★math_diff 도 같은 이유로 시스템 프롬프트가 math_opt 와 **바이트 동일**해야 한다 — 난이도
#   지시는 사용자 턴에만 있다(M_G1 대조가 성립하려면 «허가 문장의 세금»이 같아야 한다).
assert MATH_PROMPT_VARIANTS["math_diff"] == MATH_PROMPT_VARIANTS["math_opt"]
assert "difficulty: easy|medium|hard" in MATH_DIFF_ASK and "why:" in MATH_DIFF_ASK

# 허가판과 강제판은 **그 한 문장만** 달라야 한다 — 다른 데가 갈리면 "강제의 효과"와
# "프롬프트가 달라진 효과"가 섞인다(Countdown opt 조립이 같은 규약을 쓴다).
assert SOLVE_MATH_OPT.replace(_MATH_PERMISSION, _MATH_MANDATE, 1) == SOLVE_MATH_NEW

assert "<meta>" not in SOLVE_MATH_PLAIN
assert "<meta>" in SOLVE_MATH_OPT and "You MAY" in SOLVE_MATH_OPT
assert "confidence:" in SOLVE_MATH_OPT and "decision: verify" in SOLVE_MATH_OPT

# ★math_retry 는 블록 문법(«<meta>/confidence:/decision:/</meta>»)을 opt 와 글자 그대로 공유해야
#   같은 파서·자가 붙는다. 그리고 세 구조(첫 답 → 한 블록 → Second attempt)가 문장에 있어야 한다.
assert "<meta>" in SOLVE_MATH_RETRY and "EXACTLY ONE" in SOLVE_MATH_RETRY
assert "confidence:" in SOLVE_MATH_RETRY and "decision: verify" in SOLVE_MATH_RETRY
assert "Second attempt:" in SOLVE_MATH_RETRY and "LAST \\boxed" in SOLVE_MATH_RETRY
assert "You MAY" not in SOLVE_MATH_RETRY and "At least once while solving" not in SOLVE_MATH_RETRY
assert _MATH_RETRY_BLOCK.splitlines()[0:2] == _MATH_BLOCK.splitlines()[0:2]   # 머리 두 줄 동일
assert _MATH_RETRY_BLOCK.splitlines()[-2:] == _MATH_BLOCK.splitlines()[-2:]   # 꼬리 두 줄 동일

# ★math_retry_forced 는 math_retry 와 블록 문법·머리 부분이 완전히 같고 decision 지시문만
#   "반드시 redirect" 로 갈려야 한다 — 갈리면 강제탐색 효과와 프롬프트 변경 효과가 섞인다.
assert SOLVE_MATH_RETRY_FORCED.startswith(_MATH_RETRY_RULES + _MATH_RETRY_BLOCK)
assert "<meta>" in SOLVE_MATH_RETRY_FORCED and "confidence:" in SOLVE_MATH_RETRY_FORCED
assert "decision: redirect` — write it regardless" in SOLVE_MATH_RETRY_FORCED
assert "Second attempt:" in SOLVE_MATH_RETRY_FORCED and "LAST \\boxed" in SOLVE_MATH_RETRY_FORCED
assert "decision: verify" not in SOLVE_MATH_RETRY_FORCED.split("</meta>", 1)[1]  # 본문에 verify 허용 문구 없음
assert "must follow from the confidence" not in SOLVE_MATH_RETRY_FORCED  # 판단 근거 문장은 제거

# ★math_crit 는 블록 문법을 math_retry 와 공유해야 같은 파서·자가 붙는다(머리·꼬리 두 줄 동일).
#   그리고 **두 번째 시도가 없어야** 한다 — 문맥 안 재시도는 cd9 에서 값이 0 으로 실측됐다.
assert "<meta>" in SOLVE_MATH_CRIT and "EXACTLY ONE" in SOLVE_MATH_CRIT
assert "confidence:" in SOLVE_MATH_CRIT and "decision: verify" in SOLVE_MATH_CRIT
assert "Second attempt" not in SOLVE_MATH_CRIT and "LAST \\boxed" not in SOLVE_MATH_CRIT
assert "do not solve the problem again" in SOLVE_MATH_CRIT
assert _MATH_CRIT_BLOCK.splitlines()[0:2] == _MATH_BLOCK.splitlines()[0:2]   # 머리 두 줄 동일
assert _MATH_CRIT_BLOCK.splitlines()[-2:] == _MATH_BLOCK.splitlines()[-2:]   # 꼬리 두 줄 동일
assert "You MAY" not in SOLVE_MATH_CRIT and "At least once while solving" not in SOLVE_MATH_CRIT
# ★답·숫자 금지 문구가 살아 있어야 누출 가드(math_meta.critique_leaks)와 프롬프트가 같은 계약이다.
assert "Do NOT state the final answer" in SOLVE_MATH_CRIT


def build_math_prompt(problem: str, variant: str = "math_opt") -> list[dict]:
    """chat 메시지 리스트 — countdown_task.build_prompt 와 같은 모양.

    ★`math_diff` 는 사용자 턴에 난이도 지시(MATH_DIFF_ASK)를 **여기서** 붙인다 — 행마다 다른
    재료가 없으므로 문제 하나만으로 조립된다(math_dis 와 대비되는 점).

    ★`math_dis` 는 예외다: `problem` 자리에 **이미 완성된 사용자 턴**(문제 + 후보 4개 +
    진단 지시, src/training/math_dis.build_dis_user_turn)을 넘겨야 한다. 후보 없는 문제로
    부르면 즉사한다 — 조용히 통과시키면 M_DIS 가 «후보를 안 보는 M_G1» 으로 학습된다.
    """
    if variant not in MATH_PROMPT_VARIANTS:
        raise ValueError(f"unknown math prompt variant: {variant!r}")
    if variant == "math_diff":
        return [
            {"role": "system", "content": MATH_PROMPT_VARIANTS[variant]},
            {"role": "user", "content": str(problem).strip() + MATH_DIFF_ASK},
        ]
    if variant == "math_dis" and MATH_DIS_MARKER not in str(problem):
        raise ValueError(
            f"[MATH][DIS] math_dis 의 사용자 턴에 {MATH_DIS_MARKER!r} 가 없다 — "
            "src.training.math_dis.build_dis_user_turn 으로 만든 턴을 넘겨라"
            "(이 변형은 문제 하나만으로는 조립할 수 없다).")
    return [
        {"role": "system", "content": MATH_PROMPT_VARIANTS[variant]},
        {"role": "user", "content": str(problem).strip()},
    ]


# ★중복 제거(0914): math_sites.py / math_ruler_pivot.py / math_retry_eval.py / math_rollout.py
#   가 각자 (a) build_math_prompt 로 메시지를 만들고 (b) apply_chat_template(...,
#   enable_thinking=False) 를 시도하다 구 템플릿(TypeError)이면 그 인자 없이 재시도하는,
#   글자 그대로 같은 코드를 네 벌 갖고 있었다 — 하나가 고쳐지고 나머지가 안 고쳐지면
#   «같은 자리»/«같은 문맥» 전제가 조용히 깨진다(math_sites 의 own_meta 감사가 바로 이
#   전제에 기대는 측정이다). 단일 진실 원천으로 합친다.
def render_chat_messages(tok, msgs: list[dict]) -> str:
    """apply_chat_template(..., enable_thinking=False) 를 시도하고, 템플릿이 그 kwarg 를
    모르면(TypeError) 없이 재시도한다 — 네 스크립트가 각자 갖고 있던 폴백의 단일 원천.
    ★math_retry_eval.py 는 Generator 인터페이스가 이미 빌드된 messages 리스트를 받으므로
    (evaluate() 가 build_math_prompt 로 미리 만든다 — 모의 생성기 테스트가 그 계약을 검증한다)
    이 저수준 헬퍼를 직접 쓴다. 다른 세 스크립트(문제 하나 → 렌더 하나)는 render_generation_
    prompt 를 쓴다."""
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def render_generation_prompt(tok, variant: str, problem: str) -> str:
    """build_math_prompt(problem, variant) 를 tok 의 chat 템플릿으로 렌더링한 생성 프롬프트
    문자열 — math_sites.py/math_ruler_pivot.py/math_rollout.py 가 공유하는 단일 진실 원천
    (문제 하나에 프롬프트 하나를 직접 만드는 세 스크립트)."""
    return render_chat_messages(tok, build_math_prompt(problem, variant))
