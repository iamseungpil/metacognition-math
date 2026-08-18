"""Countdown — 생성기·정확 채점기·프롬프트.

왜 이 과제인가. 0817~18 실측에서 수학(MATH-500)의 자기점검은 전부 널이었는데,
**"우리 방법이 나쁜가"와 "이 과제에 도울 여지가 없는가"를 한 번도 못 갈랐다.**
Gandhi(2503.01307)는 Countdown 에서 되돌아가기가 **실제로 정답률을 올린다**고 보였다.
그러니 여기서도 널이면 방법이 문제고, 양성이면 수학이 잘못된 시험장이었다.

★이 과제의 결정적 성질: **모든 실패가 회수 가능하다.**
생성이 "무작위로 짝지어 합치기"라 **풀이가 존재함이 구성으로 보장**된다. 그래서
실패 = 탐색 실패 = 자기점검이 고칠 수 있는 유형. 수학에서는 "기법을 모름"과
"계산 실수"를 못 가르는데, 여기서는 그 물음이 설계에서 사라진다.

그리고 오늘 우리를 세 번 문 결함이 통째로 없어진다:
  \\boxed 정규식 중첩 실패 · math_verify 오탐 · 캡 절단(풀이가 짧다).

규칙: 주어진 수를 **각각 정확히 한 번씩** 쓰고 + − × ÷ 로 목표수를 만든다.
중간값은 항상 양의 정수여야 한다(뺄셈은 음수 금지, 나눗셈은 나누어떨어질 때만) —
표준 Countdown 규칙이고, 탐색 공간을 사람이 푸는 방식과 맞춘다.
"""
from __future__ import annotations

import random
import re
from fractions import Fraction

# ────────────────────────────────────────────────────────────── 생성

def gen_instance(n_nums: int, rng: random.Random, lo=1, hi=25, min_target=20):
    """무작위로 짝지어 합쳐 목표수를 만든다 → **풀이 존재가 구성으로 보장**된다.

    부산물로 증인 풀이(witness)를 함께 돌려준다. 채점에는 안 쓰고, 문제가 진짜
    풀리는지 눈으로 확인할 때만 쓴다.
    """
    for _ in range(400):
        nums = [rng.randint(lo, hi) for _ in range(n_nums)]
        cur = [(v, str(v)) for v in nums]
        ok = True
        while len(cur) > 1:
            i, j = rng.sample(range(len(cur)), 2)
            (a, ea), (b, eb) = cur[i], cur[j]
            cand = [(a + b, f"({ea}+{eb})"), (a * b, f"({ea}*{eb})")]
            if a - b > 0:
                cand.append((a - b, f"({ea}-{eb})"))
            if b - a > 0:
                cand.append((b - a, f"({eb}-{ea})"))
            if b and a % b == 0:
                cand.append((a // b, f"({ea}/{eb})"))
            if a and b % a == 0:
                cand.append((b // a, f"({eb}/{ea})"))
            v, e = rng.choice(cand)
            if v > 10000:
                ok = False
                break
            cur = [cur[k] for k in range(len(cur)) if k not in (i, j)] + [(v, e)]
        if ok and cur[0][0] >= min_target:
            return {"nums": nums, "target": cur[0][0], "witness": cur[0][1]}
    return None


# ────────────────────────────────────────────────────────────── 채점 (정확)

_EXPR_OK = re.compile(r"^[0-9+\-*/() ]+$")
_NUM = re.compile(r"\d+")


def _last_boxed(text: str):
    """마지막 \\boxed{...} 내용. 균형괄호 스캔 — 정규식판이 중첩에서 실패한 전례."""
    out, i = [], 0
    while True:
        j = text.find("\\boxed{", i)
        if j < 0:
            break
        k, dep = j + 7, 1
        while k < len(text) and dep:
            dep += (text[k] == "{") - (text[k] == "}")
            k += 1
        if dep == 0:
            out.append(text[j + 7:k - 1])
        i = j + 7
    return out[-1].strip() if out else None


def grade(text: str, nums, target) -> int:
    """식을 뽑아 **정확히** 평가한다. 셋 다 맞아야 1점.

      ① 파싱 가능한 산술식인가
      ② 주어진 수를 각각 정확히 한 번씩 썼나 (다중집합 일치)
      ③ 값이 목표수와 같은가  (Fraction 으로 계산 — 부동소수 오차 0)
    """
    e = _last_boxed(text)
    if e is None:
        return 0
    e = e.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    e = e.replace("\\left", "").replace("\\right", "").replace("$", "").strip()
    e = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", e)
    if "=" in e:                      # "(3+7)*8 = 80" 꼴이면 좌변만
        e = e.split("=")[0].strip()
    if not _EXPR_OK.match(e):
        return 0
    used = sorted(int(x) for x in _NUM.findall(e))
    if used != sorted(nums):
        return 0
    try:
        val = eval(e, {"__builtins__": {}},                       # noqa: S307
                   {"Fraction": Fraction})
    except Exception:
        return 0
    try:
        return int(Fraction(val) == Fraction(target))
    except Exception:
        return 0


def parse_ok(text: str) -> int:
    """형식 준수 — 식을 뽑아 파싱까지는 됐나 (정답 여부와 무관)."""
    e = _last_boxed(text)
    if e is None:
        return 0
    e = e.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    e = e.replace("\\left", "").replace("\\right", "").replace("$", "").strip()
    if "=" in e:
        e = e.split("=")[0].strip()
    return 1 if _EXPR_OK.match(e) else 0


def n_attempts(text: str) -> int:
    """★기전 지표 — 서로 다른 산술 시도를 몇 번 했나 (되돌아가기의 대리).

    수학에서는 못 재는 것이다. `a op b = c` 꼴을 세어 중복을 뺀다.
    """
    tries = set(re.findall(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)", text))
    return len(tries)


# ────────────────────────────────────────────────────────────── 프롬프트

# ⚠0818 파일럿 2회차에서 형식 준수율이 70~77% 에 머물렀다(사전등록 중단조건 ② < 80%).
# 실물을 보니 원인은 캡 부족이 아니라 **답을 맺지 않는 것**이었다 — 3,790 토큰을 쓰고도
# `\boxed` 가 없이 "Try: (24+2)=26  Then 26*6" 에서 잘렸다. 캡을 1536→4096 으로 올려도
# 모델이 그만큼 더 쓸 뿐이었다(토큰 1467→3154, 절단 88%→48%).
# ⇒ **탐색 예산을 프롬프트로 강제**하고, 못 찾아도 반드시 식을 내게 한다.
#   이렇게 해야 `acc=0` 이 "못 찾음"을 뜻하고 "안 맺음"을 뜻하지 않게 된다 —
#   SAVE 실험은 "못 풀다가 풀게 됨"을 재야 하지 "안 맺다가 맺음"을 재면 안 된다.
SOLVE_SYS = (
    "You are playing the Countdown numbers game. You are given a list of numbers "
    "and a target. Use EACH given number EXACTLY ONCE, combined with + - * / and "
    "parentheses, to make the target. Every intermediate value must be a positive "
    "integer (no negatives, and division only when it divides exactly).\n\n"
    "Work through AT MOST 8 candidate groupings, backtracking when one fails. Keep "
    "each attempt to one short line. Do NOT keep searching past 8 attempts.\n\n"
    "Then STOP and answer. Your response MUST end with the final arithmetic "
    "expression in \\boxed{...} — the expression itself, not its value. "
    "Example: \\boxed{(3+7)*8-25}\n"
    "If you cannot find an exact solution within 8 attempts, still end with "
    "\\boxed{...} containing your closest expression that uses each number once. "
    "Never end without a \\boxed{...}."
)

CHECK_SYS = (
    "You are reviewing a partial attempt at the Countdown numbers game. Name the "
    "single step, combination, or assumption in the work so far that is most likely "
    "to be blocking progress, in one sentence, referring to this puzzle's own "
    "numbers. Then give your confidence and a decision.\n\n"
    "Answer in exactly this form and nothing else:\n"
    "confidence: <0 to 1> | <your one sentence> | decision: verify\n"
    "  (use `decision: redirect` instead if the current branch should be abandoned)"
)

# b4p2 가 수렴했던 정형문의 Countdown 판 — 문제 정보 0, 일반적 재시도 권고.
# ★이 팔이 "아무 재시도 권고나 도움이 된다"는 귀무가설을 죽인다.
BOILER = ("confidence: 0.88 | The current combination looks plausible but should not "
          "be accepted without trying another grouping, since a different pairing may "
          "reach the target more directly. | decision: verify")


def user_msg(inst) -> str:
    return f"Numbers: {inst['nums']}\nTarget: {inst['target']}"


# ────────────────────────────────────────────────── 연산자 교체 오답 (완전 짝지음)

_OPS = "+-*/"


def swap_op_decoy(witness: str, nums, target, rng):
    """증인 식의 **연산자 하나만** 바꿔 목표를 빗나가게 만든다.

    ★수학 판의 `_rule_based_decoy` 는 정답 문자열을 규칙으로 비튼 것이라 **모델이 실제로
    고려한 적 없는** 문자열이었고, 그래서 `logp(gold) − logp(decoy)` 가 6/6 틀리는 문제에서도
    +6 nats 로 포화됐다(AUC(맞힘>틀림) 0.539). 여기서는 **같은 수·같은 구조·연산자 하나만**
    다르므로 모델이 실제로 헷갈릴 수 있는 오답이다 — PMI 포화 가설을 처음으로 제대로 검정한다.
    """
    pos = [i for i, c in enumerate(witness) if c in _OPS
           and i > 0 and witness[i - 1] not in "(+-*/"]     # 단항 부호 제외
    rng.shuffle(pos)
    for i in pos:
        for op in _OPS:
            if op == witness[i]:
                continue
            cand = witness[:i] + op + witness[i + 1:]
            try:
                v = eval(cand, {"__builtins__": {}}, {})     # noqa: S307
            except Exception:
                continue
            if isinstance(v, (int, float)) and v == int(v) and 0 < v != target:
                return cand
    return None


# ────────────────────────────────────────── 오라클 메타 (증인에서 유도) — 진단 전용

_INNER = re.compile(r"\((\d+)\s*([-+*/])\s*(\d+)\)")


def oracle_metas(witness: str, nums, rng):
    """증인 식에서 **실행 가능한 힌트**를 뽑아 강도별 메타를 만든다.

    ⚠**학습 방법이 아니라 진단이다.** 정답을 쓴다. 이것으로 성능을 주장할 수 없고,
    ***"고칠 여지가 있나"*** 만 답한다.

    여덟 판이 전부 "모델이 만든 메타"만 넣었다. 그래서 널이 **"기전이 없다"** 인지
    **"우리 메타가 나쁘다"** 인지 못 갈랐다. 오라클은 그 둘을 가른다:

        O3 (전체 식)   SAVE ~100% 여야 한다  ← ★배선검사. 아니면 실험 자체가 고장.
        O1 (첫 결합)   실행 가능한 조언의 상한
        OF (거짓 오라클) 같은 틀·**틀린 수** — "구체적 수가 있으면 도움이 된다"를 죽인다

    돌려주는 것: dict(O1, O3, OF) — 만들 수 없으면 그 키가 None.
    """
    inner = _INNER.findall(witness)
    if not inner:
        return {"O1": None, "O3": None, "OF": None}
    a, op, b = inner[0]
    val = {"+": int(a) + int(b), "-": int(a) - int(b),
           "*": int(a) * int(b), "/": int(a) // int(b) if int(b) else 0}[op]

    O1 = (f"confidence: 0.9 | The pairing to pursue is {a} {op} {b} = {val}; the "
          f"groupings tried so far do not lead to the target. | decision: redirect")
    O3 = (f"confidence: 0.95 | The full expression that reaches the target is "
          f"{witness}; stop searching and write it. | decision: redirect")

    # 거짓 오라클 — 같은 문장 틀, **틀린 결합**. 길이·구체성은 O1 과 같다.
    pool = [x for x in nums if x != int(a) and x != int(b)]
    if len(pool) >= 2:
        c, d = rng.sample(pool, 2)
    else:
        c, d = int(a) + 1, int(b) + 1
    fop = rng.choice([o for o in "+-*/" if o != op])
    fval = {"+": c + d, "-": c - d, "*": c * d, "/": c // d if d else 0}[fop]
    OF = (f"confidence: 0.9 | The pairing to pursue is {c} {fop} {d} = {fval}; the "
          f"groupings tried so far do not lead to the target. | decision: redirect")
    return {"O1": O1, "O3": O3, "OF": OF}
