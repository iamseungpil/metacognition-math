"""Countdown — 생성기 · 정확 채점기 · 프롬프트 두 벌 · verl 학습셋 빌더.

이 파일은 **과제 정의만** 담는다. 보상·텔레메트리·트레이너 배선은 다른 파일 몫이다.
출처: scratchpad `mini_abl/countdown.py` (검증된 부분만 이식). 이식하면서 고친 것과
고치지 않은 것을 아래에 전부 적는다 — 추측으로 넘어간 자리는 "미확인"이라고 적었다.

과제 규칙
    주어진 수를 **각각 정확히 한 번씩** 쓰고 + − × ÷ 로 목표수를 만든다.
    중간값은 항상 양의 정수여야 한다(뺄셈 음수 금지, 나눗셈은 나누어떨어질 때만).
    생성이 "무작위로 짝지어 합치기"라 **풀이 존재가 구성으로 보장**된다.
    ⇒ 실패 = 탐색 실패 = 자기점검이 고칠 수 있는 유형.

이식하며 **고친 것** (원본의 결함 — 넷. 전부 2026-08-18 에 원본과 나란히 실측했다.
같은 인스턴스 500개에 두 구현을 동시에 돌린 수치다.)
  ① `swap_op_decoy` — 500개 중 500개를 만들었지만 그 중 **32개(6.4%)가 Countdown
     규칙 위에서 불법**이었다. 원본은 `eval(cand)` 의 파이썬 의미로 **최종값만**
     봐서, 중간값이 **음수**이거나 나눗셈이 **나누어떨어지지 않는** 식을 통과시켰다.
     여기서는 `eval_countdown()` 로 **모든 중간값이 양의 정수**임을 강제하고
     수의 다중집합이 `nums` 와 일치하는지도 다시 확인한다 → 불법 0/500.
     구현도 문자 인덱스 치환에서 **AST 치환**으로 바꿨다(단항부호·자릿수 오탐 제거).
  ② `oracle_metas` 의 OF — 500개 중 **205개(41.0%)가 깨져 있었다**
     (등식 자체가 거짓 141 · Countdown 불법 64 · nums 밖의 수 1).
     원본이 `c - d`(음수 가능)·`c // d`(나누어떨어지지 않아도 몫)를 그대로 문장에
     박았기 때문이다. 여기서는 `nums` 안의 **실제로 합법인 짝**만 후보로 삼고 그 중
     참 힌트와 값이 다른 것을 고른다 → 500/500 정상. O1 의 값도 정확히 계산하고,
     합법 잎 결합을 못 찾으면 그 인스턴스에서는 O1 을 만들지 않는다(None).
     (사전 기록의 "OF 37.5% 가 산술적 거짓"과 같은 자리를 다시 잰 것이다. 41.0% 는
      Countdown 불법까지 포함한 수이고, 등식 거짓만 세면 28.2% 다.)
  ③ `grade` — 원본 docstring 은 "Fraction 으로 계산 — 부동소수 오차 0"이라고 적었지만
     실제로는 `eval()` 이 파이썬 `/` 로 **float** 를 만든 뒤에야 `Fraction` 으로 쌌다.
     ★실측 발산 사례: `\boxed{(((1/3)*5)*15)}` 는 정확히 25 인데 **원본은 0점**
     (거짓 음성), 수리판은 1점. R_corr 이 이 함수 위에 서므로 정답을 낸 롤아웃에서
     correctness 를 빼앗는 자리였다. 여기서는 AST 를 직접 `Fraction` 으로 접어
     오차를 0으로 만들었다(`eval` 도 함께 제거). **판정 의미는 바꾸지 않았다** —
     여전히 ⓐ파싱 ⓑ다중집합 일치 ⓒ값 일치 셋만 본다(중간값 제약은 채점에 넣지 않는다.
     사양의 R_corr 정의가 그렇다).
  ④ `parse_ok` — 문자집합만 보고 **파싱 불가 문자열에 형식 점수를 줬다**.
     `(3+*)` · `(2+3` · `3 4` · `((2+3)` · `+` 다섯 개 전부 원본 1점, 수리판 0점.
     사양의 w_format 0.35 가 이 함수 위에 서므로 쓰레기에 형식 보상이 흐르는 자리였다.

이식하며 **바꾸지 않은 것**
  `gen_instance` · `parse_ok` · `n_attempts` · `_last_boxed` 의 판정 의미.

옮기지 **않은 것**
  `CHECK_SYS` · `BOILER` — 사양 여덟 팔(A~H)에 별도 검사 프롬프트/정형문 주입 팔이
  없다. 필요해지면 그때 이 파일이 아니라 팔 정의 쪽에서 되살려라.
"""

from __future__ import annotations

import argparse
import ast
import json
import random
import re
from fractions import Fraction
from pathlib import Path

# ───────────────────────────────────────────────────────────────── 과제 상수

SEARCH_BUDGET = 16          # 사양: 탐색예산 16회
DEFAULT_N_NUMS = 5          # 사양: 수 5개
DEFAULT_LO, DEFAULT_HI = 1, 25
DEFAULT_MIN_TARGET = 20

META_OPEN_TAG = "<meta>"
META_CLOSE_TAG = "</meta>"

# 새 형식(A~G) 한 덩어리 / 옛 형식(H) 한 줄. 텔레메트리·보상이 같은 정규식을 쓰도록
# 형식 정의는 프롬프트와 같은 파일에 둔다.
META_BLOCK_RE = re.compile(r"<meta>(.*?)</meta>", re.DOTALL)
META_LINE_RE = re.compile(
    r"confidence:\s*([01](?:\.\d+)?)\s*\|(.*?)\|\s*decision:\s*(verify|redirect)",
    re.IGNORECASE)
META_CONF_RE = re.compile(r"confidence:\s*([01](?:\.\d+)?)", re.IGNORECASE)
META_DECISION_RE = re.compile(r"decision:\s*(verify|redirect)", re.IGNORECASE)


# ───────────────────────────────────────────────────────────── 정확 산술 (AST)

_BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_OP_CLASSES = (ast.Add, ast.Sub, ast.Mult, ast.Div)
_OP_CHARS = "+-*/"


class _Illegal(Exception):
    """Countdown 규칙 위반 (중간값이 양의 정수가 아니다)."""


def _parse(expr: str):
    """산술식 문자열 → AST 노드. 파싱 불가면 None."""
    try:
        return ast.parse(expr, mode="eval").body
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None


def _fold_exact(node) -> Fraction:
    """정확 유리수 접기. 중간값 제약 **없음** — 채점(`grade`)용.

    파이썬 `eval` 과 달리 `/` 가 float 를 만들지 않는다 → 부동소수 오차 0.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise _Illegal("non-integer literal")
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        v = _fold_exact(node.operand)
        return v if isinstance(node.op, ast.UAdd) else -v
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _fold_exact(node.left), _fold_exact(node.right)
        if isinstance(node.op, ast.Add):
            return a + b
        if isinstance(node.op, ast.Sub):
            return a - b
        if isinstance(node.op, ast.Mult):
            return a * b
        if b == 0:
            raise _Illegal("division by zero")
        return a / b
    raise _Illegal(f"unsupported node {type(node).__name__}")


def eval_exact(expr: str):
    """식을 정확 유리수로 평가. 평가 불가면 None. (중간값 제약 없음)"""
    node = _parse(expr)
    if node is None:
        return None
    try:
        return _fold_exact(node)
    except (_Illegal, RecursionError, ZeroDivisionError):
        return None


def _fold_countdown(node) -> int:
    """Countdown 규칙 하 평가 — **모든 중간값이 양의 정수**여야 한다."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise _Illegal("non-integer literal")
        if node.value <= 0:
            raise _Illegal("non-positive literal")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        a, b = _fold_countdown(node.left), _fold_countdown(node.right)
        if isinstance(node.op, ast.Add):
            v = a + b
        elif isinstance(node.op, ast.Sub):
            v = a - b
            if v <= 0:
                raise _Illegal("negative or zero intermediate")
        elif isinstance(node.op, ast.Mult):
            v = a * b
        else:
            if b == 0 or a % b != 0:
                raise _Illegal("inexact division")
            v = a // b
        if v <= 0:
            raise _Illegal("non-positive intermediate")
        return v
    raise _Illegal(f"unsupported node {type(node).__name__}")


def eval_countdown(expr: str):
    """Countdown 규칙을 지키며 평가한 정수. 규칙 위반·파싱 실패면 None.

    ★`swap_op_decoy` 의 수리가 이 함수 위에 서 있다.
    """
    node = _parse(expr)
    if node is None:
        return None
    try:
        return _fold_countdown(node)
    except (_Illegal, RecursionError):
        return None


def _unparse(node) -> str:
    """완전괄호 형태로 되돌린다 (증인 식과 같은 표기)."""
    if isinstance(node, ast.Constant):
        return str(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        return f"({_unparse(node.left)}{_BINOPS[type(node.op)]}{_unparse(node.right)})"
    raise _Illegal(f"unsupported node {type(node).__name__}")


def expr_numbers(expr: str) -> list[int]:
    """식에 등장한 정수 리터럴 다중집합 (정렬)."""
    return sorted(int(x) for x in _NUM.findall(expr))


# ────────────────────────────────────────────────────────────────────── 생성

def gen_instance(n_nums: int, rng: random.Random,
                 lo: int = DEFAULT_LO, hi: int = DEFAULT_HI,
                 min_target: int = DEFAULT_MIN_TARGET):
    """무작위로 짝지어 합쳐 목표수를 만든다 → **풀이 존재가 구성으로 보장**된다.

    부산물로 증인 풀이(witness)를 함께 돌려준다. 채점에는 안 쓰고, 문제가 진짜
    풀리는지 확인하는 데와 PMI 대조식을 만드는 데만 쓴다.

    돌려주는 것: {"nums": [...], "target": int, "witness": str} 또는 None(실패).
    ⚠원본과 동일한 알고리즘 — 손대지 않았다.
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


def extract_expr(text: str):
    """응답에서 최종 식 문자열을 뽑아 정규화한다. 없으면 None.

    `grade` 와 `parse_ok` 가 **같은 문자열**을 보게 하려고 하나로 뺐다
    (원본은 두 함수가 정규화 단계를 따로 갖고 있어 `\\frac` 처리에서 엇갈렸다).
    """
    e = _last_boxed(text)
    if e is None:
        return None
    e = e.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    e = e.replace("\\left", "").replace("\\right", "").replace("$", "").strip()
    e = re.sub(r"\\d?frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", e)
    if "=" in e:                      # "(3+7)*8 = 80" 꼴이면 좌변만
        e = e.split("=")[0].strip()
    return e


def grade(text: str, nums, target) -> int:
    """식을 뽑아 **정확히** 평가한다. 셋 다 맞아야 1점.

      ① 파싱 가능한 산술식인가
      ② 주어진 수를 각각 정확히 한 번씩 썼나 (다중집합 일치)
      ③ 값이 목표수와 같은가  (Fraction 접기 — 부동소수 오차 0)

    ★중간값 제약(양의 정수)은 **채점에 넣지 않는다.** 사양의
      R_corr = 1{식이 목표수를 만들고 주어진 수를 각각 한 번씩 쓴다} 를 그대로 따른다.
    """
    e = extract_expr(text)
    if e is None or not _EXPR_OK.match(e):
        return 0
    if expr_numbers(e) != sorted(int(v) for v in nums):
        return 0
    val = eval_exact(e)
    if val is None:
        return 0
    return int(val == Fraction(int(target)))


def parse_ok(text: str) -> int:
    """형식 준수 — 식을 뽑아 **정말 파싱까지 됐나** (정답 여부와 무관).

    ★수리 ④ (원본 결함): 원본은 `_EXPR_OK` 문자집합만 봤다. 그래서 `\\boxed{(3+*)}`
      처럼 **파싱조차 안 되는** 문자열이 형식 준수 1점을 받았다 — 이름과 docstring 이
      약속한 것("파싱까지는 됐나")과 다르다. 사양의 w_format 0.35 가 이 함수 위에
      서므로, 쓰레기 문자열에 형식 보상이 흐르는 자리였다.
      여기서는 문자집합 + **실제 AST 파싱** 둘 다 통과해야 1점이다.
      ⚠이 변경으로 형식 보상이 원본보다 **약간 엄격해진다**(파싱 불가 문자열이
      1→0). 값·다중집합은 여전히 보지 않는다.
    """
    e = extract_expr(text)
    if e is None or not _EXPR_OK.match(e):
        return 0
    return 1 if _parse(e) is not None else 0


def n_attempts(text: str) -> int:
    """★기전 지표 — 서로 다른 산술 시도를 몇 번 했나 (되돌아가기의 대리).

    수학에서는 못 재는 것이다. `a op b = c` 꼴을 세어 중복을 뺀다.
    """
    tries = set(re.findall(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)", text))
    return len(tries)


# ───────────────────────────────────────── ground_truth 직렬화 (verl 배선용)

def make_ground_truth(nums, target) -> str:
    """`reward_model.ground_truth` 에 실을 문자열.

    ★왜 target 만 넣지 않는가: `verl_sdc.py` 의 배치 보상 경로는 보상 함수에
    `ground_truths` **만** 넘긴다(`extra_info` 를 안 넘긴다 — 2026-08-18 확인,
    `src/training/verl_sdc.py:2255` `_compute_dcpo_heads_stash(completions,
    ground_truths, ...)` 및 `:2265` `reward_fn(completions=..., ground_truth=...)`).
    Countdown 채점에는 `nums` 가 반드시 필요하므로 **둘 다** 문자열에 실어야
    보상 쪽이 채점할 수 있다. `extra_info` 에도 같은 값을 중복으로 넣어 둔다.
    """
    return json.dumps({"target": int(target), "nums": [int(v) for v in nums]},
                      separators=(",", ":"))


def parse_ground_truth(gt):
    """`make_ground_truth` 의 역. (nums, target) 또는 (None, None)."""
    if isinstance(gt, dict):
        d = gt
    else:
        try:
            d = json.loads(str(gt))
        except (TypeError, ValueError):
            return None, None
    if not isinstance(d, dict) or "target" not in d or "nums" not in d:
        return None, None
    return [int(v) for v in d["nums"]], int(d["target"])


def grade_from_gt(text: str, gt) -> int:
    """보상 함수용 한 줄 진입점 — 직렬화된 ground_truth 로 바로 채점."""
    nums, target = parse_ground_truth(gt)
    if nums is None:
        return 0
    return grade(text, nums, target)


# ─────────────────────────────────────────────────────────────── 프롬프트 두 벌

# ⚠0818 파일럿에서 형식 준수율이 70~77% 에 머물렀다. 실물을 보니 원인은 캡 부족이
# 아니라 **답을 맺지 않는 것**이었다 — 3,790 토큰을 쓰고도 `\boxed` 없이 잘렸다.
# 캡을 1536→4096 으로 올려도 모델이 그만큼 더 쓸 뿐이었다.
# ⇒ **탐색 예산을 프롬프트로 강제**하고, 못 찾아도 반드시 식을 내게 한다.
#   이렇게 해야 `acc=0` 이 "못 찾음"을 뜻하고 "안 맺음"을 뜻하지 않는다.
_RULES = (
    "You are playing the Countdown numbers game. You are given a list of numbers "
    "and a target. Use EACH given number EXACTLY ONCE, combined with + - * / and "
    "parentheses, to make the target. Every intermediate value must be a positive "
    "integer (no negatives, and division only when it divides exactly).\n\n"
    f"Search budget: work through AT MOST {SEARCH_BUDGET} candidate groupings, "
    "backtracking when one fails. Keep each attempt to one short line. Do NOT keep "
    f"searching past {SEARCH_BUDGET} attempts.\n\n"
)

_CLOSING = (
    "\nThen STOP and answer. Your response MUST end with the final arithmetic "
    "expression in \\boxed{...} — the expression itself, not its value. "
    "Example: \\boxed{(3+7)*8-25}\n"
    f"If you cannot find an exact solution within {SEARCH_BUDGET} attempts, still "
    "end with \\boxed{...} containing your closest expression that uses each number "
    "once. Never end without a \\boxed{...}."
)

# 새 형식 — 사양의 A~G 팔. confidence → 전략 평가(★산수 금지) → decision.
# ★발화율 실측 근거: P2(형식 명시) 58.8% vs P0(요구 없음) 0.0%. 형식을 글자 그대로
#   보여주지 않으면 메타는 나오지 않는다. 그래서 예시 블록을 통째로 박는다.
SOLVE_SYS_NEW = (
    _RULES +
    "At least once during your search, stop and write a metacognitive block in "
    "EXACTLY this form, on its own lines:\n\n"
    "<meta>\n"
    "confidence: <a single number between 0 and 1>\n"
    "<One or two sentences judging YOUR OWN APPROACH so far: which family of "
    "groupings you are exploring, and whether that family is worth continuing. "
    "★Do NOT do arithmetic in here — no expressions, no equalities, no combining "
    "of numbers, no candidate answer. Assess the approach; do not solve the "
    "puzzle.>\n"
    "decision: verify\n"
    "</meta>\n\n"
    "Write `decision: verify` when the confidence you just wrote is high and the "
    "current line of search deserves to be pushed through and checked. Write "
    "`decision: redirect` when that confidence is low and the current line of "
    "search should be abandoned for a different family of groupings. The decision "
    "must follow from the confidence. Then continue the search in the way that "
    "decision commits you to.\n"
    + _CLOSING
)

# 옛 형식 — 사양의 H 팔. 한 줄. "형식의 값어치"를 재는 대조군이다.
SOLVE_SYS_OLD = (
    _RULES +
    "At least once during your search, stop and write a metacognitive line in "
    "EXACTLY this form, on its own line and nothing else on that line:\n\n"
    "confidence: 0.6 | <one sentence judging your own approach so far> | "
    "decision: verify\n\n"
    "Use `decision: redirect` instead of `decision: verify` when the current line "
    "of search should be abandoned. Then continue the search in the way that "
    "decision commits you to.\n"
    + _CLOSING
)

PROMPT_VARIANTS = {"new": SOLVE_SYS_NEW, "old": SOLVE_SYS_OLD}

# ★프롬프트 길이 실측 (2026-08-18, `Qwen/Qwen3-4B` 토크나이저 + chat template +
#   add_generation_prompt, 인스턴스 200개):
#       new  min 410 · p50 414 · p95 416 · max 417 토큰
#       old  min 310 · p50 314 · p95 316 · max 317 토큰
#   기존 `configs/verl_sdc_e21r_shared.yaml` 의 `max_prompt_length: 512` 안에 들어간다
#   (여유 95 토큰). 다만 그 config 는 `truncation: error` 라 프롬프트를 조금만 늘려도
#   학습이 죽는다. Countdown 용 config 를 새로 뜰 때 1024 로 잡아 두는 편이 안전하다.
#   ⚠A~G(new)와 H(old)는 프롬프트가 100 토큰 다르다 — H 팔은 "형식의 값어치"를 재는
#   대조군이므로 이 길이 차이 자체가 처치의 일부다(교란이 아니라 의도된 차이).


def user_msg(inst) -> str:
    return f"Numbers: {list(inst['nums'])}\nTarget: {inst['target']}"


def build_prompt(inst, variant: str = "new") -> list[dict]:
    """verl `prompt` 컬럼에 그대로 들어가는 chat 메시지 리스트(템플릿 적용 **전**)."""
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant: {variant!r}")
    return [
        {"role": "system", "content": PROMPT_VARIANTS[variant]},
        {"role": "user", "content": user_msg(inst)},
    ]


# ────────────────────────────────────── 연산자 교체 오답 (수리판) — PMI 대조식

def swap_op_decoy(witness: str, nums, target, rng: random.Random):
    """증인 식의 **연산자 하나만** 바꿔 목표를 빗나가게 만든다. 못 만들면 None.

    ★왜 이 오답인가: 수학 판의 규칙기반 오답은 정답 문자열을 규칙으로 비튼 것이라
    **모델이 실제로 고려한 적 없는** 문자열이었고, 그래서 `logp(gold) − logp(decoy)`
    가 6/6 틀리는 문제에서도 +6 nats 로 포화됐다(AUC 0.539). 여기서는 **같은 수·
    같은 구조·연산자 하나만** 다르므로 모델이 실제로 헷갈릴 수 있는 오답이다.

    ★수리(원본 대비):
      · 문자 인덱스 치환 → AST 치환. 자릿수·단항부호 오탐이 원천적으로 사라진다.
      · 합격 조건을 `eval` 최종값이 양의 정수 → **`eval_countdown` 전 중간값이
        양의 정수**로 강화. 원본은 음수 중간값·나누어떨어지지 않는 나눗셈을
        통과시켜 **Countdown 규칙 위에서 산술적으로 거짓인** 식을 만들었다.
      · 수의 다중집합이 `nums` 와 같은지 마지막에 다시 확인한다.
    """
    root = _parse(witness)
    if root is None:
        return None
    want = sorted(int(v) for v in nums)
    binops = [n for n in ast.walk(root) if isinstance(n, ast.BinOp)]
    cands = [(i, cls) for i in range(len(binops)) for cls in _OP_CLASSES]
    rng.shuffle(cands)
    for i, cls in cands:
        node = binops[i]
        if isinstance(node.op, cls):
            continue
        old = node.op
        node.op = cls()
        try:
            cand = _unparse(root)
        except _Illegal:
            cand = None
        finally:
            node.op = old
        if cand is None:
            continue
        v = eval_countdown(cand)
        if v is None or v == int(target):
            continue
        if expr_numbers(cand) != want:
            continue
        return cand
    return None


# ────────────────────────────────── 오라클 메타 (증인에서 유도) — 진단 전용

def _apply_op(a: int, b: int, op: str):
    """Countdown 합법 결합만 값을 돌려준다. 위반이면 None."""
    if op == "+":
        return a + b
    if op == "*":
        return a * b
    if op == "-":
        return a - b if a - b > 0 else None
    if op == "/":
        return a // b if b and a % b == 0 and a // b > 0 else None
    return None


def _witness_leaf_pairings(witness: str):
    """증인 식에서 **두 리터럴을 직접 결합하는** 노드들을 (a, op, b, val) 로 뽑는다.

    원본은 정규식 `\\((\\d+)\\s*([-+*/])\\s*(\\d+)\\)` 로 문자열을 훑었다. AST 로
    바꾼 이유는 결합 순서를 문자열 위치에 의존하지 않게 하려는 것뿐이고,
    "가장 먼저 나오는 잎 결합"을 고른다는 선택 자체는 원본과 같다.
    """
    root = _parse(witness)
    if root is None:
        return []
    out = []
    for node in ast.walk(root):
        if (isinstance(node, ast.BinOp) and type(node.op) in _BINOPS
                and isinstance(node.left, ast.Constant)
                and isinstance(node.right, ast.Constant)):
            a, b = node.left.value, node.right.value
            op = _BINOPS[type(node.op)]
            v = _apply_op(int(a), int(b), op)
            if v is not None:
                out.append((int(a), op, int(b), int(v)))
    return out


def _legal_pairings(nums):
    """`nums` 의 서로 다른 두 자리에서 만들 수 있는 **합법** 결합 전부.

    돌려주는 것: [(i, j, a, op, b, val), ...]  (i, j 는 nums 안의 위치)
    """
    out = []
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = int(nums[i]), int(nums[j])
            for op in _OP_CHARS:
                v = _apply_op(a, b, op)
                if v is not None:
                    out.append((i, j, a, op, b, v))
    return out


def _fmt_meta(conf: str, sentence: str, decision: str, fmt: str = "new") -> str:
    if fmt == "old":
        return f"confidence: {conf} | {sentence} | decision: {decision}"
    return (f"{META_OPEN_TAG}\nconfidence: {conf}\n{sentence}\n"
            f"decision: {decision}\n{META_CLOSE_TAG}")


def oracle_metas(witness: str, nums, rng: random.Random, fmt: str = "new"):
    """증인 식에서 **실행 가능한 힌트**를 뽑아 강도별 메타를 만든다.

    ⚠**학습 방법이 아니라 진단이다.** 정답을 쓴다. 이것으로 성능을 주장할 수 없고,
    ***"고칠 여지가 있나"*** 만 답한다. (그래서 이 문자열들만은 새 형식의
    "★산수 금지" 규칙을 일부러 어긴다 — 실행 가능한 힌트가 목적이기 때문이다.)

        O3 (전체 식)   SAVE ~100% 여야 한다  ← ★배선검사. 아니면 실험 자체가 고장.
        O1 (첫 결합)   실행 가능한 조언의 상한
        OF (거짓 오라클) 같은 틀·**다른 결합** — "구체적 수가 있으면 도움이 된다"를 죽인다

    ★수리(원본 대비): OF 의 등식이 **산술적으로 참**이도록 강제한다. 원본은
      `c - d`(음수 가능)·`c // d`(나누어떨어지지 않아도 몫)를 그대로 문장에 박아
      등식 자체가 틀린 문자열을 만들었다(OF 37.5% 거짓). 여기서는 `nums` 안의
      **합법 결합**만 후보로 삼고, 참 힌트와 값이 다른 것을 고른다. 참 힌트 O1 도
      `_apply_op` 로 검증된 결합만 쓴다.

    돌려주는 것: dict(O1, O3, OF) — 만들 수 없으면 그 키가 None.
    """
    out = {"O1": None, "O3": None, "OF": None}
    nums = [int(v) for v in nums]

    leaves = _witness_leaf_pairings(witness)
    if not leaves:
        return out
    a, op, b, val = leaves[0]

    out["O1"] = _fmt_meta(
        "0.9",
        f"The pairing to pursue is {a} {op} {b} = {val}; the groupings tried so "
        f"far do not lead to the target.",
        "redirect", fmt)
    out["O3"] = _fmt_meta(
        "0.95",
        f"The full expression that reaches the target is {witness}; stop "
        f"searching and write it.",
        "redirect", fmt)

    # 거짓 오라클 — 같은 문장 틀, **다른(그러나 참인) 결합**. 길이·구체성은 O1 과 같다.
    # 참 힌트가 쓴 두 자리는 되도록 피한다(원본의 `pool = nums - {a, b}` 의도).
    ia = next((k for k in range(len(nums)) if nums[k] == a), None)
    ib = next((k for k in range(len(nums)) if k != ia and nums[k] == b), None)
    used = {ia, ib} - {None}

    pool = [p for p in _legal_pairings(nums)
            if not (p[3] == op and {p[2], p[4]} == {a, b}) and p[5] != val]
    disjoint = [p for p in pool if p[0] not in used and p[1] not in used]
    pick_from = disjoint or pool
    if pick_from:
        _, _, c, fop, d, fval = rng.choice(pick_from)
        out["OF"] = _fmt_meta(
            "0.9",
            f"The pairing to pursue is {c} {fop} {d} = {fval}; the groupings "
            f"tried so far do not lead to the target.",
            "redirect", fmt)
    return out


# ──────────────────────────────────────────────────────── 인스턴스 배치 생성

def gen_instances(n: int, seed: int, *,
                  n_nums: int = DEFAULT_N_NUMS,
                  lo: int = DEFAULT_LO, hi: int = DEFAULT_HI,
                  min_target: int = DEFAULT_MIN_TARGET,
                  require_decoy: bool = True,
                  dedup: bool = True,
                  max_draws_per_row: int = 200) -> list[dict]:
    """증인·오답까지 붙은 인스턴스 n 개. 결정적(seed 로만 결정된다).

    각 행: {"nums", "target", "witness", "decoy"}
    `require_decoy=True` 면 연산자 교체 오답을 못 만든 인스턴스는 버리고 다시 뽑는다
    (PMI 대조식이 없는 행은 D 팔·PMI 계열에서 쓸 수 없다).
    """
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < n:
        got = None
        for _ in range(max_draws_per_row):
            inst = gen_instance(n_nums, rng, lo=lo, hi=hi, min_target=min_target)
            if inst is None:
                continue
            key = (tuple(inst["nums"]), inst["target"])
            if dedup and key in seen:
                continue
            decoy = swap_op_decoy(inst["witness"], inst["nums"], inst["target"], rng)
            if decoy is None and require_decoy:
                continue
            inst["decoy"] = decoy
            got = (key, inst)
            break
        if got is None:
            raise RuntimeError(
                f"gen_instances: could not produce row {len(rows)} within "
                f"{max_draws_per_row} draws (n_nums={n_nums}, lo={lo}, hi={hi}, "
                f"min_target={min_target}, require_decoy={require_decoy})")
        key, inst = got
        seen.add(key)
        rows.append(inst)
    return rows


# ──────────────────────────────────────────────────── verl 학습셋 (parquet)

def build_records(n: int, seed: int, *,
                  variant: str = "new",
                  n_nums: int = DEFAULT_N_NUMS,
                  data_source: str = "countdown",
                  split: str = "train",
                  include_flat_cols: bool = True,
                  **gen_kwargs) -> list[dict]:
    """verl 이 읽는 레코드 리스트. `build_parquet` 이 이걸 쓴다."""
    rows = gen_instances(n, seed, n_nums=n_nums, **gen_kwargs)
    records = []
    for idx, inst in enumerate(rows):
        nums = [int(v) for v in inst["nums"]]
        target = int(inst["target"])
        extra = {
            "index": idx,
            "split": split,
            "nums": nums,
            "target": target,
            "witness": inst["witness"],
            "decoy": inst["decoy"] if inst["decoy"] is not None else "",
            "prompt_variant": variant,
            "search_budget": SEARCH_BUDGET,
        }
        rec = {
            "data_source": data_source,
            "prompt": build_prompt(inst, variant),
            "ability": "countdown",
            "reward_model": {"style": "rule",
                             "ground_truth": make_ground_truth(nums, target)},
            "extra_info": extra,
        }
        if include_flat_cols:
            rec.update({
                "nums": nums,
                "target": target,
                "witness": inst["witness"],
                "decoy": inst["decoy"] if inst["decoy"] is not None else "",
            })
        records.append(rec)
    return records


def build_parquet(n: int, seed: int, out_path,
                  *,
                  variant: str = "new",
                  n_nums: int = DEFAULT_N_NUMS,
                  data_source: str = "countdown",
                  split: str = "train",
                  include_flat_cols: bool = True,
                  **gen_kwargs) -> dict:
    """verl 이 읽는 학습셋 parquet 을 만든다. 돌려주는 것: 요약 dict.

    ── 스키마 (2026-08-18 확인) ────────────────────────────────────────────
    이 저장소가 실제로 학습에 먹인 parquet 을 그대로 따랐다:
      `src/training/verl_gdpo_data.py:344 records_to_parquet` 의 docstring 과
      `data/verl_train_meta_mix.parquet` 실물 컬럼
      ['data_source', 'prompt', 'ability', 'reward_model', 'split_tags'].
      `configs/verl_sdc_e21r_shared.yaml:9` 이 `data.prompt_key: prompt` 로
      `prompt` 컬럼을 가리킨다.

      data_source   : str
      prompt        : list[{"role": str, "content": str}]   ← chat 템플릿 적용 **전**
      ability       : str
      reward_model  : {"style": "rule", "ground_truth": str}
      extra_info    : dict  ← verl 이 보상 함수에 넘기는 표준 통로

    ⚠**미확인**: `verl==0.7.1` 패키지가 이 머신에 설치돼 있지 않아
      (`import verl` → ModuleNotFoundError, requirements.txt:30 이 verl==0.7.1 을
      고정) `RLHFDataset` 소스를 직접 읽어 확인하지 못했다. 그래서 다음 두 가지는
      **추정**이다:
        (a) 스키마에 없는 여분 컬럼(nums/target/witness/decoy)이 `non_tensor_batch`
            로 그대로 통과하는가. 이 저장소가 `split_tags` 라는 여분 컬럼을 실제로
            실어 돌렸다는 사실이 근거이지만, `split_tags` 는 dict 이고 `nums` 는
            **길이 5 리스트**라 verl 의 `collate_fn` 이 `np.array(val, dtype=object)`
            로 (B, 5) 2차원 배열을 만들 수 있다. 그 형태가 `DataProto` 배치 검사를
            통과하는지 확인하지 못했다. 문제가 나면 `include_flat_cols=False` 로
            끄고 `extra_info` 만 쓰면 된다(같은 값이 중복 저장돼 있다).
        (b) `extra_info` 가 보상 함수까지 전달되는가 — verl 표준 규약이지만,
            **이 저장소의 배치 보상 경로는 어차피 `extra_info` 를 안 넘긴다**
            (`verl_sdc.py:2265`). 그래서 채점에 필요한 `nums`·`target` 은
            `reward_model.ground_truth` 문자열에도 실어 두었다(`make_ground_truth`).
    """
    records = build_records(
        n, seed, variant=variant, n_nums=n_nums, data_source=data_source,
        split=split, include_flat_cols=include_flat_cols, **gen_kwargs)

    import pandas as pd  # 지연 임포트: 채점기·프롬프트만 쓰는 쪽에 pandas 를 강요하지 않는다

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(out_path, index=False)

    # 왕복 검사 — verl 이 읽는 접근 패턴 그대로 (verl_gdpo_data.records_to_parquet 와 동형)
    chk = pd.read_parquet(out_path)
    s = chk.iloc[0]
    assert s["prompt"][0]["content"], "round-trip: prompt[0]['content'] empty"
    assert s["prompt"][-1]["role"] == "user", "round-trip: last message must be user"
    g_nums, g_target = parse_ground_truth(s["reward_model"]["ground_truth"])
    assert g_nums is not None and g_target is not None, "round-trip: ground_truth"
    assert grade("\\boxed{%s}" % s["extra_info"]["witness"], g_nums, g_target) == 1, \
        "round-trip: witness must grade 1 against the serialized ground_truth"

    n_decoy = sum(1 for r in records if r["extra_info"]["decoy"])
    return {
        "path": str(out_path),
        "rows": len(records),
        "variant": variant,
        "n_nums": n_nums,
        "seed": seed,
        "columns": list(df.columns),
        "with_decoy": n_decoy,
        "targets_min": min(r["extra_info"]["target"] for r in records),
        "targets_max": max(r["extra_info"]["target"] for r in records),
    }


def main():
    p = argparse.ArgumentParser(description="Build a Countdown verl parquet")
    p.add_argument("--n", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--variant", choices=sorted(PROMPT_VARIANTS), default="new")
    p.add_argument("--n_nums", type=int, default=DEFAULT_N_NUMS)
    p.add_argument("--split", default="train")
    p.add_argument("--no_flat_cols", action="store_true",
                   help="nums/target/witness/decoy 최상위 컬럼을 빼고 extra_info 만 남긴다")
    a = p.parse_args()
    info = build_parquet(a.n, a.seed, a.out, variant=a.variant, n_nums=a.n_nums,
                         split=a.split, include_flat_cols=not a.no_flat_cols)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
