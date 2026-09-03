"""조향 프롬프트 5칸 + 기계 채점기 — 혼잣말이 «다음 행동을 지정»하게 만든다.

발견(2026-09-01): 현행 프롬프트가 혼잣말의 두 필드를 각각 무력화한다.
  · "The decision must follow from the confidence."  → decision 은 confidence 의 이진화.
    확신을 통제하면 부분상관 0.020 (p=0.67) — 독립 정보 0.
  · "Do NOT do arithmetic in here — no combining of numbers."
    → «어느» 계열로 갈지 못 씀 → 선언이 다음 행동을 못 바꾼다 (novelty 결합 p=0.54).
전자는 편법 방어(answer_leak)로 넣은 것인데 조향에 필요한 내용까지 같이 막았다.

5칸 설계 — 두 축을 분리한다 (형식 × 예시):
  P0   현행                                     (P0 형식, 예시 없음)
  P0e  현행 + 예시 하나                         (P0 형식, 예시 있음)   ★예시 효과만 분리
  P1   금지를 좁힘: «완성식»만 금지              (P0 형식, 예시 없음)
  P2   P1 + ruled_out/next 구조 + decision 재배선 (P2 형식, 예시 없음)
  P3   P2 + 예시 하나                            (P2 형식, 예시 있음)
  → 금지(P1−P0) · 구조(P2−P1) · 예시(P0e−P0, P3−P2) · 헤드라인(P3−P0e)

★P2 는 P1 «에서만» 달라야 한다(감사 지적). 이전 판은 블록을 통째로 갈아끼우며 금지
  문장까지 지워서 «구조 추가»와 «금지 제거»를 뒤섞었다. 지금은 좁힌 금지를 구조 안에 남긴다.

★게이밍 방어는 금지어가 아니라 «검증»으로 한다. 감사가 찾아낸 최강 공격은
  «순응적 공허» — `ruled_out: none` + 원래 하려던 걸 `next` 에 적기. 파싱되고, 거짓도
  아니고, 준수율이 정의상 1.0 이 된다. RL 이 가장 먼저 찾을 정책이다. 그래서:
    ① next 신규성   next 가 지목한 결합이 «메타 직전 5줄»에 이미 있으면 실격
    ② ruled_out 실재 적어낸 식이 앞 텍스트에 실제로 나와야 함. `none` 은 앞에 실패가
                    하나도 없을 때만 인정. (없는 걸 버렸다고 못 함)
    ③ next 부분집합 next 가 지목한 수는 주어진 수의 «진부분집합»(≤N−1) 이어야 함
                    — 해법을 여러 조각으로 쪼개 흘리는 answer_leak 변형 차단
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.environ.get("REPO", "/home/jovyan/beomi/splee/metacognition-math"))
from src.training.countdown_task import PROMPT_VARIANTS, eval_countdown  # noqa: E402

_BAN = ("★Do NOT do arithmetic in here — no expressions, no equalities, "
        "no combining of numbers, no candidate answer. "
        "Assess the approach; do not solve the puzzle.")

_BAN_NARROW = ("★Do NOT write a complete expression that uses ALL the given numbers — "
               "that is the answer and it does not belong here. "
               "Partial groupings you have ruled out or intend to try next are fine.")

# decision 을 confidence 가 아니라 «행동 필드»에 묶는 문단 (P2/P3 전용)
_OLD_MANDATE = ("Write `decision: verify` when the confidence you just wrote is high and the "
                "current line of search deserves to be pushed through and checked. Write "
                "`decision: redirect` when that confidence is low and the current line of search "
                "should be abandoned for a different family of groupings. The decision must "
                "follow from the confidence.")
_NEW_MANDATE = ("Write `decision: redirect` when `next` names a grouping from a different family "
                "than the one you have been exploring; write `decision: verify` when `next` says "
                "you will check the current line. The decision must follow from `ruled_out` and "
                "`next`. Confidence reports how likely the current family is to succeed; it does "
                "not dictate the decision.")

P0 = PROMPT_VARIANTS["new"]
assert _BAN in P0, "현행 프롬프트에서 금지 문구를 찾지 못했다 — 프롬프트가 바뀌었다"
assert _OLD_MANDATE in P0, "decision 문단을 찾지 못했다 — 프롬프트가 바뀌었다"

P1 = P0.replace(_BAN, _BAN_NARROW)

_STRUCT = """<meta>
confidence: <a single number between 0 and 1>
ruled_out: <the specific partial groupings you have already tried and eliminated, \
comma-separated, e.g. `25*3, (25+3)*7`. Write `none` only if you have tried nothing yet. \
List only groupings that actually appear in your work above and that actually failed.>
next: <the specific partial grouping you will try next, e.g. `8*7 first`. It must be one you \
have NOT already combined above, and it must use FEWER than all of the given numbers. \
If you are going to verify instead of changing course, write what you will check.>
<One sentence judging YOUR OWN APPROACH: which family of groupings you are exploring, and \
whether that family is worth continuing. ★Do NOT write a complete expression that uses ALL \
the given numbers — that is the answer and it does not belong here.>
decision: verify
</meta>"""

_BLOCK_RE = re.compile(r"<meta>\nconfidence:.*?\n</meta>", re.S)


def _swap_block(prompt: str, new_block: str) -> str:
    out, n = _BLOCK_RE.subn(new_block, prompt, count=1)
    assert n == 1, "메타 블록 서식을 찾지 못했다"
    return out


P2 = _swap_block(P1, _STRUCT).replace(_OLD_MANDATE, _NEW_MANDATE)

# ── 예시 두 벌 — 형식만 다르고 나머지는 전부 맞춘다 ─────────────────────────
#   같은 문제 [25,3,7,8]→68 · confidence 0.3 · decision redirect · 같은 판단 문장
#   (예시의 decision 은 발화 성향을 강하게 유도하므로 반드시 일치시킨다)
_EX_HEAD = "\n\nExample of the block, for numbers [25, 3, 7, 8] and target 68:\n\n"
_EX_JUDGE = ("The multiply-25-first family overshoots badly and I keep having to subtract "
             "back, so it is not worth continuing.")

EX_P0 = _EX_HEAD + ("<meta>\nconfidence: 0.3\n" + _EX_JUDGE + "\ndecision: redirect\n</meta>")
EX_P2 = _EX_HEAD + ("<meta>\nconfidence: 0.3\nruled_out: 25*3, (25+3)*7\nnext: 8*7 first\n"
                    + _EX_JUDGE + "\ndecision: redirect\n</meta>")

P0e = P0 + EX_P0
P3 = P2 + EX_P2

VARIANTS = {"P0": P0, "P0e": P0e, "P1": P1, "P2": P2, "P3": P3}

# ══ 채점기 ════════════════════════════════════════════════════════════════
_FIELD = re.compile(r"^\s*(ruled_out|next)\s*:\s*(.*?)\s*$", re.I | re.M)   # 0902: 줄 단위 (이전 판은 개행을 삼켜 next 를 못 읽었다)
_EXPR = re.compile(r"\(?\d+\s*[+\-*/×÷]\s*\(?[\d\s+\-*/×÷()]*\d\)?")
_INT = re.compile(r"\d+")


def _norm(e: str) -> str:
    return re.sub(r"\s+", "", e.replace("×", "*").replace("÷", "/")).strip(".,")


def parse_fields(meta_body: str) -> dict:
    d = {"ruled_out": "", "next": ""}
    for k, v in _FIELD.findall(meta_body):
        d[k.lower()] = v.strip()
    return d


def exprs_in(text: str) -> list[str]:
    out = []
    for m in _EXPR.finditer(text):
        e = _norm(m.group(0))
        if any(o in e for o in "+-*/") and _INT.search(e):
            out.append(e)
    return out


def _pairs(text: str, nums) -> set:
    """텍스트에서 «실제로 연산자로 결합된» 주어진-수 쌍의 집합."""
    want = {int(v) for v in nums}
    got = set()
    for a, o, b in re.findall(r"(\d{1,4})\s*([+\-*/×÷])\s*(\d{1,4})", text):
        a, b = int(a), int(b)
        if a in want and b in want and a != b:
            got.add((min(a, b), max(a, b)))
    return got


def full_number_exprs(meta_body: str, nums) -> list[str]:
    """혼잣말 안의 «주어진 수를 전부 쓴» 식들 — 형식 위반 후보(유출과는 다르다)."""
    want = sorted(int(v) for v in nums)
    return [e for e in exprs_in(meta_body)
            if sorted(int(x) for x in _INT.findall(e)) == want]


def leaks_answer(meta_body: str, nums, target=None) -> bool:
    """★진짜 정답 유출 = 네 수를 다 쓰고 «목표에 실제로 닿는» 식이 혼잣말 안에 있다.

    이전 정의(네 수를 다 쓰기만 하면 유출)는 과탐지였다 — 실측으로 P2 에서 0.684 가
    나왔지만, 그 대부분은 «버렸다»고 나열한 실패 식이었다. 목표 도달까지 확인하면 0.139.
    실패 식 나열은 유출이 아니라 서식 위반이므로 full_number_exprs 로 따로 센다.
    """
    es = full_number_exprs(meta_body, nums)
    if target is None:
        return bool(es)
    return any(eval_countdown(e) == int(target) for e in es)


def false_ruled_out(meta_body: str, target: int) -> tuple[int, int]:
    """ruled_out 의 식 중 «사실은 목표에 닿는» 것 / 평가 가능한 총수 (거짓 주장)."""
    f = parse_fields(meta_body)["ruled_out"]
    if not f or f.strip().lower() == "none":
        return 0, 0
    bad = tot = 0
    for e in exprs_in(f):
        v = eval_countdown(e)
        if v is None:
            continue
        tot += 1
        bad += int(v == int(target))
    return bad, tot


def ruled_out_grounded(meta_body: str, before_text: str) -> bool | None:
    """②실재 검사 — 버렸다고 적은 식이 앞 텍스트에 실제로 나오는가.
    `none` 은 앞에 시도 흔적이 하나도 없을 때만 인정한다. 필드 없으면 None."""
    f = parse_fields(meta_body)["ruled_out"]
    if not f:
        return None
    pre = _norm(before_text)
    if f.strip().lower() == "none":
        return not bool(re.search(r"\d\s*[+\-*/×÷]\s*\d", before_text))
    es = exprs_in(f)
    return bool(es) and all(e in pre for e in es)


def next_ok(meta_body: str, nums, before_text: str) -> bool | None:
    """①신규성 + ③부분집합 — next 가 «앞 5줄에 없던» «진부분집합» 결합인가."""
    plan = parse_fields(meta_body)["next"]
    if not plan.strip():
        return None
    want = {int(v) for v in nums}
    named = {int(x) for x in _INT.findall(plan)} & want
    if len(named) < 2 or len(named) >= len(list(nums)):     # ③ 진부분집합
        return False
    recent = "\n".join(before_text.strip().splitlines()[-5:])
    return not (_pairs(plan, nums) & _pairs(recent, nums))  # ① 이미 한 결합이면 실격


def followed_plan(meta_body: str, before_text: str, after_text: str, nums) -> float | None:
    """준수율 — next 가 지목한 «신규» 결합이 바로 뒤에서 실제로 «연산자로» 결합되는가.

    ★정답 여부를 전혀 보지 않는다(순환 검증 차단).
    ★쌍이 «한 줄에 같이 나온다»는 이전 판정은 위양성 72.9% 로 사실상 공허했다(감사 실측).
      «연산자 결합»으로 조이면 39.7%, 여기에 «앞 5줄에 없던 결합»까지 요구하면 10.6%.
    """
    plan = parse_fields(meta_body)["next"]
    if not plan.strip():
        return None
    recent = "\n".join(before_text.strip().splitlines()[-5:])
    want = _pairs(plan, nums) - _pairs(recent, nums)        # 신규 결합만
    if not want:
        return None
    head = "\n".join(after_text.strip().splitlines()[:3])
    return len(want & _pairs(head, nums)) / len(want)


def score_meta(meta_body: str, before_text: str, after_text: str, nums, target: int) -> dict:
    """한 사이트의 전 항목 채점. false_claim_in_meta 는 산문 거짓 주장까지 잡는다."""
    try:
        from src.training.countdown_inv import false_claim_in_meta
        fc = int(bool(false_claim_in_meta(meta_body, int(target))))
    except Exception:
        fc = -1
    bad, tot = false_ruled_out(meta_body, target)
    f = parse_fields(meta_body)
    return dict(has_fields=int(bool(f["ruled_out"] and f["next"])),
                leak=int(leaks_answer(meta_body, nums, target)),
                full_expr=len(full_number_exprs(meta_body, nums)),
                false_claim=fc, false_ruled_out=bad, ruled_out_n=tot,
                grounded=ruled_out_grounded(meta_body, before_text),
                next_ok=next_ok(meta_body, nums, before_text),
                followed=followed_plan(meta_body, before_text, after_text, nums))


# ══ P4 — «인지에 대한 인지»만 남긴 판 ══════════════════════════════════════
# 사용자 지적(2026-09-01): "메타 인지 부분은 가급적 인지에 대한 내용만 들어와야 하는 거 아니야?"
# 옳다. P2/P3 는 `ruled_out: 25*3, (25+3)*7` 처럼 «대상 수준»의 산술을 메타에 넣었다.
# 그건 메타인지가 아니라 «탐색을 블록 안으로 옮긴 것»이다. 실측이 그 대가를 보여줬다 —
# P2 는 실패한 네 수 식을 14개나 나열했고(유출률 0.684), P3 는 413토큰에서 일찍 멈췄다(정답률 0.120).
#
# P4 는 산술을 «전혀» 넣지 않는다(P0 의 전면 금지를 그대로 유지). 대신 «계열 이름»을 말하게 한다.
# 계열은 인지의 대상이 아니라 «접근»이므로 메타인지다. 그러면서도 기계 채점이 된다 —
# 닫힌 어휘로 제한하고, 메타 뒤 첫 결합이 그 계열과 맞는지 대조한다.
# ★감사 수정: 평면 6단어는 (a) 이중소속(3*7 은 multiply 이면서 pair-two-small)으로
#   우연 준수를 부풀리고 (b) divide 는 모델이 사실상 안 써서 자동 실패 단어이며
#   (c) add 하나만 고정으로 외치면 0.25~0.30 을 공짜로 얻는다(순응적 공허의 계열판).
#   → 직교하는 «두 축»으로 쪼개고 준수는 «둘 다 맞을 때»만 인정한다.
#     실측 바닥 0.040 · 게이밍 천장 0.174 (평면판 0.091 / 0.30 의 절반)
OPS = ("multiply", "add", "subtract", "divide")
PAIRS = ("two-small", "two-large", "mixed")

_STRUCT_FAM = """<meta>
confidence: <a single number between 0 and 1>
next_op: <the operation you will apply FIRST in your next attempt. EXACTLY one of: \
multiply, add, subtract, divide.>
next_pair: <which two of the given numbers that first operation will combine, by size. \
EXACTLY one of: two-small, two-large, mixed.>
<One sentence judging YOUR OWN APPROACH: why that family is or is not worth continuing. \
★Do NOT do arithmetic in here — no expressions, no equalities, no combining of numbers, \
no candidate answer. Assess the approach; do not solve the puzzle.>
decision: verify
</meta>"""

_FAM_MANDATE = ("Write `decision: redirect` when the `next_op`/`next_pair` you just named differ "
                "from the ones you have been using; write `decision: verify` when they are the "
                "same because you will stay in this line and check your work. Both are equally "
                "valid moves — verifying a good line is as useful as abandoning a bad one. "
                "The decision must follow from `next_op` and `next_pair`. Confidence reports how "
                "likely the current line is to succeed; it does not dictate the decision.")

P4 = _swap_block(P0, _STRUCT_FAM).replace(_OLD_MANDATE, _FAM_MANDATE)
# ★P4e 는 «만들지 않는다» — 블록 단독 예시는 P0e/P3 에서 조기 종료를 유도해 −11~−24%p 를
#   냈다(짧은 응답 비율 9.8%→33.3%→62.7%, 그 구간 정답률 86%→23%→5%). 예시 축이 필요하면
#   시도 과정과 \boxed 가 들어간 «궤적형» 예시로 따로 만든다(P4t).
VARIANTS["P4"] = P4

# P0d — decision 을 confidence 에서 «떼기만» 한 칸. 잉여물로 판명된 그 한 문장만 바꾼다.
_P0D_MANDATE = ("Write `decision: verify` when you will stay in the current line of search and "
                "check it through; write `decision: redirect` when you will abandon it for a "
                "different family of groupings. Both are equally valid moves — verifying a good "
                "line is as useful as abandoning a bad one. Decide from what your search has "
                "actually shown, not from the confidence number you just wrote.")
VARIANTS["P0d"] = P0.replace(_OLD_MANDATE, _P0D_MANDATE)

_FAMFIELD = re.compile(r"^(ruled_out_family|next_family|next_op|next_pair)\s*:\s*([a-z\-]+)",
                       re.I | re.M)
_FAM_OP = {"multiply-first": "*", "add-first": "+", "subtract-first": "-", "divide-first": "/"}


def parse_families(meta_body: str) -> dict:
    d = {"ruled_out_family": "", "next_family": "", "next_op": "", "next_pair": ""}
    for k, v in _FAMFIELD.findall(meta_body):
        k, v = k.lower(), v.strip().lower()
        if (k.endswith("op") and v in OPS) or (k.endswith("pair") and v in PAIRS) \
                or v == "none" or v in OPS or v in PAIRS:
            d[k] = v
    return d


def _first_combo(text: str, nums):
    """첫 «주어진 수 두 개 결합» → (작은수, 큰수, 연산자).

    ★감사 수정: 이전 판은 `a != b` 를 요구해 «같은 수 두 개»를 묶는 합법 수를 통째로
      못 봤다. val 문제의 23.8%, train 의 20.7% 가 중복 수를 포함한다.
      중복 다중집합을 소모하는 방식으로 바꾼다.
    """
    from collections import Counter
    want = Counter(int(v) for v in nums)
    for a, o, b in re.findall(r"(\d{1,4})\s*([+\-*/×÷])\s*(\d{1,4})", text):
        a, b = int(a), int(b)
        c = Counter((a, b))
        if all(want.get(k, 0) >= v for k, v in c.items()):
            return (min(a, b), max(a, b), {"×": "*", "÷": "/"}.get(o, o))
    return None


def combo_axes(combo, nums):
    """결합 → (연산축, 짝크기축). ★크기 분할은 «값»이 아니라 «위치»로 한다 —
    [5,5,5,25] 에서 값집합으로 나누면 5 가 작은쪽·큰쪽 양쪽에 들어간다(감사 지적)."""
    if not combo:
        return None, None
    lo, hi, op = combo
    op_name = {"*": "multiply", "+": "add", "-": "subtract", "/": "divide"}.get(op)
    srt = sorted(int(v) for v in nums)
    half = len(srt) // 2
    small, large = srt[:half], srt[half:]           # 위치 기반 다중집합
    from collections import Counter
    cs, cl = Counter(small), Counter(large)
    in_s = lambda v: cs.get(v, 0) > 0
    in_l = lambda v: cl.get(v, 0) > 0
    if in_s(lo) and in_s(hi) and not (in_l(lo) and in_l(hi)):
        pair = "two-small"
    elif in_l(lo) and in_l(hi):
        pair = "two-large"
    else:
        pair = "mixed"
    return op_name, pair


def family_of(combo, nums) -> set:
    """한 결합이 속하는 계열들 (연산 계열 + 크기 계열)."""
    if not combo:
        return set()
    lo, hi, op = combo
    out = {k for k, v in _FAM_OP.items() if v == op}
    srt = sorted(int(v) for v in nums)
    half = len(srt) // 2
    small, large = set(srt[:half]), set(srt[half:])
    if lo in small and hi in small:
        out.add("pair-two-small")
    if lo in large and hi in large:
        out.add("pair-two-large")
    return out


def followed_axes(meta_body: str, before_text: str, after_text: str, nums):
    """양축 준수 — 선언한 (연산, 짝크기) 를 «둘 다» 맞히고 «직전과 달라야» 인정.

    ★감사 수정: 이전 판은 followed 를 novel 과 «따로» 보고해서, 원래 하던 계열을
      그대로 외치는 고정 정책이 followed=1.0 을 공짜로 받았다. 이제 결합해서 낸다.
    """
    f = parse_families(meta_body)
    op, pr = f.get("next_op"), f.get("next_pair")
    if not op or not pr:
        return None
    recent = "\n".join(before_text.strip().splitlines()[-5:])
    prev = combo_axes(_first_combo(recent, nums), nums)
    head = "\n".join(after_text.strip().splitlines()[:3])
    did = combo_axes(_first_combo(head, nums), nums)
    if did == (None, None):
        return None
    novel = (op, pr) != prev
    return float(novel and did == (op, pr))


def followed_family(meta_body: str, before_text: str, after_text: str, nums):
    """준수율(계열판) — next_family 로 «선언한» 계열을 메타 뒤 첫 결합이 실제로 따르는가.

    ★정답을 보지 않는다. ★신규성도 함께 본다 — 직전 5줄의 계열을 그대로 다시 말하면
      「원래 하려던 걸 적기」(순응적 공허)이므로 실격.
    """
    f = parse_families(meta_body)
    nxt = f["next_family"]
    if not nxt or nxt == "none":
        return None, None
    recent = "\n".join(before_text.strip().splitlines()[-5:])
    prev_fams = family_of(_first_combo(recent, nums), nums)
    novel = nxt not in prev_fams
    head = "\n".join(after_text.strip().splitlines()[:3])
    did = family_of(_first_combo(head, nums), nums)
    return (1.0 if nxt in did else 0.0), novel


def score_meta_family(meta_body: str, before_text: str, after_text: str, nums, target: int) -> dict:
    """P4 계열판 채점 — 산술이 «들어 있으면» 그 자체로 형식 위반."""
    f = parse_families(meta_body)
    # ★감사 지적: 이전 정규식은 "2-3 attempts", "0.5-0.7" 같은 산문 범위에 오탐했다.
    #   주어진 수가 실제로 결합될 때만 위반으로 센다.
    arith = _first_combo(meta_body, nums) is not None
    fol = followed_axes(meta_body, before_text, after_text, nums)
    return dict(has_fields=int(bool(f["next_op"] and f["next_pair"])),
                arith_violation=int(arith),
                leak=int(leaks_answer(meta_body, nums, target)),
                followed=fol,
                op=f["next_op"], pair=f["next_pair"])


if __name__ == "__main__":
    for k, v in VARIANTS.items():
        print(f"───── {k}  ({len(v)}자) " + "─" * 46)
        print(v[600:] if k in ("P2", "P3") else v[-620:])
        print()
    print("═" * 62)
    print("예시 길이 대조 (예시 효과 분리를 위해 맞춰야 함):",
          f"EX_P0 {len(EX_P0)}자 · EX_P2 {len(EX_P2)}자")


# ══ P5 — 전략 어휘가 있는 decision (T2: prompt-seeded strategy option) ═══════════
# P0 과 «decision 줄만» 다르다. confidence 줄과 산술 전면 금지는 그대로 둔다.
# 이진(verify/redirect) 대신 «다음에 무엇을 할지»를 닫힌 5어휘로 말하게 하고, GRPO 0스텝의
# 집단상대 이득 부호가 어휘별로 갈리는지를 본다(t2_seed_advantage.py).
DECISIONS5 = ("continue", "answer", "backward", "decompose", "constrain")

_STRUCT_P5 = """<meta>
confidence: <a single number between 0 and 1>
<One or two sentences judging YOUR OWN APPROACH so far: which family of groupings you are \
exploring, and whether that family is worth continuing. ★Do NOT do arithmetic in here — no \
expressions, no equalities, no combining of numbers, no candidate answer. Assess the approach; \
do not solve the puzzle.>
decision: <one of: continue | answer | backward | decompose | constrain>
</meta>"""

_P5_MANDATE = ("Choose the decision from what your search has actually shown, not from the "
               "confidence number. `continue` = keep the current family of groupings; "
               "`answer` = the solution is in hand, stop and answer; "
               "`backward` = start from the target and undo one operation to reach it; "
               "`decompose` = first build an intermediate value near the target and adjust it; "
               "`constrain` = place the hardest number first and fit the others around it.")

P5 = _swap_block(P0, _STRUCT_P5).replace(_OLD_MANDATE, _P5_MANDATE)
assert "backward" in P5 and "decompose" in P5 and "<meta>" in P5 and _BAN in P5
VARIANTS["P5"] = P5

_DEC5 = re.compile(r"^\s*decision\s*:\s*[`'\"<\[(*]*\s*([a-z]+)", re.I | re.M)


def parse_decision5(meta_body: str) -> str | None:
    """P5 의 decision 어휘 하나. 5어휘 밖이거나 줄이 없으면 None."""
    m = _DEC5.search(meta_body)
    if not m:
        return None
    v = m.group(1).lower()
    return v if v in DECISIONS5 else None


# ★0902 P5e: P5 + «backward» 결정 예시 1개 (산술 없음). T2 s73 에서 backward 발현 0.7% 로
#   검정력이 없었다(페이블). 예시 블록으로 발현을 올려 어드밴티지 부호를 잴 수 있게 한다.
EX_P5 = _EX_HEAD + ("<meta>\nconfidence: 0.2\nThe additive groupings I have tried all land far from "
                    "the target, so this family is not worth continuing.\ndecision: backward\n</meta>")
P5e = P5 + EX_P5
assert "decision: backward" in P5e and "<meta>" in P5e
VARIANTS["P5e"] = P5e


# ★0902 P3t: P3 + «사실 보고» 강제 — ruled_out 에는 위에서 «실제로 시도해 실패한» 묶음만 적고, 없으면 none.
P3t = P3.replace(
    "Example of the block,",
    "IMPORTANT: `ruled_out` must list ONLY groupings you ACTUALLY tried above (copy them from your own attempts). "
    "Never invent a grouping you did not try. If you have not tried anything yet, write `ruled_out: none`.\n\n"
    "Example of the block,")
assert P3t != P3
VARIANTS["P3t"] = P3t


# ★0902 궤적 수준 대조용: 메타 지시문이 «없는» 프롬프트 (countdown_task.SOLVE_SYS_PLAIN 과 동일)
from src.training.countdown_task import SOLVE_SYS_PLAIN as _PLAIN
VARIANTS["PLAIN"] = _PLAIN
assert "<meta>" not in VARIANTS["PLAIN"]
