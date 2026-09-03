"""SC(자기제어)/SCg 팔의 **행 특징 계산기** — SC_DESIGN.md(2026-09-04)의 유일한 구현처.

왜 별도 모듈인가. `countdown_rewards.py` 는 "팔 정체(TERMS/ARM_SPECS)와 조립
(arm_reward)"의 단일 정의처이고, 이 파일은 그 조립이 읽는 **원재료**(stuck·hi·
novel·followed·checked·y 등)를 텍스트에서 뽑는 자리다. 나누는 이유는 `countdown_rewards`
의 파일 머리말이 이미 선언한 것과 같다 — "팔의 정체는 한 곳에만 있어야 한다"와
"파싱은 새로 만든다(재사용이 오염이 되는 자리가 있다)"가 서로 다른 관심사이기 때문.

이 모듈은 **순수 함수**만 담는다 — torch·verl 을 import 하지 않는다(countdown_rewards
와 같은 CPU 테스트 규율).

행 특징 정의(SC_DESIGN.md §행 특징을 그대로 옮긴 것 — 사양이 원본이다):
    prefix    = 첫 <meta> **앞** 텍스트. <meta> 가 아예 없으면 prefix = 응답 전체(has_meta=0).
    n_att_pre = prefix 안의 `a op b = c` 등식 개수(참/거짓 무관 — «시도했다»의 카운트).
    pairs_pre = prefix 안에서 **연산자로 실제 결합된** 주어진-수 쌍의 집합(steer_prompts._pairs 와 동형).
    stuck     = 1[n_att_pre >= K_S]
    early     = 1[n_att_pre == 0]
    hi        = 1[confidence >= CONF_HI]  (confidence 는 클램프 없이 그대로 읽는다)
    novel     = 1[next 쌍 ∉ pairs_pre  ∧  next 의 두 수가 모두 nums 에 실재]
    followed  = 메타 뒤 첫 시도가 next 의 쌍을 실제로 잇는가(`countdown_rewards.plan_followed`)
    checked   = </meta> 와 **마지막** \\boxed{ 사이에 산술적으로 **참인** 등식이 있는가
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Mapping

from src.training.countdown_rewards import (
    _NEXT_RE,
    _PAIR_RE,
    parse_meta,
    plan_followed,
)

__all__ = [
    "prefix_features", "meta_fields", "post_meta_checked", "sc_row",
]


# ── 소수의 순수 헬퍼(countdown_rewards 의 규약과 동일: NaN/inf 는 fail-closed) ──────

def _bool01(x) -> int:
    return 1 if bool(x) else 0


# "산수 시도" 카운트 — countdown_rewards._ARITH 와 같은 모양이지만 **등식의 우변**
# (=c)까지 요구한다. n_att_pre 는 "정말로 계산해 봤다"의 카운트이고, 우변 없는
# `25+3` 같은 부분식은 "시도"로 안 센다(등식으로 매듭짓지 않았다).
_ARITH_EQ = re.compile(r"(\d+)\s*([+\-*/])\s*(\d+)\s*=\s*(\d+)")

# ★scripts/steer_prompts._pairs 를 **그대로 복사**한다(scripts/ 를 import 하지 않는다
#   — 이 파일이 torch 는 물론 scripts/ 의 어떤 것도 끌어오지 않아야 CPU 테스트가
#   가볍게 유지된다는 것이 이 저장소의 관례다). 원본이 바뀌면 이쪽도 손으로 맞춘다.
def _pairs(text: str, nums) -> set:
    """텍스트에서 «실제로 연산자로 결합된» 주어진-수 쌍의 집합."""
    want = {int(v) for v in nums}
    got = set()
    for a, o, b in re.findall(r"(\d{1,4})\s*([+\-*/×÷])\s*(\d{1,4})", text or ""):
        a, b = int(a), int(b)
        if a in want and b in want and a != b:
            got.add((min(a, b), max(a, b)))
    return got


def _pair_in_multiset(nums, a: int, b: int) -> bool:
    """`nums` 다중집합에 a·b 를 **동시에** 뽑을 수 있는가(같은 값이면 두 자리 필요).

    ★`countdown_rewards._multiset_has_pair` 와 같은 규칙이다(복제가 아니라 재확인 —
    "novel"의 «두 수가 모두 nums 에 실재」 조건이 그 함수의 존재 이유와 같기 때문에
    독립적으로 다시 적어 둔다. 둘이 갈리면 반드시 둘 다 고친다).
    """
    c = Counter(int(v) for v in nums)
    if a == b:
        return c[a] >= 2
    return c.get(a, 0) >= 1 and c.get(b, 0) >= 1


def _eval_eq(a: str, op: str, b: str, c: str) -> bool:
    """`a op b = c` 가 **정수 연산으로 정확히** 참인가. `/` 는 나머지 없이 나눠떨어질 때만."""
    ai, bi, ci = int(a), int(b), int(c)
    if op == "+":
        return ai + bi == ci
    if op == "-":
        return ai - bi == ci
    if op == "*":
        return ai * bi == ci
    if op == "/":
        return bi != 0 and ai % bi == 0 and ai // bi == ci
    return False


# ══════════════════════════════════════════════════════════════════════════════
# 프리픽스 특징
# ══════════════════════════════════════════════════════════════════════════════

def prefix_features(text: str, nums) -> dict:
    r"""첫 <meta> **앞** 텍스트에서 뽑는 특징. <meta> 가 없으면 prefix = 응답 전체.

    ★`parse_meta(text, "new")` 의 `start` 오프셋을 쓴다 — "완결된"(confidence·decision
    둘 다 있는) 메타가 아니라 **<meta> 태그가 존재하는 지점**까지를 프리픽스 경계로
    삼는다. 이유: SC 는 "막히기 전에 뭘 했나"를 재는 것이지 "형식을 완결했나"를 재는
    게 아니다 — 불완전한 메타라도 그 앞은 여전히 프리픽스다.
    """
    text = text or ""
    m = parse_meta(text, "new")
    if m.get("start") is not None:
        prefix = text[: int(m["start"])]
        has_meta = 1
    else:
        prefix = text
        has_meta = 0
    n_att_pre = len(_ARITH_EQ.findall(prefix))
    pairs_pre = _pairs(prefix, nums)
    pos_frac = (len(prefix) / len(text)) if len(text) > 0 else 0.0
    has_boxed_pre = 1 if "\\boxed{" in prefix else 0
    return {
        "n_att_pre": n_att_pre,
        "pairs_pre": pairs_pre,
        "pos_frac": pos_frac,
        "has_boxed_pre": has_boxed_pre,
        "has_meta": has_meta,
        "meta_start": m.get("start"),
        "meta_end": m.get("end"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 메타 필드
# ══════════════════════════════════════════════════════════════════════════════

# ruled_out 값 추출. body(=parse_meta 가 confidence/decision 줄을 뺀 뒤 공백으로
# 합친 것)에는 줄바꿈이 없으므로 "next:" 가 나오는 지점에서 **비탐욕적으로** 끊는다
# — 안 그러면 "ruled_out: A next: B" 에서 ruled_out 이 next 절까지 삼킨다.
_RULED_RE = re.compile(r"\bruled_out\s*:\s*(.*?)(?=\s*\bnext\s*:|$)", re.I)


def meta_fields(text: str) -> dict:
    r"""응답의 첫 메타 블록에서 confidence·decision·next 쌍·ruled_out 을 뽑는다.

    `next` 는 `countdown_rewards._NEXT_RE`(라인) → `_PAIR_RE`(그 안의 첫 쌍) 순서로
    읽는다 — `plan_next` 가 next 를 읽는 것과 **같은 두 정규식**이다(복제 금지).
    """
    text = text or ""
    m = parse_meta(text, "new")
    body = m.get("body", "") or ""

    next_pair = None
    nm = _NEXT_RE.search(body)
    if nm:
        pr = _PAIR_RE.search(nm.group(1))
        if pr:
            next_pair = (int(pr.group(1)), pr.group(2), int(pr.group(3)))

    ro = _RULED_RE.search(body)
    ruled_out = ro.group(1).strip() if ro else ""

    return {
        "confidence": m.get("confidence"),
        "decision": m.get("decision"),
        "next": next_pair,
        "ruled_out": ruled_out,
        "meta_start": m.get("start"),
        "meta_end": m.get("end"),
        "emitted": m.get("emitted"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 사후 검산(checked)
# ══════════════════════════════════════════════════════════════════════════════

def post_meta_checked(text: str, nums) -> int:
    r"""`</meta>` 와 **마지막** `\boxed{` 사이에 산술적으로 **참인** 등식이 있는가. 0/1.

    ★거짓 등식은 통과 못 한다 — `_eval_eq` 가 실제로 계산해서 비교한다(문자열
    패턴 매칭이 아니다). 메타가 없거나(=</meta> 경계가 없음) boxed 가 메타보다
    앞에 있으면(=검산 구간이 존재하지 않음) 0.
    `nums` 는 시그니처 일관성(다른 sc_* 헬퍼와 동형)을 위해 받지만 이 판정 자체는
    등식이 **어떤 수로 이뤄졌든** 참이면 인정한다 — "실제로 재계산했다"는 것이
    검산의 정의이고, 그 계산이 원래 수만 써야 한다는 제약은 SC_DESIGN.md 에 없다.
    """
    text = text or ""
    m = parse_meta(text, "new")
    end = m.get("end")
    if end is None:
        return 0
    end = int(end)
    boxed_idx = text.rfind("\\boxed{")
    if boxed_idx < 0 or boxed_idx <= end:
        return 0
    segment = text[end:boxed_idx]
    for a, op, b, c in _ARITH_EQ.findall(segment):
        if _eval_eq(a, op, b, c):
            return 1
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 조립 — SC/SCg 항이 읽는 행 하나
# ══════════════════════════════════════════════════════════════════════════════

def sc_row(text: str, nums, target, r_corr, K_S: int, CONF_HI: float) -> dict:
    r"""SC/SCg 항의 원재료 한 행. **정답 여부(r_corr)는 y 에만 쓰고, 채점 자체는 안 한다.**

    Returns dict — `countdown_rewards.TERMS["explore"/"explore_g"/"verify"/"early_cost"]`
    의 `needs` 가 요구하는 키(stuck·early·hi·dec_redirect·dec_verify·novel·followed·
    checked·y)를 전부 채우고, 진단용 원 특징(n_att_pre·pairs_pre·pos_frac·
    has_boxed_pre·confidence·decision·next·ruled_out)도 함께 돌려준다.

    ★`plan_ok`(SCg 가 추가로 필요로 하는 근거-진리)는 **여기서 계산하지 않는다** —
    그것은 `countdown_rewards.plan_next()`(완전열거, PL 팔과 공유)의 몫이고, 호출자
    (`verl_sdc._compute_countdown_arm_stash`)가 이 dict 에 병합해 넣는다. 이 함수
    안에서 다시 계산하면 "근거-진리는 SC 에서 안 쓴다"는 설계 경계가 코드로도
    흐려진다(SC_DESIGN.md: "근거-진리(완전 열거)는 SC 에서 쓰지 않는다").
    """
    text = text or ""
    pf = prefix_features(text, nums)
    mf = meta_fields(text)

    stuck = 1 if pf["n_att_pre"] >= int(K_S) else 0
    early = 1 if pf["n_att_pre"] == 0 else 0

    conf = mf["confidence"]
    hi = 1 if (conf is not None and float(conf) >= float(CONF_HI)) else 0

    dec_redirect = 1 if mf["decision"] == "redirect" else 0
    dec_verify = 1 if mf["decision"] == "verify" else 0

    novel = 0
    followed = 0
    end = mf.get("meta_end")
    after = text[int(end):] if end is not None else ""
    if mf["next"] is not None:
        a, _op, b = mf["next"]
        pair = (min(a, b), max(a, b))
        if pair not in pf["pairs_pre"] and _pair_in_multiset(nums, a, b):
            novel = 1
        followed = plan_followed(after, nums, a, b)
    elif pf["has_meta"]:
        # ★0904 «new» 프롬프트(next: 필드 없음)용 정의. gs0 실측(Qwen3-4B, new, 500×8):
        #   발화 행의 87% 가 이미 14회쯤 시도한 뒤이고 next 필드가 없으므로, «탐색»은
        #   메타 뒤 **첫 시도 쌍**이 프리픽스에 없던 쌍인가로 읽는다(실측 4.7%). 이때
        #   «이행»은 정의상 1 이다(첫 시도가 곧 계획). p3 처럼 next 가 있으면 위 분기.
        m = _PAIR_RE.search(after)
        if m:
            a, b = int(m.group(1)), int(m.group(3))
            pair = (min(a, b), max(a, b))
            if pair not in pf["pairs_pre"] and _pair_in_multiset(nums, a, b):
                novel = 1
            followed = 1

    checked = post_meta_checked(text, nums)
    y = 1 if _bool01(r_corr) else -1

    return {
        # TERMS 가 요구하는 키
        "stuck": stuck, "early": early, "hi": hi,
        "dec_redirect": dec_redirect, "dec_verify": dec_verify,
        "novel": novel, "followed": followed, "checked": checked, "y": y,
        # 진단/텔레메트리용 원 특징
        "n_att_pre": pf["n_att_pre"], "pairs_pre": pf["pairs_pre"],
        "pos_frac": pf["pos_frac"], "has_boxed_pre": pf["has_boxed_pre"],
        "has_meta": pf["has_meta"],
        "confidence": conf, "decision": mf["decision"], "next": mf["next"],
        "ruled_out": mf["ruled_out"],
    }
