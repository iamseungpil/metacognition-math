r"""도치 자(inverted ruler) — «정답을 알려준 뒤 메타 토큰이 얼마나 덜 그럴듯해지는가».

    sh_t = logP(meta_t | 프롬프트+HINT ⊕ 접두)  −  logP(meta_t | 프롬프트 ⊕ 접두)
    inv  = agg_{t ∈ 프로즈}(sh_t)

두 문맥은 **user 메시지 끝의 힌트 한 줄만** 다르고, 채점 대상 토큰열(메타)은 양쪽에서
**바이트 동일**하다. 샘플링 없음 — teacher-forced forward 두 팔뿐이다.

★이 파일이 존재하는 유일한 이유: «검증한 자» 와 «학습한 자» 를 같은 함수로 묶는 것.
  2026-08-31 설계검토가 잡은 사고가 정확히 그 분기였다:
    · `scripts/reverse_ruler.py`  : V1_prose_min = min(agg 는 gold) − min(plain)
                                    = «최소의 차이». 프로즈를 **별도 시퀀스로 재인코딩**해
                                      confidence 줄을 문맥에서 빼버린다.
    · `scripts/gaming_probe.py`   : inv_prose_min = min(hint − plain) = «차이의 최소».
                                      전체 메타를 문맥에 두고 토큰 마스크로 프로즈만 본다.
  같은 37 사이트에서 스피어만 **0.296**. 둘은 같은 자가 아니었다. 라벨 판별력(d=1.00)은
  앞의 자에서, 게이밍 취약성(conf_wrong 37/37)은 뒤의 자에서 나온 숫자다.
  ⇒ 이 모듈은 **뒤쪽(학습이 실제로 계산할 수 있는 형태)** 을 유일한 정의로 삼고,
    라벨 판별력을 그 정의로 **다시** 재도록 강제한다(`scripts/inv_ruler_unified.py`).

★게이밍 방어의 핵심은 **단측(one-sided)** 이다. 아래 `INV_SIGN` 주석 참조.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field

__all__ = [
    "INV_HINT_TMPL", "INV_ANCHOR_TMPL", "INV_AGGS", "INV_AGG", "INV_TAU",
    "INV_C", "INV_TAU_PROVISIONAL", "INV_MIN_PROSE_TOK", "INV_LEAK_NGRAM",
    "INV_SCOPES", "INV_FORMS", "INV_SCOPE", "INV_FORM", "prose_char_span",
    "inv_hint_prompt", "prose_token_flags", "inv_aggregate", "false_claim_in_meta",
    "build_inv_arms", "read_inv_from_ref_logprobs", "score_inv", "inv_signature",
    "inv_empty_row",
]

_NAN = float("nan")

# ── 힌트 문자열 · 삽입 위치 ────────────────────────────────────────────────────
# ★`scripts/reverse_ruler.py:92` 와 **바이트 동일**해야 한다. 오프라인에서 검증한 자와
#   학습이 쓰는 자가 한 글자라도 다르면 이 모듈의 존재 이유가 사라진다.
INV_HINT_TMPL = "\n\nHint: one valid solution is {w}."
# 삽입 앵커: `countdown_task.user_msg` 가 만드는 user 메시지의 **마지막 줄**.
#   프롬프트 텍스트(chat 템플릿 적용 후)에서 이 문자열의 마지막 출현 뒤에 힌트를 끼운다.
#   ⇒ 템플릿 문법을 하나도 가정하지 않는다. 앵커가 없으면 **즉사**한다(조용히 문장 끝에
#     붙이면 힌트가 assistant 발화로 들어가 다른 조건이 되고, 그 사실이 로그에 안 남는다).
INV_ANCHOR_TMPL = "Target: {t}"

# ── 집계 · 범위 · 형태 축 ─────────────────────────────────────────────────────
INV_AGGS = ("min", "bot25", "mean")

# ★«도치» 라는 이름 아래 서로 다른 자가 둘 있었다(설계검토 1번 반대, ρ=0.296).
#   여기서 **두 축으로 분해**해 넷을 전부 이름 붙인다. 어느 칸을 쓰는지 서명에 박힌다.
#   scope — 무엇을 채점 대상 토큰열로 두는가
#     "inplace"  : 메타 토큰열 그대로(태그·confidence 줄이 **문맥에 남는다**), 프로즈 마스크
#     "reencode" : 프로즈 문자 구간만 **따로 인코딩**해 채점(= `reverse_ruler` 가 한 것.
#                  confidence 줄이 문맥에서 빠진다)
#   form  — 언제 빼는가
#     "d2a"      : agg_t( logp_hint,t − logp_plain,t )   «차이의 집계» (= `gaming_probe`)
#     "a2d"      : agg_t(logp_hint,t) − agg_t(logp_plain,t)  «집계의 차이» (= `reverse_ruler`)
INV_SCOPES = ("inplace", "reencode")
INV_FORMS = ("d2a", "a2d")

# ★부호 규약 — 이 항 전체가 여기 있다.
#   라벨(Δ_abl)상 **해로운** 메타는 inv 가 **높다**(정답을 알아도 어색해지지 않는다).
#   그러므로 벌은 «inv 가 τ 를 넘은 만큼» 이다:
#       R_inv = − clip( max(0, inv − τ) / c , 0, 1 )       ∈ [−1, 0]
#   ★왜 반드시 단측인가 (실측 근거, 추측 아님):
#     양측(= −inv 를 상으로)으로 쓰면 «확신에 찬 오답» 이 실제 혼잣말을 **37/37** 사이트
#     에서 이긴다(마진 6.5배). 정답을 알면 가장 어색해지는 문장이 바로 «틀린 식을 확신에
#     차서 선언한 문장» 이므로, 그것이 이 항의 argmax 다 — 예외가 아니라 정의다.
#     단측으로 자르면 그 꼬리 전체가 페널티 0 에 눌려 이득이 정확히 사라진다.
#   ★탈출구가 있는 벌이다: 메타를 안 내면 이 항은 0 이고 손실은 `META_FLOOR`(0.02) 뿐이다.
#     그래서 발화 침식 감시(`emit_rate < 0.2` abort)를 반드시 함께 켠다.
# ★자의 정체(scope/form/agg)와 눈금(τ·c)은 **`countdown_rewards` 한 곳**에만 있다.
#   OSD 선례(`OSD_W_MAX`/`OSD_LEAK_NGRAM` 를 스코어러가 import)와 같은 규약이다 —
#   상수를 두 곳에 두면 갈리고, 갈린 순간 "선언된 판정식 != 채택된 수치" 가 성립한다.
from src.training.countdown_rewards import (            # noqa: E402
    INV_AGG, INV_C, INV_FALSE_CLAIM_PEN, INV_FORM, INV_MIN_PROSE_TOK,
    INV_SCOPE, INV_TAU, INV_TAU_PROVISIONAL,
)

# 누출 가드(G5) — OSD 와 **같은 규약**. 메타가 최종식을 담으면 이 항을 0 으로.
INV_LEAK_NGRAM = 8

# ── 프로즈 판정: confidence:/decision: 줄과 태그 줄은 프로즈가 아니다 ──────────
# `scripts/reverse_ruler.prose_span` 과 같은 규칙. 여기서는 char span 이 아니라
# **토큰 플래그**를 만든다(재인코딩 없이 원래 토큰화 위에서 판정하기 위해).
_CONF_LINE = re.compile(r"^\s*confidence\s*:", re.I)
_DEC_LINE = re.compile(r"^\s*decision\s*:", re.I)


def _prose_char_flags(meta_raw: str) -> list[bool]:
    """meta_raw 의 문자별 «프로즈인가» 플래그."""
    flags = [False] * len(meta_raw)
    off = 0
    for ln in meta_raw.split("\n"):
        s, e = off, off + len(ln)
        off = e + 1                                   # '\n' 한 글자
        st = ln.strip()
        if not st or st.startswith("<") or _CONF_LINE.match(ln) or _DEC_LINE.match(ln):
            continue
        for i in range(s, min(e, len(flags))):
            flags[i] = True
    return flags


def prose_token_flags(text: str, span, offsets) -> list[bool]:
    r"""메타 토큰 구간 `[open_start, close_end)` 각 토큰이 프로즈인가.

    판정은 **엄격**하다: 토큰의 문자 구간이 프로즈 글자와 겹치고 **비프로즈 글자와는
    한 글자도 겹치지 않을** 때만 True. 경계에 걸친 토큰(예: `>\nconfidence`)은 빠진다 —
    빠지면 표본이 줄 뿐이지만, 섞으면 confidence 줄의 logp 가 프로즈 점수에 새어든다.

    `offsets` 는 `tokenizer(text, return_offsets_mapping=True)` 의 결과이고, 그
    토큰화가 `span.ids` 와 같아야 한다(호출자가 확인한다 — OSD 의
    `_osd_encode_with_offsets` 와 같은 규약).
    """
    lo, hi = int(span.char_open), int(span.char_close_end)
    if lo < 0 or hi <= lo:
        raise ValueError(f"prose_token_flags: 메타 문자 구간이 없다 ({lo}, {hi}).")
    cflags = _prose_char_flags(text[lo:hi])
    out = []
    for i in range(int(span.open_start), int(span.close_end)):
        a, b = offsets[i]
        if b <= a:
            out.append(False)
            continue
        hit_prose = hit_other = False
        for c in range(a, b):
            if lo <= c < hi and cflags[c - lo]:
                hit_prose = True
            else:
                hit_other = True
        out.append(bool(hit_prose and not hit_other))
    return out


def inv_aggregate(sh, agg: str = None) -> float:
    """프로즈 토큰의 sh 배열 → 스칼라. 유한하지 않으면 NaN."""
    a = INV_AGG if agg is None else str(agg)
    xs = [float(v) for v in sh if math.isfinite(float(v))]
    if len(xs) < INV_MIN_PROSE_TOK:
        return _NAN
    if a == "min":
        return float(min(xs))
    if a == "mean":
        return float(sum(xs) / len(xs))
    if a == "bot25":
        xs = sorted(xs)
        k = max(1, len(xs) // 4)
        return float(sum(xs[:k]) / k)
    raise ValueError(f"inv_aggregate: 모르는 집계 {a!r}. 가능: {INV_AGGS}")


# ── G2. 메타가 «식» 을 선언하면 그 식을 실제로 검증한다 ───────────────────────
# 왜: 기존 `countdown_rewards._ARITH` 는 `a op b = c` 만 보고, 게다가 **텔레메트리
#     전용**이라 보상식·ABORT 어디에도 안 들어간다(설계검토 실측). 게이밍 프로브의
#     conf_wrong 문장 `"The expression (96-24)*3 reaches the target 216 exactly..."` 는
#     `=` 가 없어 그 정규식에 걸리지 않는다. 여기서 정면으로 막는다:
#     메타 안의 식처럼 보이는 토큰열을 뽑아 `countdown_task.eval_countdown` 으로 평가하고,
#     target 이 **아닌데** 「맞다」는 취지의 단어와 함께 있으면 거짓 선언으로 센다.
_EXPR_RE = re.compile(r"[0-9(][0-9\s+\-*/().]{1,}[0-9)]")
_CLAIM_RE = re.compile(
    r"\b(reach(es|ed)?|equals?|gives?|yields?|correct|confirmed|works?|solves?|"
    r"is\s+the\s+(answer|solution))\b", re.I)


def _balanced(expr: str) -> str:
    """앞뒤의 짝 없는 괄호를 떼어 균형 잡힌 부분식으로 만든다(못 만들면 원문)."""
    e = expr.strip()
    for _ in range(8):
        depth = lo = 0
        for ch in e:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                lo = min(lo, depth)
        if lo < 0:                                     # 여는 괄호가 모자란다 → 앞을 자른다
            i = e.find(")")
            if i < 0:
                break
            e = e[i + 1:].strip()
            continue
        if depth > 0:                                  # 닫는 괄호가 모자란다 → 뒤를 자른다
            i = e.rfind("(")
            if i < 0:
                break
            e = e[:i].strip().rstrip("+-*/ ")
            continue
        return e
    return e


_NEG_RE = re.compile(
    r"\b(not|never|no|none|fails?|failed|failing|cannot|wrong|incorrect|"
    r"unable|off\s+by|doesn|didn|isn|won|can|don)\b|n't", re.I)


def false_claim_in_meta(meta_text: str, target, *, require_claim_word: bool = True) -> int:
    r"""메타가 «target 에 닿지 않는 식» 을 **긍정형으로** 맞다고 선언하면 1.

    ★보수적으로 센다(거짓양성이 그대로 벌이 되므로 정직한 혼잣말을 때리면 안 된다):
      · 스캔 범위는 **프로즈 줄만**. `confidence: 0.6` 의 소수점이 식 후보에 빨려 들어가
        뒤의 진짜 선언을 가리는 사고를 원천 차단한다(실측으로 잡은 결함).
      · 문장에 부정어가 있으면 건너뛴다 — "I tried 24*15 but it did not work" 은
        거짓 선언이 아니라 **정직한 보고**다.
      · `eval_countdown` 이 값을 못 내면(문법 오류·나눗셈 위반) 세지 않는다.
      · target 에 **닿는** 식은 여기서 안 막는다. 그건 `answer_leak`(G5)의 소관이고,
        한 사건을 두 가드가 두 번 벌하면 그 행이 이유 없이 두 배로 맞는다.
    """
    from src.training.countdown_task import eval_countdown   # noqa: PLC0415

    txt = meta_text or ""
    if not txt:
        return 0
    try:
        tgt = int(target)
    except (TypeError, ValueError):
        return 0
    flags = _prose_char_flags(txt)
    prose_lines, cur = [], []
    for ch, ok in zip(txt, flags):
        if ok:
            cur.append(ch)
        elif cur:
            prose_lines.append("".join(cur)); cur = []
    if cur:
        prose_lines.append("".join(cur))
    for line in prose_lines:
        for sent in re.split(r"(?<=[.!?])\s+", line):
            if require_claim_word and not _CLAIM_RE.search(sent):
                continue
            if _NEG_RE.search(sent):
                continue
            for m in _EXPR_RE.finditer(sent):
                expr = _balanced(m.group(0))
                if not any(op in expr for op in "+-*/"):
                    continue
                try:
                    val = eval_countdown(expr)
                except Exception:
                    continue
                if val is None or int(val) == tgt:
                    continue
                return 1
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# 배치 조립 — 행당 **2팔**, 고정 순서 `plain, hint`. OSD 와 같은 부기(base = 2*k).
# ══════════════════════════════════════════════════════════════════════════════
# ⚠PMI 배치와 **섞지 않는다**. `countdown_pmi.read_pmi_from_ref_logprobs` 는
#   `base = 4*k` 고정 스트라이드를 가정한다(verl_sdc.py:2082 의 같은 경고).

def inv_hint_prompt(prompt_text: str, witness: str, target) -> str:
    """프롬프트 텍스트의 user 메시지 끝(`Target: N`)에 힌트 한 줄을 끼운 사본.

    앵커가 없으면 **ValueError**. 조용히 문장 끝에 붙이면 힌트가 assistant 발화가 되어
    오프라인 검증과 다른 조건이 되고, 그 사실이 어디에도 안 남는다.
    """
    anchor = INV_ANCHOR_TMPL.format(t=int(target))
    i = (prompt_text or "").rfind(anchor)
    if i < 0:
        raise ValueError(
            f"inv_hint_prompt: 프롬프트에 앵커 {anchor!r} 가 없다 — 힌트를 끼울 자리를 "
            f"모른다. countdown_task.user_msg 의 형식이 바뀌었는지 확인하라.")
    j = i + len(anchor)
    return prompt_text[:j] + INV_HINT_TMPL.format(w=str(witness)) + prompt_text[j:]


@dataclass
class _InvAttempt:
    row: int
    n_meta: int
    prose: tuple = field(repr=False)      # 메타 토큰별 프로즈 플래그


def inv_empty_row(status: str = "off") -> dict:
    """행 규약의 기본값. OSD `_osd_empty_row` 와 같은 «못 쟀다 vs 0 이다» 구분."""
    d = 0.0 if status in _INV_DEFINED_ZERO else None
    # ★거짓선언 판정도 «안 쟀다» 와 «0 이다» 를 가른다. 스코어러가 아예 안 돈 행
    #   (off / error / ref_error)에서 0 을 흘리면 `false_claim_rate` 가 «깨끗하다» 로
    #   위장하고 그 ABORT 가 무력해진다. 그런 행은 None 이고, 집계가 NaN → missing 이 된다.
    fc = 0 if status in _INV_DEFINED_ZERO else None
    return {"inv_raw": d, "inv_scored": 0, "inv_leak": 0, "inv_n_prose": 0,
            "inv_status": status, "inv_false_claim": fc}


# W 가 아니라 **프로즈 토큰**이 구조적으로 없는 사유 → inv := 0 (정상 행, 벌 없음)
_INV_DEFINED_ZERO = frozenset({"no_meta", "short_prose", "leak", "no_witness"})
# 스코어러가 안 돌았다 → None 을 유지해 fail-loud
_INV_NOT_MEASURED = frozenset({"off", "ref_error", "span_error", "pending"})


def prose_char_span(meta_raw: str):
    """`reverse_ruler.prose_span` 과 **같은 규칙**: 첫 프로즈 줄 시작 ~ 마지막 프로즈 줄 끝."""
    lines, off, spans = meta_raw.split("\n"), 0, []
    for ln in lines:
        s, e = off, off + len(ln)
        off = e + 1
        st = ln.strip()
        if not st or st.startswith("<") or _CONF_LINE.match(ln) or _DEC_LINE.match(ln):
            continue
        spans.append((s, e))
    if not spans:
        return None
    return spans[0][0], spans[-1][1]


def build_inv_arms(tokenizer, prompt_texts, response_texts, witnesses, targets,
                   final_exprs, scope: str = None):
    r"""점수를 매길 **2n 개** (문맥, 메타) 팔. GPU 를 잡지 않는다.

    행 하나당 팔 둘, 고정 순서: `plain`(힌트 없음), `hint`(힌트 있음).
    두 팔의 응답 토큰열은 **같은 객체에서 복사한 같은 열**이다 — 그것이 이 자의 전제다.

    Returns (arm_prompts, arm_resps, attempts, per_row, diag).
    """
    from src.training import countdown_pmi as _cdp             # noqa: PLC0415
    from src.training import countdown_rewards as _cdr         # noqa: PLC0415

    sc = INV_SCOPE if scope is None else str(scope)
    if sc not in INV_SCOPES:
        raise ValueError(f"build_inv_arms: 모르는 scope {sc!r}. 가능: {INV_SCOPES}")
    B = len(response_texts)
    if not (len(prompt_texts) == len(witnesses) == len(targets) == len(final_exprs) == B):
        raise ValueError(
            f"build_inv_arms: 길이 불일치 prompt={len(prompt_texts)} resp={B} "
            f"witness={len(witnesses)} target={len(targets)} expr={len(final_exprs)}")

    arm_prompts, arm_resps, attempts = [], [], []
    per_row = [inv_empty_row("no_meta") for _ in range(B)]
    diag = {"B": B, "no_meta": 0, "no_witness": 0, "short_prose": 0,
            "leak_blocked": 0, "span_error": 0, "anchor_error": 0,
            "false_claim": 0, "n_emitted": 0, "attempted": 0,
            "prose_tok_sum": 0, "fwd_tokens": 0, "leak_reasons": {}}

    for i in range(B):
        text = response_texts[i] or ""
        try:
            span = _cdp.find_meta_token_span(tokenizer, text)
        except TypeError:
            raise                                    # fast 토크나이저 없음 = 즉사(설계)
        except Exception as e:
            per_row[i] = inv_empty_row(f"span_error:{type(e).__name__}")
            diag["span_error"] += 1
            continue
        if span is None:
            diag["no_meta"] += 1
            continue
        diag["n_emitted"] += 1

        wit = str(witnesses[i] or "")
        if not wit:
            per_row[i] = inv_empty_row("no_witness")     # 힌트를 만들 수 없다 → 벌 없음
            diag["no_witness"] += 1
            continue

        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if list(enc["input_ids"]) != list(span.ids):
            per_row[i] = inv_empty_row("span_error:retokenize")
            diag["span_error"] += 1
            continue
        offsets = [tuple(o) for o in enc["offset_mapping"]]

        try:
            prose = prose_token_flags(text, span, offsets)
        except Exception as e:
            per_row[i] = inv_empty_row(f"span_error:{type(e).__name__}")
            diag["span_error"] += 1
            continue
        n_prose = sum(1 for p in prose if p)
        meta_raw = text[span.char_open:span.char_close_end]
        fclaim = false_claim_in_meta(meta_raw, targets[i])
        diag["false_claim"] += int(fclaim)

        if n_prose < INV_MIN_PROSE_TOK:
            r = inv_empty_row("short_prose")
            r["inv_n_prose"] = n_prose
            r["inv_false_claim"] = fclaim
            per_row[i] = r
            diag["short_prose"] += 1
            continue

        # ── G5. 누출 가드. OSD 와 **같은 규약**: 걸리면 forward 를 아끼고 inv := 0. ──
        #   inv 는 «벌» 이므로 0 은 «벌 없음» 이다. 누출은 여기서 봐주고
        #   `answer_leak_rate` ABORT(>0.1) 가 따로 잡는다 — 한 사건을 두 번 벌하지 않는다.
        if final_exprs[i] and _cdr.answer_leak(meta_raw, final_exprs[i]):
            r = inv_empty_row("leak")
            r["inv_leak"] = 1
            r["inv_n_prose"] = n_prose
            r["inv_false_claim"] = fclaim
            per_row[i] = r
            diag["leak_blocked"] += 1
            diag["leak_reasons"]["answer_expr"] = diag["leak_reasons"].get("answer_expr", 0) + 1
            continue

        try:
            p_plain = str(prompt_texts[i] or "")
            p_hint = inv_hint_prompt(p_plain, wit, targets[i])
        except ValueError:
            per_row[i] = inv_empty_row("span_error:anchor")
            diag["anchor_error"] += 1
            diag["span_error"] += 1
            continue

        pre = list(span.ids[:span.open_start])
        if sc == "inplace":
            meta_ids = list(span.ids[span.open_start:span.close_end])
            prose_flags = list(prose)
        else:
            # reverse_ruler 판: 프로즈 문자 구간만 따로 인코딩해 채점한다.
            ps = prose_char_span(meta_raw)
            if ps is None:
                r = inv_empty_row("short_prose"); r["inv_false_claim"] = fclaim
                per_row[i] = r; diag["short_prose"] += 1
                continue
            meta_ids = list(tokenizer(meta_raw[ps[0]:ps[1]],
                                      add_special_tokens=False)["input_ids"])
            if len(meta_ids) < INV_MIN_PROSE_TOK:
                r = inv_empty_row("short_prose"); r["inv_false_claim"] = fclaim
                per_row[i] = r; diag["short_prose"] += 1
                continue
            prose_flags = [True] * len(meta_ids)
        ctx_plain = list(tokenizer(p_plain, add_special_tokens=False)["input_ids"]) + pre
        ctx_hint = list(tokenizer(p_hint, add_special_tokens=False)["input_ids"]) + pre
        arm_prompts.append(ctx_plain); arm_resps.append(list(meta_ids))
        arm_prompts.append(ctx_hint);  arm_resps.append(list(meta_ids))
        attempts.append(_InvAttempt(row=i, n_meta=len(meta_ids), prose=tuple(prose_flags)))
        per_row[i] = {"inv_raw": None, "inv_scored": 0, "inv_leak": 0,
                      "inv_n_prose": n_prose, "inv_status": "pending",
                      "inv_false_claim": fclaim}
        diag["prose_tok_sum"] += n_prose
        diag["fwd_tokens"] += len(ctx_plain) + len(ctx_hint) + 2 * len(meta_ids)

    diag["attempted"] = len(attempts)
    return arm_prompts, arm_resps, attempts, per_row, diag


def read_inv_from_ref_logprobs(ref_lp, attempts, agg: str = None,
                               form: str = None) -> list:
    r"""ref 토큰별 logp → 행별 inv. `base = 2*k`, 팔 순서 `plain, hint`.

    유한하지 않으면 그 행만 NaN 으로 fail-closed 한다 — `countdown_pmi.
    read_pmi_from_ref_logprobs` · `_read_osd_from_ref_logprobs` 와 같은 규약.
    """
    out = []
    for k, at in enumerate(attempts):
        base = 2 * k
        L = int(at.n_meta)
        try:
            plain = [float(v) for v in list(ref_lp[base + 0])[:L]]
            hint = [float(v) for v in list(ref_lp[base + 1])[:L]]
        except Exception:
            out.append(_NAN)
            continue
        if len(plain) != L or len(hint) != L:
            out.append(_NAN)
            continue
        fm = INV_FORM if form is None else str(form)
        if fm == "d2a":
            v = inv_aggregate([hint[t] - plain[t] for t in range(L) if at.prose[t]], agg)
        elif fm == "a2d":
            hp = [hint[t] for t in range(L) if at.prose[t]]
            pp = [plain[t] for t in range(L) if at.prose[t]]
            vh, vp = inv_aggregate(hp, agg), inv_aggregate(pp, agg)
            v = vh - vp if (math.isfinite(vh) and math.isfinite(vp)) else _NAN
        else:
            raise ValueError(f"read_inv_from_ref_logprobs: 모르는 form {fm!r}. 가능: {INV_FORMS}")
        out.append(float(v) if math.isfinite(v) else _NAN)
    return out


def score_inv(*, tokenizer, trainer, prompt_texts, response_texts, witnesses,
              targets, final_exprs, step: int = 0, agg: str = None,
              scope: str = None, form: str = None, _ref_scorer=None):
    r"""행별 `inv_raw` + 진단. **여기서만 GPU 를 쓴다**(ref forward 1회).

    `_ref_scorer` 는 테스트·오프라인 프로브 주입구다(주면 verl 을 안 부른다) —
    `score_pmi_shift` · `_compute_countdown_osd` 와 같은 규약.
    """
    from src.training import countdown_pmi as _cdp              # noqa: PLC0415

    arm_prompts, arm_resps, attempts, per_row, diag = build_inv_arms(
        tokenizer, prompt_texts, response_texts, witnesses, targets, final_exprs,
        scope=scope)
    diag.update({"scored": 0, "nan_rows": 0, "ref_error": None,
                 "fwd_calls": 0, "fwd_rows": 0, "fwd_rows_pad": 0,
                 "agg": INV_AGG if agg is None else str(agg),
                 "scope": INV_SCOPE if scope is None else str(scope),
                 "form": INV_FORM if form is None else str(form)})

    if not attempts:
        for r in per_row:
            if r["inv_status"] == "pending":
                r["inv_status"] = "unscored"
        diag.update(_inv_stats([]))
        return per_row, diag

    if _ref_scorer is None:
        from src.training.verl_sdc import (          # noqa: PLC0415  지연 import
            _build_pmi_score_batches, _dcpo_v4_ref_logprobs,
        )
        _cdp.assert_pmi_config(trainer)              # config 위반은 진입 즉시 깨진다
        tensors, real_n = _build_pmi_score_batches(
            arm_prompts, arm_resps, _cdp._pad_unit(trainer))
        if real_n != 2 * len(attempts):
            raise AssertionError(
                f"INV 팔 부기가 깨졌다: {real_n} != 2*{len(attempts)}")
        diag["fwd_calls"] = 1
        diag["fwd_rows"] = real_n
        diag["fwd_rows_pad"] = int(tensors["input_ids"].shape[0]) - real_n
        try:
            ref_lp = _dcpo_v4_ref_logprobs(trainer, tensors)
        except AssertionError:
            raise                                    # config 위반은 절대 삼키지 않는다
        except Exception as e:
            diag["ref_error"] = f"{type(e).__name__}: {e}"
            for at in attempts:
                per_row[at.row]["inv_status"] = "ref_error"
            print(f"[COUNTDOWN][INV][FAIL] step={step}: ref 스코어링 실패 "
                  f"({diag['ref_error']}) — 이 배치의 inv_raw 는 전부 None.", flush=True)
            diag.update(_inv_stats([]))
            return per_row, diag
    else:
        diag["fwd_calls"] = 1
        diag["fwd_rows"] = 2 * len(attempts)
        ref_lp = _ref_scorer(arm_prompts, arm_resps)

    vals = read_inv_from_ref_logprobs(ref_lp, attempts, agg, form)
    good = []
    for at, v in zip(attempts, vals):
        r = per_row[at.row]
        if math.isfinite(v):
            r["inv_raw"] = float(v)
            r["inv_scored"] = 1
            r["inv_status"] = "ok"
            good.append(float(v))
        else:
            r["inv_raw"] = _NAN                      # 조용한 0 이 아니다
            r["inv_scored"] = 0
            r["inv_status"] = "nan"
            diag["nan_rows"] += 1
    diag["scored"] = len(good)
    diag.update(_inv_stats(good))
    return per_row, diag


def _inv_stats(vals) -> dict:
    """inv 요약. **`i_pen_p90` 이 정규화 상수 c 를 정하는 값이다**(|inv−τ|_+ 의 p90)."""
    from src.training.countdown_rewards import _quantile         # noqa: PLC0415

    xs = [float(v) for v in vals if math.isfinite(float(v))]
    n = len(xs)
    if not n:
        return {"i_mean": _NAN, "i_std": _NAN, "i_p25": _NAN, "i_p50": _NAN,
                "i_p75": _NAN, "i_pen_rate": _NAN, "i_pen_p90": _NAN, "n_inv": 0}
    m = sum(xs) / n
    var = sum((v - m) ** 2 for v in xs) / n
    pen = [max(0.0, v - INV_TAU) for v in xs]
    return {
        "i_mean": m,
        "i_std": math.sqrt(var),
        "i_p25": _quantile(xs, 0.25),
        "i_p50": _quantile(xs, 0.50),
        "i_p75": _quantile(xs, 0.75),
        "i_pen_rate": sum(1.0 for v in pen if v > 0) / n,
        "i_pen_p90": _quantile(pen, 0.90),            # ★= 정규화 상수 c 후보
        "n_inv": n,
    }


def inv_signature() -> str:
    """이 자의 정체 서명 조각. `countdown_rewards.arm_signature` 가 붙인다."""
    q = "?" if INV_TAU_PROVISIONAL else ""
    return (f"scope={INV_SCOPE},form={INV_FORM},agg={INV_AGG},tau={INV_TAU:g}{q},c={INV_C:g}{q},"
            f"minprose={INV_MIN_PROSE_TOK:d},ngram={INV_LEAK_NGRAM:d}")


# 환경변수로 «측정만» 켜는 스위치(OSD 와 같은 규약). 팔이 항을 쓰면 이 값과 무관하게 켜진다.
def inv_measure_enabled() -> bool:
    return os.environ.get("COUNTDOWN_INV", "0") == "1"
