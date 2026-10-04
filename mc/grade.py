r"""채점 — `\boxed` 추출 · math_verify 채점 · 답 동치. mc/ 의 단일 진실 원천.

`src/training/math_meta.py` 의 채점부를 그대로 옮겼다(0916 수리 포함): 맨 gold 에서
`parse` 가 빈 결과를 내 정답을 0 으로 만들던 거짓음성(566/6400)을 gold 래핑 + pred 재래핑 +
표기 정규화 세 단계로 막는다(의미 휴리스틱은 넣지 않는다).
`tests/mc/test_grade_parity.py` 가 실측 2,000행에서 구 채점기와 100% 일치를 고정한다.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

# ── \boxed 스캐너 (하나만 둔다 — \boxed 와 { 사이 공백 허용) ──────────────────
_BOXED_RE = re.compile(r"\\boxed\s*\{")


def _boxed_span_at(t: str, m: re.Match) -> tuple[str, int, int] | None:
    r"""_BOXED_RE 매치 m 에서 (내용, 시작, 닫힘 뒤 오프셋). 안 닫히면 None."""
    j, depth = m.end(), 1
    while j < len(t) and depth:
        depth += (t[j] == "{") - (t[j] == "}")
        j += 1
    return (t[m.end(): j - 1].strip(), m.start(), j) if depth == 0 else None


def boxed_spans(text: str, *, exclude: Sequence[tuple[int, int]] = ()) -> list[tuple[str, int, int]]:
    r"""모든 균형 `\boxed{...}` 의 (내용, 시작, 끝). `exclude` 구간에서 시작하는 박스는 뺀다."""
    t = text or ""
    out = []
    for m in _BOXED_RE.finditer(t):
        if any(a <= m.start() < b for a, b in exclude):
            continue
        got = _boxed_span_at(t, m)
        if got:
            out.append(got)
    return out


def boxed_answer(text: str) -> str | None:
    r"""**마지막** `\boxed{...}` 안의 문자열(최종 답 규약). 없으면 None."""
    sp = boxed_spans(text)
    return sp[-1][0] if sp else None


def first_boxed(text: str) -> str | None:
    r"""**첫** `\boxed{...}` 안의 문자열. 없으면 None."""
    sp = boxed_spans(text)
    return sp[0][0] if sp else None


# ── 표기 정규화 ──────────────────────────────────────────────────────────────
_ANS_PREFIX_RE = re.compile(r"^(?:[a-zA-Z]|\\[a-zA-Z]+)\s*=\s*")
_TEXT_CMD_RE = re.compile(r"\\(?:text|mbox|textbf|textit|mathrm|mathbf)\s*\{([^{}]*)\}")
_FRAC_NOBRACE_RE = re.compile(r"\\frac\s*([0-9a-zA-Z])\s*([0-9a-zA-Z])")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(?:,\d{3})+(?:\.\d+)?$")
_BARE_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
_DEG_RE = re.compile(r"(?:\^\s*\{?\s*\\circ\s*\}?|\\degree|\s*degrees?)\s*$")


def _norm_answer(s: str, *, gold_is_bare_num: bool = False, gold_has_prefix: bool = False) -> str:
    r"""채점용 문자열 정규화(의미 추론 없음 — 표기 차이만 지운다)."""
    t = str(s or "").strip()
    for _ in range(3):                      # 바깥 $…$ / \(…\) / \[…\]
        t = t.strip()
        if len(t) >= 2 and t[0] == "$" and t[-1] == "$":
            t = t[1:-1]
        elif t.startswith("\\(") and t.endswith("\\)"):
            t = t[2:-2]
        elif t.startswith("\\[") and t.endswith("\\]"):
            t = t[2:-2]
        else:
            break
    t = t.strip()
    for _ in range(3):                      # 바깥 \boxed{...}
        m = _BOXED_RE.match(t)
        if not m:
            break
        got = _boxed_span_at(t, m)
        if not got or got[2] != len(t):
            break
        t = got[0].strip()
    t = t.replace("\\left", "").replace("\\right", "")
    t = t.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\:", "")
    t = t.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    for _ in range(3):
        nt = _TEXT_CMD_RE.sub(r"\1", t)
        if nt == t:
            break
        t = nt
    t = _FRAC_NOBRACE_RE.sub(r"\\frac{\1}{\2}", t)
    t = t.strip()
    while t.endswith("."):
        t = t[:-1].strip()
    t = t.strip()
    if not gold_has_prefix:                 # "x=" 류 접두 — gold 에 없을 때만 벗긴다
        t = _ANS_PREFIX_RE.sub("", t, count=1).strip()
    if gold_is_bare_num:                    # 도 기호 / 퍼센트 — gold 가 맨숫자일 때만
        t = _DEG_RE.sub("", t).strip()
        while t.endswith("%") or t.endswith("\\%"):
            t = t[:-2].strip() if t.endswith("\\%") else t[:-1].strip()
    t = t.replace("\\$", "").strip()
    if t.startswith("$"):
        t = t[1:].strip()
    if _THOUSANDS_RE.match(t):
        t = t.replace(",", "")
    return re.sub(r"\s+", "", t)


# ── 객관식 선택지 표식(«(E)» 류) 벗기기 ──────────────────────────────────────
# `\boxed{\text{(E)}\ \dfrac{7}{2}}` 처럼 문항 기호가 붙은 답이 gold 와 갈려 0 으로 찍히던
# 거짓음성을 막는다. 표식을 벗긴 나머지는 **기존 경로가 전부 실패한 뒤에만** 시도하므로
# 기존 판정은 뒤집히지 않는다(오답→정답 한 방향).
_CHOICE_TEXT_RE = re.compile(r"^\\(?:text|textbf|textit|mathrm|mathbf|mbox)\s*\{([^{}]*)\}")
_CHOICE_INNER_RE = re.compile(r"^\(\s*([A-Ea-e])\s*\)\s*[:.,]?$|^([A-Ea-e])\s*[).]\s*[:,]?$")
_CHOICE_BARE_RE = re.compile(r"^\(\s*([A-Ea-e])\s*\)|^([A-Ea-e])\s*\)")
_CHOICE_SEP_RE = re.compile(r"^(?:\\[,;:!\s]|\\quad|\\qquad|~|\s|[:,.])+")


def strip_choice_marker(s: str):
    r"""답 문자열 맨 앞의 객관식 표식을 (글자, 나머지)로 가른다. 표식이 없으면 None.
    ★맨 글자(`C`)는 표식으로 보지 않는다 — 괄호나 `)` 가 있어야 한다. 숫자 gold 에 대한
    `\boxed{C}` 는 지금도 오답이어야 하기 때문이다(자가검사 케이스)."""
    t = str(s or "").strip()
    m = _CHOICE_TEXT_RE.match(t)
    if m:
        mi = _CHOICE_INNER_RE.match(m.group(1).strip())
        if not mi:
            return None
        letter = mi.group(1) or mi.group(2)
    else:
        m = _CHOICE_BARE_RE.match(t)
        if not m:
            return None
        letter = m.group(1) or m.group(2)
    return letter.upper(), _CHOICE_SEP_RE.sub("", t[m.end():]).strip()


# ── 객관식 선택지 대응(0924) — 학습 풀 r→w «망침» 82 중 66 이 값 → 글자 다시 쓰기였다 ──────────
_OPT_RE = re.compile(r"(?<!\w)\(\s*([A-E])\s*\)")
_OPT_TRIM_RE = re.compile(r"^(?:[\s$}:.~]|\\[ ,;!])+|(?:[\s${,;~]|\\[ ,;!]|\\\\|\\q?quad|\\hspace\{[^{}]*\}"
                          r"|\\(?:textbf|textrm|text|mathrm|mathbf|mbox))+$")


def parse_options(problem: str) -> dict[str, str] | None:
    r"""문제 본문의 선택지 `(A) … (E)` → {글자: 값}. `\textbf{(A) }\frac…\qquad` · `(A) 29 (B) 39` 모두.
    A 부터 순서대로 ≥4 글자가 이어지는 **마지막** 줄만 본다(본문의 `(A)` 류 오인 방지). 없으면 None."""
    t = str(problem or "").replace("\t", "\\t")          # `\textbf` 의 `\t` 가 탭으로 깨진 행(실측)
    ms, best = list(_OPT_RE.finditer(t)), []
    for k, m in enumerate(ms):
        run = [m] if m.group(1) == "A" else []
        for x in ms[k + 1:] if run else ():
            if len(run) < 5 and x.group(1) == "ABCDE"[len(run)]:
                run.append(x)
        best = run if len(run) >= max(4, len(best)) else best
    if not best:
        return None
    ends = [m.start() for m in best[1:]] + [len(t)]
    out = {}
    for m, e in zip(best, ends):
        v = t[m.end():e]
        v = re.split(r"\n|\$\s*$|\$\s*\n", v)[0] if e == len(t) else v   # 마지막 선택지: 줄·수식 끝까지
        for _ in range(4):
            v = _OPT_TRIM_RE.sub("", v)
        out[m.group(1)] = v.strip()
    return out if all(out.values()) else None


def choice_value(ans: str, options: dict | None) -> str:
    r"""답이 선택지 **글자**(`\text{B}` · `(B)` · `\textbf{(B)}\ 값`)면 그 선택지 값, 아니면 그대로."""
    a = str(ans or "").strip()
    if not options or not a:
        return a
    got = strip_choice_marker(a)
    letter = got[0] if got else _norm_answer(a).strip("()")
    return options.get(letter, a) if len(letter) == 1 else a


# ── 채점 ─────────────────────────────────────────────────────────────────────
def grade_math(pred_text: str, gold: str) -> bool:
    r"""pred 텍스트의 최종 답이 gold 와 같은가. 예외는 False.

    (a) math_verify → (b)(c) gold 래핑 → (d) pred parse 가 비면 박스 내용물 재래핑 →
    (e) 표기 정규화 문자열 동일. ★괄호 없는 쉼표 gold(`1,2`)는 math_verify 가 **집합**으로
    읽어 순서를 잃으므로 pred 가 괄호 쌍이면 래핑 경로를 끈다.
    """
    from math_verify import parse, verify  # noqa: PLC0415

    g = str(gold)
    pred = str(pred_text or "")

    def _p(x: str):
        try:
            return parse(x)
        except Exception:
            return []

    def _v(gp, pp) -> bool:
        if not gp or not pp:
            return False
        try:
            return bool(verify(gp, pp))
        except Exception:
            return False

    pp = _p(pred)
    if _v(_p(g), pp):
        return True
    boxed = boxed_answer(pred) or ""
    bare_list_gold = ("," in g) and not any(ch in g for ch in "()[]{}")
    if bare_list_gold and boxed and boxed.strip()[:1] in "([":
        gold_wrapped = []
    else:
        gold_wrapped = [_p(f"${g}$"), _p("\\boxed{" + g + "}")]
    for gp in gold_wrapped:
        if _v(gp, pp):
            return True
    if not pp and boxed:
        for pw in (_p(f"${boxed}$"), _p("\\boxed{" + boxed + "}")):
            for gp in (_p(g), *gold_wrapped):
                if _v(gp, pw):
                    return True
    if boxed:
        gold_is_num = bool(_BARE_NUM_RE.match(_norm_answer(g)))
        gold_has_prefix = bool(_ANS_PREFIX_RE.match(str(g).strip()))
        ng = _norm_answer(g, gold_is_bare_num=gold_is_num, gold_has_prefix=gold_has_prefix)
        np_ = _norm_answer(boxed, gold_is_bare_num=gold_is_num, gold_has_prefix=gold_has_prefix)
        if ng and ng == np_:
            return True
    # (f) 객관식 표식 — 표식을 벗긴 나머지(또는 글자 자체)로 한 번 더
    got = strip_choice_marker(boxed) if boxed else None
    if got:
        letter, rest = got
        for cand in ([rest] if rest else []) + [letter]:
            if grade_math("\\boxed{" + cand + "}", g):
                return True
    return False


def grade_answer(answer: str, gold: str, options: dict | None = None) -> bool:
    r"""**답 문자열** 하나를 채점(텍스트가 아니라 답만 있을 때) — `\boxed` 로 감싸 채점한다.
    `options`(`parse_options`)가 있으면 답·gold 의 글자를 선택지 값으로 바꾼 뒤(양방향) 채점한다."""
    if options:
        answer, gold = choice_value(answer, options), choice_value(gold, options)
    return grade_math("\\boxed{%s}" % (answer or ""), gold)


def answers_equivalent(a, b) -> bool:
    r"""«같은 답» 판정 = 문자열 동일 ∨ math_verify 동치(7 ≡ 7.0, 0.5 ≡ \frac{1}{2}).
    문자열만 보면 표기만 바꾼 재풀이가 «수정»으로 읽힌다. 판정기 예외는 문자열 비교로 폴백."""
    x, y = str(a or "").strip(), str(b or "").strip()
    if x == y:
        return True
    if not x or not y:
        return False
    try:
        return grade_answer(y, x)
    except Exception:
        return False


_LOOSE_WRAP_RE = re.compile(r"\\left|\\right|\\text\{[^{}]*\}|[\{\}]")
_LOOSE_SPLIT_RE = re.compile(r",|\\text\{\s*and\s*\}|\band\b")


def answers_equivalent_loose(a, b) -> bool:
    r"""표기만 다른 답 목록까지 같게 본다 — 래퍼를 벗기고 `,`/`and` 로 쪼갠 항 다중집합이
    쌍대응하면 True (`0 \text{ and } -3` ≡ `\{-3, 0\}`). 항 개수가 다르면 False.
    «수정했는가»(mc/eval.py)는 이 판정을 쓴다 — 구 `revision_zone` 과 같은 자다."""
    if answers_equivalent(a, b):
        return True
    sa, sb = str(a or "").strip(), str(b or "").strip()
    if not sa or not sb:
        return False
    ia = [p.strip() for p in _LOOSE_SPLIT_RE.split(_LOOSE_WRAP_RE.sub(" ", sa)) if p.strip()]
    ib = [p.strip() for p in _LOOSE_SPLIT_RE.split(_LOOSE_WRAP_RE.sub(" ", sb)) if p.strip()]
    if (len(ia) < 2 and len(ib) < 2) or len(ia) != len(ib):
        return False
    remaining = list(ib)
    for x in ia:
        for i, y in enumerate(remaining):
            if answers_equivalent(x, y):
                del remaining[i]
                break
        else:
            return False
    return True


def revision_zone(text: str) -> dict | None:
    r"""자발적 답 수정의 **구간** — 박스가 2개 미만이면 None.

    반환: first_answer · last_answer · zone_start(첫 박스 닫는 `}` 뒤) · zone_end(마지막
    `\boxed` 의 `\` 위치) · n_boxes · revised(첫 답과 마지막 답이 loose 동치가 **아니다**).
    ★«무엇이 수정인가»의 단일 진실 원천 — `mc/eval.py` 의 계측과 `mc/train_hook.py` 의 크레딧
    구간이 같은 자를 쓴다(구 `src/training/revision.revision_zone` 과 같은 정의).
    """
    spans = boxed_spans(text or "")
    if len(spans) < 2:
        return None
    first, last = spans[0], spans[-1]
    return {"first_answer": first[0], "last_answer": last[0],
            "zone_start": int(first[2]), "zone_end": max(int(last[1]), int(first[2])),
            "n_boxes": len(spans),
            "revised": not answers_equivalent_loose(first[0], last[0])}


def selftest() -> None:
    r"""math_verify timeout 래퍼가 워커 스레드에서 정답을 조용히 오답으로 만드는 함정
    (`scripts/patch_math_verify.py`)이 있다 — 깨져 있으면 즉사한다."""
    cases = [("\\boxed{42}", "42", True), ("\\boxed{\\frac{1}{2}}", "0.5", True),
             ("\\boxed{7}", "42", False), ("\\boxed{t^7}", "t^7", True),
             ("\\boxed{4x + 18}", "4x + 18", True), ("\\boxed{\\csc 10}", "\\csc 10", True),
             ("\\boxed{\\frac83}", "\\frac{8}{3}", True), ("\\boxed{\\$347}", "347", True),
             ("\\boxed{153}", "306", False), ("\\boxed{15}", "30^\\circ", False)]
    got = [grade_math(p, g) for p, g, _ in cases]
    want = [e for *_, e in cases]
    if got != want:
        raise RuntimeError(f"[MC] math_verify 자가검사 실패: got={got} want={want} — "
                           "scripts/patch_math_verify.py 를 먼저 적용하라.")
