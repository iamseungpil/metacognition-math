#!/usr/bin/env python
r"""verify_terms — «검산을 **말한 것**과 **수행한 것**을 텍스트에서 가르는» 항들.

왜: «검산하라»는 지시가 조건 수준에서 null 로 보여도, 그 null 이 «검산이 무용하다»를
  뜻하지 않을 수 있다 — 정책이 검산을 **말만 하고 수행하지 않았을** 수 있기 때문이다.
  이 모듈은 그 «수행했는가»를 **한 곳에서 기계적으로** 정의해, (a) 게이트가 외생화된
  템플릿 조건의 수행률을 재고 (b) 나중에 **보상 항**(수행한 검산에만 크레딧)으로 그대로
  재사용할 수 있게 한다. 그래서 순수 파이썬이다 — vLLM·torch·파일 IO 를 안 쓴다.

★PAL 경고(arXiv:2211.10435)가 이 모듈의 존재 이유다: 같은 모델이 프로그램을 «머릿속으로
  시뮬레이션»하면 23.2%, **실제로 실행**하면 72.0% 다. 검산도 같다 — «검산했다»는 문장은
  검산이 아니다. 여기의 탐지기는 전부 «**계산줄이 실제로 있는가** ∧ **그 결과에 대한 판정
  문장이 있는가**»를 요구한다. 어휘만 맞히는 것(math_meta_content_gate 의 `complied`)과
  구분되는 지점이 바로 그것이다.

★그래도 프록시다(상한): 우리는 계산이 **맞는지**는 안 본다 — «'=' 가 든 줄이 답 뒤에 3개 이상
  있고 두 결과를 같다/다르다고 말했는가»를 볼 뿐이다. 그러니 이 항은 «수행하지 않았다»는 쪽의
  증거로 강하고(계산줄이 없으면 수행이 아니다), «제대로 수행했다»는 쪽으로는 약하다.
  보상으로 쓸 때 이 비대칭을 잊으면 «'=' 를 많이 쓰면 상을 받는» 해킹이 열린다.

정의(스펙 고정 — 이 세 행위만 다룬다):
  · recompute — 후보 답 뒤에 '=' 든 줄 ≥3개 ∧ 두 방법/두 답을 대조하는 진술.
  · backward  — 역방향 어휘(work backwards / back-substitute / assuming my answer) ∧
                (matches|agrees|equals|recover) 가 (stated|given|original) 근처 ∧ 검산줄 ≥1.
  · magnitude — 첫 \boxed **앞**에 기대 진술(expect|should be|roughly|ballpark|between) ∧
                사후 대조(as expected|consistent with|matches the estimate|reasonable|
                makes sense|sanity).
  검산줄(check-line) = '=' 를 포함하면서 확인/부정 토큰(✓|holds|satisfies|checks out|
  is correct|consistent|matches|agrees|confirm|as expected · does not|not correct|wrong|
  fails|mismatch|✗)을 같이 가진 줄. 정규식은 전부 re.I.

판정(verdict)과 범주(category): 검산은 «같다/다르다»는 **판정**을 낳고, 그 판정 뒤에 답을
  **바꿨는가**가 갈린다. 네 칸이 서로 다른 이야기를 한다 —
    confirm_right(맞는 답을 맞다고 확인) · confirm_wrong(틀린 답을 맞다고 확인 = 검산의 실패)
    revise_right(고쳐서 맞음 = 검산의 값) · revise_wrong(고쳐서 틀림 = 검산의 해악).
  Δ 가 0 이어도 이 히스토그램이 revise_right ≈ revise_wrong 이면 «검산이 무용»이 아니라
  «검산이 방향 없이 흔든다»다. 그 둘을 가르려고 category 를 낸다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.math_meta import (  # noqa: E402  (박스 스캐너·동치 판정기는 하나뿐이다)
    answers_equivalent, boxed_spans, grade_math,
)

ACTS = ("recompute", "backward", "magnitude")
MIN_EQ_LINES_RECOMPUTE = 3   # 후보 답 뒤 '=' 줄 하한 — «두 번째 유도가 실제로 있었는가»
MIN_CHECK_LINES_BACKWARD = 1

# ── 정규식(전부 re.I) ─────────────────────────────────────────────────────────────
# 후보 답의 자리 = 첫 \boxed 또는 "the answer is"/"final answer" — 수행은 그 **뒤**를 본다.
_RX_ANSWER_PHRASE = re.compile(r"(the answer is|final answer)", re.I)
# 확인/부정 토큰 — 검산줄 판정용(둘 다 «판정했다»의 증거다; 부호는 verdict 가 따로 읽는다).
_RX_CONFIRM_TOK = re.compile(
    r"(✓|holds|satisfies|checks out|is correct|consistent|matches|agrees|confirm|as expected)",
    re.I)
_RX_NEGATE_TOK = re.compile(r"(does not|not correct|wrong|fails|mismatch|✗)", re.I)
# recompute 의 대조 진술 — «두 결과를 같다/다르다고 말했는가».
_RX_COMPARE_RECOMPUTE = re.compile(
    r"(both methods|same answer|agrees with|matches the (?:first|previous)|two answers|"
    r"different answer|do not match)", re.I)
# backward — 역방향 어휘 + 복원값이 원문 값과 맞는지 말한 진술.
_RX_BACKWARD_LEX = re.compile(
    r"(work backwards|back-substitut|assuming (?:my|the) answer)", re.I)
_RX_BACKWARD_CMP = re.compile(
    r"(matches|agrees|equals|recover)[^.\n]{0,60}(stated|given|original)", re.I)
# magnitude — 계산 **전** 기대 진술 / 계산 **후** 대조 진술.
_RX_EXPECT = re.compile(r"(expect|should be|roughly|ballpark|between)", re.I)
_RX_POSTHOC = re.compile(
    r"(as expected|consistent with|matches (?:the )?estimate|reasonable|makes sense|sanity)",
    re.I)
# verdict — «같다» 와 «다르다» 를 가르는 진술(둘 다 있으면 **먼저 나온 것**이 아니라 아래
#   verdict_of 규칙대로 '다르다' 를 우선한다. 이유는 그 함수의 docstring 참조).
_RX_SAME = re.compile(
    r"(the same|same answer|same result|both methods (?:give|agree)|agrees with|matches|"
    r"consistent with|as expected|checks out|holds)", re.I)
_RX_DIFFERENT = re.compile(
    r"(different answer|different result|do not match|does not match|disagree|mismatch|"
    r"they differ|not consistent|✗)", re.I)


def all_boxed(text: str) -> list[str]:
    r"""텍스트의 **모든** \boxed{...} 내용(중괄호 균형; 없으면 []).
    ★`math_meta.boxed_spans` 를 그대로 쓴다 — 박스 스캐너를 두 개 두면 "\boxed {7}" 같은
      공백형에서 두 스캐너의 판정이 갈린다(0914 검증 ②가 잡았던 함정이다)."""
    return [c for c, _, _ in boxed_spans(text or "")]


def first_candidate_pos(text: str) -> int | None:
    r"""«후보 답이 처음 제시된 자리» = 첫 \boxed 와 "the answer is"/"final answer" 중 **더
    이른 쪽**의 문자 오프셋. 둘 다 없으면 None.
    ★왜 자리가 필요한가: 수행 판정은 **답 뒤의 계산**만 센다. 푸는 과정의 '=' 줄을 같이 세면
      모든 생성이 «검산을 수행했다»가 되어 이 항이 아무것도 안 가른다."""
    t = text or ""
    cands: list[int] = [s for _, s, _ in boxed_spans(t)[:1]]
    m = _RX_ANSWER_PHRASE.search(t)
    if m:
        cands.append(m.start())
    return min(cands) if cands else None


def _lines(t: str) -> list[str]:
    return (t or "").splitlines()


def check_lines(text: str) -> list[str]:
    """검산줄 — '=' 를 **포함하면서** 확인 또는 부정 토큰을 같이 가진 줄.
    ★'=' 만으로는 «계산했다»이고, 토큰만으로는 «말했다»다. 둘이 **한 줄 안에** 있을 때만
      «그 계산의 결과를 판정했다»로 읽는다(PAL 경고의 조작적 정의)."""
    return [ln for ln in _lines(text)
            if "=" in ln and (_RX_CONFIRM_TOK.search(ln) or _RX_NEGATE_TOK.search(ln))]


def verdict_of(text: str) -> tuple[str | None, int | None]:
    """(판정, 그 자리) — "same"/"different"/None.
    ★«다르다»가 우선이다: 검산에서 불일치를 찾으면 그 뒤에 고친 답에 대해 다시 «맞다»고 쓰는
      것이 정상 서사라, 마지막/첫 매치를 쓰면 그 서사가 통째로 «같다»로 접힌다. 우리가 알고
      싶은 것은 «이 검산이 불일치를 **찾아냈는가**»이므로 different 를 먼저 본다."""
    d = _RX_DIFFERENT.search(text or "")
    if d:
        return "different", d.start()
    s = _RX_SAME.search(text or "")
    if s:
        return "same", s.start()
    return None, None


def performed_check(text: str, act: str) -> dict:
    """행위 `act` 의 검산을 **수행**했는가 → {performed, n_eq_lines_after_cand, verdict,
    verdict_pos}. 탐지 규칙은 모듈 docstring 의 정의를 그대로 코드로 옮긴 것이다.
    ★**자유서술 전용 — 템플릿 조건에 쓰지 말 것.** 0915 재분석: 이 어휘 탐지기는 구조화된
      템플릿 출력(«## Backward check» + «Recovered: … vs stated: …»)을 **못 읽는다** —
      backward 조건의 수행률이 .19 로 찍혔지만 템플릿 인식 탐지기로 다시 재면 .79 다.
      템플릿 조건은 `performed_template` 을 쓴다. 이 함수는 «지시 없이도 자연발생으로
      검산했는가»(기준선·대조군)를 재는 데만 남긴다.
    ★셋 다 «계산의 존재»(=줄) ∧ «결과에 대한 진술»(대조·판정)을 **동시에** 요구한다 —
      어느 한쪽만이면 performed=0 이다. 그게 «말한 검산»과 «수행한 검산»의 경계다."""
    if act not in ACTS:
        raise KeyError(f"[verify_terms] 모르는 act: {act!r} (아는 것: {ACTS})")
    t = text or ""
    pos = first_candidate_pos(t)
    tail = t[pos:] if pos is not None else ""
    n_eq = sum(1 for ln in _lines(tail) if "=" in ln)
    verdict, vpos = verdict_of(tail if pos is not None else t)
    if act == "recompute":
        performed = int(n_eq >= MIN_EQ_LINES_RECOMPUTE and bool(_RX_COMPARE_RECOMPUTE.search(t)))
    elif act == "backward":
        performed = int(bool(_RX_BACKWARD_LEX.search(t)) and bool(_RX_BACKWARD_CMP.search(t))
                        and len(check_lines(t)) >= MIN_CHECK_LINES_BACKWARD)
    else:  # magnitude — 기대 진술이 **첫 \boxed 앞**에 있어야 한다(사후 합리화를 빼려고).
        sp = boxed_spans(t)
        head = t[: sp[0][1]] if sp else t
        performed = int(bool(_RX_EXPECT.search(head)) and bool(_RX_POSTHOC.search(t)))
    if vpos is not None and pos is not None:
        vpos += pos
    return {"performed": performed, "n_eq_lines_after_cand": n_eq,
            "verdict": verdict, "verdict_pos": vpos}


def self_correction(text: str) -> dict:
    r"""답을 **바꿨는가** → {changed, n_boxed}. 첫 \boxed 와 마지막 \boxed 가 math_verify
    동치가 **아니면** changed=1.
    ★문자열 비교가 아니라 동치인 이유: 표기만 바꾼 재진술(7 ↔ 7.0, 0.5 ↔ \frac12)이 «고쳤다»로
      세어지면 revise_* 범주가 통째로 오염된다(math_meta.answers_equivalent 와 같은 규약)."""
    bx = all_boxed(text)
    if len(bx) < 2:
        return {"changed": 0, "n_boxed": len(bx)}
    return {"changed": int(not answers_equivalent(bx[0], bx[-1])), "n_boxed": len(bx)}


def verdict_grade(text: str, act: str, gold: str) -> dict:
    """수행·판정·정오를 한 줄로 합쳐 {performed, verdict, final_correct, category}.
    category ∈ {confirm_right, confirm_wrong, revise_right, revise_wrong, none} —
      confirm = 판정이 "same" 이거나 답을 안 바꿨다, revise = 답을 바꿨다.
    ★수행하지 않았으면 category="none" 이다(수행 안 한 생성의 «확인»은 확인이 아니다).
      이 히스토그램이 Δ 옆에 있어야 «검산이 무용»과 «검산이 방향 없이 흔든다»가 갈린다."""
    pc = performed_check(text, act)
    sc = self_correction(text)
    correct = grade_math(text or "", gold)
    if not pc["performed"]:
        cat = "none"
    elif sc["changed"]:
        cat = "revise_right" if correct else "revise_wrong"
    else:
        cat = "confirm_right" if correct else "confirm_wrong"
    return {"performed": pc["performed"], "verdict": pc["verdict"],
            "final_correct": int(correct), "changed": sc["changed"], "category": cat}


# ══ 템플릿 인식 수행 탐지 ═══════════════════════════════════════════════════════════
# 왜 두 번째 탐지기인가: 템플릿 조건은 수행을 **외생화**한다 — 섹션 이름과 «Recovered:
#   <value> vs stated: <value>» 같은 슬롯을 시킨다. 자유서술용 어휘 탐지기(`performed_check`)는
#   «work backwards» 같은 **서술 어휘**와 «'=' 와 확인 토큰이 한 줄에» 를 요구하는데, 템플릿을
#   그대로 따른 출력은 그런 문장을 안 쓰고 슬롯만 채운다 — 그래서 템플릿 조건은 이 탐지기가
#   따로 필요하다. 여기서는 **시킨 구조가 실제로 있는가**(sections: 본 섹션 ∧ 슬롯 채움 ∧
#   Compare 안의 판정)와, **그 구조 안에 실제 계산이 있는가**(n_eq_lines_main)를 나눠서 본다 —
#   전자만 보면 «형식은 맞췄지만 계산은 안 한» 출력에 크레딧을 준다(0915 감사).
# ★슬롯은 «채워졌는가»를 본다 — 템플릿 문자열을 그대로 베낀 '<value>' 는 채운 것이 아니다.
MIN_EQ_LINES_MAIN = {"recompute": 3, "backward": 1, "magnitude": 1}  # 본 섹션 안의 '=' 줄 하한
_TMPL_HEAD_RX = re.compile(
    r"^[ \t]*(?:#{1,4}|\*\*)\s*[^\n]{0,12}?"
    r"(Method\s*2|Backward\s*check|Expectation|Compare|Final|Recheck)",
    re.I | re.M)
_RX_SLOT_RECOMPUTE = re.compile(r"Method\s*2\s*result\s*:\s*(\S.*)", re.I)
_RX_SLOT_BACKWARD = re.compile(
    r"Recovered\s*:?\s*(.+?)\s*(?:vs|versus)\.?\s*stated\s*:?\s*(.+)", re.I)
# Compare 섹션 **안에서만** 읽는 판정. «다르다»를 먼저 본다(inconsistent ⊃ consistent,
#   disagree ⊃ agree, mismatch ⊃ match — 순서를 뒤집으면 전부 'same' 으로 접힌다).
_RX_TMPL_DIFF = re.compile(
    r"\b(different|differ|mismatch(?:es|ed)?|disagree\w*|inconsistent)\b", re.I)
_RX_TMPL_SAME = re.compile(r"\b(same|match(?:es|ed|ing)?|agree\w*|consistent)\b", re.I)
# 채워지지 않은 슬롯 — 템플릿의 자리표시자를 그대로 베낀 것.
_RX_PLACEHOLDER = re.compile(r"<\s*value\s*>|<\s*[a-z_ ]{0,20}\s*>", re.I)
_MAIN_HEAD = {"recompute": "method 2", "backward": "backward check",
              "magnitude": "expectation"}
_VALUE_STRIP = "$ \t✅✓✗❌.,;:!?*`'\"()[]"


def template_sections(text: str) -> dict:
    r"""머리글 이름(소문자) → 그 섹션의 본문. 같은 머리글이 여러 번이면 **마지막** 것을 쓴다
    (모델이 템플릿을 한 번 복창하고 다시 채우는 서사가 흔하다).
    ★머리글은 '#'~'####' 또는 '**' 로 시작하고, 이름 앞에 12자까지의 군더더기(이모지·번호·
      'Step ')를 허용한다 — 그보다 길면 머리글이 아니라 문장이다."""
    t = text or ""
    ms = list(_TMPL_HEAD_RX.finditer(t))
    out: dict[str, str] = {}
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(t)
        out[re.sub(r"\s+", " ", m.group(1)).strip().lower()] = t[m.end():end]
    return out


def _norm_value(v: str) -> str:
    r"""슬롯 값 정규화 — $ · \( \) · 끝의 ✅/✓/주석을 털어낸다.
    ★'120 ✅ (matches!)' 와 '$120$' 이 같은 값으로 읽혀야 `recovered_matches_stated` 가
      표기 차이로 거짓 음성을 내지 않는다."""
    s = (v or "").strip()
    s = re.split(r"\s+(?:—|–|--|//|#)\s*", s, maxsplit=1)[0]     # 뒤에 붙인 주석 컷
    s = re.sub(r"\((?![^()]*\))", "", s)                          # 짝 없는 여는 괄호
    s = s.replace(r"\(", "").replace(r"\)", "").replace(r"\[", "").replace(r"\]", "")
    s = s.replace("$", "").replace("\\,", "").replace("\\ ", " ")
    s = re.sub(r"[✅✓✗❌]", " ", s)
    s = re.sub(r"\s*\([^()]*\)\s*$", "", s)                       # 끝의 괄호 주석
    s = s.strip().strip(_VALUE_STRIP).strip()
    return re.sub(r"\s+", " ", s)


def recovered_matches_stated(recovered: str, stated: str) -> bool:
    """복원값과 원문값이 **동치**인가(문자열 동일이 아니라 math_verify 동치).
    ★검산이 «맞다»고 말한 것과 실제로 두 값이 같은 것은 다른 사건이다 — 판정(verdict)과
      이 불리언이 엇갈리는 행이 «검산이 자기 결과를 잘못 읽은» 사례다."""
    a, b = _norm_value(recovered), _norm_value(stated)
    if not a or not b:
        return False
    return bool(a == b or answers_equivalent(a, b))


def recovered_value_appears_in_problem(problem: str, value: str) -> bool:
    """복원값이 **문제 본문에 이미 적혀 있는 수**인가 → «무효 검산» 신호.
    ★0915 재분석의 핵심 의심: backward 의 이득 85%가 답을 **안 바꾼** 행에서 왔고, 그 행들의
      'Recovered:' 값은 대개 문제가 이미 준 수였다. 그렇다면 그 검산은 아무것도 복원하지
      않았고(자기가 보고 베낀 수를 자기와 대조), 이득의 정체는 «답 쓴 뒤 문제를 다시 읽은
      것»일 수 있다. 그래서 이 신호를 행마다 남겨 reread 대조와 함께 읽는다."""
    v = _norm_value(value)
    if not v:
        return False
    norm = lambda s: re.sub(r"[\s,$]", "", s or "")  # noqa: E731
    return norm(v) in norm(problem)


def _tmpl_verdict(compare_body: str) -> str | None:
    if _RX_TMPL_DIFF.search(compare_body or ""):
        return "different"
    if _RX_TMPL_SAME.search(compare_body or ""):
        return "same"
    return None


def performed_template(text: str, act: str) -> dict:
    r"""템플릿 조건의 수행 판정 → {performed, verdict, sections, n_eq_lines_main, slot_values, ...}.
      · sections = {main, slot, compare} (0/1) — **형식 준수**(시킨 구조가 실제로 있는가).
        요약에서 compliance(=sections 전부 1)와 performed 를 나란히 보여주려고 따로 낸다.
      · main   : act 의 본 섹션 머리글(Method 2 / Backward check / Expectation).
      · slot   : recompute 'Method 2 result: <채워진 값>', backward 'Recovered: A vs stated: B',
                 magnitude '## Expectation' 머리글이 **첫 \boxed 앞**에 있는 것.
      · compare: '## Compare' 섹션 **안에서만** 읽은 판정(밖의 'matches' 는 안 센다).
      · n_eq_lines_main: 본 섹션 본문에서 '=' 를 포함한 줄 수 — «형식만 맞추고 계산은
        안 했는가»를 가르는 잣대. 슬롯이 채워져도 이 수가 act 별 하한(MIN_EQ_LINES_MAIN)
        아래면 performed=0 이다(형식 준수만으로는 **수행**이 아니다).
      · performed = sections 전부 1 ∧ n_eq_lines_main ≥ 하한.
    ★backward 는 여기에 `recovered_matches_stated` 도 같이 낸다(값 동치 — 판정과 별개 사건).
    ★자유서술은 `performed_check` 다. 두 잣대를 한 표에 섞지 말 것 — 분모가 다르다."""
    if act not in ACTS:
        raise KeyError(f"[verify_terms] 모르는 act: {act!r} (아는 것: {ACTS})")
    t = text or ""
    secs = template_sections(t)
    main = int(_MAIN_HEAD[act] in secs)
    compare_body = secs.get("compare", "")
    verdict = _tmpl_verdict(compare_body)
    slot_values: dict[str, str] = {}
    if act == "recompute":
        m = _RX_SLOT_RECOMPUTE.search(t)
        val = _norm_value(m.group(1)) if m else ""
        filled = bool(m) and bool(val) and not _RX_PLACEHOLDER.search(m.group(1))
        if m:
            slot_values["method2_result"] = val
    elif act == "backward":
        m = _RX_SLOT_BACKWARD.search(t)
        rec = _norm_value(m.group(1)) if m else ""
        sta = _norm_value(m.group(2)) if m else ""
        filled = bool(m) and bool(rec) and bool(sta) and not (
            _RX_PLACEHOLDER.search(m.group(1)) or _RX_PLACEHOLDER.search(m.group(2)))
        if m:
            slot_values["recovered"], slot_values["stated"] = rec, sta
    else:  # magnitude — 기대 진술의 **자리**가 슬롯이다(첫 \boxed 앞이어야 사후 합리화가 아니다)
        sp = boxed_spans(t)
        head = t[: sp[0][1]] if sp else t
        hm = [m for m in _TMPL_HEAD_RX.finditer(head)
              if m.group(1).strip().lower() == "expectation"]
        filled = bool(hm) and bool(_norm_value(secs.get("expectation", "")))
        if secs.get("expectation"):
            slot_values["expectation"] = re.sub(r"\s+", " ", secs["expectation"]).strip()[:200]
    main_body = secs.get(_MAIN_HEAD[act], "") if main else ""
    n_eq_lines_main = sum(1 for ln in _lines(main_body) if "=" in ln)
    computed = n_eq_lines_main >= MIN_EQ_LINES_MAIN[act]
    sections = {"main": main, "slot": int(bool(filled)), "compare": int(verdict is not None)}
    out = {"performed": int(all(sections.values()) and computed),
           "verdict": verdict,
           "sections": sections,
           "n_eq_lines_main": n_eq_lines_main,
           "slot_values": slot_values}
    if act == "backward":
        out["recovered_matches_stated"] = bool(
            slot_values and recovered_matches_stated(slot_values.get("recovered", ""),
                                                     slot_values.get("stated", "")))
    return out
