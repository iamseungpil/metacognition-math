r"""decision — 설계 C «발화 = 결정, 실행 = 리셋» 의 순수 함수 층 (2026-09-17).

정책이 **자기 한 패스 풀이를 끝낸 직후** 짧은 메타 발화 하나를 낸다.

  COMMIT   `<|meta|>Decision: my answer \boxed{X} is consistent; I commit to it.<|/meta|>`
  RESTART  `<|meta|>Decision: my answer \boxed{X} may be wrong; I restart from scratch,
            and the answer is not X.<|/meta|>`
  RESTART(답 없음)  시도-1 에 \boxed 답이 아예 없으면(X 가 비어 있으면) 위 문장은
            "...\boxed{} ... the answer is not ." 라는 말이 안 되는 문장이 된다 — 그 대신
            `<|meta|>Decision: I did not reach a usable answer; I restart from scratch.<|/meta|>`.
            commit 은 X 가 비면 애초에 낼 수 없다(무엇에 그대로 갈지가 없다).

RESTART 면 **하니스**가 문맥을 버리고 배제 재시도(F1 `fact_notx` = `trial2.attempt2_prompt
(mode="notx")`)를 굴리고 최종 답은 그 재시도의 것이다. 즉 발화는 «판단»만 하고 실행은
하니스가 한다 — 모델이 자기 문맥 안에서 되짚는 것이 아니라 **리셋**이다.

라벨의 출처는 gold 가 아니라 정책 자신의 K-표본 합의다:
`build_self_traces.pseudo_label`(시도-1 답 ∪ 재시도 답의 다수결)과 그 행의 자기 답이
같으면 commit, 다르면 restart. gold 는 `--audit` 보고에서만 읽는다.

★왜 이 층이 따로 있나: 발화 문자열의 **단일 진실 원천**이다. 빌더(학습 target)와
  평가 스크립트(파싱)가 각자 정규식을 쓰면 «학습한 문장을 평가가 못 읽는» 사고가 난다.
"""
from __future__ import annotations

import re

from src.metacot.prompt import META_END, META_START
from src.training.math_meta import answers_equivalent

#: 결정 두 가지.
KINDS = ("commit", "restart")

COMMIT_TMPL = "\n\n{open}Decision: my answer \\boxed{{{x}}} is consistent; I commit to it.{close}"
RESTART_TMPL = ("\n\n{open}Decision: my answer \\boxed{{{x}}} may be wrong; I restart from "
                "scratch, and the answer is not {x}.{close}")
#: 시도-1 에 \boxed 답이 아예 없을 때의 restart — «답이 없다»를 빈 \boxed{} 로 흉내내
#: 말이 안 되는 문장("...the answer is not .")을 만드는 대신 별도 문장을 쓴다.
RESTART_NOX_TMPL = ("\n\n{open}Decision: I did not reach a usable answer; I restart from "
                    "scratch.{close}")

#: 메타 블록 — 파싱은 **뒤에서 앞으로** 훑어 결정문이 든 첫 블록을 쓴다(D3).
_META_BLOCK_RE = re.compile(re.escape(META_START) + r"(.*?)" + re.escape(META_END), re.DOTALL)
#: 블록 안에서 «(Decision:) my/the (final) answer \boxed{» 까지. 공백은 느슨하게 받고
#: 「Decision:」 머리말은 **선택**이다 — 훈련된 모델이 머리말을 떼고 같은 문장을 쓴다.
_DECISION_HEAD_RE = re.compile(
    r"(?:Decision\s*:\s*)?(?:my|the)\s+(?:final\s+)?answer\s*,?\s*\\boxed\s*\{",
    re.IGNORECASE)

#: ★패러프레이즈 목록(D3) — `\boxed{X}` **바로 뒤 꼬리의 머리**에만 맞춘다(`^` 고정).
#: 보수적으로 유지한다: 한쪽 목록의 어떤 표현도 다른 쪽 문장의 머리에 맞을 수 없어야 한다
#: («is consistent» vs «is not consistent» 처럼 부정이 낀 꼴은 각 목록에 따로 적는다).
#: 목록에 없는 꼬리는 **None**(판단 불가)이며 절대 중립 추측으로 메우지 않는다.
_COMMIT_TAILS = (
    r"is\s+consistent",             # 학습 문장 본체
    r"looks\s+consistent",
    r"seems\s+consistent",
    r"is\s+correct",
    r"looks\s+correct",
    r"seems\s+correct",
    r"is\s+right",
    r"I\s*(?:'ll|\s+will)?\s+keep\s+it",
    r"I\s+commit\s+to\s+it",
)
_RESTART_TAILS = (
    r"may\s+be\s+wrong",            # 학습 문장 본체
    r"might\s+be\s+wrong",
    r"could\s+be\s+wrong",
    r"may\s+be\s+incorrect",
    r"might\s+be\s+incorrect",
    r"is\s+(?:probably\s+|likely\s+)?wrong",
    r"is\s+incorrect",
    r"is\s+not\s+correct",
    r"is\s+not\s+consistent",
    r"I\s*(?:'ll|\s+will)?\s+restart",
    r"I\s+should\s+restart",
)
#: 꼬리 머리의 구두점·공백은 버린다(«\boxed{5}; I'll keep it.» 꼴).
_TAIL_LEAD = r"^[\s,;:.\-—]*(?:"
_COMMIT_TAIL_RE = re.compile(_TAIL_LEAD + "|".join(_COMMIT_TAILS) + r")", re.IGNORECASE)
_RESTART_TAIL_RE = re.compile(_TAIL_LEAD + "|".join(_RESTART_TAILS) + r")", re.IGNORECASE)

#: 답 없는 restart 문장 — 위 두 목록과 달리 \boxed{} 를 아예 안 쓴다(패러프레이즈 포함).
_NOX_RESTART_RE = re.compile(
    r"I\s+(?:did\s+not|didn'?t|couldn'?t|could\s+not)\s+(?:reach|get|produce)\s+"
    r"(?:a\s+usable|an?|any)\s+(?:final\s+)?answer"
    r"|I\s+have\s+no\s+usable\s+answer", re.IGNORECASE)


def has_meta_block(text: str) -> bool:
    """텍스트에 메타 블록이 **하나라도** 있나 — D3 의 «발화는 했으나 결정이 파싱되지
    않은» 위험 버킷을 세려면 «블록 없음»과 «블록 있음 + 파싱 실패»를 갈라야 한다."""
    return bool(_META_BLOCK_RE.search(text or ""))


def utterance(kind: str, x: str) -> str:
    r"""결정 발화 한 줄. `kind` ∈ {commit, restart}, `x` 는 **그 행의 자기 답**.

    `x` 가 비어 있으면(시도-1 에 \boxed 답이 없음) restart 는 `RESTART_NOX_TMPL` 로
    빠진다. commit 은 빈 `x` 를 **거부**한다 — 아무것도 아닌 것에 그대로 갈 수는 없고,
    `decision_label` 이 이미 빈 답을 항상 restart 로 매기므로 빌더는 이 경로를 절대
    타지 않아야 한다(호출 전 assert 로 지킨다)."""
    if kind not in KINDS:
        raise ValueError(f"[DECISION] 모르는 결정 {kind!r} — {'|'.join(KINDS)} 중 하나여야 한다.")
    xs = str(x or "").strip()
    if kind == "commit":
        if not xs:
            raise ValueError("[DECISION] commit 은 빈 답에 낼 수 없다 — 답 없이 그대로 갈 수 없다.")
        return COMMIT_TMPL.format(open=META_START, close=META_END, x=xs)
    if not xs:
        return RESTART_NOX_TMPL.format(open=META_START, close=META_END)
    return RESTART_TMPL.format(open=META_START, close=META_END, x=xs)


def _balanced_brace(s: str, start: int) -> tuple[str, int]:
    """`s[start]` 가 `{` 바로 **다음** 위치일 때 짝이 맞는 `}` 까지의 내용과 그 다음 인덱스.
    짝이 안 맞으면 ("", -1). `\boxed{\frac{1}{2}}` 같은 중첩을 위해 정규식 대신 쓴다."""
    depth = 1
    buf: list[str] = []
    i = start
    while i < len(s):
        c = s[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(buf), i + 1
        buf.append(c)
        i += 1
    return "", -1


def _parse_block(body: str) -> dict | None:
    """블록 하나 → 결정 dict, 결정문이 없으면 None(«이 블록은 결정이 아니다»)."""
    m = _DECISION_HEAD_RE.search(body)
    if not m:
        return {"decision": "restart", "x": ""} if _NOX_RESTART_RE.search(body) else None
    x, end = _balanced_brace(body, m.end())
    if end < 0:
        return None
    tail = body[end:]
    # restart 를 먼저 본다 — 부정형(«is not correct»)이 commit 꼬리로 새지 않게.
    if _RESTART_TAIL_RE.match(tail):
        return {"decision": "restart", "x": x.strip()}
    if _COMMIT_TAIL_RE.match(tail):
        return {"decision": "commit", "x": x.strip()}
    return None


def parse_decision(text: str) -> dict:
    r"""생성 텍스트 → {"decision": "commit"|"restart"|None, "x": str|None}.

    **뒤에서 앞으로** 메타 블록을 훑어 결정문이 든 **첫** 블록을 쓴다(D3) — 결정 발화 뒤에
    메타 블록이 하나 더 붙으면 예전 구현은 마지막 블록만 보고 None 을 돌려줬고, 그 행은
    `restart_score` 0.5(중립)로 AUC 에 동점으로 들어가고 `utter_gated` 는 조용히
    `never` 로 퇴화했다. 블록이 없거나 어느 블록에도 결정문이 없으면 둘 다 None —
    훈련 전 모델의 «발화 없음»이 그 경우이며, **판단 불가일 때는 그대로 None 을 돌려준다**
    (호출부가 버킷 크기를 보고해 퇴화를 눈에 보이게 한다)."""
    none = {"decision": None, "x": None}
    for body in reversed(_META_BLOCK_RE.findall(text or "")):
        got = _parse_block(body)
        if got is not None:
            return got
    return none


def decision_label(a1_answer: str, pseudo_label: str) -> str:
    """정답 라벨(gold 아님) — 자기 답이 의사 라벨과 **수학적으로 같으면** commit, 아니면
    restart. 자기 답이 비어 있으면(절단·`\\boxed` 없음) 무조건 restart —
    답이 없는 행은 «그대로 간다»가 될 수 없다."""
    a1 = str(a1_answer or "").strip()
    pl = str(pseudo_label or "").strip()
    if not a1:
        return "restart"
    return "commit" if (pl and answers_equivalent(a1, pl)) else "restart"
