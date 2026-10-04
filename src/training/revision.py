r"""자발적 답 수정(revision) 구간의 단일 진실 원천 — 순수 함수만(torch 없음).

정의: 한 롤아웃 안에서 첫 `\boxed{X}` 를 쓴 뒤 스스로 재검토 문장을 쓰고 마지막
`\boxed{Y}` 를 다르게 쓴 행이 «수정한 행»이다. 크레딧이 가는 곳은 **수정 구간**
(첫 박스가 끝난 지점 → 마지막 `\boxed` 가 **시작**하는 지점) 뿐이다 — 마지막 답
자체는 구간에 넣지 않는다(답을 쓰는 토큰이 아니라 «다시 따져본» 토큰을 키운다).

학습기(`verl_sdc._compute_revision_rmeta`)와 평가기(`scripts/local/math_revision_eval.py`)
가 둘 다 여기를 쓴다 — «무엇이 수정인가»를 두 곳에 베끼지 않는다.
"""
from __future__ import annotations

__all__ = ["revision_zone", "revision_cf_credit", "combo_save"]


def _loose_eq(a, b) -> bool:
    """`answers_equivalent_loose` 를 지연 import(없으면 `answers_equivalent` 폴백)."""
    from src.training import math_meta as _mm  # noqa: PLC0415

    fn = getattr(_mm, "answers_equivalent_loose", None) or _mm.answers_equivalent
    try:
        return bool(fn(a, b))
    except Exception:
        return str(a or "").strip() == str(b or "").strip()


def revision_zone(text: str) -> dict | None:
    r"""텍스트의 수정 구간. 박스가 2개 미만이면 None.

    반환 키: first_answer, last_answer, zone_start(첫 박스 끝 char 오프셋),
    zone_end(마지막 `\boxed` 가 **시작**하는 char 오프셋), n_boxes,
    n_change_points(이웃 박스 답이 loose-동치가 아닌 횟수),
    revised(= 첫 답과 마지막 답이 loose-동치가 **아니다**).
    """
    from src.training.math_meta import boxed_spans  # noqa: PLC0415

    spans = boxed_spans(text or "")
    if len(spans) < 2:
        return None
    answers = [s[0] for s in spans]
    n_change = sum(
        0 if _loose_eq(answers[i], answers[i + 1]) else 1
        for i in range(len(answers) - 1)
    )
    zone_start = int(spans[0][2])    # 첫 박스의 닫는 `}` 바로 뒤
    zone_end = int(spans[-1][1])     # 마지막 `\boxed` 의 `\` 위치
    if zone_end < zone_start:
        zone_end = zone_start
    return {
        "first_answer": answers[0],
        "last_answer": answers[-1],
        "zone_start": zone_start,
        "zone_end": zone_end,
        "n_boxes": len(spans),
        "n_change_points": n_change,
        "revised": not _loose_eq(answers[0], answers[-1]),
    }


def revision_cf_credit(first_correct, last_correct, save: float, derail: float) -> float:
    """결과-개선 크레딧(대조 팔). A = 1[Y≡gold] − 1[X≡gold] 를 비대칭으로 사상한다.

    오답→정답이면 +|save|, 정답→오답이면 −|derail|, 그 밖(둘 다 같음)은 0.0.
    """
    a = int(bool(last_correct)) - int(bool(first_correct))
    if a > 0:
        return abs(float(save))
    if a < 0:
        return -abs(float(derail))
    return 0.0


def combo_save(save: float, majority_correct) -> float:
    """`combo` 앵커의 save 크기 — 그룹 다수답이 **틀렸을 때** 구제 보너스를 두 배로."""
    s = abs(float(save))
    return s if bool(majority_correct) else 2.0 * s
