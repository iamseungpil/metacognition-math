r"""오라클 "좋은 메타" 점수 — 근거-진리(ground truth) 세 가지 체크의 합(0..3).

이것은 자(ruler)가 아니다(모델이 계산할 수 없다 — `site.family_dead`/
`live_new_moves`는 완전 탐색으로만 안다). `table.py`가 모든 자의 "무엇을 재려
했는가"를 검증하는 유일한 참값으로 쓴다 — cd6 rulers 실험에서 "계획 항"(plan)만
살아남았다는 관찰(`docs/POSTMORTEM_cd6_rulers_2026-09-03.md` §1.3)을 오라클의
두 번째 체크(plan)로 직접 반영했다.

세 체크:
  (i)   state    decision=='redirect' iff family_dead==1
                 (막혔으면 방향 전환을 선언해야 하고, 안 막혔으면 하지 말아야 한다)
  (ii)  plan     메타 뒤 첫 (a,b) 쌍(next: 필드가 있으면 그것, 없으면 메타 뒤 첫
                 시도)이 live_new_moves 안에 있다 — POSTMORTEM §1.3의 "계획 항"
                 관찰 근거(통과율 0.37, 미통과 대비 성공률 +0.13~+0.16)를 그대로
                 옮긴 정의.
  (iii) calib    |confidence − site_success_rate| < 0.25
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional

from src.rulers.base import MetaSample, Site

__all__ = ["OracleScore", "oracle_score"]

_MOVE_RE = re.compile(r"(\d+)\s*([+\-*/])\s*(\d+)")
_NEXT_RE = re.compile(r"\bnext\s*:\s*(.*)", re.I)


@dataclass(frozen=True)
class OracleScore:
    state_ok: int
    plan_ok: int
    calib_ok: int

    @property
    def total(self) -> int:
        return int(self.state_ok) + int(self.plan_ok) + int(self.calib_ok)


def _extract_pair(text: str) -> Optional[tuple[int, int]]:
    m = _MOVE_RE.search(text or "")
    if not m:
        return None
    a, _, b = m.groups()
    return int(a), int(b)


def _plan_pair(site: Site, sample: MetaSample) -> Optional[tuple[int, int]]:
    """next: 필드가 있으면 그 안의 (a,b), 없으면 메타 뒤 첫 시도의 (a,b)."""
    meta_body = sample.meta_raw or ""
    nm = _NEXT_RE.search(meta_body)
    if nm:
        p = _extract_pair(nm.group(1))
        if p:
            return p
    if sample.next_move:
        p = _extract_pair(sample.next_move)
        if p:
            return p
    cont = sample.continuation or ""
    me = sample.meta_end if sample.meta_end is not None and sample.meta_end >= 0 else 0
    return _extract_pair(cont[me:me + 128])


def _pair_in_live(pair: tuple[int, int], live_new_moves) -> bool:
    """live_new_moves는 "a op b" 정규형 문자열 리스트(`countdown_sites.oracle_for_site`
    §287-320). 연산자와 무관하게 두 수의 조합이 나열돼 있는지만 본다 — plan_ok가
    묻는 것은 "그 두 수를 결합하는 것이 해를 살리는가"이지 "정확히 그 연산자로도
    사는가"가 아니다(연산자까지 맞추는 것은 `plan_next`가 이미 별도로 재는 더 강한
    조건이고, 오라클은 여기서 완화된 버전을 쓴다 — POSTMORTEM의 plan_ok 정의가
    "그 첫수 뒤로 도달 가능한가"이지 "정확한 연산자를 골랐는가"가 아니었다)."""
    a, b = pair
    want = frozenset((a, b))
    for mv in live_new_moves:
        m = _MOVE_RE.search(str(mv))
        if not m:
            continue
        x, _, y = m.groups()
        if frozenset((int(x), int(y))) == want:
            return True
    return False


def oracle_score(site: Site, sample: MetaSample, site_success_rate: Optional[float] = None) -> OracleScore:
    r"""세 체크를 계산한다. 메타가 없으면 세 체크 전부 0(오라클은 "메타가 있고
    좋았다"를 재는 것이지 "메타가 있었다"를 재는 게 아니다 — 발화 자체의 가치는
    `dcont.py`/`baselines.meta_length`가 별도로 잰다)."""
    if not sample.meta_raw:
        return OracleScore(0, 0, 0)

    decision = (sample.decision or "").strip().lower()
    state_ok = int((decision == "redirect") == bool(int(site.family_dead)))

    pair = _plan_pair(site, sample)
    plan_ok = int(pair is not None and _pair_in_live(pair, site.live_new_moves))

    calib_ok = 0
    if sample.confidence is not None and site_success_rate is not None:
        try:
            calib_ok = int(abs(float(sample.confidence) - float(site_success_rate)) < 0.25)
        except (TypeError, ValueError):
            calib_ok = 0

    return OracleScore(state_ok=state_ok, plan_ok=plan_ok, calib_ok=calib_ok)
