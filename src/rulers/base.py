r"""자(ruler) 패키지 공통 자료형 — `Site`, `MetaSample`, `Ruler` 프로토콜.

이 패키지는 cd6 rulers 실험(`docs/POSTMORTEM_cd6_rulers_2026-09-03.md`,
`docs/VERDICT_cd6_pair_rulers.md`)에서 흩어져 있던 스코어러들을 한 곳으로 모으고,
새 자(move_kl, hidden_probe)를 더한 것이다. 기존 파일은 한 바이트도 고치지 않는다
(각 자 모듈의 docstring이 포트 출처를 `file:line`으로 인용한다).

`Site`/`MetaSample`은 `src/training/countdown_sites.py`(§2~3, `oracle_for_site`)와
`scripts/local/build_sites.py`가 만드는 parquet 스키마, 그리고 (아직 없는)
`scripts/local/gen_continuations.py`가 낼 것으로 명시된 continuations 스키마의
교집합만 담는다 — 자가 실제로 쓰는 필드만 있고, 그 이상은 `table.py`가 원본
DataFrame 행에서 직접 읽는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Sequence, runtime_checkable


@dataclass(frozen=True)
class Site:
    r"""근거-진리(ground truth)를 가진 하나의 프리픽스 절단점.

    필드는 `countdown_sites.oracle_for_site`(:287-320)와
    `countdown_sites.build_site_rows`(:374-406)가 내는 컬럼과 이름을 맞췄다:
        prompt_messages  = build_prompt() 가 만든 chat 메시지 리스트 (JSON 파싱된 것)
        prefix           = 절단된 응답 텍스트 (첫 <meta> 앞)
        nums / target    = Countdown 인스턴스
        witness          = 오라클이 찾은 해 첫 걸음 표현식 문자열 (예: "5+19")
        decoy            = `countdown_task.swap_op_decoy` 가 만든 연산자-교체 오답
        pairs_pre        = 프리픽스 안에서 이미 결합된 (a,b) 정렬쌍 집합
        family_dead      = 1 = 마지막 두 시도의 첫수 계열이 죽었다(§`family_dead_label`)
        live_new_moves   = "a op b" 정규형 문자열 리스트 — 아직 안 써본, 해가 사는 첫수
    """

    prompt_messages: list[dict]
    prefix: str
    nums: tuple[int, ...]
    target: int
    witness: str
    decoy: str
    pairs_pre: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    family_dead: int = 0
    live_new_moves: tuple[str, ...] = field(default_factory=tuple)
    site_id: Optional[str] = None
    # 채점표(table.py)가 성공률 통계에 쓰는 부가 필드. Site 자체의 정의에는 없지만
    # 여러 자·오라클이 "이 사이트 전체의 성공률"을 참조해야 해서 여기 얹는다.
    success_rate: Optional[float] = None


@dataclass(frozen=True)
class MetaSample:
    r"""한 site 에서 뽑은 (연속, 메타 블록) 표본 하나.

    필드는 (아직 없는) `scripts/local/gen_continuations.py` 산출 parquet 의
    행 스키마 — site_id/mode/policy_tag/k_index/continuation/full_text/r_corr/
    emitted/meta_raw/decision/confidence/novel/followed/checked/donor_meta_raw —
    가운데 자(ruler)가 직접 쓰는 부분집합이다. `table.py`가 나머지 컬럼을 원본
    DataFrame 행에서 읽어 오라클·시뮬레이션에 넘긴다.
    """

    continuation: str          # 절단점 이후 모델이 이어 쓴 텍스트 (메타 포함)
    meta_raw: str               # <meta>...</meta> 블록 원문 (없으면 "")
    meta_start: int              # continuation 안에서 메타 블록 시작 문자 오프셋 (-1=없음)
    meta_end: int                 # 메타 블록 끝 문자 오프셋 (배타, -1=없음)
    decision: Optional[str] = None      # "redirect" / "verify" / None
    confidence: Optional[float] = None  # 클램프 없음 — parse_meta 규약과 동일
    r_corr: Optional[int] = None        # 이 이어쓰기가 최종 정답으로 끝났는가 (0/1)
    next_move: Optional[str] = None      # meta 의 next: 필드 원문 (있으면)


@dataclass
class ScoreContext:
    r"""자가 모델 forward 를 필요로 할 때(needs_model=True) 공유하는 컨텍스트.

    `hf_ctx.HfCtx` 의 얇은 타입 별칭 — 자 모듈은 이 타입을 구조적으로만 쓴다(아래
    `Ruler.score` 시그니처가 요구하는 `ctx`). 실제 구현은 `hf_ctx.py`.
    """

    tokenizer: Any = None
    model: Any = None


@runtime_checkable
class Ruler(Protocol):
    r"""메타 블록 하나에 스칼라 점수를 매기는 자의 최소 계약.

    `needs_model=False` 인 자는 `ctx=None` 으로 호출돼도 죽지 않아야 한다
    (`ruler_table.py --no-model` 경로가 이걸 강제한다 — `table.py` 참조).
    """

    name: str
    needs_model: bool

    def score(self, site: Site, sample: MetaSample, ctx: Optional[ScoreContext]) -> float:
        ...


def pairs_pre_from_json(raw) -> frozenset:
    """`countdown_sites` 가 JSON 직렬화한 `pairs_pre`(`[[a,b],...]`) → frozenset."""
    import json
    if raw is None:
        return frozenset()
    if isinstance(raw, str):
        raw = json.loads(raw) if raw else []
    return frozenset(tuple(sorted((int(a), int(b)))) for a, b in raw)


def live_new_moves_from_json(raw) -> tuple[str, ...]:
    """`countdown_sites` 가 JSON 직렬화한 `live_new_moves`(문자열 리스트) → tuple."""
    import json
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = json.loads(raw) if raw else []
    return tuple(str(x) for x in raw)
