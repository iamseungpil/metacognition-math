r"""move_kl 자 (신규) — 메타 전/후로 "다음 첫수" 분포가 얼마나 움직였는가.

기존 자(pmi_shift/osd/inv)는 전부 하나의 gold/decoy 쌍 또는 하나의 대리 후속(W)에
매달린다. `docs/VERDICT_cd6_pair_rulers.md`의 결론("정답 경로와 안 겹치는 층에서는
전부 무너진다")이 지적하는 근본 문제는 "정답 하나를 대변하는 문자열을 고르는 순간
그 문자열과의 유사도만 재게 된다"는 것이다. move_kl은 정답 하나가 아니라 **네 수로
만들 수 있는 모든 첫수 후보의 분포**를 보고, 메타가 그 분포를 어느 후보 쪽으로
움직였는지를 재서 "정답 흉내"에 덜 취약하게 만들려는 시도다(검증되지 않은 신규
자 — 기존 25개 전멸의 재판이 될 수도 있다. `table.py`의 공격 배터리·outcome-fixed
분해를 반드시 통과해야 채택 후보가 된다).

후보 열거: 서로 다른 4개 수에 대해 **24개**
    C(4,2)=6 개의 (인덱스) 쌍 × {+,-,*,/} 4개 연산자 = 24
    (수에 중복이 있으면 문자열이 겹쳐 dedupe로 줄어든다 — `enumerate_candidate_moves`
    docstring 참조). 쌍은 항상 `nums`에 나열된 순서로 "nums[i] op nums[j]"(i<j)만
    만든다 — "24"라는 목표 개수를 지키기 위해 등가인 역순(b op a)은 별도 후보로
    세지 않는다(-, / 도 마찬가지: 역순 후보를 원하면 이후 확장에서 별도 축으로 추가).

점수 산출: 각 후보 문자열을 "\n{candidate}"로 렌더링해 그 토큰열의 합산 로그확률을
컨텍스트(메타 시작 지점 / 메타 끝 지점) 조건부로 잰다(`pair_rulers.gather_logp`와
같은 teacher-forced 방식, `hf_ctx.HfCtx.token_logprobs` 사용). 24개 로그확률을
softmax해 분포로 만들고, KL(after ‖ before)을 낸다.
"""
from __future__ import annotations

import itertools
import re
import math

from src.rulers.base import MetaSample, Site

__all__ = ["enumerate_candidate_moves", "candidate_logits", "_candidate_logits_sequential",
           "candidate_logits_batched", "softmax", "kl_divergence", "MoveKl", "MoveKlSigned"]

_OPS = ("+", "-", "*", "/")
_NAN = float("nan")


def enumerate_candidate_moves(nums) -> list[str]:
    r"""4개 수에서 "a op b" 후보를 만든다. 서로 다른 4수면 6쌍×4연산=24개.

    중복 값이 있으면 문자열이 우연히 같아질 수 있어 dedupe로 24보다 적어진다
    (예: nums=[2,2,3,5] 이면 (2,3)이 두 인덱스 쌍에서 나오지만 문자열 "2+3"은 하나로
    합쳐진다 — 그 자체가 사양이다: "fewer with duplicates").
    """
    vals = list(nums)
    seen: dict[str, None] = {}
    for i, j in itertools.combinations(range(len(vals)), 2):
        a, b = vals[i], vals[j]
        for op in _OPS:
            s = f"{a}{op}{b}"
            seen.setdefault(s, None)
    return list(seen.keys())


def softmax(xs) -> list[float]:
    xs = [float(x) for x in xs]
    if not xs:
        return []
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    z = sum(ex)
    if z <= 0 or not math.isfinite(z):
        return [1.0 / len(xs)] * len(xs)
    return [e / z for e in ex]


def kl_divergence(p_after, p_before) -> float:
    """KL(after ‖ before), 0*log(0/x)=0 규약, before 성분이 0이면 그 항을 스킵(fail-soft)."""
    if len(p_after) != len(p_before) or not p_after:
        return _NAN
    total = 0.0
    for a, b in zip(p_after, p_before):
        if a <= 0.0:
            continue
        if b <= 0.0:
            return float("inf")
        total += a * math.log(a / b)
    return float(total)


def _candidate_logits_sequential(ctx, prompt_ids, prefix_ids, candidates: list[str]) -> list[float]:
    """원래 구현(0904 이전) — 후보마다 forward 한 번, 보통 24회 순차 호출.

    ★0905 성능 수리에서 이 함수를 `HfCtx.batched_logprobs`(오른쪽 패딩 배치)로
    바꿔봤으나, 실제 `models/Qwen3-4B` bf16 GPU forward로 200행을 재현 채점해
    비교한 결과 **바이트 동일이 아니었다**(개별 후보 로그확률 합에서 최대 ~1.16,
    `move_kl` 최종 점수에서 최대 0.227 차이 — `move_novel_shift`의 AUC까지
    0.708→0.625로 갈릴 만큼 사소하지 않았다). `_lp_cache` 캐시 코드나 왼쪽-절단은
    무관하다는 것도 같은 실측에서 확인했다(pmi_shift/osd/inv/dcont — 전부 이
    캐시·절단 경로를 타지만 diff=0). 즉 오른쪽 패딩 자체가 아니라 배치 크기가
    바뀌면 이 모델의 bf16 어텐션/행렬곱 수치가 달라진다(원인 미확정 — RoPE 동적
    스케일링이 배치 내 최대 길이에 반응하는지, 커널 선택이 배치 크기로 바뀌는지
    등은 특정하지 않았다). §목표(c)="현존 자와 바이트 동일"을 배치 속도보다
    우선해 이 함수를 정본으로 되돌린다 — `candidate_logits`가 다시 이 함수의
    별칭이다. 배치판은 `candidate_logits_batched`로 남겨 두되(합성 스텁에서는
    이 함수와 일치함을 `test_rulers.py`가 여전히 확인한다) **`_dists_for`는
    부르지 않는다** — 실제 모델에서 검증되기 전에는 실사용 금지.
    """
    out = []
    ctx_ids = prompt_ids + prefix_ids
    for cand in candidates:
        ids = list(ctx.encode("\n" + cand))
        if not ids:
            out.append(_NAN)
            continue
        lp = ctx.token_logprobs(ctx_ids, ids)
        out.append(float(lp.sum()) if lp.size else _NAN)
    return out


# 정본 진입점 — `_dists_for`가 이 이름을 부른다. 위 docstring 사유로 순차 구현
# 그대로다(배치 아님).
candidate_logits = _candidate_logits_sequential


def candidate_logits_batched(ctx, prompt_ids, prefix_ids, candidates: list[str],
                             batch: int = 8) -> list[float]:
    r"""실험적 배치 경로 — **실사용 금지**(위 `_candidate_logits_sequential`
    docstring 참조: 실제 bf16 모델에서 순차 경로와 다른 숫자를 냈다). 순수
    CPU 스텁(결정적 로짓)에서는 순차 경로와 일치함을 `test_rulers.py`가 확인
    하지만, 그게 실제 GPU/bf16 모델에서의 일치를 보장하지 않는다는 것이 이번에
    실측으로 드러났다 — 이후 KV-cache 프리픽스 재사용(진짜로 같은 forward를
    재사용하는 방식, 배치 축을 늘리지 않는 방식) 등 더 안전한 대안이 나오기
    전까지는 참고용으로만 남겨 둔다."""
    ctx_ids = prompt_ids + prefix_ids
    cand_ids = [list(ctx.encode("\n" + c)) for c in candidates]
    valid = [i for i, ids in enumerate(cand_ids) if ids]
    out = [_NAN] * len(candidates)
    if not valid:
        return out
    contexts = [ctx_ids for _ in valid]
    targets = [cand_ids[i] for i in valid]
    lps = ctx.batched_logprobs(contexts, targets, batch=batch)
    for i, lp in zip(valid, lps):
        out[i] = float(lp.sum()) if lp.size else _NAN
    return out


def _prompt_ids(ctx, site: Site):
    text = ""
    if hasattr(ctx.tokenizer, "apply_chat_template"):
        text = ctx.tokenizer.apply_chat_template(
            site.prompt_messages, tokenize=False, add_generation_prompt=True)
    return list(ctx.encode(text)) if text else []


def _dists_for(site: Site, sample: MetaSample, ctx):
    """(candidates, p_before, p_after, ms) — 메타 시작 직전/끝 직후의 «다음 수» 분포. 못 재면 None.

    0905 성능 수리: `MoveKl`/`MoveKlSigned`/`MoveNovelShift`/`MoveNovelShiftStuck`
    네 자가 전부 이 함수를 호출하는데, 같은 (site, sample)이면 입력이 완전히
    같다(집계만 다르다) — `ctx`에 매단 소형 캐시로 4배 중복 forward를 없앤다.
    키는 (site_id, meta_start, meta_end, continuation) — 공격 배터리가 만드는
    가짜 continuation도 서로 다른 문자열이라 자연히 캐시 미스로 갈린다(정답은
    그대로 재계산된다, 근사 아님).
    """
    if ctx is None or not sample.meta_raw:
        return None
    cont = sample.continuation or ""
    ms = sample.meta_start if sample.meta_start is not None and sample.meta_start >= 0 else len(cont)
    me = sample.meta_end if sample.meta_end is not None and sample.meta_end >= 0 else ms

    cache = getattr(ctx, "_move_kl_dist_cache", None)
    if cache is None:
        cache = {}
        try:
            ctx._move_kl_dist_cache = cache
        except Exception:
            cache = None
    key = (site.site_id, ms, me, cont) if cache is not None else None
    if key is not None and key in cache:
        return cache[key]

    candidates = enumerate_candidate_moves(site.nums)
    if len(candidates) < 2:
        result = None
    else:
        p_ids = _prompt_ids(ctx, site)
        before_ids = list(ctx.encode(site.prefix + cont[:ms]))
        after_ids = list(ctx.encode(site.prefix + cont[:me]))
        before_logits = candidate_logits(ctx, p_ids, before_ids, candidates)
        after_logits = candidate_logits(ctx, p_ids, after_ids, candidates)
        if any(not math.isfinite(v) for v in before_logits + after_logits):
            result = None
        else:
            result = (candidates, softmax(before_logits), softmax(after_logits), ms)
    if key is not None:
        cache[key] = result
    return result


def _kl_for(site: Site, sample: MetaSample, ctx) -> float:
    d = _dists_for(site, sample, ctx)
    if d is None:
        return _NAN
    _, p_before, p_after, _ = d
    return kl_divergence(p_after, p_before)


_PAIR_IN_CAND = re.compile(r"^(\d+)[+\-*/](\d+)$")


def _novel_mask(candidates: list[str], pairs_pre) -> list[bool]:
    out = []
    for c in candidates:
        m = _PAIR_IN_CAND.match(c)
        if not m:
            out.append(False); continue
        a, b = int(m.group(1)), int(m.group(2))
        out.append((min(a, b), max(a, b)) not in set(pairs_pre))
    return out


def novel_mass_shift(site: Site, sample: MetaSample, ctx) -> float:
    r"""★«행동 변화» 자 — 메타가 «아직 안 시도한 쌍»으로 옮긴 확률 질량.

    전체 KL 은 «아무 데로나 흔들기»에 뚫린다(방향이 없다). 우리 의도는 «막히면 새 길»
    이므로, 프리픽스에서 이미 결합해 본 쌍(`pairs_pre`, 모델 자신의 글에서 읽음 — 정답표
    불필요)을 뺀 후보들에 실린 질량이 메타 앞→뒤로 얼마나 늘었는지만 센다.
    값 = Σ_{새 쌍} p_after − Σ_{새 쌍} p_before ∈ [−1, 1].
    """
    d = _dists_for(site, sample, ctx)
    if d is None:
        return _NAN
    candidates, p_before, p_after, _ = d
    mask = _novel_mask(candidates, site.pairs_pre)
    if not any(mask):
        return 0.0
    return float(sum(pa for pa, m in zip(p_after, mask) if m) - sum(pb for pb, m in zip(p_before, mask) if m))


class MoveNovelShift:
    """Σ(새 쌍) p_after − Σ(새 쌍) p_before. 정답표 불필요, 방향 있는 행동 변화."""
    name = "move_novel_shift"
    needs_model = True

    def score(self, site: Site, sample: MetaSample, ctx) -> float:
        return novel_mass_shift(site, sample, ctx)


class MoveNovelShiftStuck:
    r"""막힘 게이트 변형: `novel_mass_shift × 1[메타 앞 시도 ≥ K_S]` (K_S=4, SC 와 동일).

    막힘은 프리픽스(site.prefix + 메타 앞 이어쓰기)의 등식 수로 읽는다 — 자기 보고가
    아니라 모델 자신의 글에서 센 것이며 정답표는 쓰지 않는다.
    """
    name = "move_novel_shift_stuck"
    needs_model = True
    K_S = 4

    def score(self, site: Site, sample: MetaSample, ctx) -> float:
        v = novel_mass_shift(site, sample, ctx)
        if not math.isfinite(v):
            return _NAN
        cont = sample.continuation or ""
        ms = sample.meta_start if sample.meta_start is not None and sample.meta_start >= 0 else len(cont)
        from src.training.countdown_selfcontrol import prefix_features  # noqa: PLC0415
        pf = prefix_features(site.prefix + cont[:ms] + "<meta>", site.nums)
        return v if int(pf.get("n_att_pre", 0)) >= self.K_S else 0.0


class MoveKl:
    """KL(다음수 분포_후 ‖ 다음수 분포_전)."""
    name = "move_kl"
    needs_model = True

    def score(self, site: Site, sample: MetaSample, ctx) -> float:
        return _kl_for(site, sample, ctx)


class MoveKlSigned:
    r"""부호 변형: `move_kl × (+1 if family_dead else −1)`.

    직관(postmortem §3의 "상태 의존성" — 개입은 막혔을 때 도움, 잘 갈 때 해롭다는
    관찰과 같은 모양): 계열이 죽은 사이트에서 분포를 크게 흔드는 메타는 좋은 신호일
    수 있지만(새 방향 탐색), 계열이 살아있는데 분포를 흔드는 메타는 이미 통하는
    방향에서 이탈하는 신호일 수 있다 — 이 부호는 그 가설의 최소 구현이며 검증되지
    않았다(`table.py`가 outcome-fixed·공격 배터리로 실제로 그런지 가른다).
    """
    name = "move_kl_signed"
    needs_model = True

    def score(self, site: Site, sample: MetaSample, ctx) -> float:
        kl = _kl_for(site, sample, ctx)
        if not math.isfinite(kl):
            return _NAN
        if site.family_dead is None:
            return 0.0
        sign = 1.0 if int(site.family_dead) else -1.0
        return float(kl * sign)
