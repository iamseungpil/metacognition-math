r"""PMI-shift 자 — gold vs decoy 마지막-식 로그오즈가 메타 전/후로 얼마나 갈리는가.

포트 출처 (수학을 그대로 옮긴다, import 하지 않는다):
  `src/training/countdown_pmi.py`
    - `boxed()`                    :29-38   \boxed{expr} 문자열
    - `divergent_spans()`          :41-70   공통 접두/접미를 자른 발산 토큰 슬라이스,
                                             빈 슬라이스면 전체 문자열로 폴백("div"/"full")
    - `build_pmi_arms`/`score_pmi_shift`  :277-360, 470-  4팔(gold/decoy × open/close)
      구성과 "두 슬라이스를 각각 합산"(길이 불일치를 NaN으로 죽이지 않는다) 규약
  `scripts/pair_rulers.py`
    - `pmi_from()`                 :179-184  발산 슬라이스 PMI **와** 전체 문자열 PMI를
                                             같은 forward에서 함께 낸다(마지막-토큰/평균
                                             변형은 그 자리에 없었다 — 여기서 추가)
    - `score_site` 의 PMI 블록      :227-235  decoy = `swap_op_decoy(witness, nums, target)`,
                                             open/close 컨텍스트 = 헤드+프리픽스(+메타)

이 파일이 추가하는 것 (VERDICT의 pmi_shift 판정을 참고한 신규 변형):
  `last`  자.슬라이스의 **마지막 토큰만** 비교 — "최종 판단이 갈렸는가"에 더 가깝고
          `divergent_spans`의 길이 편향("full" 경로가 긴 문자열에 불리)에 가장 덜 민감하다.
  `mean`  자.슬라이스의 **토큰 평균** logp 차 — 길이 편향을 명시적으로 지운 변형.
  기존 `countdown_pmi.read_pmi_from_ref_logprobs`는 **합**(sum)만 냈다. VERDICT
  (`docs/VERDICT_cd6_pair_rulers.md` pmi_shift 행: "결정 안 됨, niw0 0.583")가 sum
  변형 하나만 갖고 판정했으므로, 여기서 last/mean을 나란히 내 `table.py`가 셋을
  각각 채점하게 한다 — sum 변형이 우연히 이겼는지 셋 다 비슷한지 가르기 위함이다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from src.rulers.base import MetaSample, Site

__all__ = ["boxed", "divergent_spans", "DivergentPair", "PmiShiftLast",
           "PmiShiftMean", "PmiShiftSum", "swap_op_decoy_or_none"]

_NAN = float("nan")


def boxed(expr) -> str:
    """`countdown_pmi.boxed`(:29-38)와 동일 — 여기서 다시 정의해 이 모듈이 verl/그
    구파일에 의존하지 않게 한다."""
    return r"\boxed{" + str(expr).strip() + "}"


@dataclass(frozen=True)
class DivergentPair:
    gold_ids: tuple
    decoy_ids: tuple
    gold_slice: slice
    decoy_slice: slice
    path: str


def divergent_spans(encode, gold_expr: str, decoy_expr: str) -> Optional[DivergentPair]:
    """`countdown_pmi.divergent_spans`(:216-244)와 같은 알고리즘. `encode`는
    `HfCtx.encode`처럼 문자열 -> 토큰id 리스트를 주는 콜러블."""
    if gold_expr is None or decoy_expr is None:
        return None
    g_txt, d_txt = boxed(gold_expr), boxed(decoy_expr)
    if not str(gold_expr).strip() or not str(decoy_expr).strip():
        return None
    if g_txt == d_txt:
        return None
    g, d = list(encode(g_txt)), list(encode(d_txt))
    if not g or not d:
        return None
    p = 0
    while p < min(len(g), len(d)) and g[p] == d[p]:
        p += 1
    s = 0
    while (s < min(len(g), len(d)) - p) and g[len(g) - 1 - s] == d[len(d) - 1 - s]:
        s += 1
    gs, ds = slice(p, len(g) - s), slice(p, len(d) - s)
    path = "div"
    if gs.stop <= gs.start or ds.stop <= ds.start:
        gs, ds, path = slice(0, len(g)), slice(0, len(d)), "full"
    return DivergentPair(tuple(g), tuple(d), gs, ds, path)


def swap_op_decoy_or_none(witness: str, nums, target, rng):
    """`countdown_task.swap_op_decoy`를 그대로 호출하는 얇은 래퍼(그 함수 자체는
    포트 대상이 아니다 — 검증 로직이지 자의 수학이 아니다). 실패하면 None."""
    from src.training.countdown_task import swap_op_decoy  # noqa: PLC0415
    try:
        return swap_op_decoy(witness, list(nums), int(target), rng)
    except Exception:
        return None


def _response_texts(site: Site, sample: MetaSample) -> tuple[str, str]:
    """(open, close) — 메타 전/후 텍스트. `pair_rulers.score_site`의 `ctx_open`/
    `ctx_close` 구성(:227)과 같다."""
    cont = sample.continuation or ""
    ms = max(0, int(sample.meta_start)) if sample.meta_start is not None and sample.meta_start >= 0 else len(cont)
    me = max(ms, int(sample.meta_end)) if sample.meta_end is not None and sample.meta_end >= 0 else ms
    return site.prefix + cont[:ms], site.prefix + cont[:me]


def _pmi_value(ctx, prompt_ids, open_text, close_text, pair: DivergentPair, agg: str) -> float:
    g, d = list(pair.gold_ids), list(pair.decoy_ids)
    o_ids = list(ctx.encode(open_text))
    c_ids = list(ctx.encode(close_text))

    def _score(ctx_ids, ids, sl):
        lp = ctx.token_logprobs(prompt_ids + ctx_ids, ids)
        seg = lp[sl]
        if seg.size == 0:
            return _NAN
        if agg == "sum":
            return float(seg.sum())
        if agg == "mean":
            return float(seg.mean())
        if agg == "last":
            return float(seg[-1])
        raise ValueError(f"_pmi_value: agg={agg!r} 모른다")

    g_open = _score(o_ids, g, pair.gold_slice)
    d_open = _score(o_ids, d, pair.decoy_slice)
    g_close = _score(c_ids, g, pair.gold_slice)
    d_close = _score(c_ids, d, pair.decoy_slice)
    vals = (g_open, d_open, g_close, d_close)
    if not all(math.isfinite(v) for v in vals):
        return _NAN
    return (g_close - d_close) - (g_open - d_open)


def _pair_for(ctx, site: Site) -> Optional[DivergentPair]:
    if not site.witness:
        return None
    import random
    decoy = site.decoy or swap_op_decoy_or_none(
        site.witness, site.nums, site.target, random.Random(1))
    if not decoy:
        return None
    return divergent_spans(ctx.encode, site.witness, decoy)


class _PmiShiftBase:
    needs_model = True
    agg = "sum"

    def score(self, site: Site, sample: MetaSample, ctx) -> float:
        if ctx is None:
            return _NAN
        if not sample.meta_raw:
            return _NAN
        pair = _pair_for(ctx, site)
        if pair is None:
            return _NAN
        prompt_text = ctx.tokenizer.apply_chat_template(
            site.prompt_messages, tokenize=False, add_generation_prompt=True) \
            if hasattr(ctx.tokenizer, "apply_chat_template") else ""
        prompt_ids = list(ctx.encode(prompt_text)) if prompt_text else []
        open_text, close_text = _response_texts(site, sample)
        return _pmi_value(ctx, prompt_ids, open_text, close_text, pair, self.agg)


class PmiShiftSum(_PmiShiftBase):
    """`countdown_pmi.read_pmi_from_ref_logprobs`의 원래 집계(:426-451, 슬라이스 합)."""
    name = "pmi_shift_sum"
    agg = "sum"


class PmiShiftLast(_PmiShiftBase):
    """신규: 발산 슬라이스의 마지막 토큰만."""
    name = "pmi_shift_last"
    agg = "last"


class PmiShiftMean(_PmiShiftBase):
    """신규: 발산 슬라이스 토큰 평균(길이 편향 제거)."""
    name = "pmi_shift_mean"
    agg = "mean"
