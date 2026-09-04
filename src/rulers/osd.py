r"""OSD(outcome-signed surprisal drop) 자 — 메타가 그 뒤 문장을 얼마나 덜 놀랍게
만들었는가(Δcert), 그리고 그것을 결과 부호로 곱한 보상형 변형.

포트 출처 (수학을 그대로 옮긴다, import 하지 않는다 — verl/ray 의존을 끌고 오지
않기 위해서다):
  `src/training/verl_sdc.py`
    - `_build_osd_arms()`         ~2507-2592  행당 2팔: `W@close`(메타 포함 문맥),
                                              `W@open`(메타 구간만 제거, W 토큰열은
                                              **바이트 동일**) — Δcert 정의의 전제.
    - `_read_osd_from_ref_logprobs()` ~2594-2617  `Δcert = mean(close) - mean(open)`,
                                              `mean`은 `|W|`로 나눈 것(사양이 W 길이로
                                              나눈다 — 메타 길이는 공식에 없다).
    - `_compute_countdown_osd()`  ~2619-2634  진입점 docstring이 Δcert 의 타입
                                              규약(float | 0.0(누출) | NaN(비유한) |
                                              None(못 쟀다))을 정의한다.
  `src/training/countdown_rewards.py`
    - `r_osd()`                   :525-582   `R_osd = y · clip(Δcert / c, ±1)`,
                                              `y = +1(정답)/-1(오답)` — 결과와 확신의
                                              *정렬*(calibration)을 재는 부호 곱.
    - `OSD_W_MAX = 200`           :177        W 창 최대 토큰 길이.

이 파일이 추가하는 것: 위 코드는 verl 배치 조립에 묶여 있어 사이트 하나만 채점할
수 없다. 여기서는 같은 두 수식(Δcert, 부호 곱 보상)을 `HfCtx` 위에서 사이트 단위로
다시 조립한다 — 배치 조립기는 재사용하지 않지만(재사용 대상이 아니다, 애초에 GPU
배치 최적화일 뿐 자의 정의가 아니다), Δcert·r_osd 두 수식 자체는 바이트 그대로
옮긴다.
"""
from __future__ import annotations

import math
from typing import Optional

from src.rulers.base import MetaSample, Site

__all__ = ["OSD_W_MAX", "delta_cert", "r_osd_signed", "OsdUnsigned", "OsdSigned"]

_NAN = float("nan")
OSD_W_MAX = 200  # countdown_rewards.py:177


def delta_cert(ctx, prompt_ids, prefix_open_text: str, prefix_close_text: str,
               window_text: str, w_max: int = OSD_W_MAX) -> float:
    r"""Δcert = (1/|W|)[logP(W|close) − logP(W|open)]. `_read_osd_from_ref_logprobs`
    (verl_sdc.py ~2594-2617)와 같은 정규화(‖W‖로 나눈 평균)."""
    w_ids = list(ctx.encode(window_text or ""))[:w_max]
    if not w_ids:
        return _NAN
    open_ids = list(ctx.encode(prefix_open_text))
    close_ids = list(ctx.encode(prefix_close_text))
    lp_close = ctx.token_logprobs(prompt_ids + close_ids, w_ids)
    lp_open = ctx.token_logprobs(prompt_ids + open_ids, w_ids)
    if lp_close.size == 0 or lp_open.size == 0:
        return _NAN
    d = float(lp_close.mean() - lp_open.mean())
    return d if math.isfinite(d) else _NAN


def r_osd_signed(dcert: Optional[float], r_corr: Optional[int], c: float = 1.0) -> float:
    r"""`countdown_rewards.r_osd`(:525-582)와 동일한 식.
    `R_osd = y · clip(Δcert / c, ±1)`, `y = +1(정답)/-1(오답)`.

    ★원 함수는 `delta_cert is None`을 예외로 fail-loud 처리한다("스코어러가 이
    행을 재지 못했다" — 조용한 0은 이 항을 A 항으로 위장시키므로 금지). 이 자 패키지는
    학습 항이 아니라 **채점표**용이므로 여기서는 예외 대신 NaN을 돌려주고
    `table.py`가 그 행을 스킵하게 한다 — 다른 정책이 필요하면 호출부에서 감싼다.
    """
    if dcert is None or not math.isfinite(float(dcert)):
        return _NAN
    if r_corr is None:
        return _NAN
    cc = float(c)
    if not math.isfinite(cc) or cc <= 0.0:
        raise ValueError(f"r_osd_signed: c={c!r} 는 양수 유한수여야 한다(|Δcert| 의 p90).")
    y = 1.0 if int(r_corr) else -1.0
    return float(y * max(-1.0, min(1.0, float(dcert) / cc)))


def _window_text(site: Site, sample: MetaSample) -> str:
    """메타 뒤 후속 창 W. `_build_osd_arms`의 window은 boxed 시작까지를 자르는데
    (verl_sdc.py의 `_osd_window`), 여기서는 continuation의 meta_end 이후 전부를 W
    후보로 두고 토큰 상한(`OSD_W_MAX`)에서 자른다 — 사이트 단위 채점에서는 boxed
    위치를 별도로 파싱해야 하는데 그 정보가 MetaSample에 없어, 상한 절단이 유일한
    실용적 근사다(그래서 `w_max`를 짧게 두면 boxed 유출 위험도 같이 줄어든다)."""
    cont = sample.continuation or ""
    me = sample.meta_end if (sample.meta_end is not None and sample.meta_end >= 0) else len(cont)
    return cont[me:]


def _open_close(site: Site, sample: MetaSample) -> tuple[str, str]:
    cont = sample.continuation or ""
    ms = sample.meta_start if (sample.meta_start is not None and sample.meta_start >= 0) else len(cont)
    me = sample.meta_end if (sample.meta_end is not None and sample.meta_end >= 0) else ms
    return site.prefix + cont[:ms], site.prefix + cont[:me]


def _prompt_ids(ctx, site: Site):
    text = ""
    if hasattr(ctx.tokenizer, "apply_chat_template"):
        text = ctx.tokenizer.apply_chat_template(
            site.prompt_messages, tokenize=False, add_generation_prompt=True)
    return list(ctx.encode(text)) if text else []


class OsdUnsigned:
    """Δcert 그 자체(부호 곱 없음) — "확신이 늘었는가"만 재는 진단용 원신호."""
    name = "osd_unsigned"
    needs_model = True

    def score(self, site: Site, sample: MetaSample, ctx) -> float:
        if ctx is None or not sample.meta_raw:
            return _NAN
        open_text, close_text = _open_close(site, sample)
        window_text = _window_text(site, sample)
        return delta_cert(ctx, _prompt_ids(ctx, site), open_text, close_text, window_text)


class OsdSigned:
    """`R_osd = y · clip(Δcert/c, ±1)` — 결과-정합 자(§`r_osd` 주석)."""
    name = "osd_signed"
    needs_model = True

    def __init__(self, c: float = 1.0):
        self.c = c

    def score(self, site: Site, sample: MetaSample, ctx) -> float:
        if ctx is None or not sample.meta_raw:
            return _NAN
        open_text, close_text = _open_close(site, sample)
        window_text = _window_text(site, sample)
        d = delta_cert(ctx, _prompt_ids(ctx, site), open_text, close_text, window_text)
        return r_osd_signed(d, sample.r_corr, c=self.c)
