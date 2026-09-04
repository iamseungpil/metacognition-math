r"""도치(inversion) 자 — 「정답을 알려준 뒤 메타의 프로즈 토큰이 얼마나 덜 그럴듯해지는가」.

    sh_t = logP(meta_t | 프롬프트+힌트 ⊕ 접두)  −  logP(meta_t | 프롬프트 ⊕ 접두)
    inv  = agg_{t ∈ 프로즈}(sh_t)

포트 출처 (수학을 그대로 옮긴다, import 하지 않는다):
  `src/training/countdown_inv.py`
    - `INV_HINT_TMPL`/`INV_ANCHOR_TMPL`   :44-48   힌트 문자열과 삽입 앵커
                                                   ("Target: N" 뒤에 "Hint: one valid
                                                   solution is {w}." 를 끼운다)
    - `_prose_char_flags()`               :92-103  confidence:/decision: 줄과 태그
                                                   줄을 프로즈에서 뺀다
    - `prose_token_flags()`               :106-134 문자 프로즈 플래그를 토큰 플래그로
                                                   (경계에 걸친 토큰은 엄격하게 제외)
    - `inv_aggregate()`                   :137-155 min/mean/bot25 집계
    - `inv_hint_prompt()`                 :253-265 앵커 "Target: N" 뒤에 힌트 삽입
                                                   (앵커 없으면 ValueError — fail-loud)

이 파일은 `scope="inplace", form="d2a"`(메타 토큰열을 그대로 문맥에 두고 "차이의
집계"를 쓰는 형태) 하나만 옮긴다 — `countdown_inv.py` 머리말이 판정한 대로
"학습이 실제로 계산할 수 있는 유일한 정의"이기 때문이다(다른 3칸은 VERDICT에서
전부 탈락했고, 재현 대상이 아니다).
"""
from __future__ import annotations

import math
import re
from typing import Optional

from src.rulers.base import MetaSample, Site

__all__ = ["INV_HINT_TMPL", "INV_ANCHOR_TMPL", "prose_char_flags", "prose_token_flags",
           "inv_aggregate", "InvMin", "InvMean"]

_NAN = float("nan")

INV_HINT_TMPL = "\n\nHint: one valid solution is {w}."   # countdown_inv.py:44
INV_ANCHOR_TMPL = "Target: {t}"                          # countdown_inv.py:48

_CONF_LINE = re.compile(r"^\s*confidence\s*:", re.I)
_DEC_LINE = re.compile(r"^\s*decision\s*:", re.I)


def prose_char_flags(meta_raw: str) -> list[bool]:
    """`countdown_inv._prose_char_flags`(:92-103)와 동일."""
    flags = [False] * len(meta_raw)
    off = 0
    for ln in meta_raw.split("\n"):
        s, e = off, off + len(ln)
        off = e + 1
        st = ln.strip()
        if not st or st.startswith("<") or _CONF_LINE.match(ln) or _DEC_LINE.match(ln):
            continue
        for i in range(s, min(e, len(flags))):
            flags[i] = True
    return flags


def prose_token_flags(meta_raw: str, offsets: list[tuple[int, int]],
                       meta_char_base: int = 0) -> list[bool]:
    """`countdown_inv.prose_token_flags`(:106-134)와 동일한 규약(엄격 — 경계에 걸친
    토큰은 제외). `offsets`는 메타 원문(meta_raw) 기준이 아니라 이 함수를 부르는
    쪽이 넘긴 시퀀스의 offset_mapping이므로, `meta_char_base`로 원점을 맞춘다."""
    cflags = prose_char_flags(meta_raw)
    lo, hi = meta_char_base, meta_char_base + len(meta_raw)
    out = []
    for a, b in offsets:
        if b <= a or b <= lo or a >= hi:
            out.append(False)
            continue
        hit_prose = hit_other = False
        for c in range(max(a, lo), min(b, hi)):
            if cflags[c - lo]:
                hit_prose = True
            else:
                hit_other = True
        out.append(bool(hit_prose and not hit_other))
    return out


def inv_aggregate(sh, agg: str = "min") -> float:
    """`countdown_inv.inv_aggregate`(:137-155)와 동일(min/mean/bot25)."""
    xs = [float(v) for v in sh if math.isfinite(float(v))]
    if len(xs) < 3:
        return _NAN
    if agg == "min":
        return float(min(xs))
    if agg == "mean":
        return float(sum(xs) / len(xs))
    if agg == "bot25":
        xs = sorted(xs)
        k = max(1, len(xs) // 4)
        return float(sum(xs[:k]) / k)
    raise ValueError(f"inv_aggregate: 모르는 집계 {agg!r}")


def _hint_prompt(prompt_text: str, witness: str, target: int) -> str:
    """`countdown_inv.inv_hint_prompt`(:253-265)와 동일 — 앵커 없으면 ValueError."""
    anchor = INV_ANCHOR_TMPL.format(t=int(target))
    i = (prompt_text or "").rfind(anchor)
    if i < 0:
        raise ValueError(f"_hint_prompt: 앵커 {anchor!r} 없음 — 프롬프트 형식이 다르다.")
    j = i + len(anchor)
    return prompt_text[:j] + INV_HINT_TMPL.format(w=str(witness)) + prompt_text[j:]


class _InvBase:
    needs_model = True
    agg = "min"

    def score(self, site: Site, sample: MetaSample, ctx) -> float:
        if ctx is None or not sample.meta_raw or not site.witness:
            return _NAN
        cont = sample.continuation or ""
        ms = sample.meta_start if sample.meta_start is not None and sample.meta_start >= 0 else 0
        me = sample.meta_end if sample.meta_end is not None and sample.meta_end >= 0 else ms
        prefix_text = site.prefix + cont[:ms]
        meta_text = cont[ms:me]
        if not meta_text:
            return _NAN

        prompt_text = ""
        if hasattr(ctx.tokenizer, "apply_chat_template"):
            prompt_text = ctx.tokenizer.apply_chat_template(
                site.prompt_messages, tokenize=False, add_generation_prompt=True)
        try:
            hint_prompt_text = _hint_prompt(prompt_text, site.witness, site.target)
        except ValueError:
            return _NAN

        plain_ids = list(ctx.encode(prompt_text)) if prompt_text else []
        hint_ids = list(ctx.encode(hint_prompt_text)) if hint_prompt_text else []
        pre_ids = list(ctx.encode(prefix_text))
        meta_ids, meta_offsets = ctx.encode_with_offsets(meta_text)
        if not meta_ids:
            return _NAN

        lp_plain = ctx.token_logprobs(plain_ids + pre_ids, meta_ids)
        lp_hint = ctx.token_logprobs(hint_ids + pre_ids, meta_ids)
        if lp_plain.size != len(meta_ids) or lp_hint.size != len(meta_ids):
            return _NAN

        flags = prose_token_flags(meta_text, meta_offsets, meta_char_base=0)
        sh = [float(h - p) for h, p, f in zip(lp_hint, lp_plain, flags) if f]
        return inv_aggregate(sh, self.agg)


class InvMin(_InvBase):
    """`inv_min` — VERDICT의 `inv_next_sum`과 다른 집계(min). 게이밍 방어의 원 취지
    (한 토큰만 심하게 튀어도 잡는다)를 살린 변형."""
    name = "inv_min"
    agg = "min"


class InvMean(_InvBase):
    name = "inv_mean"
    agg = "mean"
