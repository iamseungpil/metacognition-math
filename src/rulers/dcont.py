r"""dCont 자 — 메타가 "이어쓰기(EOS가 아니라 계속 쓰기)" 확률을 얼마나 올렸는가.

    p_continue(ctx) = 1 − P(EOS | ctx)
    dCont = p_continue(프롬프트⊕프리픽스⊕메타)  −  p_continue(프롬프트⊕프리픽스)

포트 출처 (수학을 그대로 옮긴다, import 하지 않는다):
  `scripts/move_space_probe.py`
    - `p_continue()`   :186-192   `1 - P(EOS 바로 다음)`. "메타가 «그만 쓰게» 만드는
                                  효과를 숨기지 않는다"는 그 파일의 주석을 그대로 지킨다
                                  — EOS 확률을 클립하거나 다른 토큰과 합치지 않는다.
    - dCont 계산 지점   :292-299  `rec["dCont"] = p_continue(ctx_post) - p_continue(ctx_pre)`
                                  (`ctx_pre`=메타 앞, `ctx_post`=메타 뒤)

★과제 지시문은 이 자가 `scripts/meta_worth_probe.py`에 있다고 했으나, 실측 결과
  그 파일에는 dCont 코드가 없다(grep 0건) — 실제 정의처는 `scripts/move_space_probe.py`
  다(위 인용). 이 docstring이 그 사실을 기록해 둔다: 지시문의 파일명은 부정확했고,
  여기서는 실제로 찾은 소스를 인용했다.

★알려진 실패 모드 (`docs/POSTMORTEM_cd6_rulers_2026-09-03.md` §2, §1.2 인용):
  - `move_space_probe.py:17` 자신이 "dCont는 죽은 열로 사전등록"했다 — 중간 문맥에서
    `<|im_end|>` 확률이 사실상 0이라 신호가 거의 없다(실측).
  - P3 사이트 배터리(FINDINGS §14, §16): **빈 메타가 dCont 최댓값**(real −0.42 <
    empty 0.00), decision을 "verify" 로 쓰면 값이 오른다(내용과 무관한 결정 레버).
  - ⇒ `table.py`의 공격 배터리(빈 메타/gibberish/verify 한 줄)가 정확히 이 실패를
    재현하는지 확인하는 것이 이 자를 여기 포함한 이유다 — "포트했다"와 "썼다"는
    다르다: 이 자는 보상 후보가 아니라 **실패 재현용 음성 대조군**으로 패키지에 있다.
"""
from __future__ import annotations

import math

from src.rulers.base import MetaSample, Site

__all__ = ["p_continue", "DCont"]

_NAN = float("nan")


def p_continue(ctx, ids: list[int]) -> float:
    """`move_space_probe.p_continue`(:186-192)와 동일: `1 - P(eos | ids)`."""
    if not ids:
        return _NAN
    eos_id = getattr(ctx.tokenizer, "eos_token_id", None)
    if eos_id is None:
        return _NAN
    dist = ctx.sequence_logprob(ids)          # log_softmax over vocab at last position
    import torch  # noqa: PLC0415  (available in this env per CLAUDE.md)
    p_eos = float(torch.exp(dist[eos_id]).item()) if hasattr(dist, "__getitem__") else _NAN
    if not math.isfinite(p_eos):
        return _NAN
    return 1.0 - p_eos


class DCont:
    """`dCont = p_continue(post-meta) - p_continue(pre-meta)`."""
    name = "dcont"
    needs_model = True

    def score(self, site: Site, sample: MetaSample, ctx) -> float:
        if ctx is None or not sample.meta_raw:
            return _NAN
        cont = sample.continuation or ""
        ms = sample.meta_start if sample.meta_start is not None and sample.meta_start >= 0 else len(cont)
        me = sample.meta_end if sample.meta_end is not None and sample.meta_end >= 0 else ms
        prompt_text = ""
        if hasattr(ctx.tokenizer, "apply_chat_template"):
            prompt_text = ctx.tokenizer.apply_chat_template(
                site.prompt_messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = list(ctx.encode(prompt_text)) if prompt_text else []
        pre_ids = prompt_ids + list(ctx.encode(site.prefix + cont[:ms]))
        post_ids = prompt_ids + list(ctx.encode(site.prefix + cont[:me]))
        p_pre = p_continue(ctx, pre_ids)
        p_post = p_continue(ctx, post_ids)
        if not (math.isfinite(p_pre) and math.isfinite(p_post)):
            return _NAN
        return float(p_post - p_pre)
