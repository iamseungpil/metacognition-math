r"""R2_full: `\boxed{expr}` 전체 토큰의 토큰당 평균 logp 로 PMI 를 읽는 변형.

divergent 슬라이스(대부분 1토큰) 대신 전체 스팬 평균:
    PMI_full(ctx) = mean_t logp(gold_t|ctx) − mean_t logp(decoy_t|ctx)
평균이므로 gold/decoy 토큰 길이 불일치(실측 3.7%)의 합-logp 길이 편향이 1차 제거된다.
배치·ref 호출은 countdown_pmi 와 동일 — 이 모듈은 읽기 함수와 진입점만 더한다.
기존 countdown_pmi.py 는 한 바이트도 수정하지 않는다.
"""
from __future__ import annotations

import math
import os

from src.training.countdown_pmi import (
    _NAN, _pad_unit, _row_sum, assert_pmi_config, build_pmi_arms,
    read_pmi_from_ref_logprobs,
)

__all__ = ["read_pmi_full_from_ref_logprobs", "score_pmi_shift_full"]


def read_pmi_full_from_ref_logprobs(ref_lp, attempts) -> list:
    """행별 (pmi_open_full, pmi_close_full). `base = 4*k` 부기는 countdown_pmi 와 동일."""
    out = []
    for k, at in enumerate(attempts):
        base = 4 * k
        Lg, Ld = len(at.pair.gold_ids), len(at.pair.decoy_ids)
        if Lg <= 0 or Ld <= 0:
            out.append((_NAN, _NAN))
            continue
        try:
            g_open = _row_sum(ref_lp, base + 0, Lg, slice(0, Lg)) / Lg
            d_open = _row_sum(ref_lp, base + 1, Ld, slice(0, Ld)) / Ld
            g_close = _row_sum(ref_lp, base + 2, Lg, slice(0, Lg)) / Lg
            d_close = _row_sum(ref_lp, base + 3, Ld, slice(0, Ld)) / Ld
        except Exception:
            out.append((_NAN, _NAN))
            continue
        po, pc = g_open - d_open, g_close - d_close
        if not (math.isfinite(po) and math.isfinite(pc)):
            out.append((_NAN, _NAN))
            continue
        out.append((float(po), float(pc)))
    return out


def score_pmi_shift_full(*, tokenizer, trainer, prompt_texts, response_texts,
                         witnesses, decoys, step=0, debug=None, _ref_scorer=None):
    """`score_pmi_shift` 와 같은 계약 + 행에 pmi_open_full/pmi_close_full 을 더한다.

    rows[i] 는 기존 키 전부(pmi_open, pmi_close, meta_n_tok, emitted, path,
    meta_first, scored)에 pmi_open_full/pmi_close_full 을 **추가**한 상위집합이다.
    """
    if debug is None:
        debug = os.environ.get("DCPO_DEBUG", "1") == "1"
    B = len(response_texts)
    rows = [{"pmi_open": _NAN, "pmi_close": _NAN,
             "pmi_open_full": _NAN, "pmi_close_full": _NAN,
             "meta_n_tok": 0, "emitted": 0, "path": None,
             "meta_first": False, "scored": False} for _ in range(B)]

    arm_prompts, arm_resps, attempts, diag = build_pmi_arms(
        tokenizer, prompt_texts, response_texts, witnesses, decoys)
    diag = dict(diag)
    diag.update(B=B, attempted=len(attempts), scored=0,
                attempted_rate=(len(attempts) / B) if B else 0.0, ref_error=None)

    for at in attempts:
        r = rows[at.row]
        r["emitted"] = 1
        r["meta_n_tok"] = at.span.n_inner_tok
        r["path"] = at.pair.path
        r["meta_first"] = at.span.meta_first

    if not attempts:
        if debug:
            print(f"[CD-PMI-FULL] step={step}: B={B} attempted=0 — {diag}", flush=True)
        return rows, diag

    if _ref_scorer is None:
        assert_pmi_config(trainer)
        from src.training.verl_sdc import (          # noqa: PLC0415  지연 import (CPU 테스트)
            _build_pmi_score_batches, _dcpo_v4_ref_logprobs,
        )
        tensors, real_n = _build_pmi_score_batches(
            arm_prompts, arm_resps, _pad_unit(trainer))
        if real_n != 4 * len(attempts):
            raise AssertionError(f"CD-PMI-FULL 팔 부기가 깨졌다: {real_n} != 4*{len(attempts)}")
        try:
            ref_lp = _dcpo_v4_ref_logprobs(trainer, tensors)
        except AssertionError:
            raise
        except Exception as e:
            diag["ref_error"] = f"{type(e).__name__}: {e}"
            print(f"[CD-PMI-FULL] step={step}: ref 실패 ({diag['ref_error']}) — 전부 NaN.",
                  flush=True)
            return rows, diag
    else:
        ref_lp = _ref_scorer(arm_prompts, arm_resps)

    pmis_div = read_pmi_from_ref_logprobs(ref_lp, attempts)     # 텔레메트리 연속성
    pmis_full = read_pmi_full_from_ref_logprobs(ref_lp, attempts)
    n_scored = 0
    for at, (po, pc), (fo, fc) in zip(attempts, pmis_div, pmis_full):
        r = rows[at.row]
        r["pmi_open"], r["pmi_close"] = po, pc
        r["pmi_open_full"], r["pmi_close_full"] = fo, fc
        ok = math.isfinite(fo) and math.isfinite(fc)
        r["scored"] = bool(ok)                       # ★scored 는 full 기준 (P 팔의 판정 축)
        n_scored += int(ok)
    diag["scored"] = n_scored
    diag["scored_rate"] = n_scored / B if B else 0.0
    if debug:
        print(f"[CD-PMI-FULL] step={step}: B={B} attempted={len(attempts)} "
              f"scored={n_scored} no_meta={diag['no_meta']}", flush=True)
    return rows, diag
