"""dcpo_pmi surviving core — split_first_meta, pmi_aggregate, PLACEBO_META.

RELOCATED 2026-08-03 from tests/test_dcpo_v4_pmi.py. That file also covered the
dense-PMI scorer (splice_and_align / sign_gate / ngram_overlap_guard /
compute_pmi_rows), which was removed with the `pmi` reward generation. The cases
below exercise symbols that are still LIVE and must not lose coverage:

  split_first_meta  — the pmi_shift scorer (verl_sdc.py) and the always-on
                      epistemic wandb block both call it.
  pmi_aggregate /
  PMI_AGG_METHODS   — still exported by src/training/dcpo_pmi.py.
  PLACEBO_META      — still imported by verl_sdc.py.
"""

import numpy as np
import pytest

from src.training.dcpo_pmi import (
    PMI_AGG_METHODS,
    pmi_aggregate,
    split_first_meta,
)


# ═══════════════════════════════════════════════════════════════════════════
# split_first_meta (round 2 M-D: ONE definition for probe + verl_sdc)
# ═══════════════════════════════════════════════════════════════════════════
def test_split_first_meta_normal_and_first_block_only():
    text = "pre<|meta|>check<|/meta|>mid<|meta|>again<|/meta|>tail"
    prefix, meta, cont = split_first_meta(text)
    assert prefix == "pre"
    assert meta == "<|meta|>check<|/meta|>"
    assert cont == "mid<|meta|>again<|/meta|>tail"   # FIRST block only
    assert prefix + meta + cont == text              # lossless 3-way split


def test_split_first_meta_rejects_malformed():
    assert split_first_meta("no tags at all") is None
    assert split_first_meta("work <|meta|>truncated at 16k cut") is None  # no close
    assert split_first_meta(None) is None
    assert split_first_meta("") is None


def test_split_first_meta_whitespace_only_continuation_is_none():
    # the STRICTER probe semantics, unified (round 2 M-D): nothing to score.
    assert split_first_meta("p<|meta|>m<|/meta|>") is None
    assert split_first_meta("p<|meta|>m<|/meta|>  \n\t") is None
    assert split_first_meta("p<|meta|>m<|/meta|> x") is not None


# ═══════════════════════════════════════════════════════════════════════════
# pmi_aggregate
# ═══════════════════════════════════════════════════════════════════════════
_DELTA = [1.0, -0.5, 3.0, 0.5]


def test_aggregate_mean_and_max():
    assert pmi_aggregate(_DELTA, "mean") == pytest.approx(1.0)
    assert pmi_aggregate(_DELTA, "max") == pytest.approx(3.0)


def test_aggregate_sum_clip_clips_per_token():
    # clip_c=2: [1, -0.5, 2, 0.5] -> 3.0 (the 3.0 outlier is bounded, not the sum)
    assert pmi_aggregate(_DELTA, "sum_clip", clip_c=2.0) == pytest.approx(3.0)
    assert pmi_aggregate(_DELTA, "sum_clip", clip_c=10.0) == pytest.approx(4.0)
    # symmetric on the negative side
    assert pmi_aggregate([-5.0, 1.0], "sum_clip", clip_c=2.0) == pytest.approx(-1.0)


def test_aggregate_topk_mean_fraction_knob():
    # frac 0.5 of 4 tokens -> k=2 -> mean(top2) = (3 + 1) / 2
    assert pmi_aggregate(_DELTA, "topk_mean", topk_frac=0.5) == pytest.approx(2.0)
    # tiny fraction still keeps k >= 1 -> max
    assert pmi_aggregate(_DELTA, "topk_mean", topk_frac=0.01) == pytest.approx(3.0)
    # frac 1.0 -> plain mean
    assert pmi_aggregate(_DELTA, "topk_mean", topk_frac=1.0) == pytest.approx(1.0)


def test_aggregate_mean_min_alpha_zero_is_clipped_mean():
    # RLT-faithful mean + alpha*min, on the per-token-CLIPPED deltas. alpha=0
    # reduces to the clipped mean (NOT the unclipped "mean" method): clip_c=2
    # bounds the 3.0 outlier -> [1, -0.5, 2, 0.5], mean = 0.75.
    assert pmi_aggregate(_DELTA, "mean_min", clip_c=2.0, alpha=0.0) == pytest.approx(0.75)
    # alpha weights the worst (clipped) token: 0.75 + 0.5 * min([1,-0.5,2,0.5]) =
    # 0.75 + 0.5*(-0.5) = 0.5.
    assert pmi_aggregate(_DELTA, "mean_min", clip_c=2.0, alpha=0.5) == pytest.approx(0.5)


def test_aggregate_mean_min_penalizes_worst_token_at_equal_mean():
    # SELECTIVITY: two metas with the SAME mean lift, but the "tanked" one drops
    # a single token. mean+alpha*min must rank the uniform one strictly higher —
    # this is what punishes generic verify that fails the hard token (Gandhi).
    uniform = [0.4, 0.4, 0.4, 0.4]  # mean 0.4, min 0.4
    tanked = [0.6, 0.6, 0.6, -0.2]  # mean 0.4, min -0.2
    u = pmi_aggregate(uniform, "mean_min", clip_c=2.0, alpha=0.5)
    t = pmi_aggregate(tanked, "mean_min", clip_c=2.0, alpha=0.5)
    assert u == pytest.approx(0.6)
    assert t == pytest.approx(0.3)
    assert u > t


def test_aggregate_mean_min_clip_bounds_outlier_min():
    # A single catastrophic token must NOT swamp the aggregate: clip_c saturates
    # the min so -100 and -1000 give the IDENTICAL result (the swamping fix).
    a = pmi_aggregate([0.5, 0.5, 0.5, -100.0], "mean_min", clip_c=2.0, alpha=0.5)
    b = pmi_aggregate([0.5, 0.5, 0.5, -1000.0], "mean_min", clip_c=2.0, alpha=0.5)
    assert a == pytest.approx(b)
    # value: clip -> [0.5,0.5,0.5,-2], mean=-0.125, min=-2 -> -0.125 + 0.5*(-2) = -1.125
    assert a == pytest.approx(-1.125)


def test_mean_min_in_agg_methods():
    assert "mean_min" in PMI_AGG_METHODS


def test_aggregate_max_minus_min_rejected():
    with pytest.raises(ValueError, match="direction-blind"):
        pmi_aggregate(_DELTA, "max_minus_min")


def test_aggregate_unknown_method_and_empty_raise():
    with pytest.raises(ValueError):
        pmi_aggregate(_DELTA, "median")
    with pytest.raises(ValueError):
        pmi_aggregate([], "mean")


def test_aggregate_accepts_numpy_input():
    assert pmi_aggregate(np.asarray(_DELTA, dtype=np.float32), "max") == pytest.approx(3.0)

def test_placebo_meta_constant_is_tag_wrapped_ssot():
    from src.training.dcpo_pmi import PLACEBO_META
    from src.metacot.prompt import META_END, META_START
    assert PLACEBO_META.startswith(META_START) and PLACEBO_META.endswith(META_END)
    assert "Let me continue." in PLACEBO_META
