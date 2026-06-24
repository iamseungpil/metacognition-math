"""CPU unit tests for the directional (gm-contrast) R_meta pure core.

Covers the testable surface of src/training/dcpo_directional.py per the spec
(2026-06-24-directional-self-distill-meta-rl-design.md §S1 TDD):
  - gm DiD mean_min aggregation (gold-favor -> +, boilerplate/empty -> NaN-fail),
  - divergent_token_mask (excludes shared \boxed structural tokens),
  - compute_directional_meta_reward sign-gate (correct -> +, wrong -> -, fail 0),
  - rlsd_meta_weight w=exp(sign(A_corr)*clip(gm)) shackling,
  - rlsd_meta_factor lam interpolation,
  - decoy construction via _rule_based_decoy + boxed_answer_string,
  - compose backward-compat byte-identity when the new flags are off.
"""
import math

import numpy as np
import torch

from src.training.dcpo_directional import (
    boxed_answer_string,
    divergent_token_mask,
    gm_contrast_row,
    compute_directional_meta_reward,
    rlsd_meta_weight,
    rlsd_meta_factor,
    gm_over_emission_penalty,
)


# ── boxed_answer_string + decoy construction ────────────────────────────────
def test_boxed_answer_string_wraps_value():
    assert boxed_answer_string("42") == r"\boxed{42}"
    assert boxed_answer_string(7) == r"\boxed{7}"
    assert boxed_answer_string("  3/4 ") == r"\boxed{3/4}"


def test_rule_based_decoy_differs_from_gold():
    from src.training._decoy_utils import _rule_based_decoy
    gold = "42"
    decoy = _rule_based_decoy(gold, seed=42)
    assert decoy != gold
    # the gm contrast wraps both into boxed strings — they must differ
    assert boxed_answer_string(gold) != boxed_answer_string(decoy)


# ── divergent_token_mask ─────────────────────────────────────────────────────
def test_divergent_mask_excludes_shared_tokens():
    # gold and decoy share a structural prefix (the \boxed{ tokens) and differ
    # only at the value token.
    gold_ids = [100, 200, 300, 999]   # ... \boxed{ ... value=300 ... }
    decoy_ids = [100, 200, 301, 999]  # same except value token differs
    mask = divergent_token_mask(gold_ids, decoy_ids)
    assert mask.tolist() == [False, False, True, False]


def test_divergent_mask_extra_gold_tokens_are_divergent():
    gold_ids = [1, 2, 3, 4]
    decoy_ids = [1, 2]
    mask = divergent_token_mask(gold_ids, decoy_ids)
    # positions past the shorter decoy length are divergent by construction
    assert mask.tolist() == [False, False, True, True]


# ── gm_contrast_row (mean_min) ───────────────────────────────────────────────
def _row(gm_meta, gm_plac, dc_meta, dc_plac, mask=None, correct=True):
    r = {
        "logp_gold_meta": np.asarray(gm_meta, dtype=np.float64),
        "logp_gold_placebo": np.asarray(gm_plac, dtype=np.float64),
        "logp_decoy_meta": np.asarray(dc_meta, dtype=np.float64),
        "logp_decoy_placebo": np.asarray(dc_plac, dtype=np.float64),
        "correct": correct,
    }
    if mask is not None:
        r["divergent_mask"] = np.asarray(mask, dtype=bool)
    return r


def test_gm_contrast_gold_favoring_is_positive():
    # token 0 shared (DiD 0, excluded by mask); token 1 = the value token: meta
    # raises gold logp and lowers decoy logp.
    #   gold DiD = (gm_meta - gm_plac) = -0.5 - (-2.0) = +1.5
    #   decoy DiD = (dc_meta - dc_plac) = -3.0 - (-2.5) = -0.5
    #   DiD = 1.5 - (-0.5) = +2.0 at the divergent token
    row = _row([-1.0, -0.5], [-1.0, -2.0], [-1.0, -3.0], [-1.0, -2.5],
               mask=[False, True])
    gm = gm_contrast_row(row, agg="mean_min", clip_c=4.0, alpha=1.0)
    # only one divergent token -> mean==min==2.0; mean_min = 2.0 + 1.0*2.0 = 4.0
    assert abs(gm - 4.0) < 1e-9


def test_gm_contrast_boilerplate_empty_divergent_is_nan():
    # all tokens shared -> empty divergent span -> NaN (caller fails closed).
    row = _row([-1.0, -0.5], [-1.0, -2.0], [-1.0, -3.0], [-1.0, -2.5],
               mask=[False, False])
    gm = gm_contrast_row(row)
    assert math.isnan(gm)


def test_gm_contrast_mean_min_excludes_dilution():
    # 3 divergent tokens: two strongly gold-favoring, one near-zero. mean alone
    # would be diluted; mean_min surfaces the worst (smallest) token.
    # DiD per token: token0=+2.0, token1=+2.0, token2=+0.1
    gm_meta = [0.0, 0.0, 0.0]
    gm_plac = [-2.0, -2.0, -0.1]
    dc_meta = [0.0, 0.0, 0.0]
    dc_plac = [0.0, 0.0, 0.0]
    row = _row(gm_meta, gm_plac, dc_meta, dc_plac, mask=[True, True, True])
    plain_mean = (2.0 + 2.0 + 0.1) / 3.0
    gm = gm_contrast_row(row, agg="mean_min", clip_c=4.0, alpha=1.0)
    # mean_min = mean + 1.0*min = plain_mean + 0.1
    assert abs(gm - (plain_mean + 0.1)) < 1e-9
    # mean_min < a pure-max read; it penalizes the weak token (anti-game)
    assert gm < 2.0 + 2.0  # not gamed by the two strong tokens


def test_gm_contrast_length_mismatch_is_nan():
    row = {
        "logp_gold_meta": np.asarray([-1.0, -0.5]),
        "logp_gold_placebo": np.asarray([-1.0]),  # mismatch
        "logp_decoy_meta": np.asarray([-1.0, -0.5]),
        "logp_decoy_placebo": np.asarray([-1.0, -0.5]),
    }
    assert math.isnan(gm_contrast_row(row))


# ── compute_directional_meta_reward (sign-gate) ──────────────────────────────
def test_directional_sign_gate_correct_positive():
    rows = [_row([-0.5], [-2.0], [-3.0], [-2.5], mask=[True], correct=True)]
    r_meta, diag = compute_directional_meta_reward(
        rows, agg="mean_min", clip_c_token=4.0, alpha=0.0, clip_c_gate=4.0)
    assert r_meta[0] > 0.0
    assert diag["failures"][0] is False


def test_directional_sign_gate_wrong_flips_negative():
    # SAME gold-favoring DiD, but the rollout was WRONG -> additive head must
    # give NEGATIVE credit (a gold-reaching meta on a wrong rollout is punished).
    rows = [_row([-0.5], [-2.0], [-3.0], [-2.5], mask=[True], correct=False)]
    r_meta, _ = compute_directional_meta_reward(
        rows, agg="mean_min", clip_c_token=4.0, alpha=0.0, clip_c_gate=4.0)
    assert r_meta[0] < 0.0


def test_directional_failed_row_scores_zero_member():
    # empty divergent span -> NaN -> member 0, r_meta 0 (fail-closed).
    rows = [_row([-0.5], [-2.0], [-3.0], [-2.5], mask=[False], correct=True)]
    r_meta, diag = compute_directional_meta_reward(rows)
    assert r_meta[0] == 0.0
    assert diag["failures"][0] is True


def test_directional_no_sign_gate_keeps_negative_did():
    # sign_gate off: a decoy-favoring DiD (negative) survives as negative.
    # gold DiD = -2.0 - (-0.5) = -1.5 ; decoy DiD = -0.5 - (-2.0) = +1.5
    # DiD = -1.5 - 1.5 = -3.0
    rows = [_row([-2.0], [-0.5], [-0.5], [-2.0], mask=[True], correct=True)]
    r_meta, _ = compute_directional_meta_reward(
        rows, agg="mean_min", clip_c_token=4.0, alpha=0.0, clip_c_gate=4.0,
        sign_gate=False)
    assert r_meta[0] < 0.0


# ── rlsd_meta_weight (multiplicative shackle) ────────────────────────────────
def test_rlsd_weight_correct_amplifies():
    # A_corr>0, gm>0 -> w = exp(+1 * 0.5) = 1.6487
    assert abs(rlsd_meta_weight(0.5, +1.0, clip_w=2.0) - math.exp(0.5)) < 1e-9


def test_rlsd_weight_wrong_shackles_below_one():
    # SAME gm>0, but A_corr<0 -> w = exp(-1 * 0.5) = 0.6065 < 1 (shackled).
    w = rlsd_meta_weight(0.5, -1.0, clip_w=2.0)
    assert abs(w - math.exp(-0.5)) < 1e-9
    assert w < 1.0


def test_rlsd_weight_flat_group_is_one():
    # A_corr==0 -> sign 0 -> w=1 (zero gradient when group correctness flat).
    assert rlsd_meta_weight(0.5, 0.0) == 1.0


def test_rlsd_weight_nan_gm_is_neutral():
    assert rlsd_meta_weight(float("nan"), +1.0) == 1.0


def test_rlsd_weight_clip_bounds_gm():
    # gm beyond clip_w is clamped: w = exp(+1 * 2.0) regardless of gm=100.
    assert abs(rlsd_meta_weight(100.0, +1.0, clip_w=2.0) - math.exp(2.0)) < 1e-9


# ── rlsd_meta_factor (lam interpolation) ─────────────────────────────────────
def test_rlsd_factor_lam0_is_identity():
    # lam=0 -> factor 1 (un-shackled Â_corr, byte-identical to additive-off).
    assert rlsd_meta_factor(0.5, +1.0, lam=0.0) == 1.0


def test_rlsd_factor_lam1_is_weight():
    # lam=1 -> factor == w
    assert abs(rlsd_meta_factor(0.5, +1.0, lam=1.0) - math.exp(0.5)) < 1e-9


def test_rlsd_factor_lam_half_interpolates():
    w = rlsd_meta_weight(0.5, +1.0)
    f = rlsd_meta_factor(0.5, +1.0, lam=0.5)
    assert abs(f - (0.5 + 0.5 * w)) < 1e-9


# ── compose backward-compat: byte-identity when new flags OFF ────────────────
def test_compose_byte_identical_when_rlsd_factor_off():
    from src.training.dcpo_region import compose_dcpo_region_advantage
    torch.manual_seed(0)
    B, T = 4, 6
    response_mask = torch.ones(B, T)
    index = np.array([0, 0, 1, 1])
    R_corr = torch.tensor([1.0, 0.0, 1.0, 0.0])
    R_meta = torch.tensor([0.5, -0.5, 0.2, -0.2])
    R_cal = torch.tensor([0.1, 0.2, 0.3, 0.4])
    answer_mask = torch.zeros(B, T); answer_mask[:, 3:] = 1.0
    meta_mask = torch.zeros(B, T); meta_mask[:, 1:3] = 1.0
    conf_mask = torch.zeros(B, T); conf_mask[:, 0] = 1.0
    kw = dict(
        response_mask=response_mask, index=index,
        R_corr=R_corr, R_meta=R_meta, R_cal=R_cal,
        answer_mask=answer_mask, meta_content_mask=meta_mask, conf_mask=conf_mask,
    )
    base, _ = compose_dcpo_region_advantage(**kw)
    # explicitly OFF (None factor) must be byte-identical to not passing it.
    off, _ = compose_dcpo_region_advantage(rlsd_meta_factor_per_row=None, **kw)
    assert torch.equal(base, off)


def test_compose_rlsd_multiply_applies_on_meta_tokens_only():
    from src.training.dcpo_region import compose_dcpo_region_advantage
    B, T = 2, 4
    response_mask = torch.ones(B, T)
    index = np.array([0, 0])
    R_corr = torch.tensor([1.0, -1.0])
    # NONZERO meta term so the multiply is observable (mutation-sensitive):
    # with w_meta=1.0 and R_meta=[+1,-1] the base meta tokens are +/-1.0, so the
    # *1.5 / *0.5 assertions below become non-trivial (0==0 would hide the bug).
    R_meta = torch.tensor([1.0, -1.0])
    R_cal = torch.zeros(B)
    answer_mask = torch.zeros(B, T); answer_mask[:, 2:] = 1.0
    meta_mask = torch.zeros(B, T); meta_mask[:, 0:2] = 1.0
    conf_mask = torch.zeros(B, T)
    factor = np.array([1.5, 0.5], dtype=np.float32)  # per-row multiplicative
    kw = dict(
        response_mask=response_mask, index=index,
        R_corr=R_corr, R_meta=R_meta, R_cal=R_cal,
        answer_mask=answer_mask, meta_content_mask=meta_mask, conf_mask=conf_mask,
        w_meta=1.0,  # full-weight additive meta head → nonzero meta-token advantage
    )
    base, _ = compose_dcpo_region_advantage(**kw)
    mult, _ = compose_dcpo_region_advantage(rlsd_meta_factor_per_row=factor, **kw)
    # GUARD: base meta tokens must be nonzero, else the scale assertions are vacuous.
    assert base[:, 0:2].abs().sum() > 0
    # ANSWER tokens (cols 2,3) unchanged; META tokens (cols 0,1) scaled by factor.
    assert torch.allclose(base[:, 2:], mult[:, 2:])
    # row0 meta tokens *1.5, row1 meta tokens *0.5
    assert torch.allclose(mult[0, 0:2], base[0, 0:2] * 1.5)
    assert torch.allclose(mult[1, 0:2], base[1, 0:2] * 0.5)


# ── gm over-emission (selectivity) penalty: CF dcpo_w_over reuse ──────────────
def test_gm_over_emission_off_is_zero_byte_identical():
    # w_over=0 (default) -> all-zero penalty so correctness is byte-identical.
    over = gm_over_emission_penalty(
        meta_member=[1.0, 1.0, 1.0, 1.0],
        c_with=[1.0, 1.0, 0.0, 1.0],
        group_index=[0, 0, 1, 1],
        w_over=0.0,
    )
    assert np.array_equal(over, np.zeros(4, dtype=np.float32))


def test_gm_over_emission_penalizes_already_solved_group():
    # group 0 fully correct (acc 1.0 >= threshold 1.0) -> both member rows pay
    # w_over; group 1 not fully solved (acc 0.5) -> no penalty.
    over = gm_over_emission_penalty(
        meta_member=[1.0, 1.0, 1.0, 1.0],
        c_with=[1.0, 1.0, 0.0, 1.0],
        group_index=[0, 0, 1, 1],
        w_over=0.05,
    )
    assert np.allclose(over, [0.05, 0.05, 0.0, 0.0])


def test_gm_over_emission_only_member_rows_pay():
    # a non-member row (meta_member=0) in an already-solved group is NOT charged.
    over = gm_over_emission_penalty(
        meta_member=[1.0, 0.0],
        c_with=[1.0, 1.0],
        group_index=[0, 0],
        w_over=0.05,
    )
    assert np.allclose(over, [0.05, 0.0])


def test_gm_over_emission_threshold_partial_group():
    # threshold 0.6: a group at acc 0.5 is below -> no penalty even with w_over.
    over = gm_over_emission_penalty(
        meta_member=[1.0, 1.0],
        c_with=[1.0, 0.0],
        group_index=[0, 0],
        w_over=0.05,
        over_threshold=0.6,
    )
    assert np.allclose(over, [0.0, 0.0])
