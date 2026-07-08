"""TDD for the asymmetric counterfactual meta-RL reward
(`dcpo_rmeta_source: asym_cf`, design 2026-06-25).

PURE PYTHON + numpy (Layer-1 gate is numpy-only, sibling of dcpo_directional.py);
the compose production-parity test pulls in torch via dcpo_region. So the file
imports torch lazily inside that one test, the gate tests stay numpy-only.

Spec (docs/superpowers/specs/2026-06-25-asymmetric-counterfactual-meta-rl-design.md):
  Layer 1 GATE — per group with c0=P(correct|meta-OFF), c1=P(correct|meta-ON):
    R_gate = alpha*max(0, c1-c0) - beta*max(0, c0-c1) - gamma*(c0>=t and c1>=t)
    beta>alpha (defaults alpha=1.0, beta=2.5, gamma=0.1, t=0.99); DERAIL term
    active only if (c0-c1) > margin (default 0.1); beta clipped; emit-floor.
    Per-row student-confidence down-weights the emit reward (suppress where
    likely-already-correct).
  Layer 2 CONTENT — independence (decoy-DiD/PMI) credited ONLY for emitted +
    net-positive groups; gate says wrong-to-emit => content reward 0.
"""
import numpy as np
import pytest

try:  # the compose production-parity test needs torch; gate tests stay numpy-only
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

requires_torch = pytest.mark.skipif(
    not _HAS_TORCH, reason="torch not available (compose routing integration test)")

try:  # the production-routing parity test imports the full verl_sdc_utils stack.
    # Reuse the verl/omegaconf/ray auto-stub installed by test_dcpo_v3_cf (same
    # pattern as test_dcpo_v4_integration) so the routing chain runs without the
    # heavy training deps installed — only torch is genuinely required.
    import tests.test_dcpo_v3_cf  # noqa: F401  (installs the auto-stub finder)
    from src.training.verl_sdc_utils import (  # noqa: F401
        _compute_dcpo_region_advantage as _PROD_ADV,
    )
    _HAS_PROD_STACK = True
except Exception:  # pragma: no cover — torch not installed in pure-numpy unit env
    _HAS_PROD_STACK = False

requires_prod_stack = pytest.mark.skipif(
    not (_HAS_TORCH and _HAS_PROD_STACK),
    reason="torch+verl_sdc_utils production stack not available")

from src.training.dcpo_asymcf import (
    asym_cf_gate_scalar,
    compute_asym_cf_gate,
    gate_emit_decision,
    apply_content_gate,
)


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — the four counterfactual regimes
# ─────────────────────────────────────────────────────────────────────────────
def test_save_regime_positive():
    """SAVE (c0=0 -> c1=1): meta turns wrong into right => positive R_gate."""
    r = asym_cf_gate_scalar(c0=0.0, c1=1.0, alpha=1.0, beta=2.5, gamma=0.1,
                            t=0.99, margin=0.1)
    assert r == pytest.approx(1.0)  # alpha * (c1-c0)


def test_derail_regime_strong_negative():
    """DERAIL (c0=1 -> c1=0): meta breaks a right answer => strong negative."""
    r = asym_cf_gate_scalar(c0=1.0, c1=0.0, alpha=1.0, beta=2.5, gamma=0.1,
                            t=0.99, margin=0.1)
    assert r == pytest.approx(-2.5)  # -beta * (c0-c1)


def test_derail_magnitude_exceeds_save_for_symmetric():
    """beta>alpha => |DERAIL| > |SAVE| for the SAME counterfactual magnitude."""
    save = asym_cf_gate_scalar(c0=0.2, c1=0.8, alpha=1.0, beta=2.5, gamma=0.1,
                               t=0.99, margin=0.1)
    derail = asym_cf_gate_scalar(c0=0.8, c1=0.2, alpha=1.0, beta=2.5, gamma=0.1,
                                 t=0.99, margin=0.1)
    assert abs(derail) > abs(save)
    assert save > 0 and derail < 0


def test_waste_regime_small_negative():
    """WASTE (c0=1 -> c1=1, both >= t): needless emission => small -gamma."""
    r = asym_cf_gate_scalar(c0=1.0, c1=1.0, alpha=1.0, beta=2.5, gamma=0.1,
                            t=0.99, margin=0.1)
    assert r == pytest.approx(-0.1)


def test_gamma_default_is_strong():
    """The WASTE penalty default is the STRONG 0.5 (suppress wasteful emission):
    goal (2) — gamma 0.1 -> 0.5 so WASTE survives to the composed advantage.
    Calling with no gamma must yield exactly -0.5 on a pure WASTE group."""
    r = asym_cf_gate_scalar(c0=1.0, c1=1.0)  # all defaults
    assert r == pytest.approx(-0.5)


def test_emit_floor_default_is_zero_no_positivity_floor():
    """The reward-positivity floor is GONE by default (goal 1): with the default
    emit_floor the WASTE scalar is <= -gamma and the DERAIL scalar <= -beta — the
    gate can express true suppression, NOT clamped non-negative."""
    waste = asym_cf_gate_scalar(c0=1.0, c1=1.0)         # defaults: gamma 0.5
    derail = asym_cf_gate_scalar(c0=1.0, c1=0.0)        # defaults: beta 2.5
    assert waste <= -0.5 + 1e-6
    assert derail <= -2.5 + 1e-6


def test_neutral_regime_near_zero():
    """NEUTRAL (c0=0 -> c1=0, both wrong): no signal => ~0."""
    r = asym_cf_gate_scalar(c0=0.0, c1=0.0, alpha=1.0, beta=2.5, gamma=0.1,
                            t=0.99, margin=0.1)
    assert r == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Margin guard, beta clip, emit floor
# ─────────────────────────────────────────────────────────────────────────────
def test_margin_guard_suppresses_small_derail():
    """A derail SMALLER than the margin does not fire the beta penalty (noise)."""
    # c0-c1 = 0.05 < margin 0.1 -> DERAIL term inactive
    r = asym_cf_gate_scalar(c0=0.55, c1=0.50, alpha=1.0, beta=2.5, gamma=0.1,
                            t=0.99, margin=0.1)
    assert r == pytest.approx(0.0)
    # just over the margin -> fires
    r2 = asym_cf_gate_scalar(c0=0.65, c1=0.50, alpha=1.0, beta=2.5, gamma=0.1,
                             t=0.99, margin=0.1)
    assert r2 < 0


def test_margin_boundary_exact():
    """At EXACTLY the margin (c0-c1 == 0.10) the strict `>` does NOT fire the
    derail penalty (a `>=` implementation would wrongly fire here)."""
    r = asym_cf_gate_scalar(c0=0.60, c1=0.50, alpha=1.0, beta=2.5, gamma=0.1,
                            t=0.99, margin=0.1)
    assert r == pytest.approx(0.0)


def test_beta_magnitude_clipped():
    """The DERAIL penalty is clipped so a single huge swing cannot dominate."""
    r = asym_cf_gate_scalar(c0=1.0, c1=0.0, alpha=1.0, beta=2.5, gamma=0.1,
                            t=0.99, margin=0.1, beta_clip=1.0)
    assert r == pytest.approx(-1.0)  # clipped at -beta_clip


def test_emit_floor_lifts_total_abstention():
    """A small emit-floor is added to every emitting member (anti-collapse)."""
    no_floor = asym_cf_gate_scalar(c0=0.0, c1=0.0, alpha=1.0, beta=2.5,
                                   gamma=0.1, t=0.99, margin=0.1, emit_floor=0.0)
    with_floor = asym_cf_gate_scalar(c0=0.0, c1=0.0, alpha=1.0, beta=2.5,
                                     gamma=0.1, t=0.99, margin=0.1,
                                     emit_floor=0.05)
    assert with_floor - no_floor == pytest.approx(0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Input validation (review): out-of-range / non-finite inputs guarded
# ─────────────────────────────────────────────────────────────────────────────
def test_out_of_range_correctness_returns_zero():
    """c0/c1 outside [0,1] (garbage) -> 0.0, no meaningless reward."""
    assert asym_cf_gate_scalar(c0=-0.1, c1=0.5) == pytest.approx(0.0)
    assert asym_cf_gate_scalar(c0=0.5, c1=1.5) == pytest.approx(0.0)


def test_confidence_clamped_out_of_range():
    """An out-of-range confidence is clamped to [0,1] (does not amplify the save
    reward or drive `scale` negative)."""
    # confidence=2.0, conf_w=1.0 -> clamped to 1.0 -> scale=max(0,1-1)=0 -> save 0.
    r = asym_cf_gate_scalar(c0=0.2, c1=0.8, alpha=1.0, beta=2.5, gamma=0.1,
                            t=0.99, margin=0.1, confidence=2.0, conf_w=1.0)
    assert r == pytest.approx(0.0)
    # confidence=-1.0 -> clamped to 0.0 -> scale=1 -> full save reward (no boost).
    r2 = asym_cf_gate_scalar(c0=0.2, c1=0.8, alpha=1.0, beta=2.5, gamma=0.1,
                             t=0.99, margin=0.1, confidence=-1.0, conf_w=1.0)
    base = asym_cf_gate_scalar(c0=0.2, c1=0.8, alpha=1.0, beta=2.5, gamma=0.1,
                               t=0.99, margin=0.1)
    assert r2 == pytest.approx(base)


def test_batch_skips_nonfinite_group():
    """A group whose counterfactual is non-finite (corrupt correctness) is skipped
    (member 0, no NaN propagated into R_gate)."""
    c_with = np.array([np.nan, 0.0, 1.0, 1.0], dtype=np.float32)
    with_meta = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    R_gate, member, diag = compute_asym_cf_gate(
        c_with=c_with, with_meta_flag=with_meta, group_index=["g"] * 4,
        alpha=1.0, beta=2.5, gamma=0.1, t=0.99, margin=0.1)
    assert member.sum() == 0.0
    assert np.all(np.isfinite(R_gate)) and np.all(R_gate == 0.0)
    assert (diag["n_save"] + diag["n_derail"] + diag["n_waste"]
            + diag["n_neutral"]) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Confidence down-weighting (suppress emit where likely-already-correct)
# ─────────────────────────────────────────────────────────────────────────────
def test_confidence_downweights_positive_emit_reward():
    """High student confidence shrinks a POSITIVE emit reward (suppress emission
    where the model is likely already correct)."""
    base = asym_cf_gate_scalar(c0=0.2, c1=0.8, alpha=1.0, beta=2.5, gamma=0.1,
                               t=0.99, margin=0.1)
    high_conf = asym_cf_gate_scalar(c0=0.2, c1=0.8, alpha=1.0, beta=2.5,
                                    gamma=0.1, t=0.99, margin=0.1,
                                    confidence=0.95, conf_w=1.0)
    low_conf = asym_cf_gate_scalar(c0=0.2, c1=0.8, alpha=1.0, beta=2.5,
                                   gamma=0.1, t=0.99, margin=0.1,
                                   confidence=0.05, conf_w=1.0)
    assert 0 < high_conf < low_conf <= base


def test_confidence_does_not_weaken_derail_penalty():
    """Confidence must NOT soften the DERAIL penalty (it only suppresses positive
    emission; a confident derail is the worst case, not let off the hook)."""
    derail = asym_cf_gate_scalar(c0=0.8, c1=0.2, alpha=1.0, beta=2.5, gamma=0.1,
                                 t=0.99, margin=0.1)
    derail_hc = asym_cf_gate_scalar(c0=0.8, c1=0.2, alpha=1.0, beta=2.5,
                                    gamma=0.1, t=0.99, margin=0.1,
                                    confidence=0.95, conf_w=1.0)
    assert derail_hc == pytest.approx(derail)


# ─────────────────────────────────────────────────────────────────────────────
# Batch driver — per-group c0/c1 from with/without arm split
# ─────────────────────────────────────────────────────────────────────────────
def test_compute_asym_cf_gate_batch_arms():
    """Group with a SAVE pattern: without-arm wrong (c0=0), with-arm right (c1=1)
    -> with-meta members carry a positive R_gate; without-meta members carry 0
    and are non-members."""
    # group g: rows 0,1 = without-meta (both wrong), rows 2,3 = with-meta (both right)
    c_with = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    with_meta = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    gid = ["g", "g", "g", "g"]
    R_gate, member, diag = compute_asym_cf_gate(
        c_with=c_with, with_meta_flag=with_meta, group_index=gid,
        alpha=1.0, beta=2.5, gamma=0.1, t=0.99, margin=0.1)
    assert member[0] == 0.0 and member[1] == 0.0
    assert member[2] == 1.0 and member[3] == 1.0
    assert R_gate[2] == pytest.approx(1.0) and R_gate[3] == pytest.approx(1.0)
    assert R_gate[0] == 0.0 and R_gate[1] == 0.0
    assert diag["n_save"] >= 1


def test_compute_asym_cf_gate_derail_group():
    """DERAIL group: without-arm right (c0=1), with-arm wrong (c1=0) -> with-meta
    members carry the strong negative -beta."""
    c_with = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)
    with_meta = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    gid = ["g"] * 4
    R_gate, member, diag = compute_asym_cf_gate(
        c_with=c_with, with_meta_flag=with_meta, group_index=gid,
        alpha=1.0, beta=2.5, gamma=0.1, t=0.99, margin=0.1)
    assert R_gate[2] == pytest.approx(-2.5)
    assert R_gate[3] == pytest.approx(-2.5)
    assert diag["n_derail"] >= 1


def test_compute_asym_cf_gate_no_sibling_skips():
    """A group with no without-meta sibling => delta undefined => member 0."""
    c_with = np.array([1.0, 1.0], dtype=np.float32)
    with_meta = np.array([1.0, 1.0], dtype=np.float32)  # all with-meta
    R_gate, member, diag = compute_asym_cf_gate(
        c_with=c_with, with_meta_flag=with_meta, group_index=["g", "g"],
        alpha=1.0, beta=2.5, gamma=0.1, t=0.99, margin=0.1)
    assert member.sum() == 0.0
    assert np.all(R_gate == 0.0)
    # a skipped (sibling-less) group must NOT increment any regime counter.
    assert (diag["n_save"] + diag["n_derail"] + diag["n_waste"]
            + diag["n_neutral"]) == 0


def test_compute_asym_cf_gate_confidence_per_row():
    """Per-row confidence array down-weights the positive emit reward per member."""
    c_with = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    with_meta = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    conf = np.array([0.0, 0.0, 0.9, 0.1], dtype=np.float32)
    R_gate, member, _ = compute_asym_cf_gate(
        c_with=c_with, with_meta_flag=with_meta, group_index=["g"] * 4,
        alpha=1.0, beta=2.5, gamma=0.1, t=0.99, margin=0.1,
        confidence=conf, conf_w=1.0)
    # EXACT magnitudes (not just ordering): save=alpha*(c1-c0)=1.0, scaled by
    # (1 - conf_w*conf). idx2 conf=0.9 -> 0.1; idx3 conf=0.1 -> 0.9.
    assert R_gate[2] == pytest.approx(0.1)
    assert R_gate[3] == pytest.approx(0.9)


def test_compute_asym_cf_gate_confidence_does_not_suppress_derail_batch():
    """Confidence must NOT change the DERAIL penalty for with-meta members
    (only the positive SAVE reward is down-weighted)."""
    c_with = np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32)  # DERAIL group
    with_meta = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    conf_hi = np.array([0.0, 0.0, 0.95, 0.95], dtype=np.float32)
    conf_lo = np.array([0.0, 0.0, 0.05, 0.05], dtype=np.float32)
    R_hi, _, _ = compute_asym_cf_gate(
        c_with=c_with, with_meta_flag=with_meta, group_index=["g"] * 4,
        alpha=1.0, beta=2.5, gamma=0.1, t=0.99, margin=0.1,
        confidence=conf_hi, conf_w=1.0)
    R_lo, _, _ = compute_asym_cf_gate(
        c_with=c_with, with_meta_flag=with_meta, group_index=["g"] * 4,
        alpha=1.0, beta=2.5, gamma=0.1, t=0.99, margin=0.1,
        confidence=conf_lo, conf_w=1.0)
    assert R_hi[2] == pytest.approx(-2.5) and R_hi[3] == pytest.approx(-2.5)
    assert R_lo[2] == pytest.approx(-2.5) and R_lo[3] == pytest.approx(-2.5)


def test_compute_asym_cf_gate_waste_regime():
    """WASTE batch (c0=c1=1, both >= t): with-meta members get -gamma and the
    n_waste counter increments (n_save/n_derail/n_neutral stay 0)."""
    c_with = np.ones(4, dtype=np.float32)  # everyone correct -> c0=c1=1
    with_meta = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    R_gate, member, diag = compute_asym_cf_gate(
        c_with=c_with, with_meta_flag=with_meta, group_index=["g"] * 4,
        alpha=1.0, beta=2.5, gamma=0.1, t=0.99, margin=0.1)
    assert R_gate[2] == pytest.approx(-0.1) and R_gate[3] == pytest.approx(-0.1)
    assert diag["n_waste"] == 1
    assert diag["n_save"] == 0 and diag["n_derail"] == 0 and diag["n_neutral"] == 0


def test_compute_asym_cf_gate_neutral_regime():
    """NEUTRAL batch (c0==c1, both below t, both wrong): no SAVE/DERAIL/WASTE
    signal -> R~0 and n_neutral increments (others stay 0). The classifier's
    NEUTRAL bucket is exactly c0==c1 below ceiling (the |delta|<margin SAVE side
    is a separate small-positive case; here delta==0)."""
    # without rows mean 0.5, with rows mean 0.5 -> delta 0, neither >= t -> NEUTRAL.
    c_with = np.array(
        [0.0, 1.0,   # without: mean 0.5
         1.0, 0.0],  # with:    mean 0.5
        dtype=np.float32)
    with_meta = np.array([0, 0, 1, 1], dtype=np.float32)
    R_gate, member, diag = compute_asym_cf_gate(
        c_with=c_with, with_meta_flag=with_meta, group_index=["g"] * 4,
        alpha=1.0, beta=2.5, gamma=0.1, t=0.99, margin=0.1)
    # with-meta members: c0==c1 (no save, no derail, not waste) -> reward 0
    for i in (2, 3):
        assert R_gate[i] == pytest.approx(0.0)
    assert diag["n_neutral"] == 1
    assert diag["n_save"] == 0 and diag["n_derail"] == 0 and diag["n_waste"] == 0


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — content gated by the Layer-1 emit decision
# ─────────────────────────────────────────────────────────────────────────────
def test_gate_emit_decision_save_is_emit():
    """gate_emit_decision: SAVE group => emit=1 (net-positive); DERAIL => 0."""
    assert gate_emit_decision(c0=0.0, c1=1.0, margin=0.1) == 1.0
    assert gate_emit_decision(c0=1.0, c1=0.0, margin=0.1) == 0.0
    # WASTE (1->1) is not a net-positive emit
    assert gate_emit_decision(c0=1.0, c1=1.0, margin=0.1) == 0.0


def test_apply_content_gate_zeros_wrong_to_emit():
    """Layer-2 content reward is zeroed where the gate says wrong-to-emit."""
    content = np.array([0.5, 0.8, -0.3, 0.4], dtype=np.float32)
    emit_ok = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    gated = apply_content_gate(content, emit_ok)
    # rows where emit_ok==0 are zeroed; others pass through
    assert gated[0] == pytest.approx(0.5)
    assert gated[1] == pytest.approx(0.0)
    assert gated[2] == pytest.approx(-0.3)
    assert gated[3] == pytest.approx(0.0)


def test_apply_content_gate_byte_identical_all_emit():
    """All-emit gate => content passes through unchanged."""
    content = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    gated = apply_content_gate(content, np.ones(3, dtype=np.float32))
    assert np.allclose(gated, content)


def test_apply_content_gate_invalid_values():
    """apply_content_gate rejects a non-binary / NaN emit_ok (must not silently
    produce NaN content rewards)."""
    content = np.array([0.5, 0.8], dtype=np.float32)
    with pytest.raises(ValueError):
        apply_content_gate(content, np.array([0.7, 1.0], dtype=np.float32))
    with pytest.raises(ValueError):
        apply_content_gate(content, np.array([np.nan, 1.0], dtype=np.float32))


# ─────────────────────────────────────────────────────────────────────────────
# PRODUCTION-PARITY: the asym_cf gate head ACTUALLY changes compose advantage
# (anti-inert / gs190-trap test). Pulls in torch via dcpo_region.
# ─────────────────────────────────────────────────────────────────────────────
@requires_torch
def test_compose_dcpo_region_advantage_ans_meta_non_inert():
    """UNIT test of compose: routing R_gate via the cf_group ANSWER-region kwargs
    CHANGES the composed advantage vs the head off (anti-inert / gs190 trap), the
    change lands ONLY on ANSWER tokens of with-meta MEMBER rows, non-member rows
    are untouched, and a DERAIL (negative R_gate) reduces the advantage.

    NOTE: this exercises compose() directly with R_ans_meta passed in; the FULL
    production routing chain (verl_sdc writes dcpo_ans_meta -> verl_sdc_utils reads
    & passes -> compose) is covered by
    test_asym_cf_production_routing_chain_non_inert below.
    """
    import torch
    from src.training.dcpo_region import compose_dcpo_region_advantage

    B, T = 4, 6
    ans = torch.tensor([[1, 1, 0, 0, 0, 0]] * B, dtype=torch.float32)
    meta_c = torch.tensor([[0, 0, 0, 1, 1, 0]] * B, dtype=torch.float32)
    conf = torch.tensor([[0, 0, 0, 0, 1, 0]] * B, dtype=torch.float32)
    rm = torch.ones(B, T, dtype=torch.float32)
    R_corr = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    R_meta = np.zeros(B, dtype=np.float32)
    R_cal = np.zeros(B, dtype=np.float32)
    index = ["g"] * B

    base_kwargs = dict(
        response_mask=rm, index=index, R_corr=R_corr, R_meta=R_meta, R_cal=R_cal,
        answer_mask=ans, meta_content_mask=meta_c, conf_mask=conf,
        w_corr=1.0, w_meta=0.5, w_cal=0.3,
    )
    A_off, _ = compose_dcpo_region_advantage(**base_kwargs)

    # rows 2,3 are with-meta members (R_gate +1 / -1); rows 0,1 are non-members.
    R_gate = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
    member = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    A_on, _ = compose_dcpo_region_advantage(
        **base_kwargs,
        R_ans_meta=R_gate, w_ans_meta=1.0, ans_meta_member_mask=member,
    )
    assert not torch.equal(A_off, A_on), "asym_cf gate head is INERT (gs190 trap)"
    # (1) non-member rows (0,1) are NOT affected by R_ans_meta
    assert torch.equal(A_off[0], A_on[0])
    assert torch.equal(A_off[1], A_on[1])
    # (2) the change lands on ANSWER tokens (cols 0,1), not META/CONF (cols 2-5)
    assert not torch.equal(A_off[2, :2], A_on[2, :2])
    assert torch.equal(A_off[2, 2:], A_on[2, 2:])
    assert torch.equal(A_off[3, 2:], A_on[3, 2:])
    # (3) group-mean-subtract over with-arm members (rows 2,3): mean(+1,-1)=0, so
    #     the centered head is +0.5/-0.5; (4) the DERAIL row (3, R_gate<mean)
    #     gets a LOWER answer advantage than the SAVE row (2).
    assert A_on[2, 0] > A_on[3, 0]


@requires_prod_stack
def test_asym_cf_production_routing_chain_non_inert():
    """TRUE production-parity: compute_asym_cf_gate -> write dcpo_ans_meta into the
    non_tensor_batch -> verl_sdc_utils._compute_dcpo_region_advantage reads & routes
    it via _cfgroup_kwargs (w_ans_meta=dcpo_w_ans_meta) -> compose -> the composed
    advantage DIFFERS from the head-absent baseline. Catches a routing bug in
    verl_sdc_utils OR the dcpo_w_ans_meta knob (NOT just compose in isolation)."""
    import torch
    from src.training.verl_sdc_utils import _compute_dcpo_region_advantage

    B, T = 4, 6
    # SAVE on rows 2,3 (with-meta), without rows 0,1 wrong -> R_gate>0 on 2,3.
    # Per-row confidence (production feature) makes the two with-meta members'
    # R_gate DIFFER, so the within-group centering does not cancel them (a single
    # group of equal-R_gate members would center to 0 — same as cf_group).
    c_with = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    arm = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    conf = np.array([0.0, 0.0, 0.9, 0.1], dtype=np.float32)
    index = np.array(["g"] * B, dtype=object)
    R_gate, member, _diag = compute_asym_cf_gate(
        c_with=c_with, with_meta_flag=arm, group_index=list(index),
        alpha=1.0, beta=2.5, gamma=0.1, t=0.99, margin=0.1,
        confidence=conf, conf_w=1.0)
    assert member[2] == 1.0 and R_gate[2] > 0  # head actually populated
    assert R_gate[2] != R_gate[3]  # confidence breaks the within-group symmetry

    rm = torch.ones(B, T, dtype=torch.float32)
    batch = {
        "dcpo_answer_mask": torch.tensor([[1, 1, 0, 0, 0, 0]] * B, dtype=torch.float32),
        "dcpo_meta_content_mask": torch.tensor([[0, 0, 0, 1, 1, 0]] * B, dtype=torch.float32),
        "dcpo_conf_mask": torch.tensor([[0, 0, 0, 0, 1, 0]] * B, dtype=torch.float32),
    }

    def _ntb(with_head):
        ntb = {
            "correctness": c_with.copy(),
            "meta_region_utility": np.zeros(B, dtype=np.float32),
            "cal_region_reward": np.zeros(B, dtype=np.float32),
        }
        if with_head:
            ntb["dcpo_ans_meta"] = np.asarray(R_gate, dtype=np.float32)
            ntb["dcpo_ans_member"] = np.asarray(member, dtype=np.float32)
        return ntb

    cfg = {"dcpo_w_corr": 1.0, "dcpo_w_meta": 0.0, "dcpo_w_cal": 0.0,
           "dcpo_w_ans_meta": 1.0}  # content OFF, gate weight independent

    A_off, _ = _compute_dcpo_region_advantage(
        response_mask=rm, index=index, batch=batch,
        non_tensor_batch=_ntb(False), config=cfg)
    A_on, _ = _compute_dcpo_region_advantage(
        response_mask=rm, index=index, batch=batch,
        non_tensor_batch=_ntb(True), config=cfg)
    assert not torch.equal(A_off, A_on), (
        "production routing chain INERT — dcpo_ans_meta did not reach compose")
    # the routed change lands on ANSWER tokens (cols 0,1) of with-meta members.
    assert not torch.equal(A_off[2, :2], A_on[2, :2])


@requires_torch
def test_gate_penalties_survive_centering_whole_group():
    """★LIVE-BUG REPRO (rmeta_neg_rate=0 at step41). The gate scalar is GROUP-
    CONSTANT (every with-arm member of a group shares the same c0/c1-derived
    value). With the OLD routing (group_mean_subtract over the with-arm MEMBER
    mask) a constant centers to ZERO -> SAVE/DERAIL/WASTE all annihilated ->
    rmeta_neg_rate=0, gate cannot suppress. The FIX centers over the WHOLE group
    (with-arm carries R_gate, without-arm carries 0) so the penalty survives, then
    masks credit onto with-arm rows. This test asserts: with conf_w=0 (the live
    default) and a batch of SAVE/DERAIL/WASTE groups, the routed (centered) gate
    advantage has neg_rate > 0 and DERAIL is more negative than WASTE."""
    import torch
    from src.training.dcpo_region import group_mean_subtract

    # Build groups of 4 (2 without, 2 with). conf_w=0 -> all with-arm members of a
    # group share the identical R_gate (the live condition that triggered the bug).
    def grp(c0, c1, gid):
        return ([c0] * 2 + [c1] * 2), [0.0, 0.0, 1.0, 1.0], [gid] * 4

    specs = ([(0.0, 1.0)] * 5      # SAVE
             + [(1.0, 0.0)] * 5    # DERAIL
             + [(1.0, 1.0)] * 20   # WASTE (live: many already-solved)
             + [(0.0, 0.0)] * 5)   # NEUTRAL
    cw, arm, gid = [], [], []
    for k, (c0, c1) in enumerate(specs):
        a, b, g = grp(c0, c1, f"g{k}")
        cw += a; arm += b; gid += g
    cw = np.array(cw, dtype=np.float32)
    arm = np.array(arm, dtype=np.float32)

    R_gate, member, diag = compute_asym_cf_gate(
        c_with=cw, with_meta_flag=arm, group_index=gid,
        alpha=1.0, beta=2.5, gamma=0.5, t=0.99, margin=0.1,
        emit_floor=0.0, conf_w=0.0)

    # OLD routing (center over member/with-arm only) -> annihilation (the bug).
    A_old = group_mean_subtract(R_gate, gid, member=member).reshape(-1).numpy()
    assert float(np.mean(A_old[arm > 0.5] < -1e-6)) == 0.0, (
        "precondition: old member-mask centering should annihilate (the bug)")

    # FIX routing (center over whole group, member=None) then mask to with-arm.
    A_fix = (group_mean_subtract(R_gate, gid, member=None).reshape(-1).numpy()
             * arm)
    on_with = A_fix[arm > 0.5]
    neg_rate = float(np.mean(on_with < -1e-6))
    assert neg_rate > 0.0, "FIX must produce NEGATIVE routed reward (suppression)"
    # DERAIL rows (group c0=1->c1=0) must be MORE negative than WASTE rows.
    derail_rows = A_fix[10 * 4:20 * 4][arm[10 * 4:20 * 4] > 0.5]  # specs[5:10] derail?
    # recompute by spec block: groups 5..9 are DERAIL (rows 20..39), 10..29 WASTE.
    derail_block = A_fix[5 * 4:10 * 4]
    derail_on = derail_block[arm[5 * 4:10 * 4] > 0.5]
    waste_block = A_fix[10 * 4:30 * 4]
    waste_on = waste_block[arm[10 * 4:30 * 4] > 0.5]
    assert derail_on.mean() < waste_on.mean(), "DERAIL must be more negative than WASTE"
    assert derail_on.max() < 0 and waste_on.max() < 0


@requires_torch
def test_whole_group_center_requires_member_mask():
    """GUARD (review 2026-06-26): ans_meta_whole_group_center=True without an
    ans_meta_member_mask must FAIL LOUDLY (assert), not silently skip the masking
    step and leak a spurious -group_mean onto without-arm rows. In production
    asym_cf always supplies the mask; this asserts the precondition for any future
    code path that forgets it."""
    import torch
    from src.training.dcpo_region import compose_dcpo_region_advantage

    B, T = 4, 6
    ans = torch.tensor([[1, 1, 0, 0, 0, 0]] * B, dtype=torch.float32)
    meta_c = torch.tensor([[0, 0, 0, 1, 1, 0]] * B, dtype=torch.float32)
    conf = torch.tensor([[0, 0, 0, 0, 1, 0]] * B, dtype=torch.float32)
    rm = torch.ones(B, T, dtype=torch.float32)
    base_kwargs = dict(
        response_mask=rm, index=["g"] * B,
        R_corr=np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        R_meta=np.zeros(B, dtype=np.float32),
        R_cal=np.zeros(B, dtype=np.float32),
        answer_mask=ans, meta_content_mask=meta_c, conf_mask=conf,
        w_corr=1.0, w_meta=0.5, w_cal=0.3,
    )
    R_gate = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
    # whole-group centering ON but member mask MISSING -> must assert.
    with pytest.raises(AssertionError):
        compose_dcpo_region_advantage(
            **base_kwargs, R_ans_meta=R_gate, w_ans_meta=1.0,
            ans_meta_member_mask=None, ans_meta_whole_group_center=True,
        )
    # with the member mask present it routes fine (no raise).
    member = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    A_base, _ = compose_dcpo_region_advantage(**base_kwargs)
    A_on, _ = compose_dcpo_region_advantage(
        **base_kwargs, R_ans_meta=R_gate, w_ans_meta=1.0,
        ans_meta_member_mask=member, ans_meta_whole_group_center=True,
    )
    # the gate changes SOMETHING (not inert) ...
    assert not torch.equal(A_base, A_on)
    # ... but the masking guarantees without-arm rows (non-members 0,1) get NO
    # gate-derived -group_mean leaked onto their answer tokens (identical to the
    # gate-absent baseline), which is exactly what the member mask protects.
    assert torch.equal(A_base[0], A_on[0])
    assert torch.equal(A_base[1], A_on[1])


def test_waste_term_uses_converted_floats_consistency():
    """Review consistency fix: the WASTE term uses the pre-converted c0f/c1f floats.
    Behavior is unchanged (pure refactor) — a pure-WASTE group still yields -gamma,
    and a just-below-ceiling group is NOT waste."""
    # both at ceiling -> WASTE -gamma
    assert asym_cf_gate_scalar(c0=1.0, c1=1.0, gamma=0.5, t=0.99) == pytest.approx(-0.5)
    # c1 just below the ceiling -> NOT waste (and equal arms -> no save/derail -> 0)
    assert asym_cf_gate_scalar(c0=0.995, c1=0.995, gamma=0.5, t=0.99) == pytest.approx(-0.5)
    assert asym_cf_gate_scalar(c0=0.98, c1=0.98, gamma=0.5, t=0.99) == pytest.approx(0.0)


@requires_prod_stack
def test_asym_cf_production_routing_neg_rate_conf_zero():
    """★LIVE-BUG REPRO through the FULL production chain (compute_asym_cf_gate ->
    write dcpo_ans_meta -> verl_sdc_utils routes -> compose). With conf_w=0 (live
    default) and a WASTE/DERAIL batch, the composed ANSWER-region advantage on
    with-arm DERAIL/WASTE rows must be NEGATIVE (neg_rate>0). BEFORE the fix the
    composed advantage was all-zero (member-mask centering of a group-constant)."""
    import torch
    from src.training.verl_sdc_utils import _compute_dcpo_region_advantage

    # one DERAIL group + one WASTE group, 4 rows each (2 without / 2 with).
    c_with = np.array([1.0, 1.0, 0.0, 0.0,   # DERAIL g0: c0=1, c1=0
                       1.0, 1.0, 1.0, 1.0],  # WASTE  g1: c0=1, c1=1
                      dtype=np.float32)
    arm = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    index = np.array(["g0"] * 4 + ["g1"] * 4, dtype=object)
    B, T = 8, 6
    R_gate, member, _diag = compute_asym_cf_gate(
        c_with=c_with, with_meta_flag=arm, group_index=list(index),
        alpha=1.0, beta=2.5, gamma=0.5, t=0.99, margin=0.1,
        emit_floor=0.0, conf_w=0.0)

    rm = torch.ones(B, T, dtype=torch.float32)
    batch = {
        "dcpo_answer_mask": torch.tensor([[1, 1, 0, 0, 0, 0]] * B, dtype=torch.float32),
        "dcpo_meta_content_mask": torch.tensor([[0, 0, 0, 1, 1, 0]] * B, dtype=torch.float32),
        "dcpo_conf_mask": torch.tensor([[0, 0, 0, 0, 1, 0]] * B, dtype=torch.float32),
    }
    ntb = {
        "correctness": c_with.copy(),
        "meta_region_utility": np.zeros(B, dtype=np.float32),
        "cal_region_reward": np.zeros(B, dtype=np.float32),
        "dcpo_ans_meta": np.asarray(R_gate, dtype=np.float32),
        "dcpo_ans_member": np.asarray(member, dtype=np.float32),
        # the asym_cf path sets this flag (whole-group centering of the gate head).
        "dcpo_ans_meta_whole_group_center": np.ones(B, dtype=np.float32),
    }
    cfg = {"dcpo_w_corr": 0.0, "dcpo_w_meta": 0.0, "dcpo_w_cal": 0.0,
           "dcpo_w_ans_meta": 1.0}  # isolate the gate head
    A_on, _ = _compute_dcpo_region_advantage(
        response_mask=rm, index=index, batch=batch,
        non_tensor_batch=ntb, config=cfg)
    # ANSWER tokens (cols 0,1) of with-arm DERAIL/WASTE rows must be NEGATIVE.
    ans_adv = A_on[:, 0].reshape(-1)  # per-row answer-token advantage
    with_rows = arm > 0.5
    neg_rate = float((ans_adv[with_rows] < -1e-6).float().mean())
    assert neg_rate > 0.0, (
        "production chain: WASTE/DERAIL with-arm rows must get NEGATIVE advantage")
    # DERAIL with-arm rows (2,3) more negative than WASTE with-arm rows (6,7).
    assert ans_adv[2] < ans_adv[6] and ans_adv[3] < ans_adv[7]


@requires_prod_stack
def test_asym_cf_gate_independent_of_content_weight():
    """Review fix: the GATE (dcpo_w_ans_meta) routes even when CONTENT is disabled
    (dcpo_w_meta=0) — the two weights are decoupled, so turning off meta content
    does NOT silently make the gate head inert (the gs190 trap)."""
    import torch
    from src.training.verl_sdc_utils import _compute_dcpo_region_advantage

    B, T = 4, 6
    index = np.array(["g"] * B, dtype=object)
    R_gate = np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32)
    member = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    rm = torch.ones(B, T, dtype=torch.float32)
    batch = {
        "dcpo_answer_mask": torch.tensor([[1, 1, 0, 0, 0, 0]] * B, dtype=torch.float32),
        "dcpo_meta_content_mask": torch.tensor([[0, 0, 0, 1, 1, 0]] * B, dtype=torch.float32),
        "dcpo_conf_mask": torch.tensor([[0, 0, 0, 0, 1, 0]] * B, dtype=torch.float32),
    }
    ntb_base = {
        "correctness": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
        "meta_region_utility": np.zeros(B, dtype=np.float32),
        "cal_region_reward": np.zeros(B, dtype=np.float32),
    }
    ntb_head = dict(ntb_base)
    ntb_head["dcpo_ans_meta"] = R_gate
    ntb_head["dcpo_ans_member"] = member

    # dcpo_w_meta=0 (content OFF) but dcpo_w_ans_meta=1 (gate ON).
    cfg = {"dcpo_w_corr": 1.0, "dcpo_w_meta": 0.0, "dcpo_w_cal": 0.0,
           "dcpo_w_ans_meta": 1.0}
    A_off, _ = _compute_dcpo_region_advantage(
        response_mask=rm, index=index, batch=batch,
        non_tensor_batch=ntb_base, config=cfg)
    A_on, _ = _compute_dcpo_region_advantage(
        response_mask=rm, index=index, batch=batch,
        non_tensor_batch=ntb_head, config=cfg)
    assert not torch.equal(A_off, A_on), (
        "gate head INERT with dcpo_w_meta=0 — weights not decoupled")
