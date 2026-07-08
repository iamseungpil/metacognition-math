r"""PMI-SHIFT-ACROSS-META — sign-reversal DCPO reward (pure numpy core).

Design (asymmetric counterfactual meta-RL, 2026-06-25 spec §3 Layer-2 refinement,
PMI-SHIFT-ACROSS-META variant):

We measure the META block's belief-update by scoring the GOLD-vs-DECOY answer
log-odds at TWO positions of the SAME rollout:

    PMI_open  = logp(gold | body-before-<|meta|>) − logp(decoy | body-before-<|meta|>)
    PMI_close = logp(gold | body+<|meta|>...<|/meta|>) − logp(decoy | body+...<|/meta|>)

PMI_open is the model's belief BEFORE reading its own meta block; PMI_close is the
belief AFTER. The SHIFT = PMI_close − PMI_open is the meta's *causal* contribution
to moving belief from one answer toward the other. We reward SIGN REVERSALS
asymmetrically:

  - decoy→gold (PMI_open < 0 and PMI_close > 0): the meta CORRECTED a wrong-leaning
    belief → +R_save_big bonus (this is exactly the "meta saves a wrong answer" event).
  - gold→decoy (PMI_open > 0 and PMI_close < 0): the meta DERAILED a right-leaning
    belief → −R_derail_big penalty, with derail magnitude ≥ save (asymmetric — the
    §4 design penalizes derail harder than it rewards save).
  - no reversal: a clipped, scaled continuous shift term (small credit/debit for the
    magnitude of the belief move in the gold direction).

This differs from the gm-contrast head (which scores a SINGLE position, after
<|/meta|>, against a contentless PLACEBO meta): PMI_SHIFT contrasts the model's OWN
two positions (open vs close), so it isolates the meta's *intervention* on belief,
not the meta-vs-placebo text presence.

CONFOUND CAVEAT (do NOT over-claim — review 2026-06-25): the signal test
(src/eval/pmi_shift_signal.py) stratifies on whether the model's own final answer
equals gold, which rules out ONE confound mechanism — the model mechanically
favoring its OWN rollout answer (answer-identity / A.6). It does NOT, by itself,
verify that meta-CONTENT (vs meta-PRESENCE-as-confidence) drives the shift: a model
that learned `emit-meta → boost-confidence` with `correctness ≈ confidence` could
still pass the own≠gold stratification while the shift is NOT driven by reading the
meta. Two additional checks are therefore required and live in the signal test:
(1) a PLACEBO test — shift(real_meta) >> shift(corrupted/randomized_meta) on the
own≠gold subset; and (2) a SAFE-DEFAULT sub-stratification — split own≠gold into
gold-is-the-default vs gold-is-not-default, since a default-to-correct correlation
also passes the bare own≠gold test. Genuine content-driven update should survive
both. On the TRAINING side, the reward additionally requires (a) a non-empty meta
block with actual content between the tags and (b) the meta NOT being a near-exact
duplicate of the body reasoning (content-integrity filter in verl_sdc.py), so a
contentless/derivative meta cannot earn shift credit via mere presence.

All knobs default to a no-op-safe regime and NaN/empty inputs FAIL CLOSED to 0.0 so
a poisoned row can never NaN its centering-group siblings downstream.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "pmi_shift_reward",
    "compute_pmi_shift_reward",
]


def pmi_shift_reward(
    pmi_open: float,
    pmi_close: float,
    *,
    scale: float = 1.0,
    reversal_save: float = 1.0,
    reversal_derail: float = 2.0,
    clip: float = 2.0,
    reversal_min_magnitude: float = 0.0,
) -> float:
    r"""Asymmetric sign-reversal PMI-SHIFT reward for ONE rollout.

    Args:
        pmi_open:  gold-minus-decoy logp at the meta-OPEN context (belief BEFORE meta).
        pmi_close: gold-minus-decoy logp at the meta-CLOSE context (belief AFTER meta).
        scale:     magnitude multiplier on the clipped continuous shift term. It is
                   taken as abs() so a config sign error cannot silently invert the
                   reward (per the asymmetry design `scale` is a magnitude, not a
                   signed multiplier — matching clip/reversal_save/reversal_derail).
        reversal_save:   bonus magnitude added on a decoy→gold sign reversal (>=0).
        reversal_derail: penalty magnitude subtracted on a gold→decoy sign reversal
                         (>=0); should be >= reversal_save (asymmetric derail>save).
        clip:      symmetric clip on the raw shift = pmi_close − pmi_open.
        reversal_min_magnitude: minimum |pmi_open| AND |pmi_close| required before a
                   sign reversal is counted (>=0). This removes the gradient
                   discontinuity / spurious-marginal-crossing incentive at the zero
                   crossing (review: a −0.9→0.0 move gives no bonus, but a 0.0→0.1
                   move would otherwise jump +reversal_save on top of the continuous
                   term). With the threshold, a reversal must clear |pmi| > eps on
                   BOTH sides — a genuine belief flip, not a marginal jitter across 0.

    Returns:
        R_shift float. NaN/inf in either input -> 0.0 (fail-closed). The continuous
        term is abs(scale)·clip(shift, −clip, clip); on top of it, a decoy→gold
        reversal adds +reversal_save and a gold→decoy reversal subtracts
        −reversal_derail. Zero-crossings exactly AT 0 (pmi==0) are NOT counted as
        reversals (a flat belief is not a correction nor a derail), and — when
        reversal_min_magnitude>0 — marginal crossings within ±eps of 0 are likewise
        NOT counted (no discontinuous bonus for a barely-over-zero crossing).
    """
    o = float(pmi_open)
    c = float(pmi_close)
    if not (np.isfinite(o) and np.isfinite(c)):
        return 0.0
    shift = c - o
    cl = abs(float(clip))
    cont = abs(float(scale)) * float(np.clip(shift, -cl, cl))
    eps = abs(float(reversal_min_magnitude))
    bonus = 0.0
    # A reversal requires a genuine flip past the eps band on BOTH positions, not a
    # marginal jitter across exactly 0 (removes the zero-crossing discontinuity).
    if o < -eps and c > eps:
        # decoy→gold: was leaning decoy, now leaning gold = SAVE.
        bonus = abs(float(reversal_save))
    elif o > eps and c < -eps:
        # gold→decoy: was leaning gold, now leaning decoy = DERAIL.
        bonus = -abs(float(reversal_derail))
    out = cont + bonus
    if not np.isfinite(out):
        return 0.0
    return float(out)


def compute_pmi_shift_reward(
    rows,
    *,
    scale: float = 1.0,
    reversal_save: float = 1.0,
    reversal_derail: float = 2.0,
    clip: float = 2.0,
    reversal_min_magnitude: float = 0.0,
):
    r"""Turn scored PMI-SHIFT rows into per-row R_shift + diagnostics.

    Each row carries scalar `pmi_open` and `pmi_close` (gold-minus-decoy logp at the
    two positions) and an optional `member` flag. A row missing either scalar, or
    non-finite, FAILS CLOSED (R 0.0, member 0).

    Returns (r_shift float32[len(rows)], diagnostics) where diagnostics carries:
        raw_shift   (per-row pmi_close−pmi_open, NaN on fail),
        failures    (per-row bool),
        n_save      (count of decoy→gold reversals),
        n_derail    (count of gold→decoy reversals).
    """
    n = len(rows)
    r_shift = np.zeros(n, dtype=np.float32)
    diag = {"raw_shift": [], "failures": [], "n_save": 0, "n_derail": 0}
    eps = abs(float(reversal_min_magnitude))
    for i, row in enumerate(rows):
        po = row.get("pmi_open", None)
        pc = row.get("pmi_close", None)
        ok = (po is not None and pc is not None
              and np.isfinite(float(po)) and np.isfinite(float(pc)))
        if not ok:
            diag["raw_shift"].append(float("nan"))
            diag["failures"].append(True)
            continue
        po = float(po)
        pc = float(pc)
        diag["raw_shift"].append(pc - po)
        diag["failures"].append(False)
        # Count reversals on the SAME eps-gated criterion the reward uses, so the
        # n_save/n_derail diagnostics match the bonuses actually paid out.
        if po < -eps and pc > eps:
            diag["n_save"] += 1
        elif po > eps and pc < -eps:
            diag["n_derail"] += 1
        r_shift[i] = pmi_shift_reward(
            po, pc, scale=scale, reversal_save=reversal_save,
            reversal_derail=reversal_derail, clip=clip,
            reversal_min_magnitude=reversal_min_magnitude)
    return r_shift, diag
