"""Asymmetric counterfactual meta-RL reward (`asym_cf`) core — pure numpy.

NEW + ADDITIVE, framework-light module (numpy only — ZERO verl / torch deps), the
sibling of `dcpo_directional` / `dcpo_pmi`. It does NOT load models or tokenizers:
callers hand it the per-row final-answer correctness (0/1) of a GRPO group whose
rows are split into a meta-ON sub-arm and a meta-OFF sub-arm (the SAME arm-split
machinery `cf_group` uses — `with_meta_flag`, real GRPO members graded for free),
and this module turns the per-group counterfactual (c0=P(correct|meta-OFF),
c1=P(correct|meta-ON)) into the asymmetric GATE reward and the emit decision.

Spec traceability
(docs/superpowers/specs/2026-06-25-asymmetric-counterfactual-meta-rl-design.md):

  LAYER 1 — GATE (timing): per group, continuous asymmetric counterfactual on the
    emit decision (the `<|meta|>`-open token):

        R_gate = alpha*max(0, c1-c0)            # SAVE  (0->1): reward
               - beta *max(0, c0-c1)            # DERAIL(1->0): STRONG penalty, beta>alpha
               - gamma*1[c0>=t and c1>=t]       # WASTE (1->1): small overhead penalty
               #  NEUTRAL (0->0): ~0

    - beta > alpha (asymmetric — derail hurts more than save helps; defaults
      alpha=1.0, beta=2.5, gamma=0.5, t=0.99). gamma raised 0.1->0.5 (2026-06-25
      live fix, goal 2) so the WASTE penalty survives to the composed advantage.
    - DERAIL margin (§4.5): the beta term fires ONLY when (c0-c1) > margin (default
      0.1) — a k-noise guard so a tiny accidental dip is not punished as a derail.
    - beta clip (§4.5): the DERAIL magnitude is clipped to beta_clip so one huge
      counterfactual swing cannot dominate the batch.
    - emit-floor: an OVERRIDABLE knob (default 0.0 = OFF). It is a group-CONSTANT
      added to every emitting member, so under the WHOLE-GROUP centering the gate
      now uses (dcpo_region compose, ans_meta_whole_group_center) it CANCELS in the
      group mean — it can NEVER make a WASTE/DERAIL row's ROUTED reward non-negative
      (the 2026-06-25 live bug: a non-negative reward cannot suppress emission).
      Total-abstention collapse is instead guarded by (a) the gate no longer being
      annihilated (whole-group centering keeps SAVE positive and present) and (b)
      real SAVE cases existing in the frontier-hard data (§6). emit_floor is kept
      only as an opt-in policy-level minimum; it is OFF by default.
    - confidence gate (§3.1 Layer-1 / §4.2): a per-row student self-confidence
      DOWN-WEIGHTS the POSITIVE emit reward (suppress emission where the model is
      likely already correct = the derail-risk region). It NEVER softens the
      DERAIL penalty (a confident derail is the worst case).

  LAYER 2 — CONTENT (quality): the decoy-DiD / PMI independence signal, but
    credited ONLY for groups the gate decided were net-positive-to-emit
    (`gate_emit_decision`): `apply_content_gate` zeros the content reward wherever
    the gate says wrong-to-emit (DERAIL / WASTE / NEUTRAL). Layer 1 decides
    *whether*; Layer 2 scores *what*, only when whether=yes.

Routing (verl_sdc / compose_dcpo_region_advantage): R_gate is routed onto the
emit/answer locus (reusing the cf_group ANSWER-region `R_ans_meta` param — the
counterfactual answer-delta locus), and the Layer-2 content reward is routed onto
META (the meta_region_utility head), gated by Layer 1. Default-OFF: the
`asym_cf` source is opt-in, every other arm stays byte-identical.
"""
from __future__ import annotations

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — asymmetric counterfactual GATE scalar
# ─────────────────────────────────────────────────────────────────────────────
def asym_cf_gate_scalar(
    *,
    c0: float,
    c1: float,
    alpha: float = 1.0,
    beta: float = 2.5,
    gamma: float = 0.5,
    t: float = 0.99,
    margin: float = 0.1,
    beta_clip: float = 1e9,
    emit_floor: float = 0.0,
    confidence: float | None = None,
    conf_w: float = 0.0,
) -> float:
    """The continuous asymmetric counterfactual reward for ONE problem group.

    c0 = P(correct | meta-OFF), c1 = P(correct | meta-ON) (both in [0, 1]).

        R = alpha*max(0, c1-c0) - beta*max(0, c0-c1) - gamma*1[c0>=t and c1>=t]

    with the §4.5 safeguards:
      - DERAIL margin: the beta term fires only when (c0-c1) > margin (noise guard).
      - beta clip: the DERAIL magnitude is clipped to beta_clip (one swing can't
        dominate).
      - emit-floor: a flat +emit_floor added to the member (default 0.0 = OFF). It
        is a group-CONSTANT, so under the whole-group centering the gate routes
        through it CANCELS — it does NOT clamp WASTE/DERAIL non-negative (the live
        bug). Kept only as an opt-in policy-level minimum.
      - confidence down-weight: a high student `confidence` (in [0,1], conf_w>0)
        SHRINKS only the POSITIVE component (1 - conf_w*confidence, clamped >=0) —
        suppress emission where the model is likely already correct. The DERAIL /
        WASTE negatives are untouched (a confident derail stays the worst case).

    Returns a python float. beta MUST exceed alpha for the intended asymmetry, but
    the function does not enforce it (the caller's config does).
    """
    # c0, c1 are empirical correctness probabilities — guard the [0,1] contract
    # (review: do not silently produce meaningless rewards from garbage input).
    c0f, c1f = float(c0), float(c1)
    if not (0.0 <= c0f <= 1.0 and 0.0 <= c1f <= 1.0):
        return 0.0

    up = max(0.0, c1f - c0f)        # SAVE magnitude
    down = max(0.0, c0f - c1f)      # DERAIL magnitude

    save_term = float(alpha) * up
    # confidence down-weights ONLY the positive (save) emission reward.
    # confidence is clamped to [0,1] here so the function is robust to
    # out-of-range callers (review: do not rely on _parse_confidence's clamp).
    if confidence is not None and conf_w:
        conf_clamped = max(0.0, min(1.0, float(confidence)))
        scale = max(0.0, 1.0 - float(conf_w) * conf_clamped)
        save_term *= scale

    # DERAIL: only counts past the margin (noise guard), magnitude clipped.
    # `down` is naturally bounded by the probability range ([0,1]); the real
    # clip is `beta_clip` on the next line (review: removed the vacuous
    # min(down, 1.0) that never fired and obscured the beta_clip intent).
    derail_term = 0.0
    if down > float(margin):
        derail_term = min(float(beta) * down, float(beta_clip))

    # WASTE: both arms already at/above the ceiling t -> needless emission.
    # Use the pre-converted c0f/c1f floats (review: consistency with the rest of
    # the function, lines 105-115 — avoid a second redundant float() conversion).
    waste_term = float(gamma) if (c0f >= float(t) and c1f >= float(t)) else 0.0

    return float(save_term - derail_term - waste_term + float(emit_floor))


def gate_emit_decision(*, c0: float, c1: float, margin: float = 0.1) -> float:
    """Layer-1 emit decision used to GATE Layer-2 content (1.0 = net-positive emit).

    A group is "right-to-emit" only when meta is net-positive past the margin
    (c1 - c0 > margin). DERAIL (c0>c1), WASTE (c0~c1~1) and NEUTRAL (c0~c1~0) are
    all NOT-right-to-emit -> 0.0. Symmetric with the SAVE branch of the gate
    scalar.
    """
    return 1.0 if (float(c1) - float(c0)) > float(margin) else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Layer 1 — batch driver over a with/without arm-split GRPO group
# ─────────────────────────────────────────────────────────────────────────────
def compute_asym_cf_gate(
    *,
    c_with,
    with_meta_flag,
    group_index,
    alpha: float = 1.0,
    beta: float = 2.5,
    gamma: float = 0.5,
    t: float = 0.99,
    margin: float = 0.1,
    beta_clip: float = 1e9,
    emit_floor: float = 0.0,
    confidence=None,
    conf_w: float = 0.0,
):
    """Per-group asymmetric counterfactual GATE reward over an arm-split batch.

    `c_with` is the standard final-answer correctness (0/1) for EVERY row — the
    meta-OFF (without-arm) rows are real GRPO group members graded for free, so
    c_with on a without-arm row IS correct_without (identical contract to
    `compute_cf_group_heads`). For each group g:
        c0 = mean(c_with over with_meta_flag==0 rows)   # P(correct | meta-OFF)
        c1 = mean(c_with over with_meta_flag==1 rows)   # P(correct | meta-ON)
    Every with-meta member of g gets `asym_cf_gate_scalar(c0, c1, ...)`; without-
    meta rows get 0 / member 0; groups missing either arm get member 0 (delta
    undefined, conservative no-gradient).

    `confidence` (optional, per-row array in [0,1]) down-weights the positive emit
    reward per member (conf_w>0). Returns (R_gate float32[B], member float32[B],
    diagnostics) — diagnostics carries n_save / n_derail / n_waste / n_neutral
    group counts + per-group emit_decision (for the Layer-2 gate).
    """
    cw = np.asarray(c_with, dtype=np.float32).reshape(-1)
    wm = np.asarray(with_meta_flag, dtype=np.float32).reshape(-1)
    B = cw.shape[0]
    conf = None
    if confidence is not None:
        conf = np.asarray(confidence, dtype=np.float32).reshape(-1)
        if conf.shape[0] != B:
            raise ValueError(
                f"confidence length {conf.shape[0]} != batch {B}")
    gid = list(
        group_index.tolist() if hasattr(group_index, "tolist") else group_index
    )
    gid = [str(g) for g in gid]

    R_gate = np.zeros(B, dtype=np.float32)
    member = np.zeros(B, dtype=np.float32)
    emit_decision = np.zeros(B, dtype=np.float32)
    diag = {"n_save": 0, "n_derail": 0, "n_waste": 0, "n_neutral": 0}

    groups: dict = {}
    for i in range(B):
        groups.setdefault(gid[i], []).append(i)

    for members in groups.values():
        with_rows = [i for i in members if wm[i] > 0.5]
        without_rows = [i for i in members if wm[i] <= 0.5]
        if not with_rows or not without_rows:
            continue  # no counterfactual sibling -> skip (member 0)
        c0 = float(np.mean([cw[i] for i in without_rows]))
        c1 = float(np.mean([cw[i] for i in with_rows]))
        # NaN/inf guard: a non-finite counterfactual (corrupt correctness) would
        # propagate NaN into rewards/advantages. Skip the group (member 0).
        if not (np.isfinite(c0) and np.isfinite(c1)):
            continue
        emit = gate_emit_decision(c0=c0, c1=c1, margin=margin)
        # classify the group for diagnostics
        if c1 - c0 > margin:
            diag["n_save"] += 1
        elif c0 - c1 > margin:
            diag["n_derail"] += 1
        elif c0 >= t and c1 >= t:
            diag["n_waste"] += 1
        else:
            diag["n_neutral"] += 1
        for i in with_rows:
            R_gate[i] = asym_cf_gate_scalar(
                c0=c0, c1=c1, alpha=alpha, beta=beta, gamma=gamma, t=t,
                margin=margin, beta_clip=beta_clip, emit_floor=emit_floor,
                confidence=(float(conf[i]) if conf is not None else None),
                conf_w=conf_w,
            )
            member[i] = 1.0
            emit_decision[i] = emit

    diag["emit_decision"] = emit_decision
    return R_gate, member, diag


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2 — content gated by the Layer-1 emit decision
# ─────────────────────────────────────────────────────────────────────────────
def apply_content_gate(content_reward, emit_ok):
    """Zero the Layer-2 content reward wherever Layer 1 says wrong-to-emit.

    `content_reward` is the decoy-DiD / PMI independence R_meta (per row);
    `emit_ok` is the per-row emit decision (1.0 = net-positive-to-emit, else 0.0,
    from `compute_asym_cf_gate`'s diagnostics). The hierarchy: Layer 1 decides
    *whether* to emit, Layer 2 only scores content where whether=yes. All-emit
    (emit_ok all 1) -> byte-identical pass-through. Returns float32[B].

    `emit_ok` MUST be a binary (0.0/1.0) decision array (the threshold below is
    `> 0.5`, so 1.0 keeps and 0.0 zeros). Non-binary / NaN values are rejected
    (review: a bad emit_ok must not silently produce NaN content rewards).
    """
    cr = np.asarray(content_reward, dtype=np.float32).reshape(-1)
    ok = np.asarray(emit_ok, dtype=np.float32).reshape(-1)
    if cr.shape[0] != ok.shape[0]:
        raise ValueError(
            f"content/emit length mismatch ({cr.shape[0]} vs {ok.shape[0]})")
    if not np.all(np.isin(ok, (0.0, 1.0))):
        raise ValueError(
            "apply_content_gate: emit_ok must be binary 0.0/1.0 "
            f"(got values {np.unique(ok)[:8]})")
    return (cr * (ok > 0.5).astype(np.float32)).astype(np.float32)


__all__ = [
    "asym_cf_gate_scalar",
    "gate_emit_decision",
    "compute_asym_cf_gate",
    "apply_content_gate",
]
