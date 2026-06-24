"""Directional self-distillation (gm-contrast) R_meta core — pure numpy.

NEW + ADDITIVE, framework-light module (numpy only — ZERO verl / torch deps),
the gm-direction sibling of `dcpo_pmi`. Like that module it does NOT load models
or tokenizers: callers hand it ROWS already carrying the per-token reference
logprobs of the two answer strings (`\\boxed{gold}` / `\\boxed{decoy}`) under the
two contexts (body+meta / body+placebo); this module does the token-level
DiD, the divergent-answer-token restriction, and the `mean_min` (RLT)
aggregation. The actual GPU teacher-forcing forward lives in verl_sdc.

Spec traceability (docs/superpowers/specs/2026-06-24-directional-self-distill-meta-rl-design.md):
  - gm direction (§FINAL 2-ARM): R_meta scored AFTER `<|/meta|>` as the
    DiD over the model's OWN meta:
        DiD_t = ( logp(gold_t | body+meta)  − logp(gold_t | body+placebo) )
              − ( logp(decoy_t| body+meta)  − logp(decoy_t| body+placebo) )
    aggregated with `mean_min` over the DIVERGENT answer-value tokens only
    (excludes the structural `\\boxed{` / `}` tokens that dilute, spec §24).
  - decoy: `_rule_based_decoy(gold, seed, checker)` — same decoy machinery as
    the contrastive RLSD line, deterministic near-miss.
  - mean_min aggregation reused VERBATIM from dcpo_pmi.pmi_aggregate (RLT
    arXiv 2506.08388 r^SS = mean(clip(d)) + alpha*min(clip(d))).
  - multiplicative (RLSD ABLATION, spec §23 / FINAL 2-ARM Arm RLSD):
        w = exp( sign(A_corr) * clip(gm, -clip_w, clip_w) )
    a per-ROW weight that MULTIPLIES the correctness advantage on META tokens
    (precedent meta_rlsd_trainer.py:896-902). Built here as a pure function so
    the shackling formula is CPU-unit-testable; the verl path threads the gm
    scalar to compose where the multiply happens per-token.
"""
from __future__ import annotations

import numpy as np

# Reuse the EXACT RLT mean_min aggregation the PMI core validated — the gm
# reward must aggregate with the same operator (mean+alpha*min over clipped
# per-token deltas) so the two directions are comparable.
from src.training.dcpo_pmi import pmi_aggregate


# Answer-string wrapper: the gm contrast scores `\boxed{value}` continuations so
# the divergent token span is the value itself (structural `\boxed{` / `}` tokens
# are shared between gold and decoy and contribute ~0 to the DiD — excluding them
# is the spec §24 dilution fix).
def boxed_answer_string(value) -> str:
    r"""The answer string scored by the gm contrast: `\boxed{value}`."""
    return r"\boxed{" + str(value).strip() + "}"


def divergent_token_mask(gold_ids, decoy_ids) -> np.ndarray:
    """Boolean mask over the gold answer tokens where gold and decoy DIFFER.

    gm scores the gold answer string under {meta, placebo}; the decoy DiD term is
    aligned positionally to the SAME gold token span (the reward credits "did the
    meta favor the gold value at the tokens that actually distinguish it from the
    near-miss"). Tokens shared between gold and decoy (the structural `\\boxed{` /
    `}` and any common prefix/suffix digits) are EXCLUDED — they carry no
    gold-vs-decoy directional information and only dilute the mean.

    Alignment is POSITIONAL over the shorter length; positions past the shorter
    string's end are divergent by construction (one string has a token the other
    lacks). Returns a bool array of length len(gold_ids).
    """
    g = list(gold_ids)
    d = list(decoy_ids)
    n = len(g)
    mask = np.ones(n, dtype=bool)
    m = min(n, len(d))
    for t in range(m):
        if g[t] == d[t]:
            mask[t] = False
    # positions [m, n) (gold longer than decoy) stay divergent (True).
    return mask


def gm_contrast_row(row, agg: str = "mean_min", clip_c: float = 2.0,
                    alpha: float = 1.0) -> float:
    r"""Per-rollout gm contrast scalar from one scored row.

    Row keys (per-token logp arrays over the GOLD answer-token span, all equal
    length = number of gold answer tokens):
      logp_gold_meta, logp_gold_placebo, logp_decoy_meta, logp_decoy_placebo
      divergent_mask   (bool array over the gold span; True = score this token)

    DiD_t = (gold_meta − gold_placebo) − (decoy_meta − decoy_placebo), restricted
    to divergent tokens, aggregated with `mean_min` (RLT). A row whose divergent
    span is empty, or whose arms are NaN/length-mismatched, returns NaN (the
    caller fails it closed — member 0).
    """
    gm_meta = np.asarray(row["logp_gold_meta"], dtype=np.float64).reshape(-1)
    gm_plac = np.asarray(row["logp_gold_placebo"], dtype=np.float64).reshape(-1)
    dc_meta = np.asarray(row["logp_decoy_meta"], dtype=np.float64).reshape(-1)
    dc_plac = np.asarray(row["logp_decoy_placebo"], dtype=np.float64).reshape(-1)
    # gold arms share the gold-answer span; decoy arms share the decoy span. The
    # DiD is positionally aligned over the GOLD span (the divergent mask lives
    # there), so the decoy arms are truncated/padded to the gold length: only the
    # divergent positions (where decoy actually has a different token) are used.
    n = gm_meta.size
    if n == 0 or gm_meta.shape != gm_plac.shape:
        return float("nan")
    if dc_meta.shape != dc_plac.shape:
        return float("nan")

    def _fit(a):  # align decoy arm to the gold span length (pad with 0 = no DiD)
        if a.size == n:
            return a
        out = np.zeros(n, dtype=np.float64)
        m = min(n, a.size)
        out[:m] = a[:m]
        return out

    dc_meta = _fit(dc_meta)
    dc_plac = _fit(dc_plac)
    did = (gm_meta - gm_plac) - (dc_meta - dc_plac)
    if not np.isfinite(did).all():
        return float("nan")
    mask = row.get("divergent_mask", None)
    if mask is None:
        mask = np.ones(n, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.size != n:
            return float("nan")
    did = did[mask]
    if did.size == 0:
        return float("nan")
    return float(pmi_aggregate(did, agg, clip_c=clip_c, alpha=alpha))


def compute_directional_meta_reward(rows, *, agg: str = "mean_min",
                                    clip_c_token: float = 2.0, alpha: float = 1.0,
                                    clip_c_gate: float = 2.0,
                                    sign_gate: bool = True):
    """Turn scored gm rows into per-row R_meta values + diagnostics.

    Each row carries the four per-token logp arrays + divergent_mask + `correct`
    (bool). gm_contrast_row computes the DiD scalar; when `sign_gate` is on the
    scalar is gated like the PMI head (correct rows keep >=0, wrong rows <=0 of
    the clipped magnitude) so a gold-favoring meta on a WRONG rollout cannot earn
    positive credit on the ADDITIVE head. A row that fails (empty divergent span,
    NaN, length mismatch) scores 0 with member 0.

    Returns (r_meta float32 [len(rows)], diagnostics) — diagnostics carries
    raw_gm (per-row scalars, NaN on fail) and `failures` (per-row bool).
    """
    n = len(rows)
    r_meta = np.zeros(n, dtype=np.float32)
    diag = {"raw_gm": [], "failures": []}
    for i, row in enumerate(rows):
        gm = gm_contrast_row(row, agg=agg, clip_c=clip_c_token, alpha=alpha)
        failed = not np.isfinite(gm)
        diag["raw_gm"].append(float(gm) if not failed else float("nan"))
        diag["failures"].append(bool(failed))
        if failed:
            continue
        if sign_gate:
            gated = float(np.clip(gm, 0.0, clip_c_gate))
            r_meta[i] = gated if bool(row.get("correct", False)) else -gated
        else:
            r_meta[i] = float(np.clip(gm, -clip_c_gate, clip_c_gate))
    return r_meta, diag


# ─────────────────────────────────────────────────────────────────────────────
# Multiplicative (RLSD ABLATION) per-row weight — spec §23 / FINAL 2-ARM Arm RLSD
# ─────────────────────────────────────────────────────────────────────────────
def rlsd_meta_weight(gm_scalar, a_corr_scalar, *, clip_w: float = 2.0) -> float:
    r"""Per-row RLSD weight w = exp( sign(A_corr) * clip(gm, -clip_w, clip_w) ).

    Precedent: meta_rlsd_trainer.py:896-902 (`A_sign = sign(advantages);
    w = exp(A_sign * delta)`). Here `delta` is the gm contrast scalar (clipped),
    so a gm-favoring meta on a correct rollout (A_corr>0) is amplified (w>1), and
    the SAME meta on a WRONG rollout (A_corr<0) is SHACKLED to negative sign ->
    w<1 (the shackling harm the ablation demonstrates). A_corr==0 -> sign 0 ->
    w=1 (zero gradient when group correctness is flat). NaN gm -> w=1 (neutral).
    """
    if not np.isfinite(gm_scalar):
        return 1.0
    s = float(np.sign(float(a_corr_scalar)))
    g = float(np.clip(float(gm_scalar), -clip_w, clip_w))
    return float(np.exp(s * g))


def rlsd_meta_factor(gm_scalar, a_corr_scalar, *, lam: float = 1.0,
                     clip_w: float = 2.0) -> float:
    r"""Per-row multiplicative META factor `(1-lam) + lam*w` (the `Â_corr` multiplier).

    `Â_t = Â_corr * ((1-lam) + lam*w)` on META tokens (meta_rlsd_trainer.py:902).
    lam interpolates between the un-shackled Â_corr (lam=0 -> factor 1) and the
    fully RLSD-weighted Â_corr*w (lam=1). Returned as the scalar factor so compose
    can route `Â_corr * factor` onto the META mask.
    """
    w = rlsd_meta_weight(gm_scalar, a_corr_scalar, clip_w=clip_w)
    return float((1.0 - lam) + lam * w)


def gm_over_emission_penalty(meta_member, c_with, group_index, *,
                             w_over: float = 0.0,
                             over_threshold: float = 1.0):
    r"""Per-row over-emission (selectivity) penalty for the gm arms.

    Mirror of `compute_cf_group_heads`'s AdaCoT `over_penalty` for the gm path,
    which has NO without-meta arm split. The gm path has no counterfactual
    sibling, so the "problem was already solvable without meta" signal is taken
    from the row's OWN GROUP accuracy: a meta-emitting (member) row whose group
    accuracy >= `over_threshold` fired meta on an already-solved problem
    (AdaCoT P_over = wasteful emission). Such rows get `w_over` subtracted from
    correctness by the caller (same fold as cf_group; no new GDPO key).

    `meta_member` (1.0 = gm-meta-bearing row), `c_with` (0/1 final correctness,
    every row), `group_index` (prompt-group id per row). Returns a length-B
    float32 penalty array (0.0 everywhere when `w_over == 0` -> selectivity off,
    byte-identical correctness). Groups are scored over ALL rows (the gm path
    has no arm split, so every row is a with-meta member of its group).
    """
    cw = np.asarray(c_with, dtype=np.float32).reshape(-1)
    mm = np.asarray(meta_member, dtype=np.float32).reshape(-1)
    B = cw.shape[0]
    over = np.zeros(B, dtype=np.float32)
    if not w_over:
        return over  # selectivity OFF -> zero penalty (correctness untouched)
    gid = list(
        group_index.tolist() if hasattr(group_index, "tolist") else group_index
    )
    gid = [str(g) for g in gid]
    groups: dict = {}
    for i in range(B):
        groups.setdefault(gid[i], []).append(i)
    for members in groups.values():
        acc = float(np.mean([cw[i] for i in members]))
        if acc >= over_threshold:
            for i in members:
                if mm[i] > 0.5:
                    over[i] = float(w_over)
    return over


__all__ = [
    "boxed_answer_string",
    "divergent_token_mask",
    "gm_contrast_row",
    "compute_directional_meta_reward",
    "rlsd_meta_weight",
    "rlsd_meta_factor",
    "gm_over_emission_penalty",
]
