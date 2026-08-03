"""TRIOBJ_DCPO_V4 — pure likelihood-delta (PMI) R_meta core.

NEW + ADDITIVE, framework-light module: numpy only — ZERO verl / torch-distributed
deps, so it is importable by BOTH the offline probe (plain HF on the local A100)
and verl_sdc's v4 populator. It does NOT load models or tokenizers: callers pass a
tokenizer object (encode/decode) and the per-token logprob arrays produced by the
frozen-reference scorer (probe forward pass or trainer._compute_ref_log_prob).

Spec traceability (docs/superpowers/specs/2026-06-11-dcpo-v4-likelihood-rmeta-design.md):
  - pmi_aggregate        §2 aggregation menu {sum_clip, topk_mean, mean, max};
                         max-minus-min REJECTED (direction-blind).
  - split_first_meta     §2 prefix/meta/continuation split around the FIRST closed
                         meta block (review round 2 M-D: ONE definition, called by
                         both the offline probe and verl_sdc's v4 scorer).
"""
from __future__ import annotations

import numpy as np

# Tag constants only (pure strings, no framework deps) — the module stays
# importable without verl/torch.
from src.metacot.prompt import META_END, META_START

# Fixed contentless-but-coherent meta for the placebo arm (spec C1): tag-wrapped
# like real metas so the only difference vs the real arm is the CONTENT. SSOT
# shared by the offline probe AND the stage-2 placebo-corrected reward — the
# correction is only valid if training subtracts the SAME placebo the probe
# validated (cross-shuffle finding 2026-06-11: raw delta is ~86% generic
# text-presence; the trainable signal is delta - delta_placebo).
PLACEBO_META = f"{META_START}\nLet me continue.\n{META_END}"

# Aggregation menu the offline probe decides among (spec §2). max-minus-min is NOT
# here on purpose: it scores a meta that makes the continuation LESS likely the same
# as one that makes it MORE likely (direction-blind, rejected in review).
PMI_AGG_METHODS = ("sum_clip", "topk_mean", "mean", "max", "mean_min")


# ─────────────────────────────────────────────────────────────────────────────
# §2 aggregation + sign gate (review M3)
# ─────────────────────────────────────────────────────────────────────────────
def pmi_aggregate(delta_per_token, method: str, topk_frac: float = 0.25,
                  clip_c: float = 2.0, alpha: float = 0.0) -> float:
    """Aggregate per-token deltas (logP_with - logP_without over the C-span).

    Methods (spec §2 probe menu):
      sum_clip   sum of PER-TOKEN deltas clipped to [-clip_c, clip_c] (outlier-robust)
      topk_mean  mean of the top ceil(topk_frac * T) deltas (>= 1 token)
      mean       plain average
      max        single best token
      mean_min   RLT (arXiv 2506.08388 r^SS = avg + alpha*min) on the per-token
                 CLIPPED deltas: mean(clip(d)) + alpha*min(clip(d)). The clip is
                 LOAD-BEARING — our delta is a real-vs-placebo DIFFERENCE that can
                 swing far negative on a single noisy token, so an unclipped min
                 would swamp the mean (sign_gate would zero every correct row).
                 Clipping bounds the worst token so the term measures "does the
                 meta also support the HARDEST token", not "find one outlier".
                 alpha=0 reduces to the clipped mean. Under placebo_correct the
                 placebo arm goes through the SAME reduction, so the worst-case
                 term is itself placebo-corrected (min_real - min_placebo).
    max-minus-min is rejected explicitly (direction-blind).
    """
    d = np.asarray(delta_per_token, dtype=np.float64).reshape(-1)
    if d.size == 0:
        raise ValueError("empty delta_per_token: nothing to aggregate")
    if method == "sum_clip":
        return float(np.clip(d, -clip_c, clip_c).sum())
    if method == "topk_mean":
        k = max(1, int(np.ceil(topk_frac * d.size)))
        return float(np.sort(d)[::-1][:k].mean())
    if method == "mean":
        return float(d.mean())
    if method == "max":
        return float(d.max())
    if method == "mean_min":
        dc = np.clip(d, -clip_c, clip_c)
        return float(dc.mean() + alpha * dc.min())
    if method in ("max_minus_min", "max-min", "maxmin"):
        raise ValueError("max-minus-min is direction-blind — rejected by spec §2")
    raise ValueError(f"unknown aggregation method {method!r}; use one of {PMI_AGG_METHODS}")


# ─────────────────────────────────────────────────────────────────────────────
# §2 first-meta split (review round 2 M-D: single definition for probe + verl_sdc)
# ─────────────────────────────────────────────────────────────────────────────
def split_first_meta(text):
    """Split `text` around its FIRST closed meta block (spec §2).

    Returns (prefix, meta, continuation) — prefix = text before <|meta|>, meta =
    the tag-INCLUSIVE block, continuation = everything after <|/meta|> — or None
    for unscorable rows: no meta, truncated meta (open without close, the
    16k-cutoff population), or a WHITESPACE-ONLY continuation (nothing to score;
    the stricter probe semantics, unified here so verl_sdc's v4 scorer cannot
    silently score whitespace tails).
    """
    text = text or ""
    o = text.find(META_START)
    if o < 0:
        return None
    c = text.find(META_END, o + len(META_START))
    if c < 0:
        return None
    end = c + len(META_END)
    continuation = text[end:]
    if not continuation.strip():
        return None
    return text[:o], text[o:end], continuation
