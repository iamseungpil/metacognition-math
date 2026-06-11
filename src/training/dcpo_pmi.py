"""TRIOBJ_DCPO_V4 — pure likelihood-delta (PMI) R_meta core.

NEW + ADDITIVE, framework-light module: numpy only — ZERO verl / torch-distributed
deps, so it is importable by BOTH the offline probe (plain HF on the local A100)
and verl_sdc's v4 populator. It does NOT load models or tokenizers: callers pass a
tokenizer object (encode/decode) and the per-token logprob arrays produced by the
frozen-reference scorer (probe forward pass or trainer._compute_ref_log_prob).

Spec traceability (docs/superpowers/specs/2026-06-11-dcpo-v4-likelihood-rmeta-design.md):
  - splice_and_align     §2 splice alignment contract (review C3): tokenize BOTH arms
                         independently, locate C by decode-and-rematch, subtract only
                         over the C-span that decodes BYTE-IDENTICALLY in both arms.
  - pmi_aggregate        §2 aggregation menu {sum_clip, topk_mean, mean, max};
                         max-minus-min REJECTED (direction-blind).
  - sign_gate            §2 sign-gate (review M3): correct rows >= 0, wrong rows <= 0.
  - ngram_overlap_guard  §2 anti-hack guards 2+3 (review C2): meta<->continuation
                         n-gram overlap + literal boxed-answer leak => delta invalid.
  - split_first_meta     §2 prefix/meta/continuation split around the FIRST closed
                         meta block (review round 2 M-D: ONE definition, called by
                         both the offline probe and verl_sdc's v4 scorer).
  - compute_pmi_rows     §3 probe orchestrator: rows in, gated R_meta + diagnostics out.
"""
from __future__ import annotations

import re

import numpy as np

# Tag constants only (pure strings, no framework deps) — the module stays
# importable without verl/torch.
from src.metacot.prompt import META_END, META_START

# Aggregation menu the offline probe decides among (spec §2). max-minus-min is NOT
# here on purpose: it scores a meta that makes the continuation LESS likely the same
# as one that makes it MORE likely (direction-blind, rejected in review).
PMI_AGG_METHODS = ("sum_clip", "topk_mean", "mean", "max")


class SpliceAlignmentError(ValueError):
    """Raised when no byte-identical C-span exists in both arms (spec C3)."""


# Review round 2 M-B: the common-tail refinement only ever needs to absorb a few
# boundary tokens (cross-splice BPE merges). A tokenizer whose two arms disagree
# DEEP into the continuation (pathological/adversarial) would otherwise walk the
# whole tail with an O(L) slice compare per drop (O(L^2)); cap the drops and
# fail LOUDLY instead — a >256-token boundary divergence is never a real splice.
_MAX_SPLICE_BOUNDARY_DROPS = 256


# ─────────────────────────────────────────────────────────────────────────────
# §2 splice alignment contract (review C3)
# ─────────────────────────────────────────────────────────────────────────────
def _first_c_start(tokenizer, ids, continuation_text: str) -> int:
    """Smallest token index whose decoded tail lies fully inside the continuation.

    Bisection on decoded-tail LENGTH (the tail shrinks monotonically as the start
    moves right), then a short forward scan for the suffix check — the scan absorbs
    boundary tokens that decode dirty (cross-boundary BPE merges, multi-byte UTF-8
    split across the cut decoding to replacement chars).
    """
    lo, hi = 0, len(ids)
    while lo < hi:
        mid = (lo + hi) // 2
        if len(tokenizer.decode(ids[mid:])) <= len(continuation_text):
            hi = mid
        else:
            lo = mid + 1
    i = lo
    while i < len(ids) and not continuation_text.endswith(tokenizer.decode(ids[i:])):
        i += 1
    return i


def splice_and_align(tokenizer, prefix_text: str, meta_text: str, continuation_text: str):
    """Build the two scoring arms and locate the SHARED C-span (spec C3).

    with-arm    = prefix + meta + continuation   (the model's own sequence)
    without-arm = prefix + continuation          (a sequence the model NEVER produced)

    Both arms are tokenized INDEPENDENTLY and C is located by decode-and-rematch —
    NEVER by token-index arithmetic: deleting the meta block can create a NEW BPE
    merge across the prefix|continuation boundary, shifting every later token (the
    v3b silent-bug class). The returned spans are token-id-IDENTICAL between arms
    (strictly stronger than byte-identical), so per-token deltas align positionally;
    continuation tokens swallowed by a boundary merge are EXCLUDED, not misaligned.

    Returns:
        dict with with_ids / without_ids (full token lists), c_span_with /
        c_span_without (half-open (start, end) over the respective ids), and c_text
        (the byte-identical decoded span, a suffix of continuation_text).

    Raises:
        SpliceAlignmentError if the continuation is empty or no non-empty common
        span exists (e.g. the whole continuation merged into a boundary token).
    """
    if not continuation_text:
        raise SpliceAlignmentError("empty continuation: nothing to score")
    with_ids = list(tokenizer.encode(prefix_text + meta_text + continuation_text,
                                     add_special_tokens=False))
    without_ids = list(tokenizer.encode(prefix_text + continuation_text,
                                        add_special_tokens=False))
    i_w = _first_c_start(tokenizer, with_ids, continuation_text)
    i_wo = _first_c_start(tokenizer, without_ids, continuation_text)
    # Refine to a common token-id tail: while the tails differ, drop one token from
    # the arm with the longer tail (ties -> with-arm). M-B: the full equality check
    # runs only at EQUAL tail lengths, and total drops are capped — boundary merges
    # cost a handful of tokens, never hundreds.
    drops = 0
    while True:
        if (len(with_ids) - i_w == len(without_ids) - i_wo
                and with_ids[i_w:] == without_ids[i_wo:]):
            break
        if drops >= _MAX_SPLICE_BOUNDARY_DROPS:
            raise SpliceAlignmentError(
                f"no common C-span within {_MAX_SPLICE_BOUNDARY_DROPS} boundary "
                "drops: arms diverge deep into the continuation (pathological "
                "tokenization, not a splice-boundary merge)")
        if len(with_ids) - i_w >= len(without_ids) - i_wo:
            i_w += 1
        else:
            i_wo += 1
        drops += 1
    c_ids = with_ids[i_w:]
    if not c_ids:
        raise SpliceAlignmentError(
            "no common C-span: continuation fully merged across the splice boundary")
    c_text = tokenizer.decode(c_ids)
    # ASSERT identity before trusting delta (spec C3): the common span must decode
    # to a suffix of the continuation (never reach back into meta/prefix bytes).
    if not continuation_text.endswith(c_text):
        raise SpliceAlignmentError(
            f"aligned span decodes outside the continuation: {c_text!r}")
    return {
        "with_ids": with_ids,
        "without_ids": without_ids,
        "c_span_with": (i_w, len(with_ids)),
        "c_span_without": (i_wo, len(without_ids)),
        "c_text": c_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §2 aggregation + sign gate (review M3)
# ─────────────────────────────────────────────────────────────────────────────
def pmi_aggregate(delta_per_token, method: str, topk_frac: float = 0.25,
                  clip_c: float = 2.0) -> float:
    """Aggregate per-token deltas (logP_with - logP_without over the C-span).

    Methods (spec §2 probe menu):
      sum_clip   sum of PER-TOKEN deltas clipped to [-clip_c, clip_c] (outlier-robust)
      topk_mean  mean of the top ceil(topk_frac * T) deltas (>= 1 token)
      mean       plain average
      max        single best token
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
    if method in ("max_minus_min", "max-min", "maxmin"):
        raise ValueError("max-minus-min is direction-blind — rejected by spec §2")
    raise ValueError(f"unknown aggregation method {method!r}; use one of {PMI_AGG_METHODS}")


def sign_gate(agg_delta: float, correct: bool, clip_c: float) -> float:
    """R_meta_row = (+1 if correct else -1) * clip(agg_delta, 0, clip_c) (review M3).

    Correct rollouts can only earn >= 0, wrong rollouts only <= 0: a meta that LOWERS
    the frozen base's likelihood of the continuation never gets credit, and a meta
    that confidently steers into a WRONG answer is punished proportionally.
    """
    gated = float(np.clip(agg_delta, 0.0, clip_c))
    return gated if correct else -gated


# ─────────────────────────────────────────────────────────────────────────────
# §2 anti-hack guards 2+3 (review C2)
# ─────────────────────────────────────────────────────────────────────────────
def ngram_overlap_guard(meta_text: str, continuation_text: str, n: int = 8,
                        threshold: float = 0.25, boxed_answer=None) -> bool:
    """True -> delta INVALID (the v2 'boilerplate detector' never existed — this is it).

    Two hacks, both trivially game the likelihood delta:
      1. answer leak: the meta states the literal boxed answer string, so the ref
         loves the continuation that echoes it (guard 3) — checked FIRST, any length;
      2. copy-through: the meta pre-states the continuation verbatim; detected as
         the fraction of word-level n-grams of the meta that also occur in the
         continuation reaching `threshold` (guard 2).
    Metas shorter than n words carry no n-grams -> valid on the overlap axis.

    Guard 1 is BOUNDARY-AWARE (review round 1): a bare `ans in meta_text` tripped
    on 36.7% of single-char-answer rows (e8_goldfree) — boxed "7" inside
    "confidence: 0.7", boxed "2" inside step numbering — silently zeroing R_meta
    + member on a GSM8K-skewed (easy/short-answer) population. The lookarounds
    keep genuine standalone answer statements ("the answer is 7") firing while
    decimal fragments and word/number substrings pass.

    Round 2 fix (sentence-final punctuation): the round-1 trailing lookaround
    (?![\\w.]) blocked '.' in EVERY context, so "the answer is 7." / "answer:
    42." passed (verified by execution). The trailing dot is now DECIMAL-AWARE:
    only a dot followed by a digit blocks ("7" inside "7.5" / "0.7" stays
    clean); a sentence-final dot after the answer flags. The leading side keeps
    blocking any adjacent dot — a dot immediately BEFORE a digit is a decimal
    point ("0.85", ".7"), never sentence punctuation.
    """
    if boxed_answer is not None:
        ans = str(boxed_answer).strip()
        if ans and re.search(rf"(?<![\w.]){re.escape(ans)}(?!\w)(?!\.\d)", meta_text):
            return True
    meta_words = meta_text.split()
    cont_words = continuation_text.split()
    if len(meta_words) < n or len(cont_words) < n:
        return False
    meta_grams = {tuple(meta_words[i:i + n]) for i in range(len(meta_words) - n + 1)}
    cont_grams = {tuple(cont_words[i:i + n]) for i in range(len(cont_words) - n + 1)}
    ratio = len(meta_grams & cont_grams) / len(meta_grams)
    return ratio >= threshold


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


# ─────────────────────────────────────────────────────────────────────────────
# §3 probe orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def compute_pmi_rows(rows, method: str = "sum_clip", topk_frac: float = 0.25,
                     clip_c_token: float = 2.0, clip_c_gate: float = 2.0,
                     ngram_n: int = 8, ngram_threshold: float = 0.25):
    """Turn scored rows into gated R_meta values + probe diagnostics.

    The module does NOT load models: each row carries the two arms' per-token
    logprobs over the SHARED C-span from splice_and_align (id-identical spans =>
    equal lengths). Row keys:
      meta_text, continuation_text, correct (bool), logp_with, logp_without
      (array-likes over the C-span), optional boxed_answer, optional
      alignment_failed (True, or logp arrays None/empty, when splice_and_align
      raised — the row scores 0 and is only counted).

    Returns:
        (r_meta float32 [len(rows)], diagnostics) where diagnostics carries
        raw_agg (per-method raw aggregates, NaN on failed rows), guard_hits,
        alignment_failures and nonfinite (per-row bools) — the probe's
        kill-or-go evidence. Round 2 IMPORTANT-3: a NaN/inf in either arm's
        logprobs marks the row `nonfinite` (R 0, raw_agg NaN) instead of
        propagating — one poisoned r_meta with member=1 would NaN every
        sibling's centered A_meta downstream (group_mean_subtract).
    """
    r_meta = np.zeros(len(rows), dtype=np.float32)
    diagnostics = {
        "raw_agg": {m: [] for m in PMI_AGG_METHODS},
        "guard_hits": [],
        "alignment_failures": [],
        "nonfinite": [],
    }
    for i, row in enumerate(rows):
        logp_w, logp_wo = row.get("logp_with"), row.get("logp_without")
        failed = bool(row.get("alignment_failed", False)) or logp_w is None or logp_wo is None
        if not failed:
            logp_w = np.asarray(logp_w, dtype=np.float64).reshape(-1)
            logp_wo = np.asarray(logp_wo, dtype=np.float64).reshape(-1)
            if logp_w.shape != logp_wo.shape:
                raise ValueError(
                    f"row {i}: arm logprob lengths differ ({logp_w.size} vs {logp_wo.size}) "
                    "— arms must be span-aligned via splice_and_align")
            failed = logp_w.size == 0
        diagnostics["alignment_failures"].append(failed)
        if failed:
            diagnostics["nonfinite"].append(False)
            diagnostics["guard_hits"].append(False)
            for m in PMI_AGG_METHODS:
                diagnostics["raw_agg"][m].append(float("nan"))
            continue  # R_meta stays 0
        delta = logp_w - logp_wo
        nonfinite = not bool(np.isfinite(delta).all())
        diagnostics["nonfinite"].append(nonfinite)
        if nonfinite:
            # IMPORTANT-3 poisoning guard: fail the ROW, never emit NaN.
            diagnostics["guard_hits"].append(False)
            for m in PMI_AGG_METHODS:
                diagnostics["raw_agg"][m].append(float("nan"))
            continue  # R_meta stays 0 (caller sets member 0 off this counter)
        for m in PMI_AGG_METHODS:
            diagnostics["raw_agg"][m].append(
                pmi_aggregate(delta, m, topk_frac=topk_frac, clip_c=clip_c_token))
        invalid = ngram_overlap_guard(
            row.get("meta_text", ""), row.get("continuation_text", ""),
            n=ngram_n, threshold=ngram_threshold, boxed_answer=row.get("boxed_answer"))
        diagnostics["guard_hits"].append(invalid)
        if invalid:
            continue  # guard hit: delta invalid, R_meta stays 0
        agg = pmi_aggregate(delta, method, topk_frac=topk_frac, clip_c=clip_c_token)
        r_meta[i] = sign_gate(agg, bool(row["correct"]), clip_c_gate)
    return r_meta, diagnostics
