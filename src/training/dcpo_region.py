"""TRIOBJ_DCPO_V2 — DCPO-style 3-region token-masked reward + mask utilities.

NEW + ADDITIVE module. Referenced ONLY by the NEW REWARD_CONFIGS['TRIOBJ_DCPO_V2']
entry in verl_sdc.py and by _compute_dcpo_region_advantage in verl_sdc_utils.py.
It does NOT modify any existing reward or mode. All correctness / parse helpers are
IMPORTED from src.training.rewards (NOT re-implemented), so train and eval grading
stay byte-identical with the rest of the pipeline.

Two public pieces (spec §2.7, §2.2):
  - build_dcpo_region_masks(resp_ids, response_mask, decode_fn, meta_open, meta_close)
        -> {META_REGION, META_CONTENT, CONF, ANSWER_REGION}  (bool [T] over response)
  - dcpo_region_rewards(completions, ground_truth, group_index, step, **cfg)
        -> {"R_corr", "R_meta", "R_cal", "p_hat", "group_acc"}  (lists, len = B)

`META_REGION` is the tag-INCLUSIVE meta block (open..close). `META_CONTENT` is the
tag-EXCLUSIVE inner span (advantage routes here). `CONF` is the confidence-number
token run inside META_CONTENT. `ANSWER_REGION` is response_mask minus META_REGION
(tags are in NEITHER answer nor content -> advantage 0).
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
import torch

# Import (do NOT rewrite) the canonical correctness / parse helpers. Train and eval
# grade the LAST \boxed via _check_correctness, so we reuse exactly those.
from src.training.rewards import (
    _check_correctness,
    _extract_answer_fallback,
    _get_text,
    _parse_confidence,
    _parse_confidence_charspan,
)
from src.training.meta_revision_rewards import _BOXED_RE

META_OPEN_DEFAULT = 151669
META_CLOSE_DEFAULT = 151670
THINK_CLOSE_DEFAULT = 151668  # </think> — the drift-clamp boundary for UNCLOSED meta


# ─────────────────────────────────────────────────────────────────────────────
# TRIOBJ_DCPO_V3 (ADDITIVE) — counterfactual meta-ablation helpers (spec §5.1-A).
# These are pure-python and dependency-free so they are importable under the
# minimal unit-test env. They do NOT change any v2 behavior.
# ─────────────────────────────────────────────────────────────────────────────
def first_meta_token_index(resp_ids, response_mask=None, meta_open: int = META_OPEN_DEFAULT):
    """Token position of the FIRST <|meta|> among real response tokens, else None.

    The counterfactual 'without-meta' prefix is resp_ids[:i] (strictly before the
    tag): the producer (verl_sdc, §3.4) regenerates from this prefix with the
    <|meta|> token suppressed. When no <|meta|> token exists -> None (the CF gen is
    skipped and R_meta = 0, the natural no-meta case).

    Args:
        resp_ids: sequence of response token ids (list / 1-D array / tensor).
        response_mask: optional bool/0-1 per-token mask (True = real token, not
            pad). Masked positions are skipped. None -> all positions are real.
        meta_open: the <|meta|> token id (default 151669).

    Returns:
        int index of the first real <|meta|> token, or None.
    """
    ids = [int(t) for t in (resp_ids.tolist() if hasattr(resp_ids, "tolist") else list(resp_ids))]
    if response_mask is None:
        rm = [True] * len(ids)
    else:
        rm = (response_mask.tolist() if hasattr(response_mask, "tolist") else list(response_mask))
    for i, t in enumerate(ids):
        if i < len(rm) and not rm[i]:
            continue
        if t == meta_open:
            return i
    return None


# Back-compat alias: spec §6 names it first_meta_index in the producer pseudo-code.
first_meta_index = first_meta_token_index


# ─────────────────────────────────────────────────────────────────────────────
# TRIOBJ_DCPO_V3k (ADDITIVE) — the ONE pure format classifier (v3k spec §2).
# Single source of truth for the three-tier REPLACE/DISCARD/REWARD strategy:
# masks, rewards, the CF producer (verl_sdc) and the offline harness ALL call
# this — NO duplicated delimiter/regex logic anywhere else (Karpathy lock).
# ─────────────────────────────────────────────────────────────────────────────
# Content signature of a meta block (measured on 512 real rollouts: lines
# "confidence: 0.NN" / "assessment: ..." / "action: ..."). Reuses the keyword
# family of _parse_confidence (rewards.py) plus the assessment/action markers.
# This is the ONLY decode the classifier performs — delimiters are detected on
# TOKEN IDS only (a literal "<|meta|>" surface string in prose tokenizes
# differently and must NOT trigger).
_META_SIGNATURE_RE = re.compile(r"(?im)^\s*(confidence|assessment|action)\s*:")


def _has_meta_signature(text: str) -> bool:
    """True iff `text` contains ≥1 meta-content line marker (v3k spec §2.1)."""
    return bool(_META_SIGNATURE_RE.search(text or ""))


# v3m: the field-label words the CF leak guard (_META_SIGNATURE_RE) detects.
_META_SIGNATURE_WORDS = ("confidence", "assessment", "action")

# v3m anti-collapse floor: rows whose meta region is TRUSTED (region routing is
# reliable) and therefore eligible for the +meta_floor emission bias. Mirrors the
# effective fmt_class names verl_sdc stashes: replaced tier-1 rows keep their
# original swapped/dup_open/reversed names; drift rows are content-anchor
# recovered. discard / truncation / no_meta are EXCLUDED (no trusted meta to lift).
TRUSTED_META_CLASSES = frozenset(
    {"wellformed", "swapped", "dup_open", "reversed", "drift"}
)


def signature_suppression_ids(encode_fn, words=_META_SIGNATURE_WORDS):
    """First-token ids of the meta field-label words (+ space / capitalized
    variants) for CF logit_bias suppression (v3m).

    Banning only the two meta TAG ids let the model leak the reflection as plain
    text ("confidence: …"); the leak guard then ungrades that CF, silencing
    R_meta (v3l: ~3/4 of CFs discarded). Suppressing the first token of each
    field label pushes the CF toward "answer directly, no reflection block" so
    more c_without grade. `encode_fn(str) -> list[int]` (e.g. a tokenizer's
    encode with add_special_tokens=False). Returns a sorted unique id list.
    """
    ids = set()
    for w in words:
        for variant in (w, " " + w, w.capitalize(), " " + w.capitalize()):
            try:
                toks = encode_fn(variant)
            except Exception:
                continue
            if toks:
                ids.add(int(toks[0]))
    return sorted(ids)


def classify_dcpo_format(
    resp_ids,
    response_mask,
    decode_fn: Callable[[list], str],
    meta_open: int = META_OPEN_DEFAULT,
    meta_close: int = META_CLOSE_DEFAULT,
    think_close: int = THINK_CLOSE_DEFAULT,
    tier1_to_discard: bool = False,
    _validate_plan: bool = True,
):
    """Classify one rollout's meta-delimiter format (v3k three-tier spec §2.1).

    Token-id-level detection over REAL (response_mask-truthy) positions; the
    content signature (`_has_meta_signature`) anchors tier-1/3 recovery so a
    replacement never promotes non-meta content into META_CONTENT.

    Args:
        resp_ids: response token ids (list / 1-D array / tensor).
        response_mask: bool/0-1 per-token mask (True = real token). None -> all real.
        decode_fn: ids(list[int]) -> str (injected; used ONLY for the signature check).
        meta_open / meta_close / think_close: delimiter token ids.
        tier1_to_discard: True = a tier-1 (replaceable) candidate classifies as
            'discard' instead of emitting a plan. Used by the CONSUMER paths
            (populator / sync __call__) when the row was NOT actually replaced
            at the CF-wrap site (knob off / wrap absent / tensor-write failed):
            replacement there is TOO LATE (old_log_prob already computed), so
            the strategy demotes unreplaced tier-1 rows to tier-2 — never
            half-replaced (v3k spec §6-3 / risk 7).
        _validate_plan: INTERNAL — the §2.2 round-trip validation re-runs this
            function on the plan-applied copy with False (no nested validation).

    Returns dict:
        fmt_class           : 'wellformed' | 'no_meta' | 'swapped' | 'dup_open' |
                              'reversed' | 'drift' | 'truncation' | 'discard'
                              (tier-1 = swapped/dup_open/reversed; tier-3 = drift;
                              tier-2 = discard).
        replacement_plan    : [(pos, old_id, new_id), ...] — 1:1 SAME-LENGTH token
                              substitutions (tier-1 only; empty otherwise).
        meta_content_span   : (lo, hi) exclusive-hi content token span — POST-plan
                              coordinates for tier-1, recovered span for drift,
                              None otherwise.
        answer_start        : first answer position AFTER the drift `</think>`
                              (drift only; None otherwise).
        violation_positions : FORMAT_VIOLATION targets — drift: the single
                              double-duty `</think>` index; discard: every garbage
                              meta-delimiter index PLUS the drift-`</think>` when
                              identifiable (spec §2.1 rule 8); else [].
        format_ok_positions : the closer `<|/meta|>` index (ORIGINALLY-wellformed
                              rows ONLY — replaced rows carry NO format positions,
                              v3k spec §3 tier-1 / risk 3).
        has_signature       : result of the signature check on the candidate
                              content span (False when no span was examined).

    §2.2 mandatory validation: a tier-1 plan is applied to a COPY and this
    function re-run on it; anything but 'wellformed' demotes the row to
    'discard' (never half-replaced).
    """
    ids = [int(t) for t in (resp_ids.tolist() if hasattr(resp_ids, "tolist") else list(resp_ids))]
    T = len(ids)
    if response_mask is None:
        rmask = [True] * T
    else:
        rm = response_mask.tolist() if hasattr(response_mask, "tolist") else list(response_mask)
        rmask = [bool(x) for x in rm]
        # Align lengths defensively (mirror build_dcpo_region_masks).
        rmask = (rmask + [False] * (T - len(rmask)))[:T]

    # Delimiter positions among REAL tokens only (v3k §2.1: O / C / K).
    O = [i for i in range(T) if rmask[i] and ids[i] == meta_open]
    C = [i for i in range(T) if rmask[i] and ids[i] == meta_close]
    K = [i for i in range(T) if rmask[i] and ids[i] == think_close]

    def _sig(lo, hi):
        # Signature check over the candidate content span (pads filtered out).
        return _has_meta_signature(decode_fn([ids[k] for k in range(lo, hi) if rmask[k]]))

    out = {
        "fmt_class": "discard",
        "replacement_plan": [],
        "meta_content_span": None,
        "answer_start": None,
        "violation_positions": [],
        "format_ok_positions": [],
        "has_signature": False,
    }

    def _discard(drift_k=None):
        # Tier-2: regions untrustworthy — flag EVERY garbage delimiter position,
        # PLUS the drift-`</think>` when identifiable (spec §2.1 rule 8): the
        # rule-4/rule-6 fall-throughs arrive here KNOWING which `</think>` did
        # double duty for a drifted span, and leaving it unflagged would leave
        # the very token the -1 exists for (the R_corr-reinforcement-leak token)
        # unpenalized on a subset the spec explicitly covers.
        out["fmt_class"] = "discard"
        out["replacement_plan"] = []
        out["meta_content_span"] = None
        out["answer_start"] = None
        out["violation_positions"] = sorted(
            set(O + C + ([drift_k] if drift_k is not None else [])))
        out["format_ok_positions"] = []
        return out

    # 1. no_meta — no meta delimiters at all (signature alone must NOT trigger).
    if not O and not C:
        out["fmt_class"] = "no_meta"
        return out

    # 2. wellformed — single pair, in order, no </think> strictly inside.
    if len(O) == 1 and len(C) == 1 and O[0] < C[0]:
        if any(O[0] < k < C[0] for k in K):
            return _discard()  # `<|meta|> .. </think> .. <|/meta|>` crossing block
        out["fmt_class"] = "wellformed"
        out["meta_content_span"] = (O[0] + 1, C[0])
        out["format_ok_positions"] = [C[0]]
        out["has_signature"] = _sig(O[0] + 1, C[0])
        return out

    # 3. SWAPPED (tier-1) — close-only; the LAST `</think>` before the close is
    #    doing opener duty for a signature block: replace that id with <|meta|>.
    if len(O) == 0 and len(C) == 1:
        before = [k for k in K if k < C[0]]
        if before:
            t = max(before)
            out["has_signature"] = _sig(t + 1, C[0])
            if out["has_signature"]:
                if tier1_to_discard:
                    return _discard()  # consumer path: unreplaced tier-1 = tier-2
                out["fmt_class"] = "swapped"
                out["replacement_plan"] = [(t, think_close, meta_open)]
                out["meta_content_span"] = (t + 1, C[0])  # post-plan coordinates
                return _validate_replacement(
                    out, ids, rmask, decode_fn, meta_open, meta_close, think_close,
                    _discard, _validate_plan,
                )
        return _discard()  # no opener candidate / signatureless -> tier-2

    # 4. DUP_OPEN (tier-1) — `<|meta|> content <|meta|>`: the second open acts as
    #    the closer. A `</think>` BETWEEN them = drifted first span (the
    #    dup-open-after-drift edge, build_dcpo_region_masks Pass A) -> discard.
    if len(O) == 2 and len(C) == 0:
        if any(O[0] < k < O[1] for k in K):
            # The first intervening `</think>` closed the drifted first span —
            # identifiable drift-K, flagged alongside O ∪ C (rule 8).
            return _discard(min(k for k in K if O[0] < k < O[1]))
        out["has_signature"] = _sig(O[0] + 1, O[1])
        if out["has_signature"]:
            if tier1_to_discard:
                return _discard()  # consumer path: unreplaced tier-1 = tier-2
            out["fmt_class"] = "dup_open"
            out["replacement_plan"] = [(O[1], meta_open, meta_close)]
            out["meta_content_span"] = (O[0] + 1, O[1])  # post-plan coordinates
            return _validate_replacement(
                out, ids, rmask, decode_fn, meta_open, meta_close, think_close,
                _discard, _validate_plan,
            )
        # Signatureless dup-open candidate: a cut run (no `</think>` after the
        # last open) gates as truncation (§2.1 rule 7); otherwise tier-2.
        if not any(k > O[1] for k in K):
            out["fmt_class"] = "truncation"
            return out
        return _discard()

    # 5. REVERSED (tier-1) — `<|/meta|> content <|meta|>`: swap the two ids.
    if len(O) == 1 and len(C) == 1 and C[0] < O[0]:
        out["has_signature"] = _sig(C[0] + 1, O[0])
        if out["has_signature"]:
            if tier1_to_discard:
                return _discard()  # consumer path: unreplaced tier-1 = tier-2
            out["fmt_class"] = "reversed"
            out["replacement_plan"] = [
                (C[0], meta_close, meta_open),
                (O[0], meta_open, meta_close),
            ]
            out["meta_content_span"] = (C[0] + 1, O[0])  # post-plan coordinates
            return _validate_replacement(
                out, ids, rmask, decode_fn, meta_open, meta_close, think_close,
                _discard, _validate_plan,
            )
        return _discard()

    # 6. DRIFT (tier-3) — open-only signature block closed by a double-duty
    #    `</think>` (the FIRST one after the open). Needs an INSERTION to fix
    #    (length change = invasive) -> no plan; lenient region recovery instead.
    if len(O) == 1 and len(C) == 0:
        after = [k for k in K if k > O[0]]
        if after:
            k = min(after)
            out["has_signature"] = _sig(O[0] + 1, k)
            if out["has_signature"]:
                out["fmt_class"] = "drift"
                out["meta_content_span"] = (O[0] + 1, k)
                out["violation_positions"] = [k]  # the wrong token ITSELF, only
                out["answer_start"] = k + 1
                return out
            # Signatureless drift candidate -> tier-2; the double-duty
            # `</think>` at k is still identifiable -> flag it (rule 8).
            return _discard(k)

    # 7. TRUNCATION — open(s), no close, NO `</think>` after the last open: the
    #    run was cut at max length (length problem, not a habit -> no penalty).
    if len(O) >= 1 and len(C) == 0 and not any(k > O[-1] for k in K):
        out["fmt_class"] = "truncation"
        return out

    # 8. DISCARD (tier-2) — everything else: multiple/crossing/interleaved
    #    blocks, >2 meta tokens not matching the shapes above, etc.
    return _discard()


def _validate_replacement(out, ids, rmask, decode_fn, meta_open, meta_close,
                          think_close, discard_fn, validate: bool):
    """§2.2 mandatory tier-1 round-trip: plan applied to a COPY must re-classify
    as 'wellformed', else the row demotes to discard (never half-replaced)."""
    if not validate:
        return out
    fixed = list(ids)
    for (pos, old_id, new_id) in out["replacement_plan"]:
        if not (0 <= pos < len(fixed)) or fixed[pos] != old_id:
            return discard_fn()  # coherence guard: plan does not match the ids
        fixed[pos] = new_id
    re_cls = classify_dcpo_format(
        fixed, rmask, decode_fn,
        meta_open=meta_open, meta_close=meta_close, think_close=think_close,
        _validate_plan=False,
    )
    if re_cls["fmt_class"] != "wellformed":
        return discard_fn()
    return out


def cf_answer_from_prefix(text: str):
    """TEXT-fallback counterfactual answer = extract from the pre-(first-)meta
    prefix only. Used by dcpo_region_rewards ONLY when no real regenerated cf
    rollout is supplied (crash-guard for the consumer). Returns None when there is
    no <|meta|> tag (the natural no-meta case -> R_meta 0) or no parseable answer
    in the pre-meta prefix (conservative under-credit, spec §10 risk 6)."""
    if not text or "<|meta|>" not in text:
        return None
    prefix = text.split("<|meta|>", 1)[0]
    return _extract_answer_fallback(prefix) or None


# ─────────────────────────────────────────────────────────────────────────────
# §2.7 / §4 — token-region mask util
# ─────────────────────────────────────────────────────────────────────────────
def build_dcpo_region_masks(
    resp_ids,
    response_mask,
    decode_fn: Callable[[list], str],
    meta_open: int = META_OPEN_DEFAULT,
    meta_close: int = META_CLOSE_DEFAULT,
    think_close: int = THINK_CLOSE_DEFAULT,
    clamp_unclosed: bool = True,
    fmt: dict | None = None,
    fmt_replaced: bool = False,
):
    """Build the four DCPO region masks over response positions.

    Args:
        resp_ids: sequence of response token ids (list / 1-D array / tensor).
        response_mask: bool/0-1 per-token mask (True = real response token, not pad).
        decode_fn: ids(list[int]) -> str. Injected so the util is importable with no
            transformers dependency (unit tests pass a fake decode_fn).
        meta_open / meta_close: the <|meta|> / <|/meta|> token ids.
        think_close: the </think> token id — the drift-clamp boundary for an
            UNCLOSED meta block (see below).
        clamp_unclosed: True (v3 default) = apply the unclosed-meta handling
            below. False (TRIOBJ_DCPO_V2 KARPATHY lock "v2 mode byte-identical"):
            LEGACY unclosed-to-end behavior verbatim — unclosed content stays in
            META_CONTENT + the conf-parse spans, FORMAT_VIOLATION stays all-zero
            and meta_unclosed/meta_drift stay False.
        fmt: optional classify_dcpo_format output for THIS row (v3k three-tier
            spec §6-1b). None (default) -> legacy behavior verbatim (v2 +
            v3-pre-k callers unchanged). When given, the parser is the single
            source of truth for {wellformed, drift, discard}:
              wellformed -> opener <|meta|> INCLUDED in META_CONTENT (R_meta
                teaches WHEN to start meta) + FORMAT_OK at the closer
                (suppressed when fmt_replaced — replaced rows carry NO format
                positions, spec §3-tier-1 / risk 3);
              drift      -> META_CONTENT = recovered span (CONF parsed inside
                it), the double-duty </think> joins META_REGION as the de-facto
                closer and is the ONLY FORMAT_VIOLATION position; answer after
                it reverts to ANSWER_REGION;
              discard    -> ANSWER/META_CONTENT/CONF all-zero (regions
                untrustworthy), FORMAT_VIOLATION = every garbage delimiter.
            no_meta / truncation fall through to the legacy scan (identical
            handling by construction).
        fmt_replaced: True = this row was tier-1 token-REPLACED at the CF-wrap
            site (fmt classifies the post-replacement ids as wellformed). Full
            normal routing, but FORMAT_OK stays empty (R_format=0 rows must not
            sit in the group-centered FORMAT_OK head).

    Returns dict of np.bool_ arrays of shape [T] (+ two python bools):
        META_REGION      : tag-inclusive block (open..close inclusive).
        META_CONTENT     : strictly-inside content (tags EXCLUDED).
        CONF             : confidence-number token run inside META_CONTENT (first per block).
        ANSWER_REGION    : response_mask & ~META_REGION (tags NOT in answer).
        FORMAT_VIOLATION : the clamped DRIFT-block tokens (case a below; zeros otherwise) —
                           the 4th routed head (R_format) lands ONLY here. With
                           `fmt`: per the v3k table (drift = the single </think>
                           index; discard = garbage delimiter indices).
        FORMAT_OK        : the closer <|/meta|> of an ORIGINALLY-wellformed row
                           (v3k two-sided format signal; +side of the R_format
                           head). All-zero unless `fmt` says wellformed and NOT
                           fmt_replaced. Legacy callers: always all-zero.
        fmt_class        : the fmt['fmt_class'] string (None for legacy callers).
        meta_unclosed    : bool — ANY unclosed meta span (case a OR b) → R_meta gate.
        meta_drift       : bool — case a only (format habit) → drives the -1 penalty.

    UNCLOSED meta handling (live-run finding: 40% of meta blocks never emit
    <|/meta|>; the old unclosed-to-end rule put the FINAL ANSWER inside
    META_CONTENT for 17% of rollouts → R_corr never reached the answer):
      a. DRIFT — a </think> token appears AFTER the open: META_REGION is clamped to
         open..(think_close-1); </think> itself and everything after revert to
         ANSWER_REGION. The clamped tokens go into META_REGION but NOT META_CONTENT
         (neutral, like tag tokens) and into FORMAT_VIOLATION. No conf parse (gated).
      b. TRUNCATION — no </think> after the open (cut at max length): META_REGION
         keeps the unclosed-to-end extent, but the tokens are EXCLUDED from
         META_CONTENT and from the conf-parse spans (gated; a truncated CF is
         useless anyway). NOT a violation — truncation is a length problem, not a
         format habit, so no penalty.

    Invariants (asserted in tests):
        CONF ⊆ META_CONTENT ⊆ META_REGION ⊆ response_mask;
        FORMAT_VIOLATION ⊆ META_REGION; FORMAT_VIOLATION ∩ META_CONTENT = ∅;
        META_CONTENT and ANSWER_REGION disjoint; ANSWER_REGION ∪ META_REGION == response_mask;
        tag tokens ∈ (META_REGION \\ META_CONTENT \\ ANSWER_REGION).
    """
    ids = [int(t) for t in (resp_ids.tolist() if hasattr(resp_ids, "tolist") else list(resp_ids))]
    rm = response_mask.tolist() if hasattr(response_mask, "tolist") else list(response_mask)
    rmask = np.asarray([bool(x) for x in rm], dtype=bool)
    T = len(ids)
    # Align lengths defensively.
    if rmask.shape[0] < T:
        rmask = np.concatenate([rmask, np.zeros(T - rmask.shape[0], dtype=bool)])
    elif rmask.shape[0] > T:
        rmask = rmask[:T]

    META_REGION = np.zeros(T, dtype=bool)
    META_CONTENT = np.zeros(T, dtype=bool)
    CONF = np.zeros(T, dtype=bool)
    FORMAT_VIOLATION = np.zeros(T, dtype=bool)
    FORMAT_OK = np.zeros(T, dtype=bool)   # v3k +side of the R_format head
    meta_unclosed = False  # case a OR b — the R_meta gate
    meta_drift = False     # case a only — drives the format penalty

    def _finalize_unclosed(o_idx, end_idx):
        """Finalize an UNCLOSED meta span (open at o_idx, last real token end_idx).

        Case a (DRIFT): a </think> appears after the open → clamp META_REGION (and
        FORMAT_VIOLATION) to open..(think_close-1); </think> + everything after
        revert to ANSWER. Case b (TRUNCATION): unclosed-to-end META_REGION kept.
        BOTH cases: NO META_CONTENT, NO conf span (gated — no spans entry).

        clamp_unclosed=False (v2 byte-identical): LEGACY unclosed-to-end —
        content + conf span kept, no gate/violation flags.
        """
        nonlocal meta_unclosed, meta_drift
        if not clamp_unclosed:
            # LEGACY (pre-v3, verbatim): close the block at the last valid token.
            if end_idx >= o_idx + 1:
                META_CONTENT[o_idx + 1 : end_idx + 1] = True
                spans.append((o_idx + 1, end_idx + 1))
            META_REGION[o_idx : end_idx + 1] = True
            return
        meta_unclosed = True
        j = None
        for k in range(o_idx + 1, end_idx + 1):
            if ids[k] == think_close:
                j = k
                break
        if j is not None:  # case a — format drift
            META_REGION[o_idx : j] = True
            FORMAT_VIOLATION[o_idx : j] = True
            meta_drift = True
        else:              # case b — true truncation (no penalty)
            META_REGION[o_idx : end_idx + 1] = True

    spans = []  # list of (content_lo, content_hi_exclusive) — Pass B conf input
    # Pass A′ — v3k fmt-DRIVEN region construction (spec §6-1b). The parser
    # (classify_dcpo_format) already resolved the regions; NO re-scan here —
    # single source of truth (Karpathy lock). Only {wellformed, drift, discard}
    # are fmt-driven: no_meta / truncation produce byte-identical masks from the
    # legacy scan by construction, so they fall through to Pass A below.
    _fmt_cls = fmt.get("fmt_class") if fmt is not None else None
    _fmt_driven = _fmt_cls in ("wellformed", "drift", "discard")
    if _fmt_driven:
        span = fmt.get("meta_content_span")
        if _fmt_cls == "wellformed":
            lo, hi = span                      # hi (exclusive) == the closer index
            opener, closer = lo - 1, hi
            META_REGION[opener : closer + 1] = True
            # v3k §3: the opener tag JOINS META_CONTENT (R_meta teaches WHEN to
            # start meta). The closer stays REGION-only — it is the FORMAT_OK
            # target instead, but ONLY for ORIGINALLY-wellformed rows: replaced
            # rows carry NO format positions (R_format=0 + membership in the
            # group-centered FORMAT_OK head would route NEGATIVE advantage onto
            # the corrected tags — spec risk 3).
            META_CONTENT[opener:hi] = True
            if hi > lo:
                spans.append((lo, hi))
            if not fmt_replaced:
                for p in (fmt.get("format_ok_positions") or []):
                    FORMAT_OK[p] = True
        elif _fmt_cls == "drift":
            lo, hi = span                      # hi == the double-duty </think> index
            opener = lo - 1
            # Tier-3 lenient recovery: META_CONTENT = the recovered signature
            # span (plays R_meta + conf). The double-duty </think> acts as the
            # de-facto closer — REGION (not CONTENT, not ANSWER) AND the single
            # FORMAT_VIOLATION position: R_format=-1 lands on the wrong token
            # ITSELF (kills the R_corr leak where a correct drifted rollout
            # reinforced </think> at w1.0). Everything after reverts to ANSWER.
            META_REGION[opener : hi + 1] = True
            META_CONTENT[lo:hi] = True
            if hi > lo:
                spans.append((lo, hi))
            for p in (fmt.get("violation_positions") or []):
                FORMAT_VIOLATION[p] = True
            meta_unclosed = True   # continuity: textual unclosed = drift + truncation
            meta_drift = True
        else:  # discard — regions untrustworthy: flow NOTHING at token level;
            #    FORMAT_VIOLATION flags every identifiable garbage delimiter.
            for p in (fmt.get("violation_positions") or []):
                FORMAT_VIOLATION[p] = True
    # Pass A — meta spans (mirrors meta_inject.meta_mask scan; unclosed spans are
    # GATED via _finalize_unclosed — drift-clamped or truncation-neutralized).
    # SKIPPED when fmt-driven (Pass A′ above already placed the regions).
    in_meta = False
    open_idx = None
    content_start = None
    last_valid = -1
    for i in range(T if not _fmt_driven else 0):
        if not rmask[i]:
            if in_meta:
                # pad while open = unclosed -> drift-clamp or truncation-gate.
                _finalize_unclosed(open_idx, last_valid)
                in_meta = False
            continue
        last_valid = i
        t = ids[i]
        if t == meta_open:
            if in_meta:
                # EDGE: nested/dup open. If a </think> INTERVENED, the previous
                # span is the SAME drift class as _finalize_unclosed case a —
                # a dup open must not silently "close" a drifted span, or the
                # post-</think> ANSWER tokens land in META_CONTENT (the exact
                # misrouting this fix kills: `open…</think>…answer…open`).
                # Clamp/violate it; otherwise force-close previous span at i-1.
                if clamp_unclosed and any(
                    ids[k] == think_close for k in range(open_idx + 1, i)
                ):
                    _finalize_unclosed(open_idx, i - 1)
                else:
                    hi = i  # exclusive
                    if hi - 1 >= content_start:
                        META_CONTENT[content_start : hi] = True
                        spans.append((content_start, hi))
                    META_REGION[open_idx : i] = True
            in_meta = True
            open_idx = i
            content_start = i + 1
            META_REGION[i] = True  # TAG in REGION, NOT CONTENT
        elif t == meta_close and in_meta:
            META_REGION[open_idx : i + 1] = True
            if i - 1 >= content_start:
                META_CONTENT[content_start : i] = True  # exclude BOTH tag tokens
            spans.append((content_start, i))  # (lo, hi) hi exclusive = close idx
            in_meta = False
        elif t == meta_close and not in_meta:
            pass  # EDGE: stray close -> ignore (default)
    if in_meta:  # EDGE: missing close — drift-clamp (case a) or truncation-gate (case b)
        _finalize_unclosed(open_idx, last_valid)

    # Pass B — confidence run via EXACT char-span -> token-span map.
    for (lo, hi) in spans:
        if hi <= lo:
            continue
        toks = ids[lo:hi]
        text = decode_fn(toks)
        span = _parse_confidence_charspan(text)
        if span is None:
            continue
        cs, ce = span
        # cumulative char-offset table over content tokens: offsets[j] = len(decode(toks[:j]))
        offsets = [0] * (len(toks) + 1)
        for j in range(1, len(toks) + 1):
            offsets[j] = len(decode_fn(toks[:j]))
        # tokens whose [offsets[j], offsets[j+1]) overlaps [cs, ce)
        k0 = None
        for j in range(len(toks)):
            if offsets[j + 1] > cs:
                k0 = j
                break
        k1 = None
        for j in range(len(toks) - 1, -1, -1):
            if offsets[j] < ce:
                k1 = j
                break
        if k0 is None or k1 is None or k1 < k0:
            continue
        CONF[lo + k0 : lo + k1 + 1] = True
        # Round-trip guard (spec §4 Pass B.3): the CONF token span must decode back
        # to a surface containing the SAME numeric literal the regex matched. We
        # compare the matched number substring (text[cs:ce]) against the decoded
        # CONF span — NOT _parse_confidence(span), because the number tokens alone
        # lack the "confidence:" keyword the keyword-gated parser needs.
        rt = decode_fn(ids[lo + k0 : lo + k1 + 1])
        if text[cs:ce] not in rt:
            # boundary mis-map (rare): drop CONF for this block rather than mislabel.
            CONF[lo + k0 : lo + k1 + 1] = False
            continue
        break  # FIRST conf per rollout only

    # Pass C — answer region = response minus the FULL meta block (tag-inclusive).
    ANSWER_REGION = rmask & ~META_REGION
    if _fmt_cls == "discard":
        # Tier-2: ANSWER zeroed too (flowing anything = misrouting). The row's
        # ONLY token-level signal is R_format=-1 on the garbage delimiters.
        ANSWER_REGION = np.zeros(T, dtype=bool)

    return {
        "META_REGION": META_REGION,
        "META_CONTENT": META_CONTENT,
        "CONF": CONF,
        "ANSWER_REGION": ANSWER_REGION,
        "FORMAT_VIOLATION": FORMAT_VIOLATION,
        "FORMAT_OK": FORMAT_OK,        # v3k +side (all-zero for legacy callers)
        "fmt_class": _fmt_cls,          # None for legacy (fmt=None) callers
        "meta_unclosed": meta_unclosed,
        "meta_drift": meta_drift,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §2.2 — the three region rewards (raw per-rollout scalars)
# ─────────────────────────────────────────────────────────────────────────────
def _all_boxed(text: str):
    return [m.strip() for m in _BOXED_RE.findall(text or "")]


def dcpo_region_rewards(
    completions,
    ground_truth=None,
    group_index=None,
    step: int = 0,
    *,
    cf_completions=None,
    cf_correct=None,
    # gate_unclosed: True (v3 default) = unclosed-meta R_meta gate + drift
    # format_penalty below. False (TRIOBJ_DCPO_V2 KARPATHY lock "v2 mode
    # byte-identical"): gate/penalty OFF — R_meta follows the plain cf path and
    # format_penalty/meta_unclosed stay all-zero.
    gate_unclosed: bool = True,
    # fmt_class: optional len-B list of classify_dcpo_format classes (v3k
    # three-tier spec §6-1c) — the parser is the single source of truth; the
    # text-level unclosed mirror below is SKIPPED when this is given. Values:
    # no_meta | wellformed | swapped | dup_open | reversed | drift | truncation
    # | discard. Tier-1 names appear ONLY for rows that were token-REPLACED at
    # the CF-wrap site (unreplaced tier-1 rows arrive demoted to 'discard' via
    # classify_dcpo_format(tier1_to_discard=True)). Per-class head routing (§4):
    #   wellformed             full heads, R_format = +1 (FORMAT_OK side)
    #   replaced (tier-1 name) full heads, R_format =  0 (no conflicting signal
    #                          on the corrected tags — replacement+advantage
    #                          does the teaching)
    #   drift                  R_meta UNGATED (tier-3 plays the CF), R_format=-1
    #   discard                R_corr = R_meta = R_cal = 0, R_format = -1
    #   truncation             R_meta gated to 0 (length, not habit), R_format=0
    #   no_meta                unchanged (R_meta naturally 0), R_format = 0
    # None (default) -> pre-v3k behavior verbatim (and gate_unclosed=False ->
    # v2 verbatim).
    fmt_class=None,
    # v2 carry-over reward knobs (eps/eps_right_right/p_lo/p_hi/warmup_steps/sandbag_*/
    # format_*) are absorbed here and IGNORED — v3's R_meta = c_with - c_without uses
    # none of them. Kept only so legacy callers don't raise TypeError.
    **cfg,
):
    """Compute the three raw region reward heads per rollout (TRIOBJ_DCPO_V3, spec §4).

    R_corr : +1 if final answer correct else -1 (routes to ANSWER span).
    R_meta : c_with - c_without = correct(main) - correct(counterfactual) ∈ {-1,0,+1}
             (routes to META_CONTENT). 0 when no counterfactual was supplied for a
             rollout (cf_correct[i] is None and no <|meta|> in the text fallback).
    R_cal  : per-instance Brier -(conf - c_with)^2 on parsed confidence; 0 if no conf
             (routes to CONF).

    Counterfactual ablation (the v3 causal meta-utility). For each rollout the
    'without-meta' correctness c_without is supplied by the PRODUCER (verl_sdc,
    §3) as either:
      - cf_correct: a parallel array of pre-graded 1.0/0.0/None floats, OR
      - cf_completions: a parallel list of regenerated CF rollout texts (graded here).
    When NEITHER is supplied, a TEXT FALLBACK grades the pre-(first-)meta prefix
    (`cf_answer_from_prefix`) so the head stays functional from a single rollout;
    when even that yields no answer, c_without is None -> R_meta = 0 (conservative
    under-credit, never wrong-sign — spec §10 risk 6).

    Args:
        completions: list of TRL-format completions (len B).
        ground_truth: list of gold strings (len B) or None.
        group_index: per-rollout group id (uid). Rollouts sharing an id form a GRPO
            group. None -> all rollouts in one group.
        step: trainer step (diagnostics only).
        cf_completions: optional parallel list (len B) of regenerated counterfactual
            rollout texts (meta suppressed). None -> use cf_correct or text fallback.
        cf_correct: optional parallel array (len B) of pre-graded CF correctness
            (1.0 / 0.0 / None). Takes precedence over cf_completions.

    UNCLOSED-meta gate + format penalty (mirrors build_dcpo_region_masks): a row
    whose ONLY meta is unclosed (no <|/meta|> in the text) has R_meta FORCED to 0
    regardless of cf grading, and earns format_penalty = -1.0 iff it DRIFTED
    (a </think> after the last open); pure truncation stays 0 (length, not habit).

    Returns dict of lists (len B): R_corr, R_meta, R_cal, p_hat, group_acc,
    format_penalty (the 4th routed head), meta_unclosed (1.0/0.0 diagnostics),
    plus constant canary_pass1_acc / sandbag_clamp stubs (kept so the existing
    wandb keys + _populate_dcpo_region_keys stay alive without touching verl_sdc).
    """
    B = len(completions)
    texts = [_get_text(c) for c in completions]
    gts = [ground_truth[i] if ground_truth is not None else "" for i in range(B)]

    def _cf_get(arr, i):
        if arr is None:
            return None
        try:
            return arr[i]
        except (IndexError, KeyError, TypeError):
            return None

    # Per-rollout primitives:
    #   c2 = correctness of the FINAL (graded) answer == c_with.
    #   conf = parsed confidence inside the meta block.
    #   c_without = counterfactual correctness (producer / text fallback), or None.
    answer2 = [None] * B   # final (graded)
    c2 = [False] * B
    conf = [None] * B
    has_meta = [False] * B
    meta_unclosed = [False] * B   # ONLY meta is unclosed -> R_meta GATE (forced 0)
    meta_drift = [False] * B      # unclosed AND </think> after it -> format penalty
    c_without = [None] * B   # None == no counterfactual available -> R_meta 0
    for i in range(B):
        t = texts[i]
        final = _extract_answer_fallback(t)
        answer2[i] = final
        c2[i] = bool(_check_correctness(final, gts[i])) if final else False
        has_meta[i] = "<|meta|>" in (t or "")
        conf[i] = _parse_confidence(t)
        # UNCLOSED-meta gate (text-level mirror of build_dcpo_region_masks'
        # meta_unclosed/meta_drift): a rollout whose ONLY meta is unclosed (no
        # <|/meta|> anywhere) gets R_meta forced to 0 — its meta content is
        # un-routable (drift-clamped / truncation-gated at the mask level), so
        # crediting/penalizing it via the CF would reward a broken block. DRIFT
        # (a </think> AFTER the last open) additionally earns the format penalty;
        # pure truncation does not (length problem, not a format habit).
        # gate_unclosed=False (v2 byte-identical): both flags stay False.
        # v3k: when fmt_class is supplied the PARSER drives the gates instead —
        # meta_unclosed keeps its textual meaning (drift + truncation) for the
        # dcpo/meta_unclosed_rate continuity, but it no longer gates drift
        # (tier-3: drift plays R_meta; only truncation stays gated below).
        if fmt_class is not None:
            meta_unclosed[i] = fmt_class[i] in ("drift", "truncation")
            meta_drift[i] = fmt_class[i] == "drift"
        elif gate_unclosed and has_meta[i] and "<|/meta|>" not in (t or ""):
            meta_unclosed[i] = True
            _last_open = (t or "").rfind("<|meta|>")
            meta_drift[i] = (t or "").find("</think>", _last_open) != -1

        # c_without[i]: precedence cf_correct -> cf_completions -> text fallback.
        # NaN-GUARD (v3b BUG-2): cf_correct may arrive as np.float32 with NaN
        # sentinels for skipped rows; np.float32 is NOT a python-float subclass so
        # isinstance-gated NaN checks upstream can miss it, and bool(nan) is True.
        # `cw == cw` is False only for NaN regardless of float type.
        cw = _cf_get(cf_correct, i)
        if cw is not None and cw == cw:
            c_without[i] = bool(cw)
        else:
            cf_txt = _cf_get(cf_completions, i)
            if cf_txt is not None:
                cf_txt = _get_text(cf_txt)
                # CF LEAK GUARD: logit_bias bans both tag ids, but the model can
                # still emit UNSTRUCTURED meta content (confidence:/assessment:/
                # action: lines — the swapped-class lesson). A leaked CF is not a
                # meta-free counterfactual; grading it corrupts c_without, so be
                # conservative: treat as ungraded (None → R_meta 0 for the row).
                if cf_txt and _has_meta_signature(cf_txt):
                    c_without[i] = None
                else:
                    cf_ans = _extract_answer_fallback(cf_txt) if cf_txt else None
                    c_without[i] = bool(_check_correctness(cf_ans, gts[i])) if cf_ans else None
            else:
                cf_ans = cf_answer_from_prefix(t)   # text-level pre-meta fallback
                c_without[i] = (
                    bool(_check_correctness(cf_ans, gts[i])) if cf_ans is not None else None
                )

    # Group ids -> p_hat / group_acc (diagnostics only; NOT an R_meta gate in v3).
    if group_index is None:
        gid = [0] * B
    else:
        gid = list(group_index.tolist() if hasattr(group_index, "tolist") else group_index)
        gid = [str(g) for g in gid]

    groups: dict = {}
    for i in range(B):
        groups.setdefault(gid[i], []).append(i)
    p_hat = [0.0] * B
    group_acc = [0.0] * B
    for members in groups.values():
        ga = float(np.mean([1.0 if c2[i] else 0.0 for i in members]))
        for i in members:
            p_hat[i] = ga
            group_acc[i] = ga

    R_corr = [0.0] * B
    R_meta = [0.0] * B
    R_cal = [0.0] * B
    for i in range(B):
        # R_corr — lenient final-answer correctness (same extractor as correctness head).
        R_corr[i] = 1.0 if c2[i] else -1.0

        # R_meta — CAUSAL meta-utility = c_with - c_without ∈ {-1,0,+1}.
        #   with-right / without-wrong -> +1  (meta turned wrong right)
        #   both-right / both-wrong     ->  0  (meta did not change the outcome)
        #   with-wrong / without-right -> -1  (meta turned right wrong)
        #   no counterfactual available ->  0  (no-meta / unparseable, conservative)
        c_with = 1.0 if c2[i] else 0.0
        # v3k gate: drift is UNGATED (tier-3 recovered span plays the CF), only
        # truncation keeps the forced-0 (length, not habit). Pre-v3k path gates
        # on meta_unclosed (drift AND truncation) as before.
        _gated = (
            fmt_class[i] == "truncation" if fmt_class is not None else meta_unclosed[i]
        )
        if _gated:
            # GATE: force 0 regardless of cf grading. has_meta stays True for
            # emission metrics.
            R_meta[i] = 0.0
        elif c_without[i] is None:
            R_meta[i] = 0.0
        else:
            R_meta[i] = c_with - (1.0 if c_without[i] else 0.0)

        # R_cal — per-instance Brier against c_with; 0 if conf missing (no floor).
        if conf[i] is not None:
            R_cal[i] = -((conf[i] - c_with) ** 2)
        else:
            R_cal[i] = 0.0

        # v3k tier-2 DISCARD: regions untrustworthy — flowing anything is
        # misrouting, so ALL THREE heads are zeroed at the scalar level too
        # (the masks are already all-zero; the 0 scalars keep the row from
        # injecting ±1 into sibling group means — spec §3-tier-2). NOTE: a 0
        # still biases the Dr.GRPO baseline toward 0 (the row's true reward was
        # ±1), so compose additionally EXCLUDES discard rows from the three
        # content-head group means via member_mask (spec §10 risk 2, CLOSED —
        # the populator writes dcpo_head_member from these fmt classes).
        if fmt_class is not None and fmt_class[i] == "discard":
            R_corr[i] = 0.0
            R_meta[i] = 0.0
            R_cal[i] = 0.0

    # DIAGNOSTIC dump (guarded by DCPO_DEBUG, default on): print what the PRODUCER
    # actually sees for sample 0 — the real RL rollout text + parsed conf/answers +
    # the counterfactual delta — so the head behavior is visible in the amlt logs.
    import os as _os
    if _os.environ.get("DCPO_DEBUG", "1") == "1" and B:
        _t = (texts[0] or "")
        _cw = c_without[0]
        _cw_s = "None" if _cw is None else ("1" if _cw else "0")
        print(
            f"[DCPO_DBG] step={step} hasMetaTag={has_meta[0]} "
            f"conf={conf[0]} c_with={1 if c2[0] else 0} c_without={_cw_s} cf={'Y' if (cf_correct is not None or cf_completions is not None) else 'fallback'} "
            f"ans2={answer2[0]!r} "
            f"R_corr={R_corr[0]:.3f} R_meta={R_meta[0]:.4f} R_cal={R_cal[0]:.4f} "
            f"p_hat={p_hat[0]:.2f} | text_tail={_t[-260:]!r}",
            flush=True,
        )

    return {
        "R_corr": R_corr,
        "R_meta": R_meta,
        "R_cal": R_cal,
        "p_hat": p_hat,
        "group_acc": group_acc,
        # Per-rollout diagnostics for the wandb rollout TABLE (observability ask
        # after the v3b silent-signal bug): c_without None -> nan so the column
        # stays numeric ("no counterfactual" is visibly distinct from 0/1).
        "c_with": [1.0 if c2[i] else 0.0 for i in range(B)],
        "c_without": [
            float("nan") if c_without[i] is None else (1.0 if c_without[i] else 0.0)
            for i in range(B)
        ],
        "conf": [float("nan") if conf[i] is None else float(conf[i]) for i in range(B)],
        "has_meta": list(has_meta),
        # FORMAT head (4th routed head, w_format). v3k (fmt_class given): the
        # full §4 table — +1 wellformed (routed onto FORMAT_OK at the closer),
        # -1 drift/discard (routed onto FORMAT_VIOLATION), 0 for replaced
        # (tier-1 names) / truncation / no_meta. ONE head, group-mean-subtracted
        # ONCE by compose; FORMAT_OK ∪ FORMAT_VIOLATION are per-row disjoint so
        # the centered value lands only on the row's own positions.
        # Pre-v3k (fmt_class=None): -1 ONLY for drift, verbatim as before.
        "format_penalty": (
            [
                1.0 if c == "wellformed" else (-1.0 if c in ("drift", "discard") else 0.0)
                for c in fmt_class
            ]
            if fmt_class is not None
            else [-1.0 if meta_drift[i] else 0.0 for i in range(B)]
        ),
        # v3k observability echo: the per-row parser class (None pre-v3k). The
        # trend scalars (replaced/discard/drift/wellformed rates) and the
        # rollout-table fmt_class column read this from the stash.
        "fmt_class": (list(fmt_class) if fmt_class is not None else None),
        # Diagnostics: 1.0 = the row's only meta is unclosed (gated R_meta) — the
        # trend scalar dcpo/meta_unclosed_rate charts this per step.
        "meta_unclosed": [1.0 if meta_unclosed[i] else 0.0 for i in range(B)],
        "answer": [a if a is not None else "" for a in answer2],
        # Constant stubs: kept so existing wandb keys + _populate_dcpo_region_keys
        # stay alive without touching verl_sdc.py (spec §5.1-B.6). v3 has no canary
        # / sandbag clamp (the counterfactual is anti-hack by construction, §8).
        "canary_pass1_acc": [1.0] * B,
        "sandbag_clamp": [1.0] * B,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §2.3 — per-region advantage composition (torch-only; no verl/omegaconf deps so
# this is unit-testable under a minimal env). verl_sdc_utils delegates here.
# ─────────────────────────────────────────────────────────────────────────────
def group_mean_subtract(values, index, member=None):
    """Dr.GRPO block-wise group centering: subtract group mean, NO /std.

    Args:
        values: [B] or [B,1] per-rollout scalars (tensor / array / list).
        index: per-rollout group id (uid array / list); rollouts sharing an id
            form a group. None -> single group.
        member: OPTIONAL [B] 0/1 membership (tensor / array / list) — v3k
            tier-2 EXCLUSION semantics (spec §10 risk 2, now CLOSED): a
            discard's forced-0 scalar is NOT a real reward (its true ±1 was
            untrustworthy), so averaging it in shifts every sibling's baseline
            by (d/n)·mean(included) — e.g. one discard in an all-correct group
            of 4 spuriously reinforces every sibling at +0.25 where exclusion
            gives the correct no-gradient 0. Rows with member==0 contribute
            NOTHING to their group's mean AND receive a centered value of 0
            (their region masks are all-zero anyway, so the row itself is
            unaffected either way). None -> all rows included (byte-identical
            pre-fix behavior; v2 / pre-k callers never pass it).

    Returns:
        [B,1] centered tensor. Degenerate (singleton / all-equal / no-member)
        groups -> 0.
    """
    v = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    B = v.shape[0]
    if index is None:
        gid = ["__g0__"] * B
    else:
        gid = list(index.tolist() if hasattr(index, "tolist") else index)
        gid = [str(g) for g in gid]
    mem = None
    if member is not None:
        mem = torch.as_tensor(member, dtype=torch.float32).reshape(-1).to(v.device)
    out = torch.zeros_like(v)
    groups: dict = {}
    for i, g in enumerate(gid):
        groups.setdefault(g, []).append(i)
    for members in groups.values():
        idx = torch.tensor(members, dtype=torch.long, device=v.device)
        if mem is not None:
            idx = idx[mem[idx] > 0.5]
            if idx.numel() == 0:
                continue  # all-discard group: nobody to center against -> all 0
        out[idx] = v[idx] - v[idx].mean()
    return out.unsqueeze(1)


def compose_dcpo_region_advantage(
    *,
    response_mask,
    index,
    R_corr,
    R_meta,
    R_cal,
    answer_mask,
    meta_content_mask,
    conf_mask,
    w_corr: float = 1.0,
    w_meta: float = 0.5,
    w_cal: float = 0.3,
    R_format=None,
    format_violation_mask=None,
    w_format: float = 0.1,
    format_ok_mask=None,
    member_mask=None,
    meta_floor: float = 0.0,
    floor_mask=None,
):
    """Independent per-head group-mean-subtract + per-region token routing (§2.3).

        A_token = ( w_corr*Â_corr*ANSWER
                  + w_meta*Â_meta*META_CONTENT
                  + w_cal *Â_cal *CONF ) * response_mask
                [ + w_format*Â_format*FORMAT_VIOLATION * response_mask ]

    TAG tokens are in NEITHER ANSWER nor META_CONTENT -> advantage 0. NO global
    re-whiten (codex-r13 LOCK). Returns (A, A).

    4th head (OPTIONAL, v3 format-penalty): R_format [B] is group-mean-subtracted
    independently (same Dr.GRPO centering) and routed ONLY onto the
    FORMAT_VIOLATION token mask (the drift-clamped block). BACKWARD COMPAT:
    R_format / format_violation_mask default None -> the term is skipped and the
    output is byte-identical to the 3-head compose (v2 mode + existing callers).

    v3k two-sided format signal (OPTIONAL `format_ok_mask`): when given, the
    SAME centered Â_format is routed onto FORMAT_OK ∪ FORMAT_VIOLATION — the
    masks are per-row disjoint by construction (a row is exactly one class), so
    positive-relative advantage lands on wellformed closers and
    negative-relative on drift `</think>` / discard garbage. format_ok_mask=None
    -> byte-identical to the pre-v3k 4-head compose.

    v3k tier-2 exclusion (OPTIONAL `member_mask`, spec §10 risk 2 CLOSED):
    [B] 0/1 — rows with 0 (DISCARD) are EXCLUDED from the R_corr/R_meta/R_cal
    group means (their forced-0 scalars are not real rewards; averaging them in
    shifts every sibling by (d/n)·mean(siblings)). The FORMAT head deliberately
    keeps EVERY row: discard's -1 vs wellformed's +1 IS the intended relative
    format signal. member_mask=None -> byte-identical to the pre-fix compose.

    v3m anti-collapse FLOOR (OPTIONAL `meta_floor` + `floor_mask`): a small
    POSITIVE, UN-CENTERED advantage bias added onto the META_CONTENT tokens of
    TRUSTED-meta rows (floor_mask[B] 0/1 = wellformed/replaced/drift-recovered).
    It is added AFTER the Dr.GRPO group-mean-subtract on purpose: a constant
    folded into R_meta BEFORE centering cancels (the group mean absorbs any term
    common to all rows), silently doing nothing. Routed post-centering it
    survives, giving "emit a trusted wellformed meta" a fixed +meta_floor pull
    that offsets the FORMAT-penalty collapse pressure (v3l: meta_emit 0.5→0 by
    step 60). The CENTERED Â_meta still rides on top, so R_meta keeps deciding
    useful-vs-harmful meta — the floor only keeps the channel OPEN, it does not
    grade content. meta_floor=0.0 / floor_mask=None -> byte-identical.
    """
    rm = torch.as_tensor(response_mask, dtype=torch.float32)
    device = rm.device

    A_corr = group_mean_subtract(R_corr, index, member=member_mask).to(device)  # [B,1]
    A_meta = group_mean_subtract(R_meta, index, member=member_mask).to(device)  # [B,1]
    A_cal = group_mean_subtract(R_cal, index, member=member_mask).to(device)    # [B,1]

    ans = torch.as_tensor(answer_mask, dtype=torch.float32).to(device)
    meta_c = torch.as_tensor(meta_content_mask, dtype=torch.float32).to(device)
    conf = torch.as_tensor(conf_mask, dtype=torch.float32).to(device)

    advantages = (
        w_corr * A_corr * ans
        + w_meta * A_meta * meta_c
        + w_cal * A_cal * conf
    ) * rm

    if R_format is not None and format_violation_mask is not None:
        A_format = group_mean_subtract(R_format, index).to(device)  # [B,1]
        fv = torch.as_tensor(format_violation_mask, dtype=torch.float32).to(device)
        if format_ok_mask is not None:
            # v3k union routing: FORMAT_OK ∩ FORMAT_VIOLATION = ∅ per row, so a
            # plain sum IS the union (no clamp needed) — ONE centered head, each
            # row's own positions only.
            fv = fv + torch.as_tensor(format_ok_mask, dtype=torch.float32).to(device)
        advantages = advantages + w_format * A_format * fv * rm

    # v3m anti-collapse floor: UN-CENTERED +meta_floor PER TRUSTED-META ROW
    # (added AFTER centering so it survives — see docstring). Spread evenly over
    # the row's META_CONTENT tokens so the row TOTAL is exactly +meta_floor,
    # length-NEUTRAL: a per-token +meta_floor would pay 0.1×(#meta tokens),
    # rewarding verbose useless meta (a length-farm hack). R_meta keeps its
    # per-token routing (Â_meta>0 only for USEFUL meta, so length there is fine);
    # the floor is a flat emission bonus, not a "write more meta" incentive.
    if meta_floor and floor_mask is not None:
        fl = torch.as_tensor(floor_mask, dtype=torch.float32).to(device).view(-1, 1)  # [B,1]
        meta_in_resp = meta_c * rm
        row_n = meta_in_resp.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B,1] meta-token count
        advantages = advantages + float(meta_floor) * fl * (meta_in_resp / row_n)

    return advantages, advantages
