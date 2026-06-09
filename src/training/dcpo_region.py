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


# ─────────────────────────────────────────────────────────────────────────────
# §2.7 / §4 — token-region mask util
# ─────────────────────────────────────────────────────────────────────────────
def build_dcpo_region_masks(
    resp_ids,
    response_mask,
    decode_fn: Callable[[list], str],
    meta_open: int = META_OPEN_DEFAULT,
    meta_close: int = META_CLOSE_DEFAULT,
):
    """Build the four DCPO region masks over response positions.

    Args:
        resp_ids: sequence of response token ids (list / 1-D array / tensor).
        response_mask: bool/0-1 per-token mask (True = real response token, not pad).
        decode_fn: ids(list[int]) -> str. Injected so the util is importable with no
            transformers dependency (unit tests pass a fake decode_fn).
        meta_open / meta_close: the <|meta|> / <|/meta|> token ids.

    Returns dict of np.bool_ arrays of shape [T]:
        META_REGION   : tag-inclusive block (open..close inclusive).
        META_CONTENT  : strictly-inside content (tags EXCLUDED).
        CONF          : confidence-number token run inside META_CONTENT (first per block).
        ANSWER_REGION : response_mask & ~META_REGION (tags NOT in answer).

    Invariants (asserted in tests):
        CONF ⊆ META_CONTENT ⊆ META_REGION ⊆ response_mask;
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

    # Pass A — meta spans (mirrors meta_inject.meta_mask scan + unclosed-to-end).
    spans = []  # list of (content_lo, content_hi_exclusive)
    in_meta = False
    open_idx = None
    content_start = None
    last_valid = -1
    for i in range(T):
        if not rmask[i]:
            if in_meta:
                # pad while open = truncation -> close the block at the last valid token.
                if last_valid >= content_start:
                    META_CONTENT[content_start : last_valid + 1] = True
                    spans.append((content_start, last_valid + 1))
                META_REGION[open_idx : last_valid + 1] = True
                in_meta = False
            continue
        last_valid = i
        t = ids[i]
        if t == meta_open:
            if in_meta:
                # EDGE: nested/dup open -> force-close previous span at i-1.
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
    if in_meta:  # EDGE: missing close (truncation)
        if last_valid >= content_start:
            META_CONTENT[content_start : last_valid + 1] = True
            spans.append((content_start, last_valid + 1))
        META_REGION[open_idx : last_valid + 1] = True

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

    return {
        "META_REGION": META_REGION,
        "META_CONTENT": META_CONTENT,
        "CONF": CONF,
        "ANSWER_REGION": ANSWER_REGION,
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
    eps: float = 0.1,
    eps_right_right: bool = False,
    p_lo: float = 0.2,
    p_hi: float = 0.8,
    warmup_steps: int = 200,
    sandbag_clamp: bool = True,
    sandbag_floor: float = 0.05,
    format_credit: float = 0.05,
    format_penalty: float = 0.05,
    **cfg,
):
    """Compute the three raw region reward heads per rollout (spec §2.2).

    R_corr : +1 if last-boxed correct else -1 (routes to ANSWER span).
    R_meta : flat-+1 transition table, group-warrant gated (routes to META_CONTENT).
    R_cal  : per-instance Brier on parsed confidence; 0 if no conf (routes to CONF).

    Group warrant: p_hat = mean over the GROUP of correct(answer1); warranted iff
    p_lo <= p_hat <= p_hi. The +eps no-harm bonus is paid ONLY when warranted.

    Args:
        completions: list of TRL-format completions (len B).
        ground_truth: list of gold strings (len B) or None.
        group_index: per-rollout group id (uid). Rollouts sharing an id form a GRPO
            group. None -> all rollouts in one group.
        step: trainer step (for w_warmup = min(1, step/warmup_steps)).

    Returns dict of lists (len B): R_corr, R_meta, R_cal, p_hat, group_acc.
    """
    B = len(completions)
    texts = [_get_text(c) for c in completions]
    gts = [ground_truth[i] if ground_truth is not None else "" for i in range(B)]

    # Per-rollout primitives.
    # B-rework: parse the PRELIMINARY answer (text before the first <|meta|>) and the
    # FINAL answer (whole text) with the SAME lenient extractor correctness uses
    # (_extract_answer_fallback handles "The answer is X" / last-\boxed / last-number),
    # NOT \boxed-only — the SFT model answers as "The answer is X" (93% emit no \boxed),
    # so the old \boxed-only path made every transition unparseable -> R_meta≡0.
    # two_pass = the model committed a preliminary answer BEFORE its <|meta|> verify
    # block AND a final answer -> only then is a revision (transition) measurable.
    answer1 = [None] * B   # preliminary (pre-meta)
    answer2 = [None] * B   # final (graded)
    c1 = [False] * B
    c2 = [False] * B
    two_pass = [False] * B
    conf = [None] * B
    for i in range(B):
        t = texts[i]
        final = _extract_answer_fallback(t)
        answer2[i] = final
        c2[i] = bool(_check_correctness(final, gts[i])) if final else False
        if "<|meta|>" in t:
            prelim = _extract_answer_fallback(t.split("<|meta|>", 1)[0])
            if prelim:
                answer1[i] = prelim
                c1[i] = bool(_check_correctness(prelim, gts[i]))
                two_pass[i] = final is not None
        conf[i] = _parse_confidence(t)

    # Group ids.
    if group_index is None:
        gid = [0] * B
    else:
        gid = list(group_index.tolist() if hasattr(group_index, "tolist") else group_index)
        gid = [str(g) for g in gid]

    # Group difficulty p_hat = mean FINAL correctness (well-defined for every rollout;
    # the preliminary answer is often absent in single-pass rollouts, so grouping on it
    # would be skewed). group_acc == p_hat here (both on the final answer).
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

    w_warmup = min(1.0, float(step) / float(warmup_steps)) if warmup_steps > 0 else 1.0

    # Sandbagging circuit-breaker (anti-inversion). canary = batch mean pass-1
    # accuracy. The flip credit (+1) rewards wrong->right; if the policy learns to
    # FAKE pass-1 errors to farm it, pass-1 accuracy collapses. When canary falls
    # below sandbag_floor (after warmup, so honest cold-start lows don't trigger),
    # ramp the ENTIRE meta head toward 0 -> removes the incentive to sandbag.
    # NOTE: the primary guard is warrant-gating the flip credit below; this clamp is
    # the backstop for COLLECTIVE sandbagging (whole group fakes pass-1 -> p_hat
    # drifts into the warranted band). Logged as canary/sandbag_clamp for wandb.
    # canary = mean PRELIMINARY accuracy over TWO-PASS rollouts (where a prelim answer
    # exists). If the policy fakes wrong prelims to farm the flip credit, this collapses.
    tp_idx = [i for i in range(B) if two_pass[i]]
    canary = float(np.mean([1.0 if c1[i] else 0.0 for i in tp_idx])) if tp_idx else 1.0
    if sandbag_clamp and step >= warmup_steps and canary < sandbag_floor:
        clamp_f = max(0.0, canary / sandbag_floor) if sandbag_floor > 0 else 0.0
    else:
        clamp_f = 1.0

    R_corr = [0.0] * B
    R_meta = [0.0] * B
    R_cal = [0.0] * B
    for i in range(B):
        # R_corr — lenient final-answer correctness (same extractor as correctness head).
        R_corr[i] = 1.0 if c2[i] else -1.0

        # R_meta (option B) — FORMAT enforcement + TRANSITION reward, routed to META_CONTENT.
        # The SFT model is single-pass (one answer, meta inside <think>); to MEASURE useful
        # metacognition we need it to commit a preliminary answer, verify in <|meta|>, then
        # (maybe) revise. So:
        #   single-pass (no preliminary answer)         -> small FORMAT PENALTY  (push to 2-pass)
        #   two-pass, identical answer (no revision)    -> small FORMAT CREDIT   (did the structure)
        #   two-pass, revised wrong->right (warranted)  -> +1.0  (useful metacognition)
        #   two-pass, revised right->wrong              -> -1.0 * warmup (destructive)
        #   two-pass, revised, other (warranted)        -> +eps  (genuine revision attempt)
        # Format terms are warmup-scaled so a cold single-pass policy isn't harshly punished.
        warranted = (p_lo <= p_hat[i] <= p_hi)
        if two_pass[i]:
            revised = (answer1[i] != answer2[i])
            if revised:
                if (not c1[i]) and c2[i]:
                    R_meta[i] = 1.0 if warranted else 0.0   # useful flip (warrant-gated anti-sandbag)
                elif c1[i] and (not c2[i]):
                    R_meta[i] = -1.0 * w_warmup             # destructive revision
                else:
                    R_meta[i] = eps if warranted else 0.0   # genuine revision attempt
            else:
                R_meta[i] = format_credit * w_warmup        # 2-pass structure, no revision
        else:
            R_meta[i] = -format_penalty * w_warmup          # single-pass -> push toward 2-pass format

        R_meta[i] *= clamp_f  # sandbagging circuit-breaker

        # R_cal — per-instance Brier; 0 if conf missing (no floor).
        if conf[i] is not None:
            tgt = 1.0 if c2[i] else 0.0
            R_cal[i] = -((conf[i] - tgt) ** 2)
        else:
            R_cal[i] = 0.0

    return {
        "R_corr": R_corr,
        "R_meta": R_meta,
        "R_cal": R_cal,
        "p_hat": p_hat,
        "group_acc": group_acc,
        "canary_pass1_acc": [canary] * B,
        "sandbag_clamp": [clamp_f] * B,
    }


# ─────────────────────────────────────────────────────────────────────────────
# §2.3 — per-region advantage composition (torch-only; no verl/omegaconf deps so
# this is unit-testable under a minimal env). verl_sdc_utils delegates here.
# ─────────────────────────────────────────────────────────────────────────────
def group_mean_subtract(values, index):
    """Dr.GRPO block-wise group centering: subtract group mean, NO /std.

    Args:
        values: [B] or [B,1] per-rollout scalars (tensor / array / list).
        index: per-rollout group id (uid array / list); rollouts sharing an id
            form a group. None -> single group.

    Returns:
        [B,1] centered tensor. Degenerate (singleton / all-equal) groups -> 0.
    """
    v = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    B = v.shape[0]
    if index is None:
        gid = ["__g0__"] * B
    else:
        gid = list(index.tolist() if hasattr(index, "tolist") else index)
        gid = [str(g) for g in gid]
    out = torch.zeros_like(v)
    groups: dict = {}
    for i, g in enumerate(gid):
        groups.setdefault(g, []).append(i)
    for members in groups.values():
        idx = torch.tensor(members, dtype=torch.long, device=v.device)
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
):
    """Independent per-head group-mean-subtract + per-region token routing (§2.3).

        A_token = ( w_corr*Â_corr*ANSWER
                  + w_meta*Â_meta*META_CONTENT
                  + w_cal *Â_cal *CONF ) * response_mask

    TAG tokens are in NEITHER ANSWER nor META_CONTENT -> advantage 0. NO global
    re-whiten (codex-r13 LOCK). Returns (A, A).
    """
    rm = torch.as_tensor(response_mask, dtype=torch.float32)
    device = rm.device

    A_corr = group_mean_subtract(R_corr, index).to(device)   # [B,1]
    A_meta = group_mean_subtract(R_meta, index).to(device)   # [B,1]
    A_cal = group_mean_subtract(R_cal, index).to(device)     # [B,1]

    ans = torch.as_tensor(answer_mask, dtype=torch.float32).to(device)
    meta_c = torch.as_tensor(meta_content_mask, dtype=torch.float32).to(device)
    conf = torch.as_tensor(conf_mask, dtype=torch.float32).to(device)

    advantages = (
        w_corr * A_corr * ans
        + w_meta * A_meta * meta_c
        + w_cal * A_cal * conf
    ) * rm

    return advantages, advantages
