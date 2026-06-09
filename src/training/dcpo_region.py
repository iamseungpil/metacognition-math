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
    cf_completions=None,
    cf_correct=None,
    # ── v2 carry-over kwargs: accepted-but-IGNORED (spec §5.1-B.7) so existing
    # callers (_compute_dcpo_heads_stash) stay byte-identical without edits. ──
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

    Returns dict of lists (len B): R_corr, R_meta, R_cal, p_hat, group_acc,
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
    c_without = [None] * B   # None == no counterfactual available -> R_meta 0
    for i in range(B):
        t = texts[i]
        final = _extract_answer_fallback(t)
        answer2[i] = final
        c2[i] = bool(_check_correctness(final, gts[i])) if final else False
        has_meta[i] = "<|meta|>" in (t or "")
        conf[i] = _parse_confidence(t)

        # c_without[i]: precedence cf_correct -> cf_completions -> text fallback.
        cw = _cf_get(cf_correct, i)
        if cw is not None:
            c_without[i] = bool(cw)
        else:
            cf_txt = _cf_get(cf_completions, i)
            if cf_txt is not None:
                cf_txt = _get_text(cf_txt)
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
        if c_without[i] is None:
            R_meta[i] = 0.0
        else:
            R_meta[i] = c_with - (1.0 if c_without[i] else 0.0)

        # R_cal — per-instance Brier against c_with; 0 if conf missing (no floor).
        if conf[i] is not None:
            R_cal[i] = -((conf[i] - c_with) ** 2)
        else:
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
