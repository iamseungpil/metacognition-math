"""Entropy-triggered force-inject of <|meta|> for CTSD training (PLAN.md §3 H2).

The A.3 probe (experiments/probes/a3_inject_causal.py) validated, offline, that
splicing a PRODUCTIVE meta block at the model's max-entropy PRE-ANSWER position
causally affects accuracy. This module is the TRAINING-side core of that same
mechanism, factored as pure, unit-testable functions so the verl rollout and the
A.3 probe share ONE implementation (DRY — Karpathy).

What this module owns (pure, tested):
  - find_inject_position: argmax body-entropy index before the first answer
  - first_boxed_token_idx: locate the answer so we never inject after it
  - build_inject_segment / splice_prefix: construct the phase-2 prompt

What it does NOT own: the two-phase vLLM rollout (generate → entropy → splice →
regenerate) lives in the trainer (SDCRayPPOTrainer) because it touches verl's
DataProto/rollout API; this module is the correct-by-construction core it calls.

This file has NO torch / transformers / verl import at module load (only numpy),
so it imports cheaply and tests without a GPU.
"""
from __future__ import annotations
import numpy as np

MIN_TOK_DEFAULT = 50  # do not inject in the first MIN_TOK response tokens

# Two inject modes (A.3 finding, 2026-05-29):
#   MARKER_ONLY (b-style) — inject ONLY the opening <|meta|>; the model fills the
#     content itself and the contrastive reward (ROD_MQ_CONTRAST) shapes it during
#     RL. This was the best A.3 condition (b: +5pp over no-inject, 7-3 wins; the
#     fixed-content c-style did NOT beat the model's own meta). DEFAULT for training.
#   GOOD_META (c-style) — inject a full fixed productive block. Used by the A.3
#     probe as the "supplied good content" arm; kept for ablation, not the default.
MARKER_ONLY = "\n<|meta|>\n"
GOOD_META = (
    "\n<|meta|>\n"
    "confidence: 0.3\n"
    "I am not fully sure this route is correct. Let me slow down, re-examine the "
    "setup, recompute the key step carefully, and verify the result before committing.\n"
    "<|/meta|>\n"
)


def first_boxed_token_idx(tokenizer, response_ids) -> int:
    """Token index where the first ``\\boxed`` appears (else len). Injecting meta
    after the answer is written is meaningless, so callers cap the search here."""
    decoded = ""
    for i, tid in enumerate(response_ids):
        decoded += tokenizer.decode([tid])
        if r"\boxed" in decoded:
            return i
    return len(response_ids)


def meta_mask(response_ids, meta_open_id: int, meta_close_id: int, length: int) -> np.ndarray:
    """Boolean mask over response_ids[:length]; True = inside/at a meta span,
    so injection never lands within an existing meta block."""
    mask = np.zeros(length, dtype=bool)
    in_meta, start = False, 0
    for i in range(length):
        t = response_ids[i]
        if t == meta_open_id:
            in_meta, start = True, i + 1
        elif t == meta_close_id and in_meta:
            mask[start:i] = True
            in_meta = False
        if t in (meta_open_id, meta_close_id):
            mask[i] = True
    if in_meta:  # unclosed <|meta|> — mask to end so injection never lands inside
        mask[start:length] = True
    return mask


def find_inject_position(entropy, response_ids, meta_open_id: int, meta_close_id: int,
                         answer_cap: int | None = None, min_tok: int = MIN_TOK_DEFAULT) -> int:
    """Argmax-entropy body position (>=min_tok, outside meta, before answer_cap).

    Returns the response-token index to inject BEFORE, or -1 if there is no valid
    position (caller skips injection for that sample). -1 (not a fallback index)
    keeps training honest: a sample with no pre-answer body simply isn't injected.
    """
    L = min(len(entropy), len(response_ids))
    hi = L if answer_cap is None else min(L, answer_cap)
    if hi <= min_tok:
        return -1
    mask = meta_mask(response_ids, meta_open_id, meta_close_id, L)
    cand = [(float(entropy[i]), i) for i in range(min_tok, hi) if not mask[i]]
    if not cand:
        return -1
    _, idx = max(cand)
    return idx


def build_inject_segment(tokenizer, template: str = GOOD_META) -> list[int]:
    """Token ids for the injected meta block."""
    return tokenizer.encode(template, add_special_tokens=False)


def splice_prefix(prompt_ids, response_ids, pos: int, segment_ids) -> list[int]:
    """Phase-2 generation prompt = prompt + response[:pos] + injected meta block."""
    return list(prompt_ids) + list(response_ids[:pos]) + list(segment_ids)


# ─── E.9 Binned-Confidence-Injection (BCI-RLVR) ────────────────────────────────
# ADDITIVE: the three helpers below are used ONLY by the BCI_RLVR rollout wrap
# (SDCRayPPOTrainer._bci_generate_sequences, installed only under the new flag
# algorithm.sdc_force_inject_conf). They do NOT touch any function above and are
# pure (numpy/str only), so they unit-test without torch/transformers/verl.
#
# Mechanism (see docs/.../e9-...-design.md §2): for a GRPO group of rollout.n
# samples (verl layout gen_batch.repeat(n, interleave=True) → within-group bin
# index = row_index % n), sample i is SEEDED with a fixed confidence statement
# `<|meta|>\nconfidence: c_i\n<|/meta|>\n`, c_i a bin center. The seed lands in the
# trained RESPONSE region so outcome_calibration_reward (rewards.py, unmodified)
# proper-scores it and REINFORCE raises the calibrated confidence.


def default_conf_bins(n: int) -> list[float]:
    """Evenly spaced confidence bin centers for n GRPO rollouts.

    Center i (0-based) = (i + 1) / (n + 1), so the n centers tile (0, 1) with
    equal spacing and equal margins at both ends. n=4 → [0.2, 0.4, 0.6, 0.8].
    """
    if n <= 0:
        raise ValueError(f"default_conf_bins needs n >= 1, got {n}")
    return [(i + 1) / (n + 1) for i in range(n)]


def conf_seed_template(c: float) -> str:
    """The seeded meta block for confidence ``c`` (2-decimal, parseable by
    rewards.outcome_calibration_reward's ``<|meta|>...confidence: X...<|/meta|>``
    regex). 2 decimals so all bins tokenize to equal length (see assert in wrap)."""
    return f"\n<|meta|>\nconfidence: {c:.2f}\n<|/meta|>\n"


def build_conf_seed_ids(tokenizer, c: float) -> list[int]:
    """Token ids for the confidence seed block (no special tokens added — the
    `<|meta|>`/`<|/meta|>` strings are themselves added-vocab tokens)."""
    return tokenizer.encode(conf_seed_template(c), add_special_tokens=False)


def plan_inject_prefixes(prompt_ids_list, response_ids_list, entropy_list,
                         tokenizer, meta_open_id: int, meta_close_id: int,
                         template: str = GOOD_META, min_tok: int = MIN_TOK_DEFAULT):
    """Batch orchestration the trainer calls BETWEEN phase-1 and phase-2 rollout.

    For each phase-1 rollout sample, return the phase-2 prompt token-ids (prompt +
    response-up-to-the-max-entropy-pre-answer-position + injected meta block), or
    None when there is no valid injection position (that sample keeps its phase-1
    rollout unchanged). This is the entire force-inject decision; the trainer only
    has to (a) compute per-token entropy from phase-1 logprobs, (b) re-pack these
    prefixes into a DataProto and regenerate. Pure + unit-tested.
    """
    assert len(prompt_ids_list) == len(response_ids_list) == len(entropy_list), (
        f"batch length mismatch: prompts={len(prompt_ids_list)} "
        f"responses={len(response_ids_list)} entropies={len(entropy_list)}"
    )
    seg = build_inject_segment(tokenizer, template)
    prefixes = []
    for p_ids, r_ids, ent in zip(prompt_ids_list, response_ids_list, entropy_list):
        cap = first_boxed_token_idx(tokenizer, r_ids)
        pos = find_inject_position(ent, r_ids, meta_open_id, meta_close_id,
                                   answer_cap=cap, min_tok=min_tok)
        prefixes.append(splice_prefix(p_ids, r_ids, pos, seg) if pos >= 0 else None)
    return prefixes
