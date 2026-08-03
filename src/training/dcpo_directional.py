"""Directional self-distillation (gm-contrast) R_meta core — pure numpy.

NEW + ADDITIVE, framework-light module (numpy only — ZERO verl / torch deps),
the gm-direction sibling of `dcpo_pmi`. Like that module it does NOT load models
or tokenizers: callers hand it ROWS already carrying the per-token reference
logprobs of the two answer strings (`\\boxed{gold}` / `\\boxed{decoy}`) under the
two contexts (body+meta / body+placebo); this module does the token-level
DiD, the divergent-answer-token restriction, and the `mean_min` (RLT)
aggregation. The actual GPU teacher-forcing forward lives in verl_sdc.

REDUCED 2026-08-03: the gm / RLSD reward generations were removed. What remains
is the answer-string + divergent-token machinery, which the LIVE pmi_shift
scorer and src/eval/pmi_shift_signal.py both call.
"""
from __future__ import annotations

import numpy as np


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


__all__ = [
    "boxed_answer_string",
    "divergent_token_mask",
]
