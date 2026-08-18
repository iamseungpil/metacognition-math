"""Context-delta R_meta ("ablation PMI") next to the legacy position-delta.

WHY THIS FILE EXISTS
--------------------
The live R_meta (src/training/dcpo_pmi_shift.py) is a POSITION delta:

    OPEN  = prompt + body-before-meta
    CLOSE = prompt + body-before-meta + meta-block
    d_pos = PMI(CLOSE) - PMI(OPEN)

Both contexts stop at the meta block, so nothing the meta *causes* downstream is
in either one, and any confident-sounding text raises PMI just by being there.
Measured 2026-08-16 on rq3v2g_b4p2: the single most common meta opener went from
14.3% of rollouts at step 5 to 67.4% at step 295, and at step 295 that template
earned R_meta +0.289 while every other meta earned -0.145 -- with R_corr
essentially equal (0.864 vs 0.839). The head was paying for the template, not
for the correction.

This module implements the CONTEXT delta instead:

    WITH = prompt + full body up to the final answer, meta slot = X
    CTRL = the same context, meta slot = a donor meta from a DIFFERENT problem
    d_abl(X) = PMI(WITH(X)) - PMI(WITH(donor))

Same measurement position, two contexts. A template that carries no
problem-specific information swaps out for a donor with no change, so its
d_abl sits at zero by construction. A meta that actually fixed a sign error
collapses the answer's probability when swapped out.

The donor (rather than deletion) is the control that holds LENGTH and STYLE
fixed, so "removing text disturbs the context" cannot masquerade as usefulness.
PG0 (2026-06-19) ran the same R / N' / Nc contrast and rejected the
any-injection confound; this is the teacher-forced version of it.

COST: identical to the live head. Both definitions need PMI at two contexts,
i.e. 4 teacher-forced forwards per rollout. No regeneration.

PMI(ctx) = sum over DIVERGENT answer tokens of
           [ logp(gold_t | ctx) - logp(decoy_t | ctx) ]
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

META_OPEN_DEFAULT = "<meta>"
META_CLOSE_DEFAULT = "</meta>"
_BOXED_RE = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")


# ---------------------------------------------------------------- text surgery

def split_meta(text: str, open_tag: str = META_OPEN_DEFAULT,
               close_tag: str = META_CLOSE_DEFAULT):
    """(before, meta_inner, after) for the FIRST closed meta block, else None.

    `meta_inner` excludes the tags so a replacement can be dropped in without
    inheriting the original's delimiters.
    """
    i = text.find(open_tag)
    if i < 0:
        return None
    j = text.find(close_tag, i + len(open_tag))
    if j < 0:
        return None
    return text[:i], text[i + len(open_tag):j], text[j + len(close_tag):]


def body_before_answer(text: str) -> Optional[str]:
    """Everything up to (excluding) the LAST \\boxed{...}; None if there is none.

    This is where both definitions teacher-force the answer, so it must be the
    same slice for d_pos and d_abl or the two are not comparable.
    """
    m = None
    for m in _BOXED_RE.finditer(text):
        pass
    if m is None:
        return None
    return text[:m.start()]


def with_meta(before: str, inner: str, after: str,
              open_tag: str = META_OPEN_DEFAULT,
              close_tag: str = META_CLOSE_DEFAULT) -> str:
    return f"{before}{open_tag}{inner}{close_tag}{after}"


def make_decoy(gold: str, seed: int = 42) -> str:
    """A wrong answer of the same shape as gold, deterministic in (gold, seed).

    Numeric gold is perturbed by a nonzero offset; anything else gets a suffix.
    The decoy only has to be a plausible competing string -- PMI is a contrast,
    not a classifier.
    """
    g = gold.strip()
    try:
        val = float(g)
    except ValueError:
        return g + "'" if not g.endswith("'") else g[:-1]
    off = 1 + (abs(hash((g, seed))) % 7)
    new = val + off
    if float(new).is_integer() and "." not in g:
        return str(int(new))
    return f"{new:g}"


# ------------------------------------------------------------- token machinery

def divergent_positions(tok, gold: str, decoy: str):
    """Token ids for \\boxed{gold} / \\boxed{decoy} plus their divergent slices.

    The shared `\\boxed{` prefix and `}` suffix are trimmed so the contrast is
    carried only by the answer itself -- the same locus the live scorer uses.
    Returns (gold_ids, decoy_ids, gold_slice, decoy_slice).
    """
    g = tok(f"\\boxed{{{gold}}}", add_special_tokens=False)["input_ids"]
    d = tok(f"\\boxed{{{decoy}}}", add_special_tokens=False)["input_ids"]
    p = 0
    while p < min(len(g), len(d)) and g[p] == d[p]:
        p += 1
    s = 0
    while (s < min(len(g), len(d)) - p) and g[len(g) - 1 - s] == d[len(d) - 1 - s]:
        s += 1
    return g, d, slice(p, len(g) - s), slice(p, len(d) - s)


@torch.inference_mode()
def _sum_logp(model, tok, ctx: str, cont_ids: Sequence[int], sl: slice,
              device: str = "cuda") -> float:
    """Sum of logp(cont_ids[sl] | ctx) under teacher forcing."""
    ctx_ids = tok(ctx, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([list(ctx_ids) + list(cont_ids)], device=device)
    logits = model(input_ids=ids).logits[0, :-1].float()
    lp = torch.log_softmax(logits, dim=-1).gather(
        -1, ids[0, 1:].unsqueeze(-1)).squeeze(-1)
    # lp[k] is the logprob of ids[k+1]; continuation starts at index len(ctx_ids)
    cont_lp = lp[len(ctx_ids) - 1:]
    return float(cont_lp[sl].sum())


def pmi_at(model, tok, ctx: str, gold: str, decoy: str, device: str = "cuda") -> float:
    """PMI(ctx) = sum_div [ logp(gold | ctx) - logp(decoy | ctx) ]."""
    g_ids, d_ids, g_sl, d_sl = divergent_positions(tok, gold, decoy)
    return (_sum_logp(model, tok, ctx, g_ids, g_sl, device)
            - _sum_logp(model, tok, ctx, d_ids, d_sl, device))


# ------------------------------------------------------------------- the deltas

@dataclass
class Deltas:
    d_pos: Optional[float]   # legacy position delta (CLOSE - OPEN)
    d_abl: Optional[float]   # proposed context delta (WITH - DONOR)
    pmi_with: Optional[float]
    pmi_ctrl: Optional[float]
    n_ctx_tokens: int


def score_row(model, tok, prompt: str, response: str, gold: str,
              donor_inner: str, *, meta_inner: Optional[str] = None,
              decoy: Optional[str] = None, device: str = "cuda",
              want_pos: bool = True) -> Optional[Deltas]:
    """Both deltas for one rollout.

    meta_inner=None uses the rollout's own meta. Passing a string SUBSTITUTES it,
    which is how the wiring conditions (boilerplate / oracle / donor) are built:
    everything except the meta slot stays byte-identical, so the contrast is the
    meta content and nothing else.
    """
    parts = split_meta(response)
    if parts is None:
        return None
    before_raw, own_inner, after_raw = parts
    inner = own_inner if meta_inner is None else meta_inner

    full = with_meta(before_raw, inner, after_raw)
    body = body_before_answer(full)
    if body is None:
        return None
    decoy = decoy or make_decoy(gold)

    ctrl_full = with_meta(before_raw, donor_inner, after_raw)
    ctrl_body = body_before_answer(ctrl_full)
    if ctrl_body is None:
        return None

    pmi_with = pmi_at(model, tok, prompt + body, gold, decoy, device)
    pmi_ctrl = pmi_at(model, tok, prompt + ctrl_body, gold, decoy, device)

    d_pos = None
    if want_pos:
        open_ctx = prompt + before_raw
        close_ctx = prompt + with_meta(before_raw, inner, "")
        d_pos = (pmi_at(model, tok, close_ctx, gold, decoy, device)
                 - pmi_at(model, tok, open_ctx, gold, decoy, device))

    return Deltas(d_pos=d_pos, d_abl=pmi_with - pmi_ctrl,
                  pmi_with=pmi_with, pmi_ctrl=pmi_ctrl,
                  n_ctx_tokens=len(tok(prompt + body,
                                       add_special_tokens=False)["input_ids"]))


def pick_donor(tok, own_inner: str, pool: Sequence[tuple[str, str]],
               own_problem_id: str) -> Optional[str]:
    """Length-matched meta from a DIFFERENT problem.

    Deterministic: sort by |token-length difference|, break ties on problem id.
    Determinism is not cosmetic here -- a control that reshuffles between runs
    cannot be compared across arms.
    """
    n = len(tok(own_inner, add_special_tokens=False)["input_ids"])
    cand = [(pid, txt) for pid, txt in pool if pid != own_problem_id]
    if not cand:
        return None
    scored = [
        (abs(len(tok(t, add_special_tokens=False)["input_ids"]) - n), pid, t)
        for pid, t in cand
    ]
    scored.sort(key=lambda x: (x[0], x[1]))
    return scored[0][2]
