"""TRIOBJ_META_V1 — meta-revision utility reward (NEW, ADDITIVE module).

This module is referenced ONLY by the NEW REWARD_CONFIGS['TRIOBJ_META_V1'] entry
in verl_sdc.py. It does NOT modify any existing reward. All parse / correctness
helpers are imported from src.training.rewards (NOT re-implemented), so train and
eval grading stay byte-identical with the rest of the pipeline.

Design (two-pass output format)
-------------------------------
The TRIOBJ_META_V1 policy emits:
    1) a PRELIMINARY boxed answer  (answer1, the FIRST \\boxed{...})
    2) a <|meta|> ... <|/meta|> block that verifies/critiques + states confidence
    3) (if needed) a revision and a FINAL boxed answer (answer2, the LAST \\boxed{...})

Grading (correctness_reward / eval) scores the LAST boxed answer = final answer
(confirmed: rewards._extract_answer_fallback returns matches[-1]).

meta_revision_utility_reward credits the revision by its CAUSAL EFFECT, not by its
presence (presence-rewards are hackable per PRM literature). It is TWO-SIDED and
OUTCOME-GATED:

    wrong->right + meta localizes error  -> +1.00  (genuine recovery; L2-gated)
    wrong->right but meta doesn't localize -> +0.30  (anti-sandbagging discount)
    right->wrong                          -> -1.00  (destructive revision; SCoRe)
    right->right, revised, genuine meta   -> +0.15  (confirmation)
    right->right, revised, no real meta   -> -0.10  (over-check)
    right->right, NOT revised (dup box)   ->  0.00  (L1: no real revision)
    both wrong                            ->  0.00
    <2 non-empty boxed answers           ->  0.00  (L3: no fabricated credit)

Output is clipped to [-1.0, 1.0].

Anti-hacking notes:
- L1: identical preliminary/final boxed answers earn nothing (no duplicate-box farm).
- L2: the +1.0 wrong->right bonus requires the meta block to localize/redirect the error
  (mitigates "sandbagging" = staging a fake-wrong preliminary answer). Residual risk is
  fundamental to pure outcome-gating; WATCH mean(answer1-correctness) during training as a
  sandbagging canary — if preliminary accuracy collapses, the policy is faking errors.
- L4: genuine/localization signals are scanned inside the meta block only (_meta_joined_text).
"""

from __future__ import annotations

import re

# Import (do NOT rewrite) detection / parse helpers from the canonical module.
from src.training.rewards import (
    _check_correctness,
    _get_text,
    _has_anomaly_notice_signal,
    _has_effective_verification_signal,
    _has_redirection_signal,
    _has_structured_meta,
    _has_verification_signal,
    _meta_joined_text,
)

# Matches a single \boxed{...} allowing up to 2 levels of nested braces — the SAME
# nested-brace convention as rewards._extract_answer_fallback, so the boxed answers
# we parse here are exactly the ones the grader sees.
_BOXED_RE = re.compile(
    r"\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"
)


def _all_boxed(text: str) -> list[str]:
    """Return every \\boxed{...} content in document order (preliminary..final)."""
    return [m.strip() for m in _BOXED_RE.findall(text or "")]


def _has_genuine_meta(text: str) -> bool:
    """Genuine metacognition = a structured <|meta|> block that actually verifies.

    Uses only existing rewards.py predicates (no new heuristics). A structured
    meta block alone is not enough — it must carry a verification/critique signal.
    L4 fix: the verification signal must live INSIDE the meta block, so we scan
    `_meta_joined_text(text)` (meta-only), not the whole completion — otherwise a
    verify keyword in the ordinary solution body would falsely satisfy the gate.
    """
    if not _has_structured_meta(text):
        return False
    meta_text = _meta_joined_text(text)
    return _has_verification_signal(meta_text) or _has_effective_verification_signal(meta_text)


def _meta_localizes_error(text: str) -> bool:
    """Does the meta block ARTICULATE what was wrong (error localization)?

    L2 anti-sandbagging gate: the full +1.0 wrong->right bonus is granted only when
    the meta block names/redirects the error (so a staged fake-wrong preliminary
    answer cannot silently farm the flip bonus). Scans meta-only text.
    """
    meta_text = _meta_joined_text(text)
    return _has_redirection_signal(meta_text) or _has_anomaly_notice_signal(meta_text)


def meta_revision_utility_reward(completions, ground_truth=None, **kwargs):
    """Two-sided, outcome-gated reward for the causal effect of meta-revision.

    Args:
        completions: TRL-format completions (one per sample in the micro-batch).
        ground_truth: list of gold strings (gold[i] for sample i). May be None.
        **kwargs: extra reward-loop kwargs (completion_lengths, answer_extracted,
            ...) — accepted and ignored, per the reward-fn call contract.

    Returns:
        list[float] of length len(completions), each in [-1.0, 1.0].
    """
    scores: list[float] = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""

        # L3: drop empty boxed contents so "\boxed{}" cannot count as a sincere attempt.
        boxed = [b for b in _all_boxed(text) if b]
        # No real two-pass (zero or one non-empty boxed answer) -> no fabricated credit.
        if len(boxed) < 2:
            scores.append(0.0)
            continue

        answer1 = boxed[0]   # preliminary
        answer2 = boxed[-1]  # final (the graded answer)
        revised = (answer1 != answer2)   # L1: was there an actual change in the boxed answer?

        c1 = _check_correctness(answer1, gt)
        c2 = _check_correctness(answer2, gt)

        if (not c1) and c2:
            # Useful metacognition: wrong -> right. L2 anti-sandbagging: full credit
            # only if the meta block genuinely localizes/redirects the error; otherwise
            # a staged fake-wrong preliminary answer could farm the flip bonus.
            score = 1.0 if (_has_genuine_meta(text) and _meta_localizes_error(text)) else 0.30
        elif c1 and (not c2):
            score = -1.0                     # destructive revision: right -> wrong (SCoRe)
        elif c1 and c2:
            # L1: identical boxed answers (no actual revision) earn nothing — kills the
            # "duplicate the right box + decorative meta" farm. A genuine re-derivation
            # that still lands correct earns a small confirmation credit iff meta is real.
            if not revised:
                score = 0.0
            else:
                score = 0.15 if _has_genuine_meta(text) else -0.10
        else:
            score = 0.0                      # both wrong: no signal

        # Defensive clip (all branches already in-range; keeps the contract explicit).
        score = max(-1.0, min(1.0, score))
        scores.append(score)

    return scores
