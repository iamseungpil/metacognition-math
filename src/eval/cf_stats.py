"""Pure-python verification primitives for the R-B' counterfactual eval.

scipy AND statsmodels are absent in-env (review C-3), so McNemar is an exact
two-sided binomial computed by hand. Also: parse gate (drop rows where an arm
produced no answer) and a degeneracy health gate (an off-policy suppressed arm
that loops / truncates / gives no answer must NOT be scored as a "meta saved
it" win — review round-4 C-2 / round-5 I-2). Spec 2026-06-18 REV-6 §5.
"""
import math
import re


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar (binomial sign test on discordant pairs):
    under H0, min(b,c) ~ Binomial(n=b+c, p=0.5). Returns the two-sided p-value.
    b = saved (wrong->right), c = broke (right->wrong)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return min(1.0, 2.0 * tail)


def is_parsed(text: str) -> bool:
    """True if a final answer can be extracted (else the row is dropped before
    scoring, so a no-answer arm can't inflate `saved`). Delegates to the
    project's extractor when importable; else a light boxed/number fallback."""
    if not text or not text.strip():
        return False
    try:
        from src.training.rewards import _extract_answer_fallback
        return _extract_answer_fallback(text) not in (None, "")
    except Exception:
        if re.search(r"\\boxed\{", text):
            return True
        return bool(re.search(r"-?\d", text))


def degeneracy_flags(text: str, min_len: int = 20) -> dict:
    """Flag off-policy degeneracy of a suppressed arm: loop/repetition,
    too-short, no-final-answer. A flagged row must not count as a redirect win."""
    t = text or ""
    toks = t.split()
    repetition = False
    if len(toks) >= 8:
        # max single-token run length relative to total
        run = best = 1
        for i in range(1, len(toks)):
            run = run + 1 if toks[i] == toks[i - 1] else 1
            best = max(best, run)
        # or low unique ratio
        repetition = best >= 6 or (len(set(toks)) / len(toks) < 0.25)
    return {
        "repetition": repetition,
        "too_short": len(toks) < min_len,
        "no_answer": not is_parsed(t),
    }
