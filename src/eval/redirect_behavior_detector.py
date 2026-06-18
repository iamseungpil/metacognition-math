"""Redirect-BEHAVIOR detector (not token presence).

Banning the `<|switch|>` token does not ban the redirect *behavior*: a primed
model routes around it in plain prose ("wait, that's wrong, let me try a
different approach"). The R-B' causal measure and PG1 separability gate require
detecting that behavior in BOTH arms — spec 2026-06-18 REV-6 §5.3, review
round-5 M-3. Primary signal = an LLM judge; regex is a cheap high-precision
pre-filter. `measure_recall` reports detector recall on hand-labeled redirects
(PG1 requires >= ~0.8).
"""
import re

_REDIRECT_RE = re.compile(
    r"(?i)("
    r"instead|different (approach|method)|another (approach|method|way)|"
    r"let me reconsider|reconsider|that('?s| is) wrong|start over|"
    r"backtrack|scrap that|on second thought|wait,? (that|this)|"
    r"<\|switch\|>"
    r")"
)


def _regex_hit(text: str) -> bool:
    return bool(_REDIRECT_RE.search(text or ""))


def detect_redirect(text: str, llm_judge=None) -> bool:
    """True if the continuation exhibits redirect behavior. regex pre-filter OR
    LLM judge. `llm_judge(text)->bool` is injectable (None = regex only)."""
    if _regex_hit(text):
        return True
    if llm_judge is not None:
        return bool(llm_judge(text))
    return False


def measure_recall(labeled_redirects, llm_judge=None) -> float:
    """Fraction of KNOWN redirect traces the detector catches (PG1 gate)."""
    if not labeled_redirects:
        return float("nan")
    hits = sum(1 for t in labeled_redirects if detect_redirect(t, llm_judge=llm_judge))
    return hits / len(labeled_redirects)
