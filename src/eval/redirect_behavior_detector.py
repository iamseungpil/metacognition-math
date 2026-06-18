"""Redirect-BEHAVIOR detector (prose, NOT token presence).

Banning the `<|switch|>` token does not ban the redirect *behavior*: a primed
model routes around it in plain prose ("wait, that's wrong, let me try a
different approach"). The R-B' causal measure and PG1 separability gate require
detecting that BEHAVIOR in BOTH arms on identical footing — so the detector must
NOT key on the `<|switch|>` token (which is present in arm R/A but -inf-masked in
B'/c_without; keying on it would inflate R and fake separability — intent-check
wbitrlry0 CRITICAL). The token is reported on a SEPARATE channel `emitted_switch`.

Primary signal = an LLM judge tied to behavior-that-changes-the-attempt. The
regex is only a cheap high-precision PRE-FILTER to pick candidates for the judge
(it never overrides the judge). Spec 2026-06-18 REV-6 §5.3, review round-5 M-3,
intent-check wbitrlry0 (CRITICAL, E, F).
"""
import re

# Prose redirect cues ONLY — no <|switch|> token (intent-check wbitrlry0 CRITICAL).
_REDIRECT_RE = re.compile(
    r"(?i)("
    r"instead|different (approach|method)|another (approach|method|way)|"
    r"let me reconsider|reconsider|that('?s| is) wrong|start over|"
    r"backtrack|scrap that|on second thought|wait,? (that|this)"
    r")"
)


def emitted_switch(text: str) -> bool:
    """Separate channel: did the `<|switch|>` TOKEN appear (form, not behavior)."""
    return "<|switch|>" in (text or "")


def regex_prefilter(text: str) -> bool:
    """Cheap high-precision pre-filter — selects candidates FOR the judge.
    NOT a behavior decision on its own (review F: regex must not override judge)."""
    return bool(_REDIRECT_RE.search(text or ""))


def detect_redirect(text: str, llm_judge=None, regex_only: bool = False) -> bool:
    """Behavior decision. On the LIVE R-B' / c_without path the LLM judge is the
    primary signal and is REQUIRED (review E). `regex_only=True` is for the
    documented pre-filter and unit tests only."""
    if regex_only:
        return regex_prefilter(text)
    if llm_judge is None:
        raise ValueError(
            "detect_redirect: LLM judge required on the live behavior path "
            "(pass regex_only=True only for pre-filter/tests)."
        )
    return bool(llm_judge(text))


def measure_recall(labeled_redirects, llm_judge=None, regex_only: bool = False) -> float:
    """Recall on KNOWN redirects (PG1 gate, must measure with the production judge)."""
    if not labeled_redirects:
        return float("nan")
    hits = sum(1 for t in labeled_redirects if detect_redirect(t, llm_judge, regex_only))
    return hits / len(labeled_redirects)


def measure_precision(labeled_non_redirects, llm_judge=None, regex_only: bool = False) -> float:
    """1 - false-positive rate on KNOWN non-redirects (review F: regex precision
    must be measured, not assumed)."""
    if not labeled_non_redirects:
        return float("nan")
    fp = sum(1 for t in labeled_non_redirects if detect_redirect(t, llm_judge, regex_only))
    return 1.0 - fp / len(labeled_non_redirects)
