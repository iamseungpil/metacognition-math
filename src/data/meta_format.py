"""Inspect-and-substitute filter for teacher-emitted ``<|meta|>`` blocks.

A real teacher run produced a PERFECT redirect/verify demo but closed the meta
block with ``</|meta|>`` (an HTML-ish variant) instead of the canonical
``<|/meta|>``. The strict structural checker (``META_BLOCK_RE`` in
``build_v8_strict_paired_data``) does not match that variant, so the demo was
silently DROPPED even though it was correct.

Two pure functions fix this:

  * ``normalize_meta_format(text)`` REPAIRS the repairable variants into the
    canonical form: close-tag variants ``</|meta|>`` / ``</meta>`` / ``<|meta/|>``
    -> ``<|/meta|>``; canonicalize the ``confidence:`` / ``decision:`` line
    casing + spacing; collapse stray inner whitespace. It does NOT invent missing
    content (a missing confidence line is left missing for validate to catch).

  * ``validate_meta_structure(text)`` drops FATAL cases after normalization:
    zero or more-than-one meta block, a missing ``confidence:`` line, a missing
    ``decision:`` line, or a ``decision:`` value not in {redirect, verify}.

Wired BEFORE the structural/causal filter in the build driver so a ``</|meta|>``
demo is repaired + kept rather than dropped. No I/O, no network, no GPU.
"""
from __future__ import annotations

import re

META_START = "<|meta|>"
META_END = "<|/meta|>"

VALID_DECISIONS = ("redirect", "verify")

# Close-tag variants a teacher emits instead of the canonical <|/meta|>. Ordered
# so the most specific patterns are tried first; each maps to META_END.
_CLOSE_TAG_VARIANTS = (
    "</|meta|>",   # the exact variant a real teacher run produced
    "<|meta/|>",   # slash before the closing pipe
    "</meta>",     # plain HTML-ish close
)

# A confidence / decision line ANYWHERE in the text, case-insensitive, tolerant of
# missing space after the colon. Canonicalized to 'confidence: 0.xx' / 'decision: x'.
_CONF_LINE_RE = re.compile(r"(?im)^[ \t]*confidence[ \t]*:[ \t]*([0-9]*\.?[0-9]+)[ \t]*$")
_DECISION_LINE_RE = re.compile(r"(?im)^[ \t]*decision[ \t]*:[ \t]*([A-Za-z]+)[ \t]*$")


def normalize_meta_format(text):
    """Repair repairable meta-format variants into the canonical form.

    ``None`` / ``""`` pass through unchanged. Returns the repaired string;
    structural fatality (0/>1 blocks, missing lines) is left for
    ``validate_meta_structure``.
    """
    if not text:
        return text

    out = text
    for variant in _CLOSE_TAG_VARIANTS:
        if variant in out:
            out = out.replace(variant, META_END)

    # canonicalize the confidence / decision line casing + spacing.
    out = _CONF_LINE_RE.sub(lambda m: f"confidence: {m.group(1)}", out)
    out = _DECISION_LINE_RE.sub(lambda m: f"decision: {m.group(1).lower()}", out)
    return out


def validate_meta_structure(text):
    """Validate a (normalized) demo's meta structure.

    Returns ``(ok, reason)``. ``ok`` is False with a human-readable reason for the
    FATAL cases: zero or >1 meta blocks, missing ``confidence:`` line, missing
    ``decision:`` line, or a decision value not in {redirect, verify}. A
    well-formed block returns ``(True, "")``.
    """
    if not text:
        return False, "empty text: no meta block"

    n_open = text.count(META_START)
    n_close = text.count(META_END)
    if n_open == 0 or n_close == 0:
        return False, "zero meta blocks"
    if n_open > 1 or n_close > 1:
        return False, "more than one meta block"

    start = text.index(META_START) + len(META_START)
    end = text.index(META_END)
    if end <= start:
        return False, "malformed meta block (close before open)"
    body = text[start:end]

    if not _CONF_LINE_RE.search(body):
        return False, "missing confidence line"

    dm = _DECISION_LINE_RE.search(body)
    if not dm:
        return False, "missing decision line"
    if dm.group(1).lower() not in VALID_DECISIONS:
        return False, f"invalid decision value: {dm.group(1)!r}"

    return True, ""


# Markers that unambiguously mean SOLVING (a calculation or the final answer) leaked
# INTO the meta block. The meta block must hold only the judgment (confidence +
# reason + decision); all solving belongs AFTER <|/meta|> so the region-routed RL
# reward (R_meta on meta, R_corr on the answer) is not confounded.
_SOLVING_IN_META = (r"\boxed", r"\[", r"\]", "$$")


def meta_is_pure_judgment(text):
    """Return ``(ok, reason)``: the meta block must be JUDGMENT ONLY.

    Rejects when the meta body contains a solving marker (``\\boxed``, display math
    ``\\[`` / ``\\]`` / ``$$``) — i.e. a calculation or the final answer leaked into
    the meta block. Prose strategy is NOT caught here (handled by the teacher
    prompt); this is the unambiguous backstop. A text with no parseable single meta
    block returns ``(False, ...)`` (validate_meta_structure runs first upstream).
    """
    if not text or META_START not in text or META_END not in text:
        return False, "no meta block"
    start = text.index(META_START) + len(META_START)
    end = text.index(META_END)
    if end <= start:
        return False, "malformed meta block"
    body = text[start:end]
    for marker in _SOLVING_IN_META:
        if marker in body:
            return False, f"solving leaked into meta block ({marker!r})"
    return True, ""


def strip_preamble_before_meta(text):
    """Drop any text BEFORE the first ``<|meta|>`` so the teacher's continuation
    begins at the meta block.

    A redirect demo is assembled as ``wrong_prefix + teacher_text``; if the teacher
    restates the prefix before opening the meta block, the assistant carries the
    prefix twice (the masked copy + the teacher's trained copy). Returning the text
    from the first ``<|meta|>`` removes that duplicate (and any teacher preamble).
    ``None`` / no-meta text passes through unchanged.
    """
    if not text or META_START not in text:
        return text
    return text[text.index(META_START):]
