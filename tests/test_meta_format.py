"""Tests for the meta-format NORMALIZE + VALIDATE filter.

A real teacher run produced a perfect demo but with the close tag ``</|meta|>``
(instead of the canonical ``<|/meta|>``) which the strict META_BLOCK_RE DROPPED.
``normalize_meta_format`` repairs the repairable close-tag variants (and
canonicalizes the confidence/decision lines + stray whitespace) so the demo is
kept; ``validate_meta_structure`` drops only the FATAL cases (0 or >1 blocks,
missing confidence, missing/invalid decision).

Pure (no I/O, no network, no GPU).
"""
from scripts.build_v8_strict_paired_data import META_BLOCK_RE
from src.data.meta_format import normalize_meta_format, validate_meta_structure


# --------------------------------------------------------------------------- #
# normalize: repair the close-tag variants -> canonical <|/meta|>
# --------------------------------------------------------------------------- #
def test_normalize_repairs_slash_after_pipe_close_tag():
    # the EXACT variant a real teacher run produced: </|meta|>
    text = "<|meta|>\nconfidence: 0.25\ndecision: redirect\n</|meta|>\nThe answer is $7$."
    out = normalize_meta_format(text)
    assert "<|/meta|>" in out
    assert "</|meta|>" not in out
    # and the canonical regex now MATCHES the repaired block (it did not before)
    assert META_BLOCK_RE.search(out) is not None
    assert META_BLOCK_RE.search(text) is None


def test_normalize_repairs_plain_html_close_tag():
    text = "<|meta|>\nconfidence: 0.25\ndecision: redirect\n</meta>\nok"
    out = normalize_meta_format(text)
    assert "<|/meta|>" in out
    assert "</meta>" not in out
    assert META_BLOCK_RE.search(out) is not None


def test_normalize_repairs_slash_before_pipe_close_tag():
    text = "<|meta|>\nconfidence: 0.25\ndecision: verify\n<|meta/|>\nok"
    out = normalize_meta_format(text)
    assert "<|/meta|>" in out
    assert "<|meta/|>" not in out
    assert META_BLOCK_RE.search(out) is not None


def test_normalize_leaves_canonical_block_untouched_structurally():
    text = "<|meta|>\nconfidence: 0.25\ndecision: redirect\n<|/meta|>\nThe answer is $7$."
    out = normalize_meta_format(text)
    assert META_BLOCK_RE.search(out) is not None
    assert "confidence: 0.25" in out
    assert "decision: redirect" in out


def test_normalize_canonicalizes_confidence_and_decision_line_casing():
    text = "<|meta|>\nConfidence:0.25\nDecision:  Redirect\n<|/meta|>\nx"
    out = normalize_meta_format(text)
    assert "confidence: 0.25" in out
    assert "decision: redirect" in out


def test_normalize_handles_none_and_empty():
    assert normalize_meta_format("") == ""
    assert normalize_meta_format(None) is None


# --------------------------------------------------------------------------- #
# validate: keep well-formed, drop fatal
# --------------------------------------------------------------------------- #
def test_validate_accepts_wellformed_block():
    text = "<|meta|>\nconfidence: 0.25\ndecision: redirect\n<|/meta|>\nThe answer is $7$."
    ok, reason = validate_meta_structure(text)
    assert ok is True
    assert reason == ""


def test_validate_accepts_verify_decision():
    text = "<|meta|>\nconfidence: 0.85\ndecision: verify\n<|/meta|>\nok"
    ok, _ = validate_meta_structure(text)
    assert ok is True


def test_validate_drops_zero_blocks():
    ok, reason = validate_meta_structure("no meta here. The answer is $7$.")
    assert ok is False
    assert "block" in reason.lower()


def test_validate_drops_more_than_one_block():
    text = (
        "<|meta|>\nconfidence: 0.2\ndecision: redirect\n<|/meta|>\n"
        "<|meta|>\nconfidence: 0.9\ndecision: verify\n<|/meta|>\n"
    )
    ok, reason = validate_meta_structure(text)
    assert ok is False
    assert "block" in reason.lower()


def test_validate_drops_missing_confidence():
    text = "<|meta|>\ndecision: redirect\n<|/meta|>\nok"
    ok, reason = validate_meta_structure(text)
    assert ok is False
    assert "confidence" in reason.lower()


def test_validate_drops_missing_decision():
    text = "<|meta|>\nconfidence: 0.25\n<|/meta|>\nok"
    ok, reason = validate_meta_structure(text)
    assert ok is False
    assert "decision" in reason.lower()


def test_validate_drops_invalid_decision_value():
    text = "<|meta|>\nconfidence: 0.25\ndecision: ponder\n<|/meta|>\nok"
    ok, reason = validate_meta_structure(text)
    assert ok is False
    assert "decision" in reason.lower()


def test_validate_handles_none_and_empty():
    ok, reason = validate_meta_structure("")
    assert ok is False
    ok2, _ = validate_meta_structure(None)
    assert ok2 is False


# --------------------------------------------------------------------------- #
# the load-bearing case: a </|meta|> demo is REPAIRED then PASSES validation
# --------------------------------------------------------------------------- #
def test_repaired_close_tag_demo_passes_validation():
    raw = "<|meta|>\nconfidence: 0.25\ndecision: redirect\n</|meta|>\nThe answer is $7$."
    # raw is fatal (the strict regex sees zero well-formed blocks)
    assert validate_meta_structure(raw)[0] is False
    repaired = normalize_meta_format(raw)
    assert validate_meta_structure(repaired)[0] is True
