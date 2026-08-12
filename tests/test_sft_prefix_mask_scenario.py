"""Regression: the wrong_prefix loss mask must be scenario-aware (EXP-0812c).

Masking the prefix is only correct when the prefix is genuinely flawed
(redirect rows). SFT2 applied it to ALL 1,763 rows including the 1,209 verify
rows whose prefix is the model's own sound first pass, which removed every
"reason first" example from the corpus and produced meta-first generation from
RL step 1 (97.2% of on-policy rollouts open with <|meta|>).

Contract pinned here:
  - redirect + prefix  -> masked   (original intent: never teach a flawed prefix)
  - verify   + prefix  -> NOT masked (the fix: train reason->meta->verification)
  - no scenario field  -> masked   (legacy corpora train byte-identically)
  - empty prefix       -> never masked, any scenario
"""
from src.training.sft import _should_mask_prefix


def test_redirect_prefix_is_masked():
    assert _should_mask_prefix("I started with 2+2=5 so", "redirect") is True


def test_verify_prefix_is_not_masked():
    assert _should_mask_prefix("<think>40*5=200 ...", "verify") is False


def test_missing_scenario_keeps_legacy_mask():
    assert _should_mask_prefix("some prefix", "") is True


def test_empty_prefix_never_masks():
    assert _should_mask_prefix("", "redirect") is False
    assert _should_mask_prefix("", "verify") is False
    assert _should_mask_prefix("", "") is False


def test_unknown_scenario_is_not_masked():
    # Any explicitly-labeled non-redirect scenario carries a sound prefix by
    # contract; only redirect (and the legacy no-field case) masks.
    assert _should_mask_prefix("prefix", "plain") is False
