"""CPU unit tests for the directional (gm-contrast) R_meta pure core.

Covers what SURVIVES of src/training/dcpo_directional.py after the gm / RLSD
reward generations were removed (2026-08-03):
  - boxed_answer_string + decoy construction via _rule_based_decoy,
  - divergent_token_mask (excludes shared \boxed structural tokens).

Both are called by the LIVE pmi_shift scorer, so this is not legacy coverage.
"""
import math

import numpy as np
import torch

from src.training.dcpo_directional import (
    boxed_answer_string,
    divergent_token_mask,
)


# ── boxed_answer_string + decoy construction ────────────────────────────────
def test_boxed_answer_string_wraps_value():
    assert boxed_answer_string("42") == r"\boxed{42}"
    assert boxed_answer_string(7) == r"\boxed{7}"
    assert boxed_answer_string("  3/4 ") == r"\boxed{3/4}"


def test_rule_based_decoy_differs_from_gold():
    from src.training._decoy_utils import _rule_based_decoy
    gold = "42"
    decoy = _rule_based_decoy(gold, seed=42)
    assert decoy != gold
    # the gm contrast wraps both into boxed strings — they must differ
    assert boxed_answer_string(gold) != boxed_answer_string(decoy)


# ── divergent_token_mask ─────────────────────────────────────────────────────
def test_divergent_mask_excludes_shared_tokens():
    # gold and decoy share a structural prefix (the \boxed{ tokens) and differ
    # only at the value token.
    gold_ids = [100, 200, 300, 999]   # ... \boxed{ ... value=300 ... }
    decoy_ids = [100, 200, 301, 999]  # same except value token differs
    mask = divergent_token_mask(gold_ids, decoy_ids)
    assert mask.tolist() == [False, False, True, False]


def test_divergent_mask_extra_gold_tokens_are_divergent():
    gold_ids = [1, 2, 3, 4]
    decoy_ids = [1, 2]
    mask = divergent_token_mask(gold_ids, decoy_ids)
    # positions past the shorter decoy length are divergent by construction
    assert mask.tolist() == [False, False, True, True]
