"""Tests for build_segment_loss_mask / redirect_train_spans.

Redirect-priming SFT (spec docs/superpowers/specs/2026-06-18-redirect-priming-
counterfactual-design.md §4 Stage B) trains traces of the form:

    [prompt][wrong_prefix]<|meta|>...<|switch|>...<|/meta|>[correct continuation]

We must MASK loss on [prompt]+[wrong_prefix] (context only — do NOT teach the
model to PRODUCE bad reasoning / performative struggling) and train loss ONLY on
[meta..switch..]+[correct continuation]. Boundaries are TOKEN-ID-INDEX based
(chat-template-robust), not char offsets.
"""

import pytest

from src.training.segment_loss_mask import (
    build_segment_loss_mask,
    redirect_train_spans,
)

IGN = -100


class TestBuildSegmentLossMask:
    def test_prompt_prefix_masked_continuation_trained(self):
        # seq_len 10; prompt+prefix = first 4 tokens masked, train on [4,10)
        mask = build_segment_loss_mask(10, [(4, 10)])
        assert mask == [IGN, IGN, IGN, IGN, 1, 1, 1, 1, 1, 1]

    def test_empty_spans_all_masked(self):
        assert build_segment_loss_mask(5, []) == [IGN] * 5

    def test_overlap_union(self):
        # overlapping spans should union, not double-count or error
        mask = build_segment_loss_mask(8, [(1, 4), (3, 6)])
        assert mask == [IGN, 1, 1, 1, 1, 1, IGN, IGN]

    def test_adjacent_spans_union(self):
        mask = build_segment_loss_mask(6, [(0, 2), (2, 4)])
        assert mask == [1, 1, 1, 1, IGN, IGN]

    def test_out_of_range_clamped(self):
        # end beyond seq_len clamps; negative start clamps to 0
        mask = build_segment_loss_mask(5, [(3, 99)])
        assert mask == [IGN, IGN, IGN, 1, 1]
        mask2 = build_segment_loss_mask(5, [(-3, 2)])
        assert mask2 == [1, 1, IGN, IGN, IGN]

    def test_fully_out_of_range_span_skipped(self):
        # span entirely past the end contributes nothing
        assert build_segment_loss_mask(4, [(10, 20)]) == [IGN] * 4

    def test_start_ge_end_skipped(self):
        assert build_segment_loss_mask(5, [(3, 3)]) == [IGN] * 5
        assert build_segment_loss_mask(5, [(4, 2)]) == [IGN] * 5

    def test_custom_ignore_index(self):
        mask = build_segment_loss_mask(4, [(2, 4)], ignore_index=-1)
        assert mask == [-1, -1, 1, 1]

    def test_zero_seq_len(self):
        assert build_segment_loss_mask(0, [(0, 5)]) == []

    def test_boundary_token_trained_prior_masked(self):
        # The token at boundary index is trained; the one before is masked.
        prompt_len, prefix_len, total = 3, 2, 9
        start = prompt_len + prefix_len  # 5
        mask = build_segment_loss_mask(total, redirect_train_spans(prompt_len, prefix_len, total))
        assert mask[start] == 1          # boundary token trained
        assert mask[start - 1] == IGN    # token just before masked
        assert mask[total - 1] == 1      # last continuation token trained


class TestRedirectTrainSpans:
    def test_single_span_after_prompt_and_prefix(self):
        spans = redirect_train_spans(prompt_len=3, prefix_len=2, total_len=9)
        assert spans == [(5, 9)]

    def test_no_prefix(self):
        spans = redirect_train_spans(prompt_len=4, prefix_len=0, total_len=10)
        assert spans == [(4, 10)]

    def test_full_mask_when_nothing_after(self):
        # prompt+prefix consume the whole sequence -> empty train region
        spans = redirect_train_spans(prompt_len=5, prefix_len=5, total_len=10)
        mask = build_segment_loss_mask(10, spans)
        assert mask == [IGN] * 10

    def test_integration_with_build(self):
        spans = redirect_train_spans(prompt_len=2, prefix_len=3, total_len=8)
        mask = build_segment_loss_mask(8, spans)
        assert mask == [IGN, IGN, IGN, IGN, IGN, 1, 1, 1]
