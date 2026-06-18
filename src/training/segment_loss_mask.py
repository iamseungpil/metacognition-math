"""Token-id-index segment loss masking for redirect-priming SFT.

Spec: docs/superpowers/specs/2026-06-18-redirect-priming-counterfactual-design.md
§4 Stage B ("Segment loss-mask (I7, M5)").

Redirect-priming traces have the shape::

    [prompt][wrong_prefix]<|meta|>...<|switch|>...<|/meta|>[correct continuation]

The model must NOT be taught to PRODUCE the wrong prefix (that would reinforce
bad reasoning / performative struggling). So loss is MASKED on
``[prompt]+[wrong_prefix]`` and trained ONLY on the meta block + the correct
continuation.

The existing ``tokenize_row`` in ``sft.py`` masks by a single ``prompt_len``
boundary, which is insufficient here because the wrong prefix sits AFTER the
prompt and must also be masked. These helpers express the mask via TOKEN-ID
INDEX spans (chat-template-robust, not char offsets) so a caller can build a
labels-mask of arbitrary, possibly disjoint, trained regions.

These are pure functions with no torch / tokenizer dependency.

Integration (NOT wired here — sft.py left untouched):
    In ``sft.py`` ``tokenize_row``, after computing ``prompt_len`` and the
    tokenized full sequence, compute ``prefix_len`` (the wrong-prefix token
    count, e.g. len(tokenize(wrong_prefix))) and then::

        from src.training.segment_loss_mask import (
            build_segment_loss_mask, redirect_train_spans,
        )
        spans = redirect_train_spans(prompt_len, prefix_len, len(full_ids))
        keep = build_segment_loss_mask(len(full_ids), spans)
        labels = [tok if k == 1 else -100 for tok, k in zip(full_ids, keep)]

    Precedence: teacher_kl MUST be OFF for redirect-priming SFT — the teacher_kl
    control-span path keys off a single prompt_len boundary and the assistant
    text, and would re-introduce loss on the masked wrong-prefix. Keep
    teacher_kl.enabled = false whenever the redirect segment mask is used.
"""

from __future__ import annotations

__all__ = ["build_segment_loss_mask", "redirect_train_spans"]


def build_segment_loss_mask(
    seq_len: int,
    train_spans: list[tuple[int, int]],
    ignore_index: int = -100,
) -> list[int]:
    """Build a per-token labels-mask of length ``seq_len``.

    Position ``i`` gets the sentinel value ``1`` ("train here") iff ``i`` falls
    inside any ``[start, end)`` half-open span in ``train_spans``; otherwise it
    gets ``ignore_index``. The caller applies this by zipping with the real
    labels (``tok if keep == 1 else ignore_index``).

    Edge cases:
      * empty ``train_spans``     -> all ``ignore_index``
      * overlapping spans         -> union (no double counting / no error)
      * out-of-range spans        -> clamped to ``[0, seq_len)``
      * ``start >= end``          -> skipped
      * fully out-of-range span   -> contributes nothing
    """
    if seq_len <= 0:
        return []

    mask = [ignore_index] * seq_len
    for start, end in train_spans:
        # Clamp into range and skip degenerate / out-of-range spans.
        lo = max(0, int(start))
        hi = min(seq_len, int(end))
        if lo >= hi:
            continue
        for i in range(lo, hi):
            mask[i] = 1
    return mask


def redirect_train_spans(
    prompt_len: int,
    prefix_len: int,
    total_len: int,
) -> list[tuple[int, int]]:
    """Train spans for a redirect-priming trace.

    Everything AFTER ``[prompt]+[wrong_prefix]`` is trained; the prompt and the
    wrong prefix are masked. Encodes the spec's §4 Stage B masking decision as
    the single half-open span ``[prompt_len + prefix_len, total_len)``.

    If ``prompt_len + prefix_len >= total_len`` (nothing left to train), the
    returned span is degenerate and ``build_segment_loss_mask`` will mask the
    whole sequence.
    """
    start = int(prompt_len) + int(prefix_len)
    return [(start, int(total_len))]
