"""Tokenizer compatibility helpers for metacognition training."""

from __future__ import annotations

import inspect


def _get_additional_special_tokens(tokenizer) -> list[str]:
    special_map = getattr(tokenizer, "special_tokens_map", {}) or {}
    tokens = special_map.get("additional_special_tokens")
    if tokens is None:
        tokens = getattr(tokenizer, "additional_special_tokens", None)
    return list(tokens or [])


def _supports_replace_kwarg(tokenizer) -> bool:
    try:
        signature = inspect.signature(tokenizer.add_special_tokens)
    except (TypeError, ValueError):
        return False
    return "replace_additional_special_tokens" in signature.parameters


def ensure_meta_tokens_not_special(tokenizer, meta_tokens: list[str]) -> None:
    """Keep meta markers in the vocab but out of additional special tokens.

    Some tokenizer implementations expose `special_tokens_map` but not the
    `additional_special_tokens` attribute. Reward parsing relies on TRL not
    stripping the meta markers, so we normalize through the portable API.
    """
    additional_special = _get_additional_special_tokens(tokenizer)
    for token in meta_tokens:
        if token in tokenizer.get_vocab():
            if token in additional_special:
                additional_special = [t for t in additional_special if t != token]
        else:
            tokenizer.add_tokens([token])

    payload = {"additional_special_tokens": additional_special}
    if _supports_replace_kwarg(tokenizer):
        tokenizer.add_special_tokens(payload, replace_additional_special_tokens=True)
    else:
        tokenizer.add_special_tokens(payload)

    # Older tokenizers may not expose a full replacement API. Keep the
    # observable tokenizer state consistent for downstream checks and smoke
    # scripts that rely on either attribute.
    if hasattr(tokenizer, "special_tokens_map") and isinstance(tokenizer.special_tokens_map, dict):
        tokenizer.special_tokens_map["additional_special_tokens"] = list(additional_special)
    if hasattr(tokenizer, "additional_special_tokens"):
        try:
            tokenizer.additional_special_tokens = list(additional_special)
        except Exception:
            pass
