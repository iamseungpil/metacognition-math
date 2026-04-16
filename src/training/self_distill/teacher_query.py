"""Teacher top-k query utilities for OPD-style dense targets."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from src.curriculum.control_rag import render_messages_as_text
from src.training.self_distill.trace import load_messages, read_table


def _normalize_token_ids(encoded) -> list[int]:
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict) and "input_ids" in encoded:
        encoded = encoded["input_ids"]
    if torch is not None and isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()
    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]
    return [int(x) for x in encoded]


def tokenize_chat_messages(
    tokenizer,
    messages: list[dict[str, str]],
    *,
    max_length: int = 4096,
) -> tuple[list[int], int]:
    prompt_messages = messages[:-1]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            prompt_ids = _normalize_token_ids(
                tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True)
            )
            full_ids = _normalize_token_ids(
                tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
            )
        except Exception:
            prompt_text = render_messages_as_text(prompt_messages, add_generation_prompt=True)
            full_text = render_messages_as_text(messages, add_generation_prompt=False)
            prompt_ids = _normalize_token_ids(tokenizer(prompt_text, return_tensors=None))
            full_ids = _normalize_token_ids(tokenizer(full_text, return_tensors=None))
    else:
        prompt_text = render_messages_as_text(prompt_messages, add_generation_prompt=True)
        full_text = render_messages_as_text(messages, add_generation_prompt=False)
        prompt_ids = _normalize_token_ids(tokenizer(prompt_text, return_tensors=None))
        full_ids = _normalize_token_ids(tokenizer(full_text, return_tensors=None))

    if len(full_ids) > max_length:
        full_ids = full_ids[:max_length]
    return full_ids, min(len(prompt_ids), len(full_ids))


def extract_topk_targets(
    logits,
    target_token_ids: list[int],
    *,
    top_k: int,
) -> dict[str, Any]:
    if torch is None:
        raise ImportError("torch is required for teacher top-k extraction")
    if logits.ndim != 2:
        raise ValueError(f"Expected (T, V) logits, got shape {tuple(logits.shape)}")
    if logits.shape[0] != len(target_token_ids):
        raise ValueError(
            f"Mismatch between logits positions ({logits.shape[0]}) and target tokens ({len(target_token_ids)})"
        )

    log_probs = torch.log_softmax(logits, dim=-1)
    top_vals, top_idx = torch.topk(log_probs, k=min(top_k, log_probs.shape[-1]), dim=-1)
    target_log_probs = log_probs.gather(
        dim=-1,
        index=torch.tensor(target_token_ids, device=log_probs.device, dtype=torch.long).unsqueeze(-1),
    ).squeeze(-1)
    return {
        "teacher_topk_token_ids": top_idx.cpu().tolist(),
        "teacher_topk_logprobs": top_vals.cpu().tolist(),
        "teacher_target_logprobs": target_log_probs.cpu().tolist(),
        "assistant_token_ids": [int(x) for x in target_token_ids],
        "num_positions": len(target_token_ids),
    }


def query_teacher_topk_for_messages(
    *,
    model,
    tokenizer,
    messages: list[dict[str, str]],
    top_k: int = 16,
    max_length: int = 4096,
) -> dict[str, Any]:
    if torch is None:
        raise ImportError("torch is required for teacher query")
    full_ids, prompt_len = tokenize_chat_messages(tokenizer, messages, max_length=max_length)
    if len(full_ids) <= prompt_len:
        raise ValueError("No assistant tokens available for teacher query")

    input_ids = torch.tensor([full_ids], device=model.device, dtype=torch.long)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits[0]

    assistant_token_ids = full_ids[prompt_len:]
    # logits[pos-1] predicts token at pos
    start = max(prompt_len - 1, 0)
    end = len(full_ids) - 1
    teacher_logits = logits[start:end, :]
    payload = extract_topk_targets(teacher_logits, assistant_token_ids, top_k=top_k)
    payload["prompt_len_tokens"] = int(prompt_len)
    payload["completion_len_tokens"] = int(len(assistant_token_ids))
    return payload


def build_teacher_query_dataframe(
    rows: list[dict[str, Any]],
    *,
    query_fn: Callable[[list[dict[str, str]]], dict[str, Any]],
    top_k: int,
    source_tag: str | None = None,
) -> pd.DataFrame:
    built_rows = []
    for row in rows:
        raw_messages = row.get("messages")
        if raw_messages is None:
            continue
        messages = load_messages(raw_messages)
        payload = query_fn(messages)
        built = dict(row)
        built["teacher_query_source"] = source_tag or "teacher_topk_query"
        built["teacher_query_top_k"] = int(top_k)
        built["teacher_topk_token_ids_json"] = json.dumps(payload["teacher_topk_token_ids"], ensure_ascii=False)
        built["teacher_topk_logprobs_json"] = json.dumps(payload["teacher_topk_logprobs"], ensure_ascii=False)
        built["teacher_target_logprobs_json"] = json.dumps(payload["teacher_target_logprobs"], ensure_ascii=False)
        built["assistant_token_ids_json"] = json.dumps(payload["assistant_token_ids"], ensure_ascii=False)
        built["teacher_num_positions"] = int(payload["num_positions"])
        built["teacher_prompt_len_tokens"] = int(payload["prompt_len_tokens"])
        built["teacher_completion_len_tokens"] = int(payload["completion_len_tokens"])
        built_rows.append(built)
    return pd.DataFrame(built_rows)


def build_teacher_query_dataset(
    input_path: str | Path,
    *,
    query_fn: Callable[[list[dict[str, str]]], dict[str, Any]],
    top_k: int,
    source_tag: str | None = None,
) -> pd.DataFrame:
    return build_teacher_query_dataframe(
        read_table(input_path),
        query_fn=query_fn,
        top_k=top_k,
        source_tag=source_tag,
    )


__all__ = [
    "build_teacher_query_dataframe",
    "build_teacher_query_dataset",
    "extract_topk_targets",
    "query_teacher_topk_for_messages",
    "tokenize_chat_messages",
]
