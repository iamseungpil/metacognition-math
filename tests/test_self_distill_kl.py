import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.sft import prepare_sft_dataset
from src.training.self_distill.kl import (
    build_control_span_weights,
    load_teacher_topk_payload,
    trim_teacher_payload,
)


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        tokens = text.split()
        input_ids = list(range(len(tokens)))
        if not return_offsets_mapping:
            return {"input_ids": input_ids}
        offsets = []
        cursor = 0
        for token in tokens:
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            cursor = end
        return {"input_ids": input_ids, "offset_mapping": offsets}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = " ".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += " assistant:"
        if tokenize:
            return list(range(len(text.split())))
        return text


def test_load_teacher_topk_payload_from_json_strings():
    row = {
        "teacher_topk_token_ids_json": json.dumps([[1, 2], [3, 4]]),
        "teacher_topk_logprobs_json": json.dumps([[-0.1, -0.2], [-0.3, -0.4]]),
        "teacher_target_logprobs_json": json.dumps([-0.15, -0.35]),
        "assistant_token_ids_json": json.dumps([10, 11]),
    }
    payload = load_teacher_topk_payload(row)
    assert payload is not None
    assert payload.assistant_token_ids == [10, 11]
    assert payload.token_ids[0] == [1, 2]


def test_trim_teacher_payload_respects_target_length():
    payload = load_teacher_topk_payload({
        "teacher_topk_token_ids_json": json.dumps([[1], [2], [3]]),
        "teacher_topk_logprobs_json": json.dumps([[-0.1], [-0.2], [-0.3]]),
        "teacher_target_logprobs_json": json.dumps([-0.1, -0.2, -0.3]),
        "assistant_token_ids_json": json.dumps([5, 6, 7]),
    })
    trimmed = trim_teacher_payload(payload, target_length=2)
    assert trimmed.assistant_token_ids == [5, 6]
    assert len(trimmed.token_ids) == 2


def test_build_control_span_weights_prioritizes_meta_and_study_need():
    tokenizer = DummyTokenizer()
    assistant_text = (
        "<|meta|> confidence: 0.31 The earlier route is weak. "
        "study_need: factor the constrained count <|/meta|> "
        "Re-solve carefully, verify by substituting back, and end with \\boxed{4}."
    )
    weights = build_control_span_weights(
        tokenizer=tokenizer,
        assistant_text=assistant_text,
        expected_length=len(assistant_text.split()),
        diagnosis_text="earlier route is weak",
        study_need="factor the constrained count",
    )
    assert len(weights) == len(assistant_text.split())
    assert max(weights) >= 1.4
    assert any(weight > 0 for weight in weights)


def test_build_control_span_weights_meta_only_masks_non_meta_tokens():
    tokenizer = DummyTokenizer()
    assistant_text = (
        "<|meta|> confidence: 0.31 study_need: factor the count <|/meta|> "
        "Re-solve carefully and verify by substituting back. \\boxed{4}"
    )
    weights = build_control_span_weights(
        tokenizer=tokenizer,
        assistant_text=assistant_text,
        expected_length=len(assistant_text.split()),
        mask_mode="meta_only",
        diagnosis_text="factor",
        study_need="factor the count",
    )

    meta_token_count = len("<|meta|> confidence: 0.31 study_need: factor the count <|/meta|>".split())
    assert any(weight > 0 for weight in weights[:meta_token_count])
    assert all(weight == 0.0 for weight in weights[meta_token_count:])


def test_prepare_sft_dataset_fails_closed_when_teacher_kl_enabled_without_targets(tmp_path):
    path = tmp_path / "missing_teacher_targets.parquet"
    pd.DataFrame([
        {
            "messages": json.dumps(
                [
                    {"role": "user", "content": "Solve x+3=7."},
                    {"role": "assistant", "content": "<|meta|> confidence: 0.4 <|/meta|> x=4 \\boxed{4}"},
                ]
            ),
            "diagnosis_text": "guessing",
            "study_need": "direct isolation",
        }
    ]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="teacher_kl is enabled"):
        prepare_sft_dataset(
            str(path),
            DummyTokenizer(),
            teacher_kl={"enabled": True, "coef": 0.1, "require_targets": True},
        )


def test_prepare_sft_dataset_meta_only_fails_closed_without_meta_spans(tmp_path):
    path = tmp_path / "meta_only_without_wrapped_meta.parquet"
    pd.DataFrame([
        {
            "messages": json.dumps(
                [
                    {"role": "user", "content": "Solve x+3=7."},
                    {"role": "assistant", "content": "confidence: 0.4 x=4 \\boxed{4}"},
                ]
            ),
            "diagnosis_text": "guessing",
            "study_need": "direct isolation",
            "teacher_topk_token_ids_json": json.dumps([[1], [2], [3], [4]]),
            "teacher_topk_logprobs_json": json.dumps([[-0.1], [-0.2], [-0.3], [-0.4]]),
            "teacher_target_logprobs_json": json.dumps([-0.1, -0.2, -0.3, -0.4]),
            "assistant_token_ids_json": json.dumps([10, 11, 12, 13]),
        }
    ]).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="teacher_kl is enabled"):
        prepare_sft_dataset(
            str(path),
            DummyTokenizer(),
            teacher_kl={
                "enabled": True,
                "coef": 0.1,
                "require_targets": True,
                "mask_mode": "meta_only",
            },
        )
