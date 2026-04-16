import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from src.training.self_distill.teacher_query import (
    build_teacher_query_dataframe,
    extract_topk_targets,
)


def test_extract_topk_targets_shapes():
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.9, -0.3],
            [1.2, 0.0, -0.2, 0.4],
        ],
        dtype=torch.float32,
    )
    payload = extract_topk_targets(logits, [2, 0], top_k=2)
    assert payload["num_positions"] == 2
    assert len(payload["teacher_topk_token_ids"]) == 2
    assert len(payload["teacher_topk_token_ids"][0]) == 2
    assert payload["assistant_token_ids"] == [2, 0]
    assert len(payload["teacher_target_logprobs"]) == 2


def test_build_teacher_query_dataframe_with_fake_query():
    rows = [
        {
            "messages": json.dumps(
                [
                    {"role": "user", "content": "Solve x+7=12."},
                    {"role": "assistant", "content": "Subtract 7, so x=5. \\boxed{5}"},
                ]
            ),
            "self_distill_mode": "sdpo_regen",
        }
    ]

    def fake_query(messages):
        assert messages[0]["role"] == "user"
        return {
            "teacher_topk_token_ids": [[1, 2], [3, 4]],
            "teacher_topk_logprobs": [[-0.1, -0.2], [-0.3, -0.4]],
            "teacher_target_logprobs": [-0.15, -0.35],
            "assistant_token_ids": [10, 11],
            "num_positions": 2,
            "prompt_len_tokens": 5,
            "completion_len_tokens": 2,
        }

    df = build_teacher_query_dataframe(rows, query_fn=fake_query, top_k=2)
    assert len(df) == 1
    assert df.iloc[0]["teacher_query_top_k"] == 2
    assert json.loads(df.iloc[0]["teacher_topk_token_ids_json"])[0] == [1, 2]
    assert json.loads(df.iloc[0]["assistant_token_ids_json"]) == [10, 11]
