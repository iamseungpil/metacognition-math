"""prepare_sft_dataset must mask the REDIRECT wrong_prefix (train only meta+recovery)
while leaving VERIFY / plain rows on the prompt-only boundary mask.

GPU-free + network-free: a FAKE char-level tokenizer mimics apply_chat_template /
encode deterministically so the masking WIRING (sft.py -> segment_loss_mask) is tested
without a real model. The pure span math is covered by test_segment_loss_mask.py.
"""
from __future__ import annotations

import pandas as pd

from src.training.sft import prepare_sft_dataset


class _FakeTokenizer:
    """1 char -> 1 token id. The 'chat template' is just the concatenated message
    contents (no special wrappers), so token boundaries == char boundaries — enough
    to assert WHICH region is masked."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        # Mirror a real template: the generation-prompt opener "|G|" precedes the
        # assistant turn, so prompt_len (prompt + opener) lands exactly where the
        # assistant CONTENT begins in the full sequence.
        parts = []
        for m in messages:
            if m["role"] == "assistant":
                parts.append("|G|")
            parts.append(str(m["content"]))
        if add_generation_prompt:
            parts.append("|G|")
        return [ord(c) for c in "".join(parts)]


def test_redirect_row_masks_prompt_and_wrong_prefix(tmp_path):
    # assistant target = wrong_prefix + recovery; prompt = the user turn.
    user = "Q: 2+2?"
    wrong_prefix = "I think 2+2=5 so"
    recovery = "<|meta|>switch<|/meta|> The answer is 4."
    df = pd.DataFrame([{
        "messages": [{"role": "user", "content": user},
                     {"role": "assistant", "content": wrong_prefix + recovery}],
        "wrong_prefix": wrong_prefix,
    }])
    p = tmp_path / "r.parquet"
    df.to_parquet(p)

    ds = prepare_sft_dataset(str(p), _FakeTokenizer(), max_length=4096)
    row = ds[0]
    trained = "".join(chr(t) for t, lab in zip(row["input_ids"], row["labels"]) if lab != -100)
    # ONLY the recovery (meta + correct continuation) is trained; never the prompt,
    # never the wrong prefix.
    assert trained == recovery, trained
    assert wrong_prefix not in trained
    assert "Q: 2+2?" not in trained


def test_verify_row_trains_whole_assistant(tmp_path):
    user = "Q: sum?"
    assistant = "<|meta|>check<|/meta|> The answer is 210."
    df = pd.DataFrame([{
        "messages": [{"role": "user", "content": user},
                     {"role": "assistant", "content": assistant}],
        "wrong_prefix": "",  # verify -> no prefix to mask
    }])
    p = tmp_path / "v.parquet"
    df.to_parquet(p)

    ds = prepare_sft_dataset(str(p), _FakeTokenizer(), max_length=4096)
    row = ds[0]
    trained = "".join(chr(t) for t, lab in zip(row["input_ids"], row["labels"]) if lab != -100)
    assert trained == assistant, trained
