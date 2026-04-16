import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.eval_hf import build_run_metadata, evaluate


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 0
    pad_token = "<pad>"
    eos_token = "<eos>"
    name_or_path = "dummy-tokenizer"
    additional_special_tokens = []

    def __init__(self, completion_text: str):
        self.completion_text = completion_text
        self.generated_ids = [101, 102, 103]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "\n".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += "\nassistant:"
        return text if not tokenize else list(range(len(text.split())))

    def __call__(self, text, return_tensors="pt", truncation=False, max_length=None):
        tokens = list(range(len(str(text).split())))
        if truncation and max_length is not None:
            tokens = tokens[:max_length]
        return {
            "input_ids": torch.tensor([tokens], dtype=torch.long),
            "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=False):
        if list(token_ids) == self.generated_ids:
            return self.completion_text
        return " ".join(str(x) for x in token_ids)

    def convert_tokens_to_ids(self, token):
        return 1


class DummyModel:
    def __init__(self, generated_ids):
        self.device = torch.device("cpu")
        self.generated_ids = torch.tensor([generated_ids], dtype=torch.long)

    def generate(self, input_ids=None, attention_mask=None, max_new_tokens=None, **kwargs):
        return torch.cat([input_ids, self.generated_ids], dim=1)


def test_evaluate_records_prompt_truncation_and_token_limit():
    completion = "<|meta|>\nconfidence: 0.2\n<|/meta|>\n\\boxed{4}"
    tokenizer = DummyTokenizer(completion)
    model = DummyModel(tokenizer.generated_ids)
    problems = [{"question": "word " * 20, "gold_answer": "4", "benchmark": "math500"}]

    results = evaluate(
        model,
        tokenizer,
        problems,
        max_tokens=3,
        max_prompt_tokens=5,
        do_sample=False,
    )

    row = results[0]
    assert row["prompt_was_truncated"] is True
    assert row["prompt_total_tokens_before_truncation"] > row["prompt_length_tokens"]
    assert row["hit_max_new_tokens"] is True


def test_build_run_metadata_includes_prompt_budget():
    class Args:
        model_path = "m"
        base_model = None
        is_lora = False
        benchmarks = ["math500"]
        max_problems = 10
        num_samples = 1
        max_new_tokens = 4096
        max_prompt_tokens = 16384
        do_sample = False
        temperature = 0.0
        top_p = 1.0
        seed = 0
        device_map = "single"

    tokenizer = DummyTokenizer("\\boxed{1}")
    metadata = build_run_metadata(Args, "dummy", [{}], tokenizer, resolved_do_sample=False)
    assert metadata["max_prompt_tokens"] == 16384
