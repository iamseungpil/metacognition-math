import torch
from src.training.meta_token_init import transplant_meta_embeddings_from_think


class _FakeEmb:
    def __init__(self, n, d):
        self.weight = torch.nn.Parameter(torch.randn(n, d))


class _FakeModel:
    def __init__(self, n, d):
        self._e = _FakeEmb(n, d)

    def get_input_embeddings(self):
        return self._e

    def get_output_embeddings(self):
        return self._e


class _FakeTok:
    def __init__(self, m):
        self.m = m

    def convert_tokens_to_ids(self, t):
        return self.m[t]


def test_meta_rows_become_think_rows():
    tok = _FakeTok({"<think>": 10, "</think>": 11, "<|meta|>": 12, "<|/meta|>": 13})
    model = _FakeModel(20, 4)
    transplant_meta_embeddings_from_think(model, tok)
    w = model.get_input_embeddings().weight
    assert torch.allclose(w[12], w[10])   # <|meta|> <- <think>
    assert torch.allclose(w[13], w[11])   # <|/meta|> <- </think>
