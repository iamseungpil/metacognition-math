"""TDD for the cf_group PLACEBO-META without-arm (design 2026-06-22).

PURE PYTHON (no verl/vLLM/torch needed) — runs under metaprobe. Tests the two
unit-testable pieces of the placebo fix:
  - the placebo opener STRING round-trips through a (fake) tokenizer and contains
    the meta tags + the contentless body;
  - build_placebo_output(...) — the pure helper that decides what the agent loop
    returns: prompt_ids UNCHANGED (placebo NOT in prompt), response_ids =
    (placebo + continuation)[:response_length], response_mask all-1 over them,
    response_logprobs None (verl actor recomputes).

HONEST LIMIT (stated in code + report): the DEFINITIVE check — that forcing the
placebo opener actually makes the without-arm SOLVE on-distribution (non-empty,
acc_without > 0) on the live v8_rv_confidence_warmup model — is NOT unit-testable
(needs verl + vLLM + the real model + GPU). The live special-token ids
151669/151670 are only checkable with the real tokenizer; here we assert on the
STRING + a fake-tokenizer round-trip.
"""
from src.training.cf_placebo_agent import build_placebo_output, placebo_opener_str
from src.training.dcpo_pmi import PLACEBO_META


class _FakeTokenizer:
    """Minimal whitespace/segment tokenizer that treats the meta tags + body as
    atomic 'words' and assigns each a stable id. encode/decode round-trip cleanly
    so we can assert the placebo opener tokenizes and decodes back."""

    def __init__(self):
        self._vocab = {}
        self._inv = {}

    def _id(self, tok):
        if tok not in self._vocab:
            i = len(self._vocab) + 1000
            self._vocab[tok] = i
            self._inv[i] = tok
        return self._vocab[tok]

    def encode(self, text, add_special_tokens=False):
        # Split on whitespace but keep the segments; good enough to round-trip.
        return [self._id(t) for t in text.split(" ") if t != ""]

    def decode(self, ids):
        return " ".join(self._inv[i] for i in ids)


# ─────────────────────────────────────────────────────────────────────────────
# Placebo opener string
# ─────────────────────────────────────────────────────────────────────────────
def test_placebo_meta_constant():
    # SSOT: same placebo the offline probe validated.
    assert PLACEBO_META == "<|meta|>\nLet me continue.\n<|/meta|>"


def test_placebo_opener_string_shape():
    s = placebo_opener_str()
    # The SFT response body starts with '<think>\n' (meta inside think); the
    # contentless opener mirrors it then closes the placebo block.
    assert s.startswith("<think>")
    assert "<|meta|>" in s
    assert "Let me continue." in s
    assert "<|/meta|>" in s
    # exact string the implementation forces
    assert s == "<think>\n" + PLACEBO_META + "\n"
    assert s == "<think>\n<|meta|>\nLet me continue.\n<|/meta|>\n"


def test_placebo_opener_roundtrips_through_tokenizer():
    tok = _FakeTokenizer()
    s = placebo_opener_str()
    ids = tok.encode(s, add_special_tokens=False)
    dec = tok.decode(ids)
    assert "<|meta|>" in dec
    assert "Let me continue." in dec
    assert "<|/meta|>" in dec


# ─────────────────────────────────────────────────────────────────────────────
# build_placebo_output contract — the GRPO-member split
# ─────────────────────────────────────────────────────────────────────────────
def test_build_placebo_output_response_is_placebo_plus_continuation():
    prompt_ids = [10, 11, 12]
    placebo_ids = [20, 21, 22]
    gen_ids = [30, 31, 32, 33]
    out = build_placebo_output(prompt_ids, placebo_ids, gen_ids, response_length=100)
    # prompt UNCHANGED — placebo is NOT in the prompt
    assert out["prompt_ids"] == prompt_ids
    # placebo IS the trained response prefix, then the continuation
    assert out["response_ids"] == placebo_ids + gen_ids
    # all-1 mask over placebo + continuation (matches single_turn masking its own opener)
    assert out["response_mask"] == [1] * len(placebo_ids + gen_ids)
    # rollout logprobs dropped -> verl actor recomputes over the full response
    assert out["response_logprobs"] is None


def test_build_placebo_output_truncates_to_response_length():
    prompt_ids = [1, 2]
    placebo_ids = [20, 21, 22]
    gen_ids = [30, 31, 32, 33, 34, 35]
    out = build_placebo_output(prompt_ids, placebo_ids, gen_ids, response_length=5)
    assert out["response_ids"] == [20, 21, 22, 30, 31]
    assert out["response_mask"] == [1, 1, 1, 1, 1]
    assert out["prompt_ids"] == prompt_ids


def test_build_placebo_output_does_not_mutate_inputs():
    prompt_ids = [1, 2]
    placebo_ids = [20, 21]
    gen_ids = [30, 31]
    build_placebo_output(prompt_ids, placebo_ids, gen_ids, response_length=100)
    assert prompt_ids == [1, 2]
    assert placebo_ids == [20, 21]
    assert gen_ids == [30, 31]
