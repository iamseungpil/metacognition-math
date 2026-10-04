"""관문 G4 — 1턴 프롬프트가 현행 `math_opt` 와 바이트 동일 · 합의 상태 자 · 프롬프트 변형 손잡이."""
from __future__ import annotations

import pytest

from mc import context as ctx

PROBLEMS = [
    "What is $1+1$?",
    "Find the minimum value of $x^2+2x+3$.",
    r"Let $f(x)=\frac{1}{x}$. Compute $f(2)$.",
    "How many primes are below 10?",
    r"Solve $\sin\theta = \tfrac12$ for $0\le\theta<2\pi$.",
]


class FakeTok:
    """chat 템플릿 대역 — 두 경로가 **같은 메시지**를 넘기는지만 본다(실 토크나이저는
    test_context_real_tokenizer 가 있으면 쓴다)."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True,
                            enable_thinking=False):
        return "".join(f"<|{m['role']}|>{m['content']}" for m in msgs) + "<|assistant|>"


@pytest.fixture
def tok():
    return FakeTok()


def test_turn1_prompt_byte_identical_to_math_opt(tok):
    from src.metacot.math_meta_prompt import render_generation_prompt
    for p in PROBLEMS:
        assert ctx.turn1_prompt(tok, p) == render_generation_prompt(tok, "math_opt", p)


def test_agreement_state_is_the_shared_ruler():
    from src.training.trial2 import agreement_state as ref
    for ans in (["3"] * 8, ["3"] * 6 + ["9", "9"], ["3"] * 3 + ["9"] * 3 + ["1", "2"],
                ["", ""] + ["3"] * 6, [""] * 8):
        assert ctx.agreement_state(ans) == ref(ans)


def test_plain_prompt_is_math_opt_minus_the_paragraph():
    removed = ctx._PERMISSION + ctx._BLOCK + ctx._DECISION
    assert ctx.SOLVE_MATH_OPT.count(removed) == 1
    assert ctx.SOLVE_MATH_PLAIN == ctx.SOLVE_MATH_OPT.replace(removed, "")
    assert "<meta>" not in ctx.SOLVE_MATH_PLAIN and "metacognitive" not in ctx.SOLVE_MATH_PLAIN


def test_prompt_variant_knob_drives_every_consumer(monkeypatch):
    monkeypatch.delenv("PROMPT_VARIANT", raising=False)
    assert ctx.default_variant() == "math_opt"                 # 미설정 = 옛 동작
    assert ctx.build_math_prompt("q")[0]["content"] == ctx.SOLVE_MATH_OPT
    monkeypatch.setenv("PROMPT_VARIANT", "plain")
    assert ctx.build_math_prompt("q")[0]["content"] == ctx.SOLVE_MATH_PLAIN
    assert ctx.build_math_prompt("q", "math_opt")[0]["content"] == ctx.SOLVE_MATH_OPT
    monkeypatch.setenv("PROMPT_VARIANT", "nope")
    with pytest.raises(ValueError):
        ctx.default_variant()
