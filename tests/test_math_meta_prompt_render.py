"""render_generation_prompt/render_chat_messages 회귀 시험 — math_sites.py/math_ruler_pivot.py/
math_retry_eval.py/math_rollout.py 가 각자 갖고 있던 `_gen_prompt`/`chat` 네 벌(apply_chat_template
+ enable_thinking=False 폴백)을 src/metacot/math_meta_prompt.py 로 합친 뒤 출력이 예전 구현과
바이트 단위로 같은지 확인한다."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metacot.math_meta_prompt import (
    build_math_prompt, render_chat_messages, render_generation_prompt,
)


class _TokWithThinking:
    """enable_thinking kwarg 를 받는(신 템플릿) 가짜 토크나이저."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=None):
        assert tokenize is False and add_generation_prompt is True
        parts = [f"<{m['role']}>{m['content']}" for m in msgs]
        return "|".join(parts) + f"|thinking={enable_thinking}|GEN"


class _TokWithoutThinking:
    """enable_thinking kwarg 를 모르는(구 템플릿) 가짜 토크나이저 — TypeError 폴백 경로."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
        parts = [f"<{m['role']}>{m['content']}" for m in msgs]
        return "|".join(parts) + "|GEN"


def _old_gen_prompt(tok, msgs) -> str:
    """math_sites.py 가 갖고 있던 예전 구현(제거되기 전)의 정확한 재현 — 회귀 기준선."""
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def test_render_chat_messages_matches_old_implementation_new_template():
    tok = _TokWithThinking()
    msgs = build_math_prompt("2+2=?", "math_opt")
    assert render_chat_messages(tok, msgs) == _old_gen_prompt(tok, msgs)


def test_render_chat_messages_matches_old_implementation_legacy_template():
    tok = _TokWithoutThinking()
    msgs = build_math_prompt("2+2=?", "math_opt")
    assert render_chat_messages(tok, msgs) == _old_gen_prompt(tok, msgs)


def test_render_generation_prompt_matches_build_then_render():
    tok = _TokWithThinking()
    problem = "What is 6*7?"
    variant = "math_retry"
    expect = _old_gen_prompt(tok, build_math_prompt(problem, variant))
    assert render_generation_prompt(tok, variant, problem) == expect


def test_render_generation_prompt_matches_math_ruler_pivot_old_signature():
    """math_ruler_pivot._gen_prompt(tok, variant, problem) 은 build_math_prompt 대신 인라인으로
    {"role": "system"/"user"} 을 만들었지만, 두 문제 모두 앞뒤 공백이 없는 한(실무 입력이 늘
    그렇다 — sites.jsonl 의 problem 은 parquet 원문) build_math_prompt 의 .strip() 은 결과에
    영향을 주지 않는다."""
    tok = _TokWithThinking()
    problem, variant = "solve for x: x+1=2", "math_new"
    old_msgs = [{"role": "system", "content": __import__(
        "src.metacot.math_meta_prompt", fromlist=["MATH_PROMPT_VARIANTS"]).MATH_PROMPT_VARIANTS[variant]},
        {"role": "user", "content": problem}]
    assert render_generation_prompt(tok, variant, problem) == _old_gen_prompt(tok, old_msgs)
