r"""hint 모드 렌더 바이트 동일성 — `scripts/local/gen_continuations.py::build_hint_messages`
가 조립한 메시지를 실제 토크나이저로 렌더한 결과가, 힌트를 삽입하지 않은 평범한 렌더
(meta 모드와 같은 조립)와 **힌트 삽입 부분만 빼면** 바이트 동일한지 확인한다.

★이 테스트가 검증하는 것. `src/training/countdown_sites.py::render_prefix_prompt` 의
바이트-동일성을 검증한 `test_countdown_site_prompt_render.py` 와 같은 패턴이다 — 여기서는
"hint 모드가 meta 모드와 다른 유일한 지점이 user 메시지 끝의 힌트 텍스트"라는 과제
지시(Task A: "byte-identical rendered prompts... except for the hint insertion")를
토크나이저 수준에서 실측한다.

토크나이저가 없으면(이 CPU 환경엔 보통 없다 — GPU 박스 전용 경로) 스킵한다.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root

TOKENIZER_PATH = "/hdd_data/seungpil/scratch/models/Qwen3-4B"


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed in this environment")
    if not Path(TOKENIZER_PATH).is_dir():
        pytest.skip(f"tokenizer not found at {TOKENIZER_PATH} (GPU box only)")
    return AutoTokenizer.from_pretrained(TOKENIZER_PATH)


def _render(tok, msgs) -> str:
    return tok.apply_chat_template(
        msgs, tokenize=False, continue_final_message=True,
        add_generation_prompt=False, enable_thinking=False)


def test_hint_render_equals_plain_render_except_hint_insertion():
    import scripts.local.gen_continuations as gc
    from src.training.countdown_sites import render_prefix_prompt

    tok = _load_tokenizer()
    nums = [1, 2, 3, 4]
    target = 10
    prefix = "2-1=1\n"                    # family_dead=1, live_new_moves 있음(실측 — test_countdown_opd.py 와 같은 인스턴스)
    inst = {"nums": nums, "target": target}

    plain_msgs = render_prefix_prompt(inst, prefix, "new")   # [system,user,assistant(prefix)]
    hint_msgs, hint = gc.build_hint_messages(plain_msgs, nums, target, prefix)
    assert hint_msgs is not None and hint != ""

    plain_rendered = _render(tok, plain_msgs)
    hint_rendered = _render(tok, hint_msgs)

    assert hint_rendered != plain_rendered            # 힌트가 실제로 무언가를 바꿨다
    # 힌트가 user 메시지 content 뒤에 "\n\n" + hint 로 붙는다 — 렌더된 문자열에서
    # 그 삽입 지점을 빼면 두 렌더가 바이트 동일해야 한다.
    inserted = "\n\n" + hint
    assert inserted in hint_rendered
    assert hint_rendered.replace(inserted, "", 1) == plain_rendered
    # 프리픽스는 두 렌더 모두 꼬리에 그대로 남아 있다(assistant 프리픽스는 손대지 않았다).
    assert plain_rendered.endswith(prefix)
    assert hint_rendered.endswith(prefix)


def test_hint_render_skipped_site_has_no_hint_messages():
    import scripts.local.gen_continuations as gc

    nums = [24, 15, 22, 7]
    target = 331
    prefix = "Let me think about this."
    msgs = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "Numbers: [24, 15, 22, 7]\nTarget: 331"},
        {"role": "assistant", "content": prefix},
    ]
    out, hint = gc.build_hint_messages(msgs, nums, target, prefix)
    assert out is None and hint == ""
