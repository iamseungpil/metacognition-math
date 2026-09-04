r"""site 프롬프트 렌더 바이트 동일성 — `countdown_sites.render_prefix_prompt` 가 만드는
[system,user,assistant(prefix)] 메시지를, verl 의 agent-loop 가 실제로 부르는 것과 같은
호출 형태(`tokenizer.apply_chat_template(..., continue_final_message=True,
add_generation_prompt=False)`)로 렌더한 결과가 `render_prompt + prefix`(프리픽스 없는
프롬프트를 generation-prompt 로 렌더한 것 + 프리픽스 원문)와 **바이트 동일**한지 확인한다.

★이 테스트가 검증하는 것과 안 하는 것.
  검증한다 — HF 토크나이저 수준에서 `continue_final_message=True` 이어붙이기가
  `countdown_sites.py` 모듈 docstring §6 이 "이 모듈은 검증하지 않는다"고 명시한
  바로 그 바이트-동일성 가정을 만족하는가(Qwen3-4B 템플릿 기준).
  검증하지 않는다 — `sitecustomize._patch_verl_agent_loop_chat_template` 이 실제
  verl `AgentLoopBase.apply_chat_template` 안에서 같은 인자로 불리는가(verl 이
  이 CPU 테스트 환경에 없다 — `sitecustomize.py` 의 패치 자체는 이 값들을 그대로
  HF `apply_chat_template` 에 전달할 뿐이므로, 이 테스트가 통과하면 그 전달의
  "받는 쪽"이 옳다는 것은 보장된다).

토크나이저가 없으면(이 CPU 환경엔 보통 없다 — GPU 박스 전용 경로) 스킵한다.
"""
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


def test_site_prompt_continue_final_message_matches_render_prompt_plus_prefix():
    from src.training.countdown_sites import render_prefix_prompt
    from src.training.countdown_task import build_prompt

    tok = _load_tokenizer()
    inst = {"nums": [3, 5, 7, 9], "target": 24}
    prefix = "Let me try 5+9=14 first, then 14+... "

    site_msgs = render_prefix_prompt(inst, prefix, "new")
    # ★run_arm.sh 가 이미 쓰는 것과 같은 override
    #   (+data.apply_chat_template_kwargs.enable_thinking=false) — 조건을 맞춘다.
    rendered = tok.apply_chat_template(
        site_msgs, tokenize=False, add_generation_prompt=False,
        continue_final_message=True, enable_thinking=False)

    base_msgs = build_prompt(inst, "new")
    render_prompt = tok.apply_chat_template(
        base_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    assert rendered == render_prompt + prefix


def test_site_prompt_render_is_not_trivially_equal_without_prefix():
    """음성 대조 — 프리픽스가 빈 문자열이면 두 렌더가 (당연히) 같다는 것만으로
    위 테스트가 항진명제가 아님을 확인한다(프리픽스가 실제로 꼬리에 붙는지 검사)."""
    from src.training.countdown_sites import render_prefix_prompt
    from src.training.countdown_task import build_prompt

    tok = _load_tokenizer()
    inst = {"nums": [2, 4, 6, 8], "target": 16}
    prefix = "some distinguishing prefix text 4*4=16? "

    site_msgs = render_prefix_prompt(inst, prefix, "new")
    rendered = tok.apply_chat_template(
        site_msgs, tokenize=False, add_generation_prompt=False,
        continue_final_message=True, enable_thinking=False)
    base_msgs = build_prompt(inst, "new")
    render_prompt = tok.apply_chat_template(
        base_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)

    assert rendered != render_prompt          # 프리픽스가 실제로 꼬리에 더해졌다
    assert rendered.endswith(prefix)
