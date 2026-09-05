"""E-131: 과장 프롬프트 행이 배치 폭을 늘리면 모든 행의 응답 꼬리가 채점에서 사라진다.
가드는 (1) 배치 프롬프트 폭 ≠ max_prompt_length, (2) 길이 0 응답(abort) 에서 즉시 죽어야 한다."""
import pytest

from src.training.verl_sdc import _countdown_batch_geometry_guard as guard


def test_clean_batch_passes():
    guard(prompts_width=2048, expected_width=2048, response_valid_lengths=[1, 500, 2048], step=7)


def test_widened_batch_raises():
    with pytest.raises(RuntimeError, match=r"E-131.*2205.*2048"):
        guard(prompts_width=2205, expected_width=2048, response_valid_lengths=[10, 10], step=54)


def test_aborted_rows_raise_even_when_width_ok():
    with pytest.raises(RuntimeError, match=r"E-131.*abort.*2/3"):
        guard(prompts_width=2048, expected_width=2048, response_valid_lengths=[0, 0, 12], step=54)


def test_unknown_expected_width_skips_width_check_but_not_abort_check():
    guard(prompts_width=2205, expected_width=0, response_valid_lengths=[3], step=1)
    with pytest.raises(RuntimeError):
        guard(prompts_width=2205, expected_width=0, response_valid_lengths=[0], step=1)
