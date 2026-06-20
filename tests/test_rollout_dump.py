"""Unit tests for the pure answer-extractor in scripts/rollout_dump.py
(the GPU/vLLM rollout itself is exercised on the node, not here)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.rollout_dump import _extract_answer


def test_extract_answer_prefers_boxed():
    assert _extract_answer("work... \\boxed{42} done") == "42"
    assert _extract_answer("\\boxed{6} then \\boxed{13}") == "13"  # last boxed


def test_extract_answer_falls_back_to_answer_is():
    assert _extract_answer("The answer is $7$.") == "7"
    assert _extract_answer("So the answer is 160") == "160"


def test_extract_answer_empty_when_none():
    assert _extract_answer("just reasoning, no final answer") == ""
    assert _extract_answer("") == ""
