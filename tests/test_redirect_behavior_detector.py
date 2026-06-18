import pytest

from src.eval.redirect_behavior_detector import (
    detect_redirect,
    emitted_switch,
    regex_prefilter,
    measure_recall,
    measure_precision,
)

REDIRECTS = [
    "wait, that's wrong, let me try a different approach",
    "this isn't working — instead I'll use an invariant",
    "let me reconsider and start over with substitution",
]
NON = [
    "so the total is 16 - 7 = 9, times 2 is 18",
    "next, factor the quadratic to get (x-2)(x-3)",
    "therefore the answer is 42",
]


def test_regex_prefilter_detects_prose_redirect():
    for t in REDIRECTS:
        assert detect_redirect(t, regex_only=True) is True


def test_regex_prefilter_clean_continuation():
    for t in NON:
        assert detect_redirect(t, regex_only=True) is False


def test_switch_token_is_NOT_a_behavior_signal():
    # token alone, no prose redirect cue -> behavior detector must NOT fire
    assert detect_redirect("<|switch|> change method", regex_only=True) is False
    # but the separate token channel sees it
    assert emitted_switch("<|switch|> change method") is True
    assert emitted_switch("just continue normally") is False


def test_live_path_requires_judge():
    with pytest.raises(ValueError):
        detect_redirect("wait, that's wrong", llm_judge=None)  # regex_only=False


def test_judge_is_primary_on_live_path():
    # judge decides; a prose-cue string the judge vetoes -> False
    assert detect_redirect("wait, that's wrong", llm_judge=lambda s: False) is False
    assert detect_redirect("subtle keyword-free restart", llm_judge=lambda s: True) is True


def test_recall_and_precision():
    assert measure_recall(REDIRECTS, regex_only=True) == 1.0
    assert measure_precision(NON, regex_only=True) == 1.0
