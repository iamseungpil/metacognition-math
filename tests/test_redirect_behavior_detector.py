from src.eval.redirect_behavior_detector import detect_redirect, measure_recall

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


def test_detects_prose_redirect():
    for t in REDIRECTS:
        assert detect_redirect(t) is True


def test_does_not_flag_plain_continuation():
    for t in NON:
        assert detect_redirect(t) is False


def test_switch_token_surface_form_detected():
    assert detect_redirect("<|switch|> change method") is True


def test_measure_recall_reports_fraction():
    # stub judge that never fires → recall driven by regex alone
    r = measure_recall(REDIRECTS, llm_judge=lambda s: False)
    assert 0.0 <= r <= 1.0 and r == 1.0  # all 3 caught by regex
