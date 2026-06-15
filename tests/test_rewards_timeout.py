from src.training import rewards


def test_check_correctness_no_signal_alarm_typeerror(monkeypatch):
    """Even off the main thread / with a math_verify that calls signal.alarm,
    grading must not raise and must grade a correct numeric answer True."""
    # correct numeric answer should grade True without TypeError flood
    assert rewards._check_correctness("\\boxed{42}", "42") is True
    assert rewards._check_correctness("\\boxed{1/2}", "0.5") is True
    # wrong stays False
    assert rewards._check_correctness("\\boxed{7}", "42") is False


def test_verify_called_with_positive_timeout(monkeypatch):
    calls = {}
    import src.training.rewards as R
    if not R.HAS_MATH_VERIFY:
        return
    def fake_verify(g, p, timeout_seconds=None):
        calls["t"] = timeout_seconds
        return True
    monkeypatch.setattr(R, "verify", fake_verify)
    R._check_correctness("\\boxed{42}", "42")
    assert calls["t"] is not None and calls["t"] > 0   # FAILS now (None)
