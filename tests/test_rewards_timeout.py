import threading

from src.training import rewards


def _run_in_worker_thread(fn):
    """Run fn() on a non-main thread (the real Ray RewardLoopWorker condition)
    and return its result. signal.signal/alarm only work on the main thread, so
    this exercises the path that a positive math_verify timeout breaks."""
    out = {}
    err = {}

    def target():
        try:
            out["r"] = fn()
        except BaseException as e:  # noqa: BLE001 - surface any propagated error
            err["e"] = e

    t = threading.Thread(target=target)
    t.start()
    t.join()
    if "e" in err:
        raise err["e"]
    return out["r"]


def test_check_correctness_off_main_thread_grades_symbolic_true():
    """Off the main thread (Ray RewardLoopWorker), grading must not raise and
    must grade correct symbolic/numeric answers True. With a positive
    math_verify timeout this FAILS because SIGALRM re-raises in worker threads
    and grading silently degrades to string-match."""
    # numeric
    assert _run_in_worker_thread(
        lambda: rewards._check_correctness("\\boxed{42}", "42")
    ) is True
    # symbolic equivalence that string-match cannot recover
    assert _run_in_worker_thread(
        lambda: rewards._check_correctness("\\boxed{1/2}", "0.5")
    ) is True
    assert _run_in_worker_thread(
        lambda: rewards._check_correctness("\\boxed{\\frac{1}{2}}", "0.5")
    ) is True
    assert _run_in_worker_thread(
        lambda: rewards._check_correctness("\\boxed{2/4}", "1/2")
    ) is True
    # wrong stays False
    assert _run_in_worker_thread(
        lambda: rewards._check_correctness("\\boxed{7}", "42")
    ) is False


def test_check_correctness_main_thread_grades_symbolic_true():
    """Main-thread behaviour is preserved (sanity)."""
    assert rewards._check_correctness("\\boxed{42}", "42") is True
    assert rewards._check_correctness("\\boxed{1/2}", "0.5") is True
    assert rewards._check_correctness("\\boxed{7}", "42") is False


def test_alarm_shim_tolerates_none_seconds():
    """Regression: newer math_verify calls signal.alarm(None) to DISABLE its
    timeout. The real signal.alarm requires an int, so on the MAIN thread the
    shim raised TypeError('NoneType ... integer') that aborted EVERY comparison
    (even "2"=="2") -> pg0 pilot graded all 200 wrong -> spurious STOP. The shim
    must coerce None -> 0 (cancel) without raising."""
    import signal
    # importing rewards installed the patch; alarm(None) must not raise
    assert signal.alarm(None) == 0
    signal.alarm(0)  # cleanup any pending alarm
    # and grading a trivially-correct pair must return True on the main thread
    assert rewards._check_correctness("2", "2") is True


def test_verify_called_with_disabled_timeout(monkeypatch):
    """math_verify must be invoked with the SIGALRM timeout disabled (None) so
    it is thread-safe in Ray worker threads."""
    calls = {}
    import src.training.rewards as R
    if not R.HAS_MATH_VERIFY:
        return

    def fake_verify(g, p, timeout_seconds="MISSING"):
        calls["verify_t"] = timeout_seconds
        return True

    def fake_parse(s, extraction_mode=None, parsing_timeout="MISSING"):
        calls["parse_t"] = parsing_timeout
        return s

    monkeypatch.setattr(R, "verify", fake_verify)
    monkeypatch.setattr(R, "parse", fake_parse)
    R._check_correctness("\\boxed{42}", "42")
    assert calls["verify_t"] is None
    assert calls["parse_t"] is None
