"""TDD tests for the sample-generate entry + the real-teacher wiring helpers.

GPU-free + network-free: the TEACHER is a MOCK callable (deterministic strings).
The REAL TRAPI(Entra) teacher fn (``make_trapi_teacher_fn``) is exercised only for
its PROMPT-BRANCHING + lazy client construction via an injected fake AzureOpenAI;
no network/credential call happens in the suite.

sample_generate(anchors, teacher_fn) drives the SAME normalize -> validate ->
causal/structural filter the build driver uses (no duplicated filter code) over a
hand-anchored, GPU-free list of {problem, gold, wrong_prefix, conf, difficulty,
action} dicts and returns (kept_rows, quality_report).
"""
from __future__ import annotations

import pytest

from scripts.build_confidence_redirect_verify_sft import (
    sample_generate,
    make_trapi_teacher_fn,
)
from src.metacot.prompt_redirect_verify import META_START, META_END


# --------------------------------------------------------------------------- #
# hand-anchored toy anchors (GPU-free, no rollout)
# --------------------------------------------------------------------------- #
def _anchors():
    return [
        {  # A1 redirect: teacher flips wrong->right with a real decision: redirect
            "problem": "A1 redirect easy",
            "gold": "7",
            "wrong_prefix": "I started with 2+2=5 so",
            "conf": 0.0,
            "difficulty": "easy",
            "action": "redirect",
        },
        {  # A2 verify: a perfect demo but with the BROKEN close tag </|meta|>
            "problem": "A2 verify medium",
            "gold": "42",
            "wrong_prefix": "<think>slip is 41</think> The answer is $41$.",
            "conf": 0.75,
            "difficulty": "medium",
            "action": "verify",
        },
        {  # A3 redirect: DECORATIVE (decision: verify, no real redirect) -> dropped
            "problem": "A3 decorative easy",
            "gold": "5",
            "wrong_prefix": "I think the route is X so",
            "conf": 0.0,
            "difficulty": "easy",
            "action": "redirect",
        },
    ]


def _mock_teacher(payload: dict):
    """Deterministic per (problem, arm). Includes a '</|meta|>' close-tag variant
    (A2), a decorative redirect (A3), and a real flip (A1). The driver builds the
    payload via _teacher_payload, which stores the anchor 'problem' under 'question'."""
    p = payload["question"]
    arm = payload["arm"]
    if p.startswith("A1"):
        if arm == "redirect":
            return (
                "<|meta|>\nconfidence: 0.0\ndecision: redirect\n"
                "Weak route; switching to a different method.\n<|/meta|>\n"
                "Now correct. The answer is $7$."
            )
        if arm == "control":
            return "Continuing the same way. The answer is $1$."  # stays wrong
    if p.startswith("A2") and arm == "verify":
        # BROKEN close tag '</|meta|>' -> must be repaired + kept.
        return (
            "<|meta|>\nconfidence: 0.75\ndecision: verify\n"
            "Looks right but I must check; substitute back.\n</|meta|>\n"
            "Substituting recomputes 6 * 7 = 42. The answer is $42$."
        )
    if p.startswith("A3"):
        if arm == "redirect":
            # decorative: a meta block but decision is 'verify', no real redirect.
            return (
                "<|meta|>\nconfidence: 0.0\ndecision: verify\nLooks weak.\n<|/meta|>\n"
                "The answer is $5$."
            )
        if arm == "control":
            return "Same way. The answer is $2$."
    raise AssertionError(f"unexpected teacher call: p={p!r} arm={arm!r}")


def test_sample_generate_repairs_keeps_and_drops():
    kept, report = sample_generate(_anchors(), _mock_teacher)

    # A1 redirect kept (real flip) + A2 verify kept (close tag REPAIRED).
    scenarios = sorted(r["scenario"] for r in kept)
    assert scenarios == ["redirect", "verify"], scenarios
    assert len(kept) == 2

    # the repaired verify carries the CANONICAL close tag, not the broken one.
    verify_row = next(r for r in kept if r["scenario"] == "verify")
    assert META_END in verify_row["messages"][1]["content"]
    assert "</|meta|>" not in verify_row["messages"][1]["content"]

    # A3 decorative redirect dropped.
    assert report["n_redirect"] == 1
    assert report["n_verify"] == 1
    assert report["n_dropped_by_reason"]["decorative"] == 1

    # quality_report numbers.
    # functional_rate = kept / (anchors that produced a teacher demo) = 2/3.
    assert report["functional_rate"] == pytest.approx(2 / 3)
    # exactly the A2 verify was repaired (1 of the 3 demos touched normalize).
    assert report["format_repaired_rate"] == pytest.approx(1 / 3)
    # all three stated confidences are within tol of the anchor conf -> 1.0.
    assert report["calibration_ok_rate"] == pytest.approx(1.0)


def test_sample_generate_calibration_flags_inflated_conf():
    """A redirect whose STATED confidence is inflated above the anchor conf is
    NOT calibration_ok and is dropped by the stated-conf gate."""
    anchors = [{
        "problem": "B1 redirect easy",
        "gold": "7",
        "wrong_prefix": "started wrong so",
        "conf": 0.10,
        "difficulty": "easy",
        "action": "redirect",
    }]

    def inflated_teacher(payload):
        if payload["arm"] == "control":
            return "Same way. The answer is $1$."
        return (  # flips + real decision, but states INFLATED 0.90
            "<|meta|>\nconfidence: 0.90\ndecision: redirect\nSwitching.\n<|/meta|>\n"
            "The answer is $7$."
        )

    kept, report = sample_generate(anchors, inflated_teacher)
    assert kept == []
    assert report["calibration_ok_rate"] == pytest.approx(0.0)
    assert report["n_dropped_by_reason"]["conf_mismatch"] == 1


def test_sample_generate_draws_k_distinct_control_seeds():
    """Regression: sample_generate and build_dataset share ONE control-draw loop
    (_draw_control_grades), so the redirect arm here must draw CONTROL_K control
    samples carrying distinct seed indices 0..K-1 (not a single noisy control)."""
    from scripts.build_confidence_redirect_verify_sft import CONTROL_K

    anchors = [{
        "problem": "S1 redirect easy",
        "gold": "7",
        "wrong_prefix": "started wrong so",
        "conf": 0.0,
        "difficulty": "easy",
        "action": "redirect",
    }]
    control_samples = []

    def teacher(payload):
        if payload["arm"] == "control":
            control_samples.append(payload.get("sample"))
            return "Same way. The answer is $1$."  # stays wrong
        return (
            "<|meta|>\nconfidence: 0.0\ndecision: redirect\nSwitching.\n<|/meta|>\n"
            "The answer is $7$."
        )

    sample_generate(anchors, teacher)
    assert len(control_samples) == CONTROL_K
    assert sorted(s for s in control_samples if s is not None) == list(range(CONTROL_K))


def test_sample_generate_empty_anchors_safe():
    kept, report = sample_generate([], _mock_teacher)
    assert kept == []
    assert report["functional_rate"] == 0.0
    assert report["format_repaired_rate"] == 0.0
    assert report["calibration_ok_rate"] == 0.0


# --------------------------------------------------------------------------- #
# REAL teacher fn (Entra) — branch the prompt on arm + lazy/cached client.
# We inject a FAKE AzureOpenAI so NO network/credential call happens.
# --------------------------------------------------------------------------- #
class _FakeResp:
    def __init__(self, text):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})()})()]


class _FakeChatCompletions:
    def __init__(self, parent):
        self.parent = parent

    def create(self, *, model, messages, **kw):
        self.parent.calls.append({"model": model, "messages": messages})
        # echo back which arm's SYSTEM prompt was used so the test can assert
        # the branch; the system prompt is messages[0].
        system = messages[0]["content"]
        return _FakeResp(f"<|meta|>\nconfidence: 0.00\ndecision: redirect\n"
                         f"SYS={system[:24]}\n<|/meta|>\nThe answer is $7$.")


class _FakeClient:
    def __init__(self):
        self.calls = []
        self.chat = type("Chat", (), {"completions": _FakeChatCompletions(self)})()


def test_make_trapi_teacher_fn_branches_on_arm_and_caches_client():
    built = {"n": 0}
    fake = _FakeClient()

    def fake_client_factory():
        built["n"] += 1
        return fake

    teacher = make_trapi_teacher_fn(
        model_list=["m1", "m2"], client_factory=fake_client_factory
    )

    # redirect arm
    out_r = teacher({
        "question": "P", "gold": "7", "confidence": 0.0, "arm": "redirect",
        "wrong_prefix": "started wrong so",
    })
    assert META_START in out_r and META_END in out_r
    # the redirect SYSTEM prompt (TEACHER_DISTILL) was used, not the control one.
    assert "math teacher" in fake.calls[-1]["messages"][0]["content"].lower()

    # control arm -> the CONTROL continuation system prompt
    teacher({
        "question": "P", "gold": "7", "confidence": 0.0, "arm": "control",
        "wrong_prefix": "started wrong so", "sample": 0,
    })
    assert "control continuation" in fake.calls[-1]["messages"][0]["content"].lower()

    # verify arm -> verify prompt on the attempt (wrong_prefix carries the attempt)
    teacher({
        "question": "P", "gold": "42", "confidence": 0.8, "arm": "verify",
        "wrong_prefix": "the student attempt text",
    })
    assert "verify" in fake.calls[-1]["messages"][1]["content"].lower()

    # client built ONCE (lazy + cached), reused across all three calls.
    assert built["n"] == 1


def test_make_trapi_teacher_fn_falls_back_on_404():
    fake = _FakeClient()

    class _Flaky(_FakeChatCompletions):
        def create(self, *, model, messages, **kw):
            if model == "m1":
                raise RuntimeError("Error code: 404 - model not deployed")
            return super().create(model=model, messages=messages, **kw)

    fake.chat.completions = _Flaky(fake)

    teacher = make_trapi_teacher_fn(
        model_list=["m1", "m2"], client_factory=lambda: fake, max_retries=2,
    )
    out = teacher({
        "question": "P", "gold": "7", "confidence": 0.0, "arm": "redirect",
        "wrong_prefix": "started wrong so",
    })
    assert META_START in out
    # the first model 404'd, the second model served the completion.
    assert fake.calls[-1]["model"] == "m2"


def test_make_trapi_teacher_fn_retries_429_then_succeeds(monkeypatch):
    import scripts.build_confidence_redirect_verify_sft as mod
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)
    fake = _FakeClient()
    state = {"n": 0}

    class _RateLimited(_FakeChatCompletions):
        def create(self, *, model, messages, **kw):
            state["n"] += 1
            if state["n"] == 1:
                raise RuntimeError("Error code: 429 - rate limit")
            return super().create(model=model, messages=messages, **kw)

    fake.chat.completions = _RateLimited(fake)
    teacher = make_trapi_teacher_fn(
        model_list=["m1"], client_factory=lambda: fake, max_retries=3,
    )
    out = teacher({
        "question": "P", "gold": "7", "confidence": 0.0, "arm": "redirect",
        "wrong_prefix": "started wrong so",
    })
    assert META_START in out
    assert state["n"] == 2  # one 429 retry, then success
