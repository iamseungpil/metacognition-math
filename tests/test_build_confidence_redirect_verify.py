"""TDD tests for scripts/build_confidence_redirect_verify_sft.py.

GPU-free + network-free: the STUDENT rollout and the TEACHER generation are both
injected as mock callables. The smoke test drives the WHOLE pipeline end to end
on 3 toy problems and asserts:
  * a functional REDIRECT row is produced (student wrong -> teacher redirect flips
    wrong->right AND the no-redirect control stays wrong);
  * a functional VERIFY row is produced (high student confidence, teacher confirms);
  * a HARD problem is dropped (anchor on easy/medium only — teacher capability-gap);
  * a DECORATIVE redirect is dropped (teacher trace stays wrong / control already
    right => no causal flip);
  * the output parquet mirrors build_v8 schema (messages JSON + split_tags) and the
    redirect row carries a loss-mask that masks the wrong prefix.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.build_confidence_redirect_verify_sft import (
    action_bucket,
    build_dataset,
    confidence_from_grades,
    stated_conf_matches,
    _strip_final_answer,
    _pick_wrong_prefix,
    BUCKET_REDIRECT,
    BUCKET_VERIFY,
    BUCKET_NONE,
)


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_confidence_from_grades_is_student_pass_rate():
    assert confidence_from_grades([1, 1, 1, 1]) == pytest.approx(1.0)
    assert confidence_from_grades([0, 0, 0, 0]) == pytest.approx(0.0)
    assert confidence_from_grades([1, 0, 1, 0]) == pytest.approx(0.5)
    assert confidence_from_grades([]) == pytest.approx(0.0)


def test_action_bucket_low_conf_is_redirect():
    # low/confidently-wrong pass-rate -> REDIRECT
    assert action_bucket(0.0) == BUCKET_REDIRECT
    assert action_bucket(0.25) == BUCKET_REDIRECT


def test_action_bucket_high_conf_is_verify():
    # high-but-checkable -> VERIFY
    assert action_bucket(0.75) == BUCKET_VERIFY
    assert action_bucket(1.0) == BUCKET_VERIFY


def test_action_bucket_mid_is_none():
    # ambiguous middle -> nothing useful
    assert action_bucket(0.5) == BUCKET_NONE


# --------------------------------------------------------------------------- #
# end-to-end smoke (mock rollout + mock teacher)
# --------------------------------------------------------------------------- #
def _toy_problems():
    return [
        {  # P1 easy, student almost always WRONG -> redirect bucket, FUNCTIONAL
            "question": "P1 redirect easy",
            "gold": "7",
            "tags": {"difficulty": "easy", "scenario": "redirect", "trigger": "t"},
        },
        {  # P2 medium, student almost always RIGHT -> verify bucket, FUNCTIONAL
            "question": "P2 verify medium",
            "gold": "42",
            "tags": {"difficulty": "medium", "scenario": "verify", "trigger": "t"},
        },
        {  # P3 hard, student wrong -> MUST be dropped (capability gap, anchor easy/medium)
            "question": "P3 redirect hard",
            "gold": "99",
            "tags": {"difficulty": "hard", "scenario": "redirect", "trigger": "t"},
        },
        {  # P4 easy, student wrong -> redirect bucket but teacher trace stays WRONG
            "question": "P4 decorative easy",
            "gold": "5",
            "tags": {"difficulty": "easy", "scenario": "redirect", "trigger": "t"},
        },
    ]


def _mock_rollout(question: str, gold: str, n: int):
    """Return n student rollouts (text, is_correct, answer). Deterministic per
    problem. The answer string feeds confidently_wrong / majority gating."""
    if question.startswith("P1"):  # low conf: 1/4 correct -> redirect
        return [
            ("<think>wrong P1 attempt one</think> The answer is $3$.", False, "3"),
            ("<think>wrong P1 attempt two</think> The answer is $4$.", False, "4"),
            ("<think>wrong P1 attempt three</think> The answer is $2$.", False, "2"),
            ("<think>lucky P1</think> The answer is $7$.", True, "7"),
        ][:n]
    if question.startswith("P2"):  # high-conf-but-CHECKABLE: 3/4 correct -> verify
        # one wrong sample so there is something to verify against (a 4/4-perfect
        # problem is trivially solved -> NONE, not a decorative verify demo).
        return [
            ("<think>P2 solve</think> The answer is $42$.", True, "42"),
            ("<think>P2 solve</think> The answer is $42$.", True, "42"),
            ("<think>P2 solve</think> The answer is $42$.", True, "42"),
            ("<think>P2 slip</think> The answer is $41$.", False, "41"),
        ][:n]
    if question.startswith("P3"):  # hard, wrong -> would be redirect but dropped on difficulty
        return [("<think>P3 wrong</think> The answer is $1$.", False, "1") for _ in range(n)]
    if question.startswith("P4"):  # low conf redirect, but teacher will fail
        return [
            ("<think>P4 wrong a</think> The answer is $1$.", False, "1"),
            ("<think>P4 wrong b</think> The answer is $2$.", False, "2"),
            ("<think>P4 wrong c</think> The answer is $3$.", False, "3"),
            ("<think>P4 wrong d</think> The answer is $4$.", False, "4"),
        ][:n]
    return [("<think>x</think> The answer is $0$.", False, "0") for _ in range(n)]


def _mock_teacher(payload: dict):
    """Conditional teacher. payload carries: question, gold, confidence, bucket,
    wrong_prefix (redirect only), and arm in {"redirect","control","verify"}.

    The driver must call the teacher for the REDIRECT arm AND a no-redirect
    CONTROL arm (to prove causality), and once for VERIFY.
    Returns a continuation string with <|meta|>...<|/meta|> and a boxed answer.
    """
    q = payload["question"]
    arm = payload["arm"]
    if q.startswith("P1"):
        if arm == "redirect":
            return (
                "<|meta|>\nconfidence: 0.25\n"
                "Something is off; I will switch to a different method.\n"
                "<|switch|>\n<|/meta|>\n"
                "Using the right method now. The answer is $7$."
            )
        if arm == "control":  # no-redirect control stays WRONG
            return "Continuing the same way. The answer is $3$."
    if q.startswith("P2") and arm == "verify":
        return (
            "<|meta|>\nconfidence: 0.85\n"
            "This looks right; I will substitute to verify.\n"
            "<|/meta|>\n"
            "Substituting confirms it. The answer is $42$."
        )
    if q.startswith("P4"):
        if arm == "redirect":  # teacher ALSO fails -> decorative, must be dropped
            return (
                "<|meta|>\nconfidence: 0.2\n"
                "Switching method.\n<|switch|>\n<|/meta|>\n"
                "The answer is $8$."
            )
        if arm == "control":
            return "Same way. The answer is $1$."
    # P3 should never be queried (dropped on difficulty before the teacher call)
    raise AssertionError(f"unexpected teacher call: q={q!r} arm={arm!r}")


def test_smoke_end_to_end(tmp_path: Path):
    out = tmp_path / "confidence_rv_sft.parquet"
    summary = build_dataset(
        problems=_toy_problems(),
        rollout_fn=_mock_rollout,
        teacher_fn=_mock_teacher,
        out_path=str(out),
        n_rollouts=4,
    )

    assert out.exists(), "parquet must be written"
    df = pd.read_parquet(out)

    # exactly two functional rows: P1 redirect + P2 verify. P3 (hard) and P4
    # (decorative) dropped.
    scenarios = sorted(df["scenario"].tolist())
    assert scenarios == ["redirect", "verify"], scenarios
    assert len(df) == 2

    # schema mirrors build_v8: messages JSON (user+assistant), split_tags present
    for _, row in df.iterrows():
        msgs = json.loads(row["messages"])
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        assert "<|meta|>" in msgs[1]["content"]
        assert "<|/meta|>" in msgs[1]["content"]
        st = row["split_tags"]
        st = json.loads(st) if isinstance(st, str) else st
        assert st["difficulty"] in ("easy", "medium")

    # redirect row: confidence label ~ student pass rate (0.25). The row carries
    # what the SFT collator needs to recompute the TOKEN mask (the wrong-prefix
    # TEXT + a char split marker) — NOT a char mask masquerading as a token mask.
    redirect_row = df[df["scenario"] == "redirect"].iloc[0]
    assert redirect_row["confidence_label"] == pytest.approx(0.25)
    assert "loss_mask" not in df.columns, "must not persist a char mask as a token mask"
    wrong_prefix = redirect_row["wrong_prefix"]
    assert isinstance(wrong_prefix, str) and wrong_prefix, "wrong prefix text must be carried"
    # the wrong prefix is the head of the assistant target (so the collator can
    # find + mask it), and it must NOT contain the wrong boxed/final answer.
    assistant = json.loads(redirect_row["messages"])[1]["content"]
    assert assistant.startswith(wrong_prefix)
    assert "The answer is" not in wrong_prefix
    assert int(redirect_row["prefix_split_char"]) == len(wrong_prefix)

    # verify row: high confidence label
    verify_row = df[df["scenario"] == "verify"].iloc[0]
    assert verify_row["confidence_label"] >= 0.65

    # summary accounting
    assert summary["kept_redirect"] == 1
    assert summary["kept_verify"] == 1
    assert summary["dropped_hard"] >= 1
    assert summary["dropped_decorative"] >= 1


def test_hard_problems_never_query_teacher(tmp_path: Path):
    """A hard problem must be dropped BEFORE any teacher call (capability-gap OOD)."""
    out = tmp_path / "hard_only.parquet"
    calls = []

    def spy_teacher(payload):
        calls.append(payload)
        return _mock_teacher(payload)

    hard_only = [p for p in _toy_problems() if p["tags"]["difficulty"] == "hard"]
    summary = build_dataset(
        problems=hard_only,
        rollout_fn=_mock_rollout,
        teacher_fn=spy_teacher,
        out_path=str(out),
        n_rollouts=4,
    )
    assert calls == [], "teacher must never be called for hard problems"
    assert summary["kept_redirect"] == 0 and summary["kept_verify"] == 0
    assert summary["dropped_hard"] == 1


def test_perfect_problem_is_not_a_decorative_verify(tmp_path: Path):
    """A 4/4-correct (high-conf, NOTHING-to-check) problem must NOT become a
    verify demo: verify is high-confidence-but-CHECKABLE, a perfect problem is
    trivially solved -> NONE (no decorative verify meta)."""
    out = tmp_path / "perfect.parquet"
    calls = []

    def spy_teacher(payload):
        calls.append(payload)
        return _mock_teacher(payload)

    perfect = [{
        "question": "PP perfect easy",
        "gold": "42",
        "tags": {"difficulty": "easy", "scenario": "verify", "trigger": "t"},
    }]

    def perfect_rollout(question, gold, n):
        return [("<think>solve</think> The answer is $42$.", True) for _ in range(n)]

    summary = build_dataset(
        problems=perfect, rollout_fn=perfect_rollout, teacher_fn=spy_teacher,
        out_path=str(out), n_rollouts=4,
    )
    assert calls == [], "teacher must not be called for a trivially-solved problem"
    assert summary["kept_verify"] == 0
    assert summary["dropped_bucket_none"] == 1


def test_verify_without_real_check_is_dropped(tmp_path: Path):
    """A verify trace that ends correct but has NO meta block (no actual check)
    is decorative and must be dropped (correctness alone is gameable: the teacher
    is told to always end correct)."""
    out = tmp_path / "hollow_verify.parquet"
    prob = [{
        "question": "HV checkable medium",
        "gold": "42",
        "tags": {"difficulty": "medium", "scenario": "verify", "trigger": "t"},
    }]

    def rollout(question, gold, n):  # 3/4 -> verify bucket, has a wrong sample
        return [
            ("<think>a</think> The answer is $42$.", True),
            ("<think>b</think> The answer is $42$.", True),
            ("<think>c</think> The answer is $42$.", True),
            ("<think>d</think> The answer is $41$.", False),
        ][:n]

    def hollow_teacher(payload):  # correct final answer, NO meta block / no check
        return "The answer is $42$."

    summary = build_dataset(
        problems=prob, rollout_fn=rollout, teacher_fn=hollow_teacher,
        out_path=str(out), n_rollouts=4,
    )
    assert summary["kept_verify"] == 0
    assert summary["dropped_decorative"] == 1


def test_redirect_without_switch_is_dropped(tmp_path: Path):
    """A redirect trace that flips wrong->right but never switches method (no
    <|switch|>) is decorative recovery, not a redirect, and must be dropped."""
    out = tmp_path / "hollow_redirect.parquet"
    prob = [{
        "question": "HR low easy",
        "gold": "7",
        "tags": {"difficulty": "easy", "scenario": "redirect", "trigger": "t"},
    }]

    def rollout(question, gold, n):  # all wrong -> redirect bucket
        return [("<think>wrong</think> The answer is $1$.", False) for _ in range(n)]

    def no_switch_teacher(payload):
        if payload["arm"] == "control":
            return "Same way. The answer is $1$."
        # redirect arm: correct + a confidence line but NO <|switch|> method change
        return (
            "<|meta|>\nconfidence: 0.1\nLooks weak.\n<|/meta|>\n"
            "The answer is $7$."
        )

    summary = build_dataset(
        problems=prob, rollout_fn=rollout, teacher_fn=no_switch_teacher,
        out_path=str(out), n_rollouts=4,
    )
    assert summary["kept_redirect"] == 0
    assert summary["dropped_decorative"] == 1


# --------------------------------------------------------------------------- #
# Fix #1 — the causal filter must be NON-VACUOUS: an 'always-correct' control
# (the old bug: control reused the 'always end correct' teacher) keeps ZERO
# redirects, because the redirect can no longer beat a control that also recovers.
# --------------------------------------------------------------------------- #
def test_always_correct_control_yields_zero_kept_redirects(tmp_path: Path):
    out = tmp_path / "vacuous_control.parquet"
    prob = [{
        "question": "AC low easy",
        "gold": "7",
        "tags": {"difficulty": "easy", "scenario": "redirect", "trigger": "t"},
    }]

    def rollout(question, gold, n):  # all wrong -> redirect bucket, conf 0.0
        return [("<think>wrong</think> The answer is $1$.", False, "1") for _ in range(n)]

    def always_correct_control_teacher(payload):
        if payload["arm"] == "control":
            # BUG REPRODUCTION: the control also "always ends correct".
            return "Continuing. The answer is $7$."
        # genuine redirect that flips wrong->right and states the right confidence
        return (
            "<|meta|>\nconfidence: 0.0\nWeak route; switching.\n<|switch|>\n<|/meta|>\n"
            "Now correct. The answer is $7$."
        )

    summary = build_dataset(
        problems=prob, rollout_fn=rollout,
        teacher_fn=always_correct_control_teacher,
        out_path=str(out), n_rollouts=4,
    )
    assert summary["kept_redirect"] == 0, "non-vacuous filter must reject when control recovers"
    assert summary["dropped_control_recovers"] == 1


def test_genuinely_wrong_control_keeps_the_redirect(tmp_path: Path):
    """Twin of the above: the SAME redirect is KEPT when the control genuinely
    stays wrong (proving it was the control, not the redirect, that changed)."""
    out = tmp_path / "good_control.parquet"
    prob = [{
        "question": "GC low easy",
        "gold": "7",
        "tags": {"difficulty": "easy", "scenario": "redirect", "trigger": "t"},
    }]

    def rollout(question, gold, n):
        return [("<think>wrong</think> The answer is $1$.", False, "1") for _ in range(n)]

    def good_control_teacher(payload):
        if payload["arm"] == "control":
            return "Continuing the same way. The answer is $1$."  # stays wrong
        return (
            "<|meta|>\nconfidence: 0.0\nWeak route; switching.\n<|switch|>\n<|/meta|>\n"
            "Now correct. The answer is $7$."
        )

    summary = build_dataset(
        problems=prob, rollout_fn=rollout, teacher_fn=good_control_teacher,
        out_path=str(out), n_rollouts=4,
    )
    assert summary["kept_redirect"] == 1
    assert summary["dropped_control_recovers"] == 0


def test_control_arm_drawn_k_times(tmp_path: Path):
    """The control arm is sampled k>=4 times (a single control sample is noisy)."""
    out = tmp_path / "k_control.parquet"
    prob = [{
        "question": "KC low easy",
        "gold": "7",
        "tags": {"difficulty": "easy", "scenario": "redirect", "trigger": "t"},
    }]
    control_samples = []

    def rollout(question, gold, n):
        return [("<think>wrong</think> The answer is $1$.", False, "1") for _ in range(n)]

    def teacher(payload):
        if payload["arm"] == "control":
            control_samples.append(payload.get("sample"))
            return "Same way. The answer is $1$."
        return (
            "<|meta|>\nconfidence: 0.0\nSwitching.\n<|switch|>\n<|/meta|>\n"
            "The answer is $7$."
        )

    build_dataset(problems=prob, rollout_fn=rollout, teacher_fn=teacher,
                  out_path=str(out), n_rollouts=4)
    assert len(control_samples) >= 4, control_samples
    # the control draws carry a distinct sample index (so a real teacher can vary seed)
    assert sorted(s for s in control_samples if s is not None) == [0, 1, 2, 3]


# --------------------------------------------------------------------------- #
# Fix #4 — the teacher's STATED confidence must match the student's MEASURED
# value. A teacher stating 0.90 on a pass_rate=0.10 redirect is dropped.
# --------------------------------------------------------------------------- #
def test_inflated_stated_confidence_is_dropped(tmp_path: Path):
    out = tmp_path / "conf_mismatch.parquet"
    prob = [{
        "question": "CM low easy",
        "gold": "7",
        "tags": {"difficulty": "easy", "scenario": "redirect", "trigger": "t"},
    }]

    def rollout(question, gold, n):  # 1/10 correct -> pass_rate 0.10, redirect
        rs = [("<think>w</think> The answer is $1$.", False, "1") for _ in range(9)]
        rs.append(("<think>lucky</think> The answer is $7$.", True, "7"))
        return rs[:n]

    def inflated_teacher(payload):
        if payload["arm"] == "control":
            return "Same way. The answer is $1$."
        # flips to right + a real switch, BUT states an INFLATED 0.90 confidence
        return (
            "<|meta|>\nconfidence: 0.90\nSwitching.\n<|switch|>\n<|/meta|>\n"
            "The answer is $7$."
        )

    summary = build_dataset(
        problems=prob, rollout_fn=rollout, teacher_fn=inflated_teacher,
        out_path=str(out), n_rollouts=10,
    )
    assert summary["kept_redirect"] == 0
    assert summary["dropped_conf_mismatch"] == 1


def test_missing_stated_confidence_is_dropped(tmp_path: Path):
    out = tmp_path / "conf_missing.parquet"
    prob = [{
        "question": "MM low easy",
        "gold": "7",
        "tags": {"difficulty": "easy", "scenario": "redirect", "trigger": "t"},
    }]

    def rollout(question, gold, n):
        return [("<think>w</think> The answer is $1$.", False, "1") for _ in range(n)]

    def no_conf_teacher(payload):
        if payload["arm"] == "control":
            return "Same way. The answer is $1$."
        # real switch + flips, but NO `confidence:` line in the meta block
        return (
            "<|meta|>\nSwitching to a better method.\n<|switch|>\n<|/meta|>\n"
            "The answer is $7$."
        )

    summary = build_dataset(
        problems=prob, rollout_fn=rollout, teacher_fn=no_conf_teacher,
        out_path=str(out), n_rollouts=4,
    )
    assert summary["kept_redirect"] == 0
    assert summary["dropped_conf_mismatch"] == 1


def test_stated_conf_matches_helper():
    assert stated_conf_matches("<|meta|>\nconfidence: 0.10\n...<|/meta|>", 0.0) is True
    assert stated_conf_matches("<|meta|>\nconfidence: 0.90\n...<|/meta|>", 0.10) is False
    assert stated_conf_matches("<|meta|>\nno conf here\n<|/meta|>", 0.10) is False
    # a stray confidence OUTSIDE the meta block does not count
    assert stated_conf_matches("no meta. confidence: 0.10", 0.10) is False


# --------------------------------------------------------------------------- #
# Fix #5 — splice point: the wrong prefix must EXCLUDE the (wrong) boxed/final
# answer so the prefix never carries the answer the student already committed.
# --------------------------------------------------------------------------- #
def test_strip_final_answer_removes_boxed_and_final_line():
    txt = "Step one. Step two.\nThe answer is $5$.\n\\boxed{5}"
    out = _strip_final_answer(txt)
    assert "\\boxed" not in out
    assert "The answer is" not in out
    assert "Step one" in out


def test_strip_final_answer_handles_nested_boxed_braces():
    # NESTED braces: a naive \boxed\{[^}]*\} stops at the first '}' and LEAKS the
    # answer fragment. The balanced strip must remove the whole \boxed{...}.
    txt = r"I work the route here. The answer is \boxed{\frac{1}{2}}"
    out = _strip_final_answer(txt)
    assert "\\boxed" not in out
    assert "\\frac" not in out
    assert "2}" not in out and "{2" not in out
    assert "route here" in out
    # \boxed{\sqrt{2}} likewise fully removed
    out2 = _strip_final_answer(r"steps and more steps \boxed{\sqrt{2}}")
    assert "boxed" not in out2 and "sqrt" not in out2 and "2}" not in out2


def test_pick_wrong_prefix_does_not_leak_nested_boxed_answer():
    # a SHORT wrong rollout whose answer is a nested-brace box: the spliced prefix
    # must NOT carry the answer fragment.
    rollouts = [(r"2+2=5 so done. \boxed{\frac{5}{1}}", False, "5")]
    prefix = _pick_wrong_prefix(rollouts)
    assert prefix is not None
    assert "\\boxed" not in prefix
    assert "frac" not in prefix
    assert "5}" not in prefix


def test_short_wrong_rollout_prefix_excludes_boxed_answer():
    # a SHORT wrong rollout whose 0.5 char-splice would otherwise land past the box
    rollouts = [("Let me compute. 2+2=5. \\boxed{5}", False, "5")]
    prefix = _pick_wrong_prefix(rollouts)
    assert prefix is not None
    assert "\\boxed" not in prefix
    assert "5}" not in prefix


def test_pick_wrong_prefix_excludes_the_answer_is_line():
    rollouts = [("I think the route is X. The answer is $13$.", False, "13")]
    prefix = _pick_wrong_prefix(rollouts)
    assert prefix is not None
    assert "The answer is" not in prefix
