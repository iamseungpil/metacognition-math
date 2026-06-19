"""CPU unit tests for the student-calibrated confidence labeler (pure module).

Covers self_consistency (pass_rate), majority_answer (tie / empty / case),
confidently_wrong (high-agreement-on-wrong vs confidently-right vs scattered),
and every action_bucket outcome incl. the PG0 HARD-exclusion from redirect.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.confidence_label import (
    DEFAULT_HI,
    DEFAULT_LO,
    action_bucket,
    confidently_wrong,
    majority_answer,
    self_consistency,
)


# ── self_consistency ────────────────────────────────────────────────────────
def test_self_consistency_all_correct():
    assert self_consistency([1, 1, 1, 1]) == 1.0


def test_self_consistency_all_wrong():
    assert self_consistency([0, 0, 0, 0]) == 0.0


def test_self_consistency_half():
    assert self_consistency([1, 0, 1, 0]) == 0.5


def test_self_consistency_bool_grades():
    assert self_consistency([True, False, True, True]) == 0.75


def test_self_consistency_empty_is_zero():
    assert self_consistency([]) == 0.0


# ── majority_answer ─────────────────────────────────────────────────────────
def test_majority_answer_basic():
    assert majority_answer(["42", "42", "7"]) == "42"


def test_majority_answer_case_and_whitespace_normalized():
    assert majority_answer([" Yes ", "yes", "no"]) == "yes"


def test_majority_answer_ignores_empty_and_none():
    assert majority_answer(["", None, "5", "5"]) == "5"


def test_majority_answer_tie_first_appearance():
    # stable on insertion order -> first-seen wins the tie
    assert majority_answer(["a", "b"]) == "a"


def test_majority_answer_all_empty_is_blank():
    assert majority_answer(["", None, "  "]) == ""


# ── confidently_wrong ───────────────────────────────────────────────────────
def test_confidently_wrong_high_agreement_on_wrong():
    # 3/4 share answer "13", all graded wrong -> confidently wrong
    answers = ["13", "13", "13", "9"]
    grades = [0, 0, 0, 0]
    assert confidently_wrong(answers, grades, thr=0.6) is True


def test_confidently_right_is_not_confidently_wrong():
    # 3/4 share "12" and those are CORRECT -> not confidently wrong
    answers = ["12", "12", "12", "9"]
    grades = [1, 1, 1, 0]
    assert confidently_wrong(answers, grades, thr=0.6) is False


def test_scattered_wrong_is_not_confidently_wrong():
    # all wrong but no majority answer clears the agreement threshold
    answers = ["1", "2", "3", "4"]
    grades = [0, 0, 0, 0]
    assert confidently_wrong(answers, grades, thr=0.6) is False


def test_confidently_wrong_below_threshold():
    # majority "5" is only 2/4 = 0.5 < 0.6 threshold
    answers = ["5", "5", "6", "7"]
    grades = [0, 0, 0, 0]
    assert confidently_wrong(answers, grades, thr=0.6) is False


def test_confidently_wrong_mismatched_lengths_false():
    assert confidently_wrong(["1", "1"], [0], thr=0.6) is False


def test_confidently_wrong_empty_false():
    assert confidently_wrong([], [], thr=0.6) is False


# ── action_bucket ───────────────────────────────────────────────────────────
def test_bucket_none_trivially_solved():
    # all correct -> nothing to demo
    assert action_bucket([1, 1, 1, 1], ["7", "7", "7", "7"]) == "none"


def test_bucket_redirect_low_confidence():
    # pass_rate 0.25 <= LO (0.30) -> redirect
    grades = [1, 0, 0, 0]
    answers = ["7", "3", "4", "5"]
    assert self_consistency(grades) <= DEFAULT_LO
    assert action_bucket(grades, answers) == "redirect"


def test_bucket_redirect_confidently_wrong_even_if_midconf():
    # pass_rate 0.5 (not <= LO, not >= HI) but confidently wrong -> redirect
    grades = [0, 0, 0, 1, 1, 1]
    answers = ["13", "13", "13", "12", "8", "9"]  # "13" 3/6=0.5 all wrong
    pr = self_consistency(grades)
    assert DEFAULT_LO < pr < DEFAULT_HI
    assert confidently_wrong(answers, grades, thr=0.5) is True
    assert action_bucket(grades, answers, confwrong_thr=0.5) == "redirect"


def test_bucket_verify_high_confidence_with_a_wrong_sample():
    # pass_rate 0.75 >= HI and one wrong sample to check -> verify
    grades = [1, 1, 1, 0]
    answers = ["12", "12", "12", "9"]
    assert self_consistency(grades) >= DEFAULT_HI
    assert action_bucket(grades, answers) == "verify"


def test_bucket_none_midconf_no_confwrong():
    # pass_rate 0.5, not confidently wrong (scattered) -> none
    grades = [1, 1, 0, 0]
    answers = ["7", "7", "1", "2"]
    pr = self_consistency(grades)
    assert DEFAULT_LO < pr < DEFAULT_HI
    assert action_bucket(grades, answers) == "none"


def test_bucket_hard_excluded_from_redirect():
    # SAME low-confidence input that would be 'redirect' for easy/medium,
    # but difficulty=hard -> NOT redirect (PG0: forced redirect hurts hard).
    grades = [1, 0, 0, 0]
    answers = ["7", "3", "4", "5"]
    assert action_bucket(grades, answers, difficulty="easy") == "redirect"
    assert action_bucket(grades, answers, difficulty="hard") == "none"


def test_bucket_hard_confidently_wrong_still_excluded():
    grades = [0, 0, 0, 0]
    answers = ["13", "13", "13", "13"]
    assert confidently_wrong(answers, grades) is True
    # easy/medium would redirect; hard is excluded -> none
    assert action_bucket(grades, answers, difficulty="medium") == "redirect"
    assert action_bucket(grades, answers, difficulty="hard") == "none"


def test_bucket_hard_can_still_verify():
    # hard is only excluded from REDIRECT; high-conf verify still allowed
    grades = [1, 1, 1, 0]
    answers = ["12", "12", "12", "9"]
    assert action_bucket(grades, answers, difficulty="hard") == "verify"


def test_bucket_empty_is_none():
    assert action_bucket([], []) == "none"


def test_bucket_difficulty_case_insensitive():
    grades = [1, 0, 0, 0]
    answers = ["7", "3", "4", "5"]
    assert action_bucket(grades, answers, difficulty="HARD") == "none"
