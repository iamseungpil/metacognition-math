"""Student-calibrated confidence labels + demo ACTION bucketing.

Turns a STUDENT's N answer-grades (and the N answer strings) for one problem
into the labels the teacher-distill pipeline needs (CONVERGED DESIGN, memory
pg0-raw-onpolicy-harvest-infeasible):

  (i)  CONFIDENCE LABEL = the student's self-consistency / pass_rate over the N
       rollouts. This is the STUDENT-CALIBRATED confidence target the teacher
       conditions on (gold is used ONLY for the per-rollout grade -> correctness,
       never to *measure* confidence, so there is no confidence leak).
  (ii) MAJORITY answer + a "confidently wrong" flag = high agreement on a WRONG
       answer (the student is sure, and sure of the wrong thing -> a redirect
       demo, not a verify demo).
  (iii) an ACTION bucket for the demo:
         'redirect' — low self-consistency OR confidently wrong (switch method)
         'verify'   — high self-consistency but at least one wrong sample to
                      confirm against (checkable, an independent check pays off)
         'none'     — trivially solved (all correct, nothing to check) OR a HARD
                      problem (PG0: forced redirect HURTS hard, mean_gap -0.034,
                      so hard is excluded from redirect -> 'none' / skip).

All functions are PURE (no I/O, no network, no GPU). The per-rollout grading
(answer string -> 0/1 vs gold) is done UPSTREAM with
rewards._check_correctness; this module never sees gold, only the grades and the
answer strings, so it cannot leak the gold into the confidence target.

`grades` is a list of 0/1 (or bool); `answers` is the matching list of answer
strings (the student's extracted answers, parallel to `grades`).
"""
from collections import Counter

# Self-consistency thresholds for the action bucket (CONVERGED DESIGN).
# pass_rate <= LO  -> low confidence -> redirect
# pass_rate >= HI  -> high confidence -> verify (if any wrong sample to check)
#
# SINGLE SOURCE OF TRUTH: this module owns the confidence thresholds. The build
# driver (scripts/build_confidence_redirect_verify_sft.py) IMPORTS CONF_LO /
# CONF_HI / CONFWRONG_THR from here rather than re-declaring its own divergent
# CONF_LOW=0.45 / CONF_HIGH=0.65 (that drift split the same problem into two
# different buckets depending on which file you read).
CONF_LO = 0.30
CONF_HI = 0.70

# Agreement threshold for "confidently wrong": the majority WRONG answer must
# command at least this fraction of the samples for the student to count as
# *confidently* (not merely incidentally) wrong.
CONFWRONG_THR = 0.60

# Backwards-compatible aliases (older callers / tests use the DEFAULT_* names).
DEFAULT_LO = CONF_LO
DEFAULT_HI = CONF_HI
DEFAULT_CONFWRONG_THR = CONFWRONG_THR


def _norm(ans):
    """Normalize an answer string for majority counting (whitespace + case)."""
    return "" if ans is None else str(ans).strip().lower()


def self_consistency(grades) -> float:
    """Student-calibrated confidence = pass_rate = fraction correct over N.

    Empty -> 0.0 (no evidence the student can solve it). Returns a value in
    [0, 1]; this is the confidence TARGET the teacher conditions on.
    """
    if not grades:
        return 0.0
    return sum(1 for g in grades if g) / len(grades)


def majority_answer(answers) -> str:
    """Most common (normalized) non-empty answer string, '' if none.

    Ties broken by first appearance (Counter.most_common is stable on insertion
    order in CPython). Empty / None answers are ignored so a model that emitted
    no parseable answer does not win the vote.
    """
    counts = Counter(_norm(a) for a in (answers or []) if _norm(a))
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def confidently_wrong(answers, grades, thr: float = DEFAULT_CONFWRONG_THR) -> bool:
    """True if the student AGREES (>= thr of samples) on a WRONG answer.

    "Wrong" is read off the grades aligned to `answers`: the majority answer is
    confidently wrong iff (a) it is shared by >= thr of all samples and (b) the
    samples carrying that exact answer are graded incorrect. This is the
    redirect trigger "confidently wrong" — the student is sure, and wrong.
    """
    if not answers or not grades or len(answers) != len(grades):
        return False
    norm = [_norm(a) for a in answers]
    counts = Counter(a for a in norm if a)
    if not counts:
        return False
    maj, maj_n = counts.most_common(1)[0]
    if maj_n / len(answers) < thr:
        return False
    # Is the majority answer wrong? Look at the grades of the samples that
    # carry exactly that answer; if any of them is graded correct, the majority
    # answer is the RIGHT answer, so the student is confidently RIGHT, not wrong.
    for a, g in zip(norm, grades):
        if a == maj and g:
            return False
    return True


def action_bucket(
    grades,
    answers,
    difficulty=None,
    lo: float = DEFAULT_LO,
    hi: float = DEFAULT_HI,
    confwrong_thr: float = DEFAULT_CONFWRONG_THR,
) -> str:
    """Pick the demo ACTION bucket for one problem from the student's N samples.

    Returns one of 'redirect' / 'verify' / 'none':

      * HARD is excluded from redirect (PG0: forced redirect HURTS hard). A hard
        problem never yields a 'redirect' demo: it falls through to 'verify' only
        when high-confidence-with-a-wrong-sample, else 'none'. (Per PG0 we do not
        anchor redirect demos on hard at all.)
      * 'redirect' — low self-consistency (pass_rate <= lo) OR confidently wrong.
      * 'verify'   — high self-consistency (pass_rate >= hi) AND at least one
                     wrong sample to confirm against (checkable).
      * 'none'     — trivially solved (all correct) or nothing useful to demo.
    """
    if not grades:
        return "none"

    pass_rate = self_consistency(grades)
    is_hard = difficulty is not None and str(difficulty).strip().lower() == "hard"
    conf_wrong = confidently_wrong(answers, grades, thr=confwrong_thr)
    any_wrong = any(not g for g in grades)

    # Trivially solved: every sample correct -> no meta needed.
    if not any_wrong:
        return "none"

    # Redirect: low confidence OR confidently wrong — but NOT on hard problems.
    if not is_hard and (pass_rate <= lo or conf_wrong):
        return "redirect"

    # Verify: high confidence and a wrong sample to check against.
    if pass_rate >= hi and any_wrong:
        return "verify"

    return "none"
