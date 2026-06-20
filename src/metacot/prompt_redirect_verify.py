"""Teacher-prompt builders CONDITIONED on the student's measured state.

These build the TRAPI (GPT-5.4) prompts for the converged teacher-distill design:
given a problem, the student's REAL wrong prefix (redirect) or attempt (verify),
and the student's MEASURED confidence (self-consistency pass_rate), the teacher
produces an on-distribution demo that

  - states confidence ~= the STUDENT's measured value (never the teacher's own,
    never inflated above it),
  - then takes the right action: redirect (switch to a genuinely different
    method) for low-conf/confidently-wrong prefixes, or verify (independent
    recheck) for high-conf attempts,
  - and finishes correctly with a final \\boxed{} answer.

The <|meta|>/<|/meta|>/`confidence:`/`decision:` format mirrors
``prompt_behavior.py`` and ``prompt_control_v4.py`` so the resulting demos
assemble into the v8 SFT parquet format unchanged. The redirect-vs-verify choice
is a TEXT field ``decision: redirect`` / ``decision: verify`` INSIDE the meta
block (no special vocab token): causality is measured by ablating the whole meta
block + the wrong->right flip, not by banning a token.

Pure string builders: no network, no GPU. The TRAPI call happens elsewhere.
"""

META_START = "<|meta|>"
META_END = "<|/meta|>"


TEACHER_DISTILL_SYSTEM_PROMPT = f"""\
You are a math teacher writing a single training demonstration for a STUDENT model.
The demonstration must teach genuine metacognitive control, not decorative self-talk.

You are given the student's own work and the student's MEASURED confidence (its
self-consistency pass rate). The confidence number is the STUDENT's, not yours:
even if you personally find the problem easy, you must NOT inflate the confidence
above the given student value. Report confidence at (approximately) the given
value, never higher.

Hard rules:
1. Always reach the correct result and end with a final \\boxed{{answer}}.
2. Use a {META_START} ... {META_END} block only when it changes behavior or
   checks behavior. Every meta block must contain an explicit `confidence: 0.xx`
   line (approximately the given student confidence) AND an explicit
   `decision: redirect` or `decision: verify` line stating the action taken.
3. No decorative or filler meta. Never write fake doubt, "let me think again",
   or any meta that does not lead to a concrete action.
4. The confidence you state is the STUDENT's measured confidence, not your own,
   and you must not inflate it above the value you were given.
5. Meta text is natural language. Do not dump rigid templates like
   `trigger:`/`confidence_before:`/`confidence_after:`.
6. A redirect demo is only valid if it states `decision: redirect` AND the later
   reasoning genuinely uses a DIFFERENT method, not a rephrasing.
7. A verify demo states `decision: verify` and performs a truly INDEPENDENT check
   (substitution or recomputation by another route), then confirms or corrects.
"""


CONTROL_CONTINUATION_SYSTEM_PROMPT = """\
You are continuing a STUDENT's partial solution to a math problem, in the
student's OWN voice. This is a CONTROL continuation: your job is to carry the
existing line of work to its natural conclusion WITHOUT any metacognition.

Hard rules:
1. Continue the student's existing approach exactly as it was going. Use the
   SAME method/route the student already chose, even if it looks flawed.
2. Do NOT open a meta block. Never write <|meta|>, `confidence:`, or
   `decision:`. Do NOT switch method.
3. Do NOT switch to a different method, do NOT backtrack, do NOT second-guess.
   No "wait", no "let me reconsider" — just keep going on the current route.
4. Do not silently fix the student's mistakes. If the route leads to a wrong
   answer, let it reach that wrong answer; carry it to its natural conclusion.
5. End with a final \\boxed{answer} that follows from the SAME approach.

This continuation is the counterfactual baseline: it shows what happens if the
student keeps going WITHOUT redirecting. It must NOT recover by switching method.
"""


def _fmt_conf(conf: float) -> str:
    if not isinstance(conf, (int, float)):
        raise ValueError(f"conf must be a number, got {type(conf)!r}")
    if not (0.0 <= float(conf) <= 1.0):
        raise ValueError(f"conf must be in [0, 1], got {conf}")
    return f"{float(conf):.2f}"


def _messages(user: str):
    return [
        {"role": "system", "content": TEACHER_DISTILL_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_redirect_demo_prompt(problem: str, student_wrong_prefix: str, conf: float):
    """Teacher prompt for a REDIRECT demo conditioned on the student's wrong prefix.

    The teacher continues FROM the student's wrong prefix, detects the path is
    weak, states confidence ~= ``conf`` (the student's measured value, low /
    confidently-wrong), writes ``decision: redirect`` and switches to a genuinely
    different method, and solves correctly with \\boxed{{}}.

    Returns chat messages ``[system, user]``.
    """
    if not student_wrong_prefix or not student_wrong_prefix.strip():
        raise ValueError("student_wrong_prefix must be non-empty for a redirect demo")
    c = _fmt_conf(conf)

    user = f"""\
Scenario: redirect (low-confidence / confidently-wrong path).

Problem:
{problem}

The student already started solving and produced this WRONG prefix:
---
{student_wrong_prefix}
---

The student's MEASURED confidence on this attempt is {c}. This is the STUDENT's
self-consistency value, NOT your own confidence. Do not inflate it above {c}.

Write the continuation as a demonstration:
1. Continue FROM the student's wrong prefix (do not restart from scratch; pick up
   where the student left off and react to that work).
2. Open a {META_START} ... {META_END} block. State `confidence: {c}` and, in
   natural language, diagnose the concrete reason the current route is weak
   (e.g. a failed substitution, a contradiction, an unsupported assumption).
3. Inside the meta block, write the line `decision: redirect` and name a
   genuinely different method to switch to (a different strategy, not a rephrasing).
4. After the meta block, solve the problem correctly using that different method
   and end with the final \\boxed{{answer}}.

Remember: the meta block must change behavior (a real method switch), the
confidence is the student's value {c} and must not be inflated, and no decorative
filler.
"""
    return _messages(user)


def build_control_continuation_prompt(problem: str, student_wrong_prefix: str):
    """Teacher prompt for the no-redirect CONTROL arm of the causal filter.

    This is the counterfactual baseline that makes ``redirect_ok and not
    control_ok`` falsifiable: the teacher CONTINUES the student's wrong prefix
    using the SAME flawed approach, with NO meta block, NO ``confidence:`` line,
    NO ``decision:`` line, and no method switch — carrying the (likely wrong)
    route to its natural conclusion. If this control STILL recovers, the problem was
    self-recovering and the redirect demo is not credited.

    Uses ``CONTROL_CONTINUATION_SYSTEM_PROMPT`` (NOT the distill 'always end
    correct' prompt, which is exactly what made the old control non-falsifiable).

    Returns chat messages ``[system, user]``.
    """
    if not student_wrong_prefix or not student_wrong_prefix.strip():
        raise ValueError("student_wrong_prefix must be non-empty for a control continuation")

    user = f"""\
Continue the student's partial solution below WITHOUT any metacognition.

Problem:
{problem}

The student's work so far (continue from here, do not restart):
---
{student_wrong_prefix}
---

Continue this SAME approach to its natural conclusion:
1. Stay on the student's current route/method. Do NOT switch methods, do NOT
   backtrack, do NOT reconsider.
2. Do NOT open a {META_START} block, do NOT write a `confidence:` line, and do
   NOT write a `decision:` line.
3. Finish with a final \\boxed{{answer}} that follows from this same approach,
   even if that approach leads to a wrong answer.
"""
    return [
        {"role": "system", "content": CONTROL_CONTINUATION_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def build_verify_demo_prompt(problem: str, student_attempt: str, conf: float):
    """Teacher prompt for a VERIFY demo conditioned on a high-confidence attempt.

    The teacher states confidence ~= ``conf`` (high), performs an INDEPENDENT
    check (substitution / recomputation by another route), then confirms or
    corrects, ending with \\boxed{{}}.

    Returns chat messages ``[system, user]``.
    """
    if not student_attempt or not student_attempt.strip():
        raise ValueError("student_attempt must be non-empty for a verify demo")
    c = _fmt_conf(conf)

    user = f"""\
Scenario: verify (high-confidence attempt that should be independently checked).

Problem:
{problem}

The student produced this attempt:
---
{student_attempt}
---

The student's MEASURED confidence on this attempt is {c}. This is the STUDENT's
self-consistency value, NOT your own confidence. Report it at about {c} and do
not inflate it above {c}.

Write the demonstration:
1. Open a {META_START} ... {META_END} block. State `confidence: {c}`, write the
   line `decision: verify`, and explain in natural language that the answer looks
   right but must not be committed without an INDEPENDENT check.
2. Perform a truly independent verification: substitute the candidate value back
   into the original problem, or recompute the answer by a different route. Do
   not simply repeat the same steps.
3. If the check confirms the attempt, finalize it; if the check reveals an error,
   correct it. Either way end with the final \\boxed{{answer}}.

Remember: the confidence is the student's value {c} and must not be inflated, the
check must be genuinely independent, and no decorative filler.
"""
    return _messages(user)
