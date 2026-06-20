#!/usr/bin/env python3
"""Build the CONFIDENCE-conditioned redirect/verify SFT corpus (teacher distill).

Converged design (CLAUDE.md, memory pg0-raw-onpolicy-harvest-infeasible):

  (1) CONFIDENCE LABEL = STUDENT self-consistency. Roll the student out N times
      per problem; pass_rate = the student-CALIBRATED confidence target. The gold
      is used ONLY to grade correctness/causality, NEVER to measure confidence —
      so the confidence label is leak-free.
  (2) ANCHOR on the student's REAL wrong rollouts, EASY/MEDIUM only. Hard problems
      are dropped before any teacher call (teacher capability-gap -> OOD demos).
  (3) TEACHER (TRAPI GPT-5.4) CONDITIONAL generation given
      [problem + student wrong prefix + measured confidence]:
        low-conf / confidently-wrong  -> REDIRECT (switch to a different method)
        high-conf-but-checkable       -> VERIFY  (independent check)
  (4) CAUSAL FILTER with gold: keep a redirect demo only if the teacher trace
      flips wrong->right, states a confidence matching the student's measured
      value, AND a no-redirect CONTROL (k>=4 samples, CONTINUING the SAME flawed
      approach — NOT the 'always end correct' teacher) stays MAJORITY-wrong by a
      lower-CI margin; keep a verify demo only if it confirms/corrects.
  (5) LOSS-MASK the student's bad prefix (train only meta + recovery). The row
      carries the wrong_prefix TEXT + a char split marker; the SFT collator
      recomputes the TOKEN mask with the real tokenizer (src.training.
      segment_loss_mask.redirect_train_spans, which operates on TOKEN indices) —
      we do NOT persist a char mask as if it were a token mask.
  (6) Assemble SFT rows mirroring build_v8_strict_paired_data (messages JSON +
      split_tags), write a parquet.

ALL heavy IO is behind INJECTABLE callables so the pipeline is unit-tested
GPU-free / network-free:
  * ``rollout_fn(question, gold, n) -> list[(text, is_correct, answer)]`` — real
    path uses vLLM exactly like scripts/pg0_yield_pilot.py; the test passes a
    mock. The per-sample ANSWER string is required so confidently_wrong can gate
    redirect minting (a legacy (text, is_correct) 2-tuple is tolerated).
  * ``teacher_fn(payload) -> str`` — real path uses generator.get_trapi_client
    (model gpt-5.4); the test passes a mock. ``payload`` carries question, gold,
    confidence, bucket, arm in {"redirect","control","verify"}, wrong_prefix, and
    (control arm) a ``sample`` index. The 'control' arm MUST use the CONTROL-
    specific system prompt (continue the same flawed approach, no meta/switch),
    NOT the 'always end correct' distill prompt.

Reuses (does NOT reinvent):
  scripts/harvest_redirect_cf.splice_index   (wrong-prefix cut point)
  src/training/rewards._check_correctness     (answer-blind grading)
  src/training/segment_loss_mask              (loss-mask spans)
  scripts/build_v8_strict_paired_data.dump_messages (messages JSON)
  src/metacot/prompt_control_v4.build_control_v4_prompt (teacher prompt body)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v8_strict_paired_data import META_BLOCK_RE, dump_messages
from src.data.meta_format import (
    normalize_meta_format, validate_meta_structure,
    meta_is_pure_judgment, strip_preamble_before_meta, META_END,
)
from scripts.harvest_redirect_cf import splice_index, lower_ci_diff
from src.training.rewards import _check_correctness
# NOTE: the TOKEN loss-mask spans (src.training.segment_loss_mask.redirect_train_spans)
# are recomputed by the SFT collator from the persisted wrong_prefix + prefix_split_char,
# NOT here — so this driver does not import/call redirect_train_spans (it would be dead).
# SINGLE SOURCE OF TRUTH for the confidence thresholds + the action bucket.
# This driver does NOT re-declare its own (previously divergent) CONF_LOW=0.45 /
# CONF_HIGH=0.65; it imports the canonical values + the bucketer from
# src.data.confidence_label (CONF_LO=0.30 / CONF_HI=0.70) so the build path and
# the labeler agree on which bucket a problem falls into.
from src.data.confidence_label import (
    CONF_LO,
    CONF_HI,
    CONFWRONG_THR,
    action_bucket as _label_action_bucket,
    majority_answer,
    _norm as _norm_ans,
)

# --------------------------------------------------------------------------- #
# action bucketing from the student-self-consistency confidence
# --------------------------------------------------------------------------- #
BUCKET_REDIRECT = "redirect"
BUCKET_VERIFY = "verify"
BUCKET_NONE = "none"

# Map a SHARED-acceptance drop reason -> the build_dataset summary counter key.
# (sample_generate uses the bare reason strings; build_dataset keeps the longer
# 'dropped_*' names its tests assert on.)
_DROP_SUMMARY_KEY = {
    "malformed": "dropped_malformed",
    "solving_in_meta": "dropped_solving_in_meta",
    "decorative_decision": "dropped_decorative_decision",
    "decorative_norecover": "dropped_decorative_norecover",
    "decorative_verify": "dropped_decorative_verify",
    "conf_mismatch": "dropped_conf_mismatch",
    "control_recovers": "dropped_control_recovers",
}

# Anchor only on on-distribution difficulties (teacher capability-gap on hard).
ANCHOR_DIFFICULTIES = {"easy", "medium"}

# Control arm: how many samples to draw + the lower-CI margin redirect must beat
# the control pass-rate by. A single control sample is noisy (one lucky control
# recovery wrongly kills a genuine redirect), so draw k>=4 and require the
# lower 95% CI bound of (redirect_rate - control_rate) to clear CONTROL_MARGIN.
CONTROL_K = 4
CONTROL_MARGIN = 0.50

# Multi-anchor: mint up to this many redirect demos per redirect-band problem (one
# per DISTINCT wrong approach). redirect is rare (only ~18% of easy/medium land in
# the band, and the causal filter kills most), so a single anchor/problem under-
# yields; distinct wrong rollouts are genuinely different redirects (recognize THIS
# flawed path + switch). verify stays single-anchor (it is already abundant).
REDIRECT_MAX_ANCHORS = 4

# Tolerance for the teacher's STATED confidence vs the student's MEASURED value.
# The meta block must carry a `confidence: 0.xx` line within this of the
# student's pass-rate; otherwise the demo states a confidence the student does
# not hold (e.g. teacher says 0.90 on a pass_rate=0.10 redirect) and is dropped.
CONF_STATED_TOL = 0.15

# `confidence: 0.xx` extractor (mirrors harvest_redirect_cf._CONF_RE).
_CONF_LINE_RE = re.compile(r"confidence:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# `decision: redirect|verify` extractor (the TEXT field that replaced <|switch|>).
_DECISION_LINE_RE = re.compile(r"decision:\s*([A-Za-z]+)", re.IGNORECASE)

# Independent-check cue for the VERIFY causal filter. A genuine verify performs an
# INDEPENDENT check — substitute the candidate back, recompute by another route,
# or re-derive — rather than merely restating "confidence high, answer correct".
# A genuine independent check in the answer region needs BOTH a check-INTENT word
# AND an actual re-COMPUTATION. A decorative verify that only asserts the answer is
# right (intent word but no re-derivation) is dropped; a genuine check always
# recomputes (so carries computation). This is the verify analog of the redirect
# wrong->right flip — correctness alone is gameable (the teacher always ends correct).
_VERIFY_INTENT_RE = re.compile(
    r"\b(substitut\w*|plug\w*|recomput\w*|re-?check\w*|re-?deriv\w*|re-?work\w*|"
    r"cross-?check\w*|verif\w*|independent\w*|consisten\w*|convers\w*|confirm\w*|"
    r"check\w*|test\w*|another\s+(route|method|way|approach)|"
    r"different\s+(route|method|way|approach))\b",
    re.IGNORECASE,
)
# a display-math block or a digit-bearing equation = an actual computation.
_COMPUTATION_RE = re.compile(r"\\\[|\$\$|\d\s*[-+*/=×÷^]\s*\\?\d")


def _has_independent_check_cue(text: str) -> bool:
    """True iff the verify demo performs a genuine INDEPENDENT check in the ANSWER
    region (the text AFTER ``<|/meta|>``): a substitution / recompute / re-derive /
    cross-check cue.

    The meta block now holds only the JUDGMENT (confidence + 'check it' + decision),
    so the actual check lives in the answer region — scoping the cue there. A verify
    that merely states 'looks correct' with no real check in the answer region has no
    cue and is decorative -> dropped. This is the verify analog of requiring a real
    wrong->right flip for redirect (correctness alone is gameable: the teacher is
    told to always end correct)."""
    if not text:
        return False
    idx = text.find(META_END)
    answer_region = text[idx + len(META_END):] if idx != -1 else text
    return bool(_VERIFY_INTENT_RE.search(answer_region)) and bool(
        _COMPUTATION_RE.search(answer_region))


def _has_meta_block(text: str) -> bool:
    """True iff the trace contains a well-formed <|meta|>...<|/meta|> block.

    Used as a STRUCTURAL precondition on the verify causal filter so a trace that
    just ends with the correct boxed answer (no actual verification step) is not
    accepted as a 'verify' demo (correctness alone is gameable — the teacher is
    told to always end correct)."""
    return bool(text) and bool(META_BLOCK_RE.search(text))


def _meta_decision(text: str):
    """Parse the `decision: redirect|verify` text field from the meta block, or
    None. Reads only INSIDE the <|meta|>...<|/meta|> block (the TEXT field that
    replaced the old <|switch|> token); a stray 'decision:' in the recovery prose
    does not count. Returns the lower-cased value, else None."""
    if not text:
        return None
    m = META_BLOCK_RE.search(text)
    if not m:
        return None
    dm = _DECISION_LINE_RE.search(m.group(1))
    return dm.group(1).lower() if dm else None


def _stated_confidence(text: str):
    """Parse the emitted `confidence: 0.xx` from the meta block, or None.

    Reads only inside the <|meta|>...<|/meta|> block so a stray 'confidence:' in
    the recovery prose does not count. Returns the first parseable float in
    [0, 1], else None (missing / malformed)."""
    if not text:
        return None
    m = META_BLOCK_RE.search(text)
    if not m:
        # No meta block -> the demo has no place to state a calibrated confidence.
        return None
    cm = _CONF_LINE_RE.search(m.group(1))
    if not cm:
        return None
    try:
        val = float(cm.group(1))
    except ValueError:
        return None
    return val if 0.0 <= val <= 1.0 else None


def stated_conf_matches(text: str, student_confidence: float,
                        tol: float = CONF_STATED_TOL) -> bool:
    """True iff the trace states a `confidence:` line within ``tol`` of the
    student's MEASURED confidence. Missing / malformed line -> False (drop).

    This guards the no-leak invariant's twin: the demo must report the STUDENT's
    confidence, not the teacher's own (a teacher stating 0.90 on a pass_rate=0.10
    redirect is teaching the wrong, inflated number)."""
    stated = _stated_confidence(text)
    if stated is None:
        return False
    return abs(stated - float(student_confidence)) <= tol


def confidence_from_grades(grades: list[int]) -> float:
    """STUDENT-calibrated confidence = pass-rate over the N self-consistency
    rollouts (gold used only to grade, never shown -> leak-free)."""
    if not grades:
        return 0.0
    return sum(grades) / len(grades)


def action_bucket(confidence: float, any_wrong: bool = True) -> str:
    """Map student confidence to the metacognitive action to demonstrate.

    Thin confidence-only wrapper kept for callers/tests that only have a
    pass-rate (no per-sample answers). Delegates the thresholds to the single
    source of truth (CONF_LO / CONF_HI in src.data.confidence_label): confidence
    <= CONF_LO -> REDIRECT, >= CONF_HI -> VERIFY (iff there is a wrong sample to
    check), else NONE. The FULL build path calls
    confidence_label.action_bucket(grades, answers, difficulty) directly so the
    confidently-wrong signal (which needs the answer strings) gates redirect.
    """
    if confidence <= CONF_LO:
        return BUCKET_REDIRECT
    if confidence >= CONF_HI:
        return BUCKET_VERIFY if any_wrong else BUCKET_NONE
    return BUCKET_NONE


# --------------------------------------------------------------------------- #
# teacher payload + causal acceptance
# --------------------------------------------------------------------------- #
def _teacher_payload(question, gold, confidence, bucket, arm, wrong_prefix):
    return {
        "question": question,
        "gold": gold,
        "confidence": confidence,
        "bucket": bucket,
        "arm": arm,
        "wrong_prefix": wrong_prefix,
    }


def _draw_control_grades(teacher_fn, question, gold, confidence, wrong_prefix):
    """Draw the no-redirect CONTROL arm: CONTROL_K samples that CONTINUE the SAME
    flawed approach (arm='control', NO meta/switch), graded vs gold. The per-sample
    index perturbs the teacher seed so the k draws are independent. Returns the list
    of CONTROL_K correctness ints — the counterfactual the causal filter compares
    redirect against. Shared by build_dataset AND sample_generate (one loop, not two)."""
    grades = []
    for k in range(CONTROL_K):
        ctrl_text = teacher_fn(
            _teacher_payload(question, gold, confidence, BUCKET_REDIRECT, "control", wrong_prefix)
            | {"sample": k}
        )
        grades.append(1 if _grade_answer(ctrl_text, gold) else 0)
    return grades


# Final-answer / boxed line stripped off the tail before splicing so the wrong
# prefix never carries the (wrong) final answer the student already committed.
_FINAL_ANS_RE = re.compile(
    r"(?im)^\s*(the\s+answer\s+is|final\s+answer\s*:?|answer\s*:)\b.*$"
)


def _strip_boxed(text: str) -> str:
    """Remove every ``\\boxed{...}`` (and the related ``\\fbox{...}``) with
    BRACE-BALANCED matching so NESTED braces are handled. A naive
    ``\\boxed\\{[^}]*\\}`` stops at the first ``}``, so on the very common
    ``\\boxed{\\frac{1}{2}}`` it would strip only ``\\boxed{\\frac{1}`` and LEAK
    the answer fragment ``{2}}`` into the wrong prefix (a no-leak / redirect-
    defeating inversion). This scanner walks balanced braces and drops the whole
    macro argument."""
    if not text:
        return text
    out = []
    i = 0
    n = len(text)
    while i < n:
        matched = False
        for macro in ("\\boxed", "\\fbox"):
            if text.startswith(macro, i):
                j = i + len(macro)
                # skip optional whitespace between the macro and its '{'
                while j < n and text[j] in " \t":
                    j += 1
                if j < n and text[j] == "{":
                    depth = 0
                    k = j
                    while k < n:
                        if text[k] == "{":
                            depth += 1
                        elif text[k] == "}":
                            depth -= 1
                            if depth == 0:
                                k += 1
                                break
                        k += 1
                    # k now past the matched (or run-away unbalanced) close brace
                    i = k
                    matched = True
                    break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)


def _strip_final_answer(text: str) -> str:
    """Drop any trailing \\boxed{...} expression and 'The answer is ...' / 'Final
    answer:' line so a spliced prefix cannot already contain the wrong final
    answer (which would defeat the redirect — the student already 'answered').

    The boxed strip is brace-BALANCED (see ``_strip_boxed``) so a nested
    ``\\boxed{\\frac{1}{2}}`` does not leak its answer fragment."""
    if not text:
        return text
    out = _strip_boxed(text)
    out = _FINAL_ANS_RE.sub("", out)
    return out.rstrip()


# --------------------------------------------------------------------------- #
# SAFE grading: parse ONLY the extracted SHORT final answer, never the full essay.
# Feeding the whole teacher output to math_verify (whose SIGALRM timeout is disabled
# in worker threads) let a pathological intermediate expression (e.g. a control sample
# that "continues a flawed approach" and emits a huge exponent/factorial) make sympy
# allocate unboundedly -> the full build grew to ~117GB RSS and was OOM-killed.
# --------------------------------------------------------------------------- #
_DANGEROUS_MATH = re.compile(
    r"\d\s*\^\s*[({]?\s*-?\d*\s*\^"   # nested exponent a^b^c
    r"|\^\s*[({]?\s*-?\d{4,}"          # exponent >= 1000
    r"|\d{6,}\s*!"                      # factorial of a large number
)


def _last_boxed_value(text: str) -> str:
    """Inner argument of the LAST ``\\boxed{...}`` / ``\\fbox{...}`` (brace-balanced),
    or '' if none. Mirrors _strip_boxed's balanced scan but KEEPS the argument."""
    if not text:
        return ""
    last = ""
    i, n = 0, len(text)
    while i < n:
        macro = "\\boxed" if text.startswith("\\boxed", i) else (
            "\\fbox" if text.startswith("\\fbox", i) else "")
        if macro:
            j = i + len(macro)
            while j < n and text[j] in " \t":
                j += 1
            if j < n and text[j] == "{":
                depth, k = 0, j
                while k < n:
                    if text[k] == "{":
                        depth += 1
                    elif text[k] == "}":
                        depth -= 1
                        if depth == 0:
                            last = text[j + 1:k]
                            break
                    k += 1
                i = k + 1
                continue
        i += 1
    return last.strip()


def _extract_final_answer(text: str) -> str:
    """A SHORT final-answer string: the last ``\\boxed{...}``, else the 'answer is X'
    tail, else ''."""
    if not text:
        return ""
    b = _last_boxed_value(text)
    if b:
        return b
    m = list(re.finditer(r"(?i)answer\s+is\s*[:=]?\s*\$?([^\n$.]+)", text))
    return m[-1].group(1).strip() if m else ""


def _numeq(a: str, b: str) -> bool:
    try:
        return abs(float(a) - float(b)) < 1e-6
    except (ValueError, TypeError):
        return False


def _grade_answer(text: str, gold) -> bool:
    """SAFELY grade whether ``text`` reaches ``gold``. Parse only the EXTRACTED final
    answer (short) with math_verify; a long or pathological candidate (nested/huge
    exponent, big factorial) skips sympy and uses plain/numeric equality. NEVER hands
    the full essay to math_verify (the OOM path)."""
    cand = _extract_final_answer(text)
    g = str(gold).strip()
    if cand and len(cand) <= 80 and not _DANGEROUS_MATH.search(cand):
        return _check_correctness(cand, gold)
    return bool(cand) and (cand.strip() == g or _numeq(cand, g))


def _pick_wrong_prefixes(rollouts, max_n: int = 1) -> list[str]:
    """Up to ``max_n`` DISTINCT wrong-rollout prefixes (each a different flawed
    approach), spliced at 0.5 (the moment the trace went bad) and final-answer-
    stripped. Multi-anchor: a redirect-band problem has several wrong rollouts, so
    minting one redirect demo per DISTINCT wrong approach multiplies genuine
    redirect yield without touching the causal filter. De-dupes by stripped text."""
    out: list[str] = []
    seen: set[str] = set()
    for r in rollouts:
        text, correct, _answer = _unpack_rollout(r)
        if correct:
            continue
        stripped = _strip_final_answer(text)
        if not stripped.strip():
            continue
        cut = splice_index(len(stripped), 0.5)
        prefix = stripped[:cut]
        key = prefix.strip()
        if key in seen:
            continue
        seen.add(key)
        out.append(prefix)
        if len(out) >= max_n:
            break
    return out


def _pick_wrong_prefix(rollouts) -> str | None:
    """First WRONG rollout prefix (back-compat single-anchor wrapper). Returns None
    if there is no wrong rollout (nothing to redirect from) or nothing survives the
    strip."""
    picks = _pick_wrong_prefixes(rollouts, 1)
    return picks[0] if picks else None


def _pick_verify_attempt(rollouts):
    """Pick the REAL student attempt the VERIFY demo is anchored on.

    The verify scenario is high-confidence WITH a wrong sample. To make the
    verify causally load-bearing (the slip the check must catch), PREFER a WRONG
    sample with usable text — that sample's ``is_correct`` is the no-verify
    CONTROL (the raw attempt committed as-is). If the drawn samples carry no
    usable wrong text, fall back to the MAJORITY attempt (a correct sample whose
    answer is the majority) so the anchor is still a REAL attempt, never the
    empty string (the ungrounded / off-distribution bug we fixed for redirect).

    Returns ``(attempt_text, control_is_correct)`` or ``None`` if no rollout has
    usable text at all.
    """
    rs = [_unpack_rollout(r) for r in rollouts]
    # 1) prefer a WRONG sample with usable (non-blank) text — this is the slip.
    for text, correct, _answer in rs:
        if not correct and text and text.strip():
            return text, False
    # 2) else anchor on the majority attempt (still a REAL attempt).
    answers = [a for _, _, a in rs]
    maj = majority_answer(answers)
    if maj:
        for text, correct, answer in rs:
            if _norm_ans(answer) == maj and text and text.strip():
                return text, bool(correct)
    # 3) last resort: any sample with usable text.
    for text, correct, _answer in rs:
        if text and text.strip():
            return text, bool(correct)
    return None


def _unpack_rollout(r):
    """Normalize a rollout tuple to (text, is_correct, answer).

    The contract is now ``(text, is_correct, answer)`` so the confidently-wrong
    signal (which needs the per-sample ANSWER strings) can gate redirect. A
    legacy ``(text, is_correct)`` 2-tuple is tolerated (answer = '')."""
    if len(r) >= 3:
        text, correct, answer = r[0], r[1], r[2]
    else:
        text, correct, answer = r[0], r[1], ""
    return text, correct, answer


def _build_messages(question: str, wrong_prefix: str, teacher_text: str, bucket: str):
    """Assistant target = [wrong_prefix][teacher meta + recovery]. For verify there
    is no wrong prefix to keep (the student was right), so the assistant is just the
    teacher's verify trace."""
    if bucket == BUCKET_REDIRECT:
        assistant = f"{wrong_prefix}{teacher_text}"
    else:
        assistant = teacher_text
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant},
    ]
    return messages, assistant


# --------------------------------------------------------------------------- #
# SHARED acceptance filters (one source of truth for build_dataset AND
# sample_generate — the normalize -> validate -> structural/causal gate).
# Each returns an Outcome(row, reason, repaired, calibration_ok):
#   * row != None  -> KEPT (reason is the keep kind: 'redirect'|'verify_catch'|
#                     'verify_confirm'); else dropped (reason is the drop bucket).
#   * repaired     -> normalize_meta_format changed the text (close-tag repair).
#   * calibration_ok -> the STATED confidence matched the student's MEASURED value
#                     (tracked even on drops so quality_report can report it).
# --------------------------------------------------------------------------- #
class _Outcome:
    __slots__ = ("row", "reason", "repaired", "calibration_ok")

    def __init__(self, row, reason, repaired, calibration_ok):
        self.row = row
        self.reason = reason
        self.repaired = repaired
        self.calibration_ok = calibration_ok


def _normalize_and_validate(raw_text):
    """Shared step (3.5): repair repairable close-tag variants, then validate.
    Returns ``(text, repaired_bool, ok_struct)``."""
    normed = normalize_meta_format(raw_text)
    repaired = normed != raw_text  # close-tag/casing repair (NOT the preamble strip)
    text = strip_preamble_before_meta(normed)  # drop teacher prefix-repeat / preamble
    ok_struct, _reason = validate_meta_structure(text)
    return text, repaired, ok_struct


def _accept_redirect_demo(question, gold, confidence, wrong_prefix,
                          raw_redirect, control_grades):
    """Shared REDIRECT acceptance (structure + causality + calibration).

    ``control_grades`` is the list of CONTROL-arm correctness ints (the no-redirect
    counterfactual); the caller draws them so this function stays pure of the
    teacher. Returns an ``_Outcome``; on keep, ``row`` is the assembled SFT row."""
    redirect_text, repaired, ok_struct = _normalize_and_validate(raw_redirect)
    cal_ok = stated_conf_matches(redirect_text, confidence)
    if not ok_struct:
        return _Outcome(None, "malformed", repaired, cal_ok)
    # meta must be JUDGMENT ONLY (no calculation/answer leaked inside the block).
    if not meta_is_pure_judgment(redirect_text)[0]:
        return _Outcome(None, "solving_in_meta", repaired, cal_ok)
    # (a) a real redirect decision AND a wrong->right flip. Split the two failure
    # modes: decorative_decision (the teacher did not emit decision: redirect —
    # PROMPT-fixable) vs decorative_norecover (decision ok but the recovery answer is
    # wrong — the TEACHER could not solve it, a capability cap multi-anchor amplifies).
    if _meta_decision(redirect_text) != "redirect":
        return _Outcome(None, "decorative_decision", repaired, cal_ok)
    if not _grade_answer(redirect_text, gold):
        return _Outcome(None, "decorative_norecover", repaired, cal_ok)
    # (b) STATED confidence must match the student's MEASURED value.
    if not cal_ok:
        return _Outcome(None, "conf_mismatch", repaired, cal_ok)
    # (c) the no-redirect CONTROL must stay MAJORITY-wrong (lower-CI margin).
    redirect_grades = [1] * len(control_grades)
    if lower_ci_diff(redirect_grades, control_grades) < CONTROL_MARGIN:
        return _Outcome(None, "control_recovers", repaired, cal_ok)
    messages, _assistant = _build_messages(question, wrong_prefix, redirect_text,
                                           BUCKET_REDIRECT)
    row = {
        "messages": messages,
        "scenario": BUCKET_REDIRECT,
        "confidence_label": float(confidence),
        "wrong_prefix": wrong_prefix,
        "prefix_split_char": len(wrong_prefix),
    }
    return _Outcome(row, "redirect", repaired, cal_ok)


def _accept_verify_demo(question, gold, confidence, verify_attempt,
                        control_correct, raw_verify):
    """Shared VERIFY acceptance (symmetric to redirect). ``control_correct`` is the
    correctness of the raw anchored attempt (the no-verify counterfactual): a wrong
    attempt that the check corrects -> 'verify_catch'; an already-right attempt
    -> 'verify_confirm'. Returns an ``_Outcome``."""
    verify_text, repaired, ok_struct = _normalize_and_validate(raw_verify)
    cal_ok = stated_conf_matches(verify_text, confidence)
    if not ok_struct:
        return _Outcome(None, "malformed", repaired, cal_ok)
    # meta must be JUDGMENT ONLY (no calculation/answer leaked inside the block).
    if not meta_is_pure_judgment(verify_text)[0]:
        return _Outcome(None, "solving_in_meta", repaired, cal_ok)
    # structure + correct final + a GENUINE independent-check cue.
    if not (_has_meta_block(verify_text) and _grade_answer(verify_text, gold)
            and _has_independent_check_cue(verify_text)):
        return _Outcome(None, "decorative_verify", repaired, cal_ok)
    if not cal_ok:
        return _Outcome(None, "conf_mismatch", repaired, cal_ok)
    messages, _assistant = _build_messages(question, "", verify_text, BUCKET_VERIFY)
    row = {
        "messages": messages,
        "scenario": BUCKET_VERIFY,
        "confidence_label": float(confidence),
        "wrong_prefix": "",
        "prefix_split_char": 0,
    }
    kind = "verify_confirm" if control_correct else "verify_catch"
    return _Outcome(row, kind, repaired, cal_ok)


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def _process_problem(prob, rollout_fn, teacher_fn, n_rollouts):
    """Per-problem worker. Returns ``(kept_rows, counts)`` — NO shared state, so the
    driver can run it sequentially OR across a thread pool (the full build is ~1.6e4
    teacher calls; problems are independent). ``counts`` maps summary keys to deltas.

    Mirrors the inline loop body exactly: hard-drop -> rollout -> action bucket ->
    (multi-anchor redirect | verify) -> structural/causal/calibration gate."""
    counts: dict = {}
    kept_rows: list = []

    def bump(key):
        counts[key] = counts.get(key, 0) + 1

    question = prob["question"]
    gold = prob["gold"]
    tags = dict(prob.get("tags") or {})
    difficulty = str(tags.get("difficulty", "")).lower()

    # (2) ANCHOR easy/medium only — drop hard BEFORE any teacher call.
    if difficulty not in ANCHOR_DIFFICULTIES:
        bump("dropped_hard")
        return kept_rows, counts

    # (1) STUDENT self-consistency rollout -> confidence label.
    rollouts = list(rollout_fn(question, gold, n_rollouts))
    unpacked = [_unpack_rollout(r) for r in rollouts]
    grades = [1 if c else 0 for _, c, _ in unpacked]
    answers = [a for _, _, a in unpacked]
    confidence = confidence_from_grades(grades)

    # (2) ACTION BUCKET via the single source of truth (confidently_wrong + hard-
    # exclusion applied inside the labeler, not re-implemented here).
    bucket = _label_action_bucket(
        grades, answers, difficulty=difficulty,
        lo=CONF_LO, hi=CONF_HI, confwrong_thr=CONFWRONG_THR,
    )

    if bucket == BUCKET_NONE:
        bump("dropped_bucket_none")
        return kept_rows, counts

    try:
        if bucket == BUCKET_REDIRECT:
            # MULTI-ANCHOR: one redirect demo per DISTINCT wrong approach (up to
            # REDIRECT_MAX_ANCHORS) — redirect is rare, so a single anchor/problem
            # under-yields. Each anchor runs the full causal filter independently.
            wrong_prefixes = _pick_wrong_prefixes(rollouts, REDIRECT_MAX_ANCHORS)
            if not wrong_prefixes:
                bump("dropped_no_wrong_prefix")
                return kept_rows, counts
            outcomes = []
            for wrong_prefix in wrong_prefixes:
                # (3) TEACHER conditional redirect + (3.5/4) the no-redirect CONTROL
                # arm (CONTINUE the same flawed approach) so the filter is non-vacuous.
                raw_redirect = teacher_fn(
                    _teacher_payload(question, gold, confidence, bucket, "redirect", wrong_prefix)
                )
                control_grades = _draw_control_grades(
                    teacher_fn, question, gold, confidence, wrong_prefix
                )
                outcomes.append(_accept_redirect_demo(
                    question, gold, confidence, wrong_prefix, raw_redirect, control_grades
                ))

        else:  # BUCKET_VERIFY
            # (1') ANCHOR the verify demo on a REAL sampled student attempt (prefer a
            # WRONG sample — the slip the verify must catch). The picked sample's
            # is_correct is the no-verify CONTROL (the raw attempt committed as-is).
            picked = _pick_verify_attempt(rollouts)
            if picked is None:
                bump("dropped_no_verify_attempt")
                return kept_rows, counts
            verify_attempt, control_correct = picked
            raw_verify = teacher_fn(
                _teacher_payload(question, gold, confidence, bucket, "verify", verify_attempt)
            )
            outcomes = [_accept_verify_demo(
                question, gold, confidence, verify_attempt, control_correct, raw_verify
            )]
    except Exception:
        # A per-problem teacher failure (e.g. a 400 content-filter on the problem text,
        # or transient errors exhausted inside make_trapi_teacher_fn) must DROP this one
        # problem, NOT kill the whole concurrent batch. The count surfaces the magnitude
        # (a few = content filter; thousands = systemic -> investigate, do not mask).
        bump("dropped_teacher_error")
        return kept_rows, counts

    # ----- bookkeeping: count repairs/drops, collect kept rows. One outcome for
    # verify; up to REDIRECT_MAX_ANCHORS for multi-anchor redirect. -----
    tags.setdefault("difficulty", difficulty)
    for outcome in outcomes:
        if outcome.repaired:
            bump("repaired")
        if outcome.row is None:
            bump(_DROP_SUMMARY_KEY[outcome.reason])
            continue
        if outcome.reason == BUCKET_REDIRECT:
            bump("kept_redirect")
        else:  # verify_catch | verify_confirm
            bump(outcome.reason)
            bump("kept_verify")

        row = outcome.row
        kept_rows.append(
            {
                "messages": dump_messages(row["messages"]),
                "scenario": row["scenario"],
                "confidence_label": row["confidence_label"],
                # The SFT collator recomputes the TOKEN loss-mask from these with the
                # real tokenizer (prefix_len = len(tokenize(wrong_prefix)); spans =
                # redirect_train_spans(...)) — prefix_split_char is only a cheap CHAR
                # split marker, never assumed equal to the token boundary.
                "wrong_prefix": row["wrong_prefix"],
                "prefix_split_char": int(row["prefix_split_char"]),
                "split_tags": json.dumps(tags, ensure_ascii=False),
            }
        )
    return kept_rows, counts


def build_dataset(
    problems,
    rollout_fn,
    teacher_fn,
    out_path: str,
    n_rollouts: int = 8,
    max_workers: int = 1,
):
    """Orchestrate the full pipeline (see module docstring). Returns a summary dict
    and writes the SFT parquet to ``out_path``.

    ``rollout_fn(question, gold, n) -> [(text, is_correct), ...]``
    ``teacher_fn(payload) -> continuation_text``
    """
    import pandas as pd

    kept_rows = []
    summary = {
        "n_problems": len(problems),
        "kept_redirect": 0,
        "kept_verify": 0,
        "verify_catch": 0,
        "verify_confirm": 0,
        "dropped_hard": 0,
        "dropped_bucket_none": 0,
        "dropped_solving_in_meta": 0,
        "dropped_decorative_decision": 0,
        "dropped_decorative_norecover": 0,
        "dropped_decorative_verify": 0,
        "dropped_no_wrong_prefix": 0,
        "dropped_no_verify_attempt": 0,
        "dropped_control_recovers": 0,
        "dropped_conf_mismatch": 0,
        "dropped_malformed": 0,
        "dropped_teacher_error": 0,
        "repaired": 0,
    }

    # Each problem is independent -> sequential (max_workers=1, deterministic order)
    # OR a thread pool for the ~1.6e4-teacher-call full build. Aggregation happens in
    # the driver thread so the summary + kept_rows stay race-free.
    def _run(prob):
        return _process_problem(prob, rollout_fn, teacher_fn, n_rollouts)

    if max_workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            results = list(ex.map(_run, problems))
    else:
        results = [_run(prob) for prob in problems]

    for rows, counts in results:
        for key, delta in counts.items():
            summary[key] = summary.get(key, 0) + delta
        kept_rows.extend(rows)

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        kept_rows,
        columns=["messages", "scenario", "confidence_label",
                 "wrong_prefix", "prefix_split_char", "split_tags"],
    )
    df.to_parquet(out_path_p, index=False)
    summary["kept_rows"] = len(kept_rows)
    summary["out_path"] = str(out_path_p)
    return summary


# --------------------------------------------------------------------------- #
# sample-generate entry: run a hand-anchored (GPU-free) batch through the SAME
# normalize -> validate -> structural/causal filter as build_dataset, and report
# the demo QUALITY. Lets us inspect a small real-teacher batch before the full run.
# --------------------------------------------------------------------------- #
def sample_generate(anchors, teacher_fn):
    """Generate + filter teacher demos for hand-anchored cases (no student rollout).

    ``anchors`` is a list of dicts:
        {problem, gold, wrong_prefix, conf, difficulty, action}
      * action == 'redirect' -> wrong_prefix is the student's WRONG prefix; draws
        the redirect arm + CONTROL_K control arms and runs ``_accept_redirect_demo``.
      * action == 'verify'   -> wrong_prefix is the student's ATTEMPT; runs the
        verify arm + ``_accept_verify_demo``. The anchor may carry
        ``control_correct`` (default False = the attempt is a slip the check catches).

    Shares the SAME filter helpers as ``build_dataset`` (no duplicated logic).
    Returns ``(kept_rows, quality_report)`` where each kept row carries the LIST
    ``messages`` (not a JSON dump) + scenario + confidence_label + wrong_prefix +
    prefix_split_char, and ``quality_report`` has::

        functional_rate     kept / n_demos (anchors that produced a teacher demo)
        format_repaired_rate demos whose meta-format needed repair / n_demos
        calibration_ok_rate  demos whose STATED conf matched the anchor conf / n_demos
        n_redirect, n_verify kept counts by scenario
        n_dropped_by_reason  {reason: count} over dropped demos
    """
    kept_rows = []
    n_demos = 0
    n_repaired = 0
    n_cal_ok = 0
    n_redirect = 0
    n_verify = 0
    dropped = {}

    for a in anchors:
        problem = a["problem"]
        gold = a["gold"]
        conf = float(a["conf"])
        wrong_prefix = a["wrong_prefix"]
        action = a["action"]

        if action == BUCKET_REDIRECT:
            raw = teacher_fn(
                _teacher_payload(problem, gold, conf, BUCKET_REDIRECT, "redirect", wrong_prefix)
            )
            control_grades = _draw_control_grades(
                teacher_fn, problem, gold, conf, wrong_prefix
            )
            outcome = _accept_redirect_demo(
                problem, gold, conf, wrong_prefix, raw, control_grades
            )
        elif action == BUCKET_VERIFY:
            raw = teacher_fn(
                _teacher_payload(problem, gold, conf, BUCKET_VERIFY, "verify", wrong_prefix)
            )
            outcome = _accept_verify_demo(
                problem, gold, conf, wrong_prefix,
                bool(a.get("control_correct", False)), raw,
            )
        else:
            raise ValueError(f"anchor action must be redirect|verify, got {action!r}")

        n_demos += 1
        if outcome.repaired:
            n_repaired += 1
        if outcome.calibration_ok:
            n_cal_ok += 1
        if outcome.row is None:
            dropped[outcome.reason] = dropped.get(outcome.reason, 0) + 1
            continue
        if outcome.reason == BUCKET_REDIRECT:
            n_redirect += 1
        else:
            n_verify += 1
        kept_rows.append(outcome.row)

    def _rate(num):
        return (num / n_demos) if n_demos else 0.0

    report = {
        "n_demos": n_demos,
        "n_kept": len(kept_rows),
        "n_redirect": n_redirect,
        "n_verify": n_verify,
        "functional_rate": _rate(len(kept_rows)),
        "format_repaired_rate": _rate(n_repaired),
        "calibration_ok_rate": _rate(n_cal_ok),
        "n_dropped_by_reason": dropped,
    }
    return kept_rows, report


# --------------------------------------------------------------------------- #
# real wiring (vLLM student rollout + TRAPI teacher) — not unit-tested
# --------------------------------------------------------------------------- #
def _real_rollout_fn(model_path):  # pragma: no cover - GPU
    """Build a vLLM-backed rollout_fn mirroring scripts/pg0_yield_pilot.py."""
    raise NotImplementedError(
        "Wire vLLM like scripts/pg0_yield_pilot.py: load model_path, "
        "generate n samples temp 0.8, grade each with _check_correctness vs gold, "
        "extract the boxed answer per sample, "
        "return [(text, is_correct, answer), ...]."
    )


# Default TRAPI model fallback list (gpt-5.5 -> 404 not-deployed; gpt-5.4 ->
# sometimes 503; gpt-5.4-mini works). Tried in order on 404/503/health errors.
TRAPI_MODEL_FALLBACK = [
    "gpt-5.4-mini_2026-03-17",
    "gpt-5.3-chat_2026-03-03",
    "gpt-5.4_2026-03-05",
]

# Errors that mean "this MODEL is unavailable" -> advance the fallback list.
_MAX_COMPLETION_TOKENS = 4000
_MODEL_UNAVAILABLE = ("404", "503", "not deployed", "not found", "unavailable")
# Errors that mean "transient, retry the SAME model with backoff".
_TRANSIENT = ("429", "403", "rate", "timeout", "500", "502")


def _trapi_client_factory():  # pragma: no cover - network/credential
    """Construct the AzureOpenAI TRAPI client via ENTRA (no static token):
    ChainedTokenCredential(AzureCli, ManagedIdentity) -> bearer provider for
    'api://trapi/.default'. Mirrors bestiary scripts/trapi_openai_proxy.py."""
    from openai import AzureOpenAI
    from azure.identity import (
        AzureCliCredential,
        ManagedIdentityCredential,
        ChainedTokenCredential,
        get_bearer_token_provider,
    )

    cred = ChainedTokenCredential(AzureCliCredential(), ManagedIdentityCredential())
    tp = get_bearer_token_provider(cred, "api://trapi/.default")
    return AzureOpenAI(
        azure_endpoint="https://trapi.research.microsoft.com/gcr/shared",
        azure_ad_token_provider=tp,
        api_version="2025-04-01-preview",
    )


def _arm_messages(payload):
    """Build the chat messages for the teacher call, branching on payload['arm']:
      * redirect -> build_redirect_demo_prompt(question, wrong_prefix, confidence)
      * verify   -> build_verify_demo_prompt(question, attempt, confidence)
                    (attempt = payload['wrong_prefix'], the REAL anchored attempt)
      * control  -> build_control_continuation_prompt(question, wrong_prefix)
                    (CONTINUE the same flawed approach, NO meta/decision/switch —
                    CONTROL_CONTINUATION_SYSTEM_PROMPT, NOT 'always end correct';
                    this is what makes the causal filter falsifiable)."""
    from src.metacot.prompt_redirect_verify import (
        build_redirect_demo_prompt,
        build_verify_demo_prompt,
        build_control_continuation_prompt,
    )

    arm = payload["arm"]
    question = payload["question"]
    if arm == "redirect":
        return build_redirect_demo_prompt(question, payload["wrong_prefix"], payload["confidence"])
    if arm == "verify":
        return build_verify_demo_prompt(question, payload["wrong_prefix"], payload["confidence"])
    if arm == "control":
        return build_control_continuation_prompt(question, payload["wrong_prefix"])
    raise ValueError(f"unknown teacher arm: {arm!r}")


def make_trapi_teacher_fn(model_list=None, client_factory=None, max_retries=8):
    """Build a TRAPI(Entra)-backed ``teacher_fn(payload) -> str``.

    The AzureOpenAI client + Entra token provider are constructed LAZILY on the
    first call and CACHED (one client for the whole run). ``client_factory`` is
    injectable (tests pass a fake -> no network/credential); it defaults to the
    Entra constructor above.

    Per call: branch the prompt on ``payload['arm']`` (``_arm_messages``), then try
    the MODEL FALLBACK LIST in order — advance to the next model on a 404/503/health
    error (model not deployed), retry the SAME model with capped exponential backoff
    on a transient 429/403/5xx. Uses ``chat.completions`` (the API TRAPI serves these
    deployments on); ``seed``/``temperature`` are intentionally NOT sent — the proven
    call is ``chat.completions.create(model, messages, max_completion_tokens)``.

    Returns the completion text (raises if every model is exhausted)."""
    models = list(model_list or TRAPI_MODEL_FALLBACK)
    factory = client_factory or _trapi_client_factory
    state = {"client": None}

    def _client():
        if state["client"] is None:
            state["client"] = factory()
        return state["client"]

    def teacher_fn(payload):
        messages = _arm_messages(payload)
        last_err = None
        for model in models:
            for attempt in range(max_retries):
                try:
                    resp = _client().chat.completions.create(
                        model=model, messages=messages,
                        max_completion_tokens=_MAX_COMPLETION_TOKENS)
                    text = resp.choices[0].message.content
                    if text:
                        return text
                    last_err = RuntimeError("empty completion")
                except Exception as e:  # noqa: BLE001 - classify by message
                    last_err = e
                    msg = str(e).lower()
                    if any(t in msg for t in _MODEL_UNAVAILABLE):
                        break  # next model in the fallback list
                    if any(t in msg for t in _TRANSIENT):
                        wait = min(60.0, 2.0 * (2 ** attempt) + random.uniform(0, 2))
                        time.sleep(wait)
                        continue
                    raise  # non-transient, non-model error -> surface immediately
        raise RuntimeError(f"all TRAPI models exhausted ({models}); last error: {last_err}")

    return teacher_fn


def main():  # pragma: no cover - wires GPU + network; logic above is unit-tested
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_path", default="/scratch/models/v8_meta_inside_strict_sft")
    parser.add_argument("--train_parquet", default="/scratch/metacognition/data/verl_train_meta_mix.parquet")
    parser.add_argument("--out", default="data/v8_confidence_redirect_verify.parquet")
    parser.add_argument("--pool_size", type=int, default=2000)
    parser.add_argument("--n_rollouts", type=int, default=8)
    args = parser.parse_args()

    from scripts.pg0_yield_pilot import _load_pool

    problems = _load_pool(args.train_parquet, args.pool_size)
    summary = build_dataset(
        problems=problems,
        rollout_fn=_real_rollout_fn(args.model_path),
        teacher_fn=make_trapi_teacher_fn(),
        out_path=args.out,
        n_rollouts=args.n_rollouts,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
