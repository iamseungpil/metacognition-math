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
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v8_strict_paired_data import META_BLOCK_RE, dump_messages
from src.metacot.prompt_redirect_verify import SWITCH_TOKEN
from scripts.harvest_redirect_cf import splice_index, lower_ci_diff
from src.training.rewards import _check_correctness
from src.training.segment_loss_mask import redirect_train_spans
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
)

# --------------------------------------------------------------------------- #
# action bucketing from the student-self-consistency confidence
# --------------------------------------------------------------------------- #
BUCKET_REDIRECT = "redirect"
BUCKET_VERIFY = "verify"
BUCKET_NONE = "none"

# Anchor only on on-distribution difficulties (teacher capability-gap on hard).
ANCHOR_DIFFICULTIES = {"easy", "medium"}

# Control arm: how many samples to draw + the lower-CI margin redirect must beat
# the control pass-rate by. A single control sample is noisy (one lucky control
# recovery wrongly kills a genuine redirect), so draw k>=4 and require the
# lower 95% CI bound of (redirect_rate - control_rate) to clear CONTROL_MARGIN.
CONTROL_K = 4
CONTROL_MARGIN = 0.50

# Tolerance for the teacher's STATED confidence vs the student's MEASURED value.
# The meta block must carry a `confidence: 0.xx` line within this of the
# student's pass-rate; otherwise the demo states a confidence the student does
# not hold (e.g. teacher says 0.90 on a pass_rate=0.10 redirect) and is dropped.
CONF_STATED_TOL = 0.15

# `confidence: 0.xx` extractor (mirrors harvest_redirect_cf._CONF_RE).
_CONF_LINE_RE = re.compile(r"confidence:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)


def _has_meta_block(text: str) -> bool:
    """True iff the trace contains a well-formed <|meta|>...<|/meta|> block.

    Used as a STRUCTURAL precondition on the verify causal filter so a trace that
    just ends with the correct boxed answer (no actual verification step) is not
    accepted as a 'verify' demo (correctness alone is gameable — the teacher is
    told to always end correct)."""
    return bool(text) and bool(META_BLOCK_RE.search(text))


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


def _pick_wrong_prefix(rollouts) -> str | None:
    """First WRONG rollout text spliced at a fraction within [SPLICE_LO,SPLICE_HI]
    (the moment the trace went bad), AFTER stripping any trailing boxed / final-
    answer line. Returns None if there is no wrong rollout (nothing to redirect
    from) or nothing survives the strip."""
    for r in rollouts:
        text, correct, _answer = _unpack_rollout(r)
        if not correct:
            stripped = _strip_final_answer(text)
            if not stripped.strip():
                continue
            cut = splice_index(len(stripped), 0.5)
            return stripped[:cut]
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
# driver
# --------------------------------------------------------------------------- #
def build_dataset(
    problems,
    rollout_fn,
    teacher_fn,
    out_path: str,
    n_rollouts: int = 8,
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
        "dropped_hard": 0,
        "dropped_bucket_none": 0,
        "dropped_decorative": 0,
        "dropped_no_wrong_prefix": 0,
        "dropped_control_recovers": 0,
        "dropped_conf_mismatch": 0,
    }

    for prob in problems:
        question = prob["question"]
        gold = prob["gold"]
        tags = dict(prob.get("tags") or {})
        difficulty = str(tags.get("difficulty", "")).lower()

        # (2) ANCHOR easy/medium only — drop hard BEFORE any teacher call.
        if difficulty not in ANCHOR_DIFFICULTIES:
            summary["dropped_hard"] += 1
            continue

        # (1) STUDENT self-consistency rollout -> confidence label.
        # Contract: rollout_fn -> [(text, is_correct, answer), ...]; the per-sample
        # answer strings feed confidently_wrong (gates redirect minting).
        rollouts = list(rollout_fn(question, gold, n_rollouts))
        unpacked = [_unpack_rollout(r) for r in rollouts]
        grades = [1 if c else 0 for _, c, _ in unpacked]
        answers = [a for _, _, a in unpacked]
        confidence = confidence_from_grades(grades)

        # (2) ACTION BUCKET via the single source of truth: pass the ANSWER strings
        # AND the difficulty so confidently_wrong + the hard-exclusion are applied
        # here (not re-implemented). The labeler's 'redirect' covers low-pass-rate
        # OR confidently-wrong; 'verify' covers high-pass-rate-with-a-wrong-sample.
        bucket = _label_action_bucket(
            grades, answers, difficulty=difficulty,
            lo=CONF_LO, hi=CONF_HI, confwrong_thr=CONFWRONG_THR,
        )

        if bucket == BUCKET_NONE:
            summary["dropped_bucket_none"] += 1
            continue

        if bucket == BUCKET_REDIRECT:
            wrong_prefix = _pick_wrong_prefix(rollouts)
            if wrong_prefix is None:
                summary["dropped_no_wrong_prefix"] += 1
                continue
            # (3) TEACHER conditional redirect from the wrong prefix.
            redirect_text = teacher_fn(
                _teacher_payload(question, gold, confidence, bucket, "redirect", wrong_prefix)
            )
            # (4) CAUSAL FILTER (structure + confidence + causality):
            # (a) a real method switch (a <|switch|> inside a meta block, not a
            #     decorative confidence line that silently continues to the right
            #     answer), and the trace must flip wrong->right;
            redirect_ok = _check_correctness(redirect_text, gold)
            has_switch = _has_meta_block(redirect_text) and (SWITCH_TOKEN in (redirect_text or ""))
            if not (has_switch and redirect_ok):
                summary["dropped_decorative"] += 1
                continue
            # (b) the STATED confidence must match the student's MEASURED value
            #     (no inflated 0.90 on a pass_rate=0.10 redirect).
            if not stated_conf_matches(redirect_text, confidence):
                summary["dropped_conf_mismatch"] += 1
                continue
            # (c) the no-redirect CONTROL must stay MAJORITY-wrong. A single control
            #     sample is noisy, so draw CONTROL_K samples (arm='control' ->
            #     CONTROL-specific prompt that CONTINUES the same flawed approach,
            #     NO meta / NO switch) and require redirect to beat the control
            #     pass-rate by CONTROL_MARGIN on the lower 95% CI bound. With a
            #     genuinely-wrong control this clears; with an 'always-correct'
            #     control it never does (the filter is non-vacuous only then).
            control_grades = []
            for k in range(CONTROL_K):
                ctrl_text = teacher_fn(
                    _teacher_payload(question, gold, confidence, bucket, "control", wrong_prefix)
                    | {"sample": k}
                )
                control_grades.append(1 if _check_correctness(ctrl_text, gold) else 0)
            # redirect arm rate = 1.0 (it flipped to right, verified above).
            redirect_grades = [1] * CONTROL_K
            if lower_ci_diff(redirect_grades, control_grades) < CONTROL_MARGIN:
                summary["dropped_control_recovers"] += 1
                continue

            messages, assistant = _build_messages(question, wrong_prefix, redirect_text, bucket)
            # (5) LOSS-MASK the wrong prefix. We do NOT persist a CHAR mask as if it
            # were a token mask (the old bug: a char-length mask applied at TOKEN
            # positions could leave the bad prefix TRAINED). Instead carry the
            # wrong_prefix text + split marker so the SFT collator recomputes the
            # token mask with the REAL tokenizer via redirect_train_spans (which
            # operates on TOKEN indices supplied at SFT time).
            scenario = BUCKET_REDIRECT
            wrong_prefix_text = wrong_prefix
            prefix_split_char = len(wrong_prefix)
            summary["kept_redirect"] += 1

        else:  # BUCKET_VERIFY
            # (3) TEACHER conditional verify. (4) keep only if it actually VERIFIES
            # and confirms/corrects. Correctness alone is NOT a causal filter here:
            # the teacher is instructed to "always end correct", so a hollow demo
            # that emits a decorative confidence line and no real check would pass
            # `_check_correctness` trivially. Require a structural verify step (a
            # <|meta|> block) AND a correct final answer.
            verify_text = teacher_fn(
                _teacher_payload(question, gold, confidence, bucket, "verify", "")
            )
            if not (_has_meta_block(verify_text) and _check_correctness(verify_text, gold)):
                summary["dropped_decorative"] += 1
                continue
            # (4b) STATED confidence must match the student's MEASURED value.
            if not stated_conf_matches(verify_text, confidence):
                summary["dropped_conf_mismatch"] += 1
                continue
            messages, assistant = _build_messages(question, "", verify_text, bucket)
            # verify: no wrong prefix -> train the whole assistant target (split=0).
            scenario = BUCKET_VERIFY
            wrong_prefix_text = ""
            prefix_split_char = 0
            summary["kept_verify"] += 1

        tags.setdefault("difficulty", difficulty)
        kept_rows.append(
            {
                "messages": dump_messages(messages),
                "scenario": scenario,
                "confidence_label": float(confidence),
                # The SFT collator recomputes the TOKEN loss-mask from these with
                # the real tokenizer (NOT a char mask masquerading as a token mask):
                #   prefix_len = len(tokenize(wrong_prefix_text))
                #   spans = redirect_train_spans(prompt_len, prefix_len, len(full_ids))
                # so loss is masked on prompt+wrong_prefix and trained on
                # meta+recovery. prefix_split_char is the CHAR boundary in the
                # assistant string (a cheap split marker; the token boundary is
                # recomputed, never assumed equal to this char index).
                "wrong_prefix": wrong_prefix_text,
                "prefix_split_char": int(prefix_split_char),
                "split_tags": json.dumps(tags, ensure_ascii=False),
            }
        )

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


def _real_teacher_fn():  # pragma: no cover - network
    """Build a TRAPI-backed teacher_fn using generator.get_trapi_client (gpt-5.4).

    Branch the SYSTEM PROMPT on ``payload['arm']``:
      * arm == 'redirect' -> prompt_redirect_verify.build_redirect_demo_prompt(
          question, wrong_prefix, confidence)   (distill 'always end correct' + switch)
      * arm == 'verify'   -> prompt_redirect_verify.build_verify_demo_prompt(
          question, attempt, confidence)
      * arm == 'control'  -> prompt_redirect_verify.build_control_continuation_prompt(
          question, wrong_prefix)               (CONTINUE the same flawed approach,
          NO meta, NO <|switch|>, do NOT switch method; uses
          CONTROL_CONTINUATION_SYSTEM_PROMPT, NOT the 'always end correct' prompt —
          this is what makes the causal filter falsifiable). For the control arm,
          vary the sampling seed/temperature by ``payload['sample']`` so the k>=4
          control draws are independent (the build driver requires the control to
          be MAJORITY-wrong by a lower-CI margin).
    """
    raise NotImplementedError(
        "Wire generator.get_trapi_client(model='gpt-5.4_2026-03-05'); branch the "
        "SYSTEM PROMPT on payload['arm']: redirect=build_redirect_demo_prompt, "
        "verify=build_verify_demo_prompt, control=build_control_continuation_prompt "
        "(SAME flawed approach, NO meta/switch); vary control seed by payload['sample']."
    )


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
        teacher_fn=_real_teacher_fn(),
        out_path=args.out,
        n_rollouts=args.n_rollouts,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
