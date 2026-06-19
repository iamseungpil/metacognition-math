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
      flips wrong->right AND a no-redirect CONTROL stays wrong; keep a verify demo
      only if it confirms/corrects (teacher trace correct).
  (5) LOSS-MASK the student's bad prefix (train only meta + recovery) — via
      src.training.segment_loss_mask.
  (6) Assemble SFT rows mirroring build_v8_strict_paired_data (messages JSON +
      split_tags), write a parquet.

ALL heavy IO is behind INJECTABLE callables so the pipeline is unit-tested
GPU-free / network-free:
  * ``rollout_fn(question, gold, n) -> list[(text, is_correct)]`` — real path uses
    vLLM exactly like scripts/pg0_yield_pilot.py; the test passes a mock.
  * ``teacher_fn(payload) -> str`` — real path uses generator.get_trapi_client
    (model gpt-5.4); the test passes a mock. ``payload`` carries question, gold,
    confidence, bucket, arm in {"redirect","control","verify"}, wrong_prefix.

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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_v8_strict_paired_data import META_BLOCK_RE, dump_messages
from src.metacot.prompt_redirect_verify import SWITCH_TOKEN
from scripts.harvest_redirect_cf import splice_index
from src.training.rewards import _check_correctness
from src.training.segment_loss_mask import (
    build_segment_loss_mask,
    redirect_train_spans,
)

# --------------------------------------------------------------------------- #
# action bucketing from the student-self-consistency confidence
# --------------------------------------------------------------------------- #
BUCKET_REDIRECT = "redirect"
BUCKET_VERIFY = "verify"
BUCKET_NONE = "none"

# Confidence thresholds (mirror the redirect/verify confidence semantics in
# prompt_control_v4: redirect intervenes at low conf, verify confirms at high conf).
CONF_LOW = 0.45   # <= -> REDIRECT (confidently-wrong / low pass-rate)
CONF_HIGH = 0.65  # >= -> VERIFY (high-but-checkable)

# Anchor only on on-distribution difficulties (teacher capability-gap on hard).
ANCHOR_DIFFICULTIES = {"easy", "medium"}


def _has_meta_block(text: str) -> bool:
    """True iff the trace contains a well-formed <|meta|>...<|/meta|> block.

    Used as a STRUCTURAL precondition on the verify causal filter so a trace that
    just ends with the correct boxed answer (no actual verification step) is not
    accepted as a 'verify' demo (correctness alone is gameable — the teacher is
    told to always end correct)."""
    return bool(text) and bool(META_BLOCK_RE.search(text))


def confidence_from_grades(grades: list[int]) -> float:
    """STUDENT-calibrated confidence = pass-rate over the N self-consistency
    rollouts (gold used only to grade, never shown -> leak-free)."""
    if not grades:
        return 0.0
    return sum(grades) / len(grades)


def action_bucket(confidence: float, any_wrong: bool = True) -> str:
    """Map student confidence to the metacognitive action to demonstrate.

    ``any_wrong`` = did at least one of the N student rollouts come out wrong.
    A VERIFY demo only makes sense when there is something to check against: a
    problem the student solves 4/4 is trivially solved and a "verify" block there
    is decorative meta (the converged design buckets it to NONE — verify is
    "high-confidence-but-CHECKABLE", not "high-confidence-and-already-perfect").
    """
    if confidence <= CONF_LOW:
        return BUCKET_REDIRECT
    if confidence >= CONF_HIGH:
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


def _pick_wrong_prefix(rollouts) -> str | None:
    """First WRONG rollout text spliced at splice_index (the moment the trace went
    bad). Returns None if there is no wrong rollout (nothing to redirect from)."""
    for text, correct in rollouts:
        if not correct:
            cut = splice_index(len(text), 0.5)
            return text[:cut]
    return None


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


def _approx_token_len(text: str) -> int:
    """Whitespace token proxy (the real path passes true tokenizer lengths; for the
    SFT-row mask we record CHAR-index spans so it is tokenizer-agnostic here)."""
    return len(text)


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
        rollouts = rollout_fn(question, gold, n_rollouts)
        grades = [1 if c else 0 for _, c in rollouts]
        confidence = confidence_from_grades(grades)
        any_wrong = any(g == 0 for g in grades)
        bucket = action_bucket(confidence, any_wrong=any_wrong)

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
            # no-redirect CONTROL from the same wrong prefix.
            control_text = teacher_fn(
                _teacher_payload(question, gold, confidence, bucket, "control", wrong_prefix)
            )
            # (4) CAUSAL FILTER: redirect must (a) contain a real method switch
            # (a <|switch|> decision inside a meta block — not a decorative
            # confidence line that silently continues to the right answer),
            # (b) flip wrong->right, AND (c) the no-redirect control stays wrong.
            redirect_ok = _check_correctness(redirect_text, gold)
            control_ok = _check_correctness(control_text, gold)
            has_switch = _has_meta_block(redirect_text) and (SWITCH_TOKEN in (redirect_text or ""))
            if not (has_switch and redirect_ok and not control_ok):
                summary["dropped_decorative"] += 1
                continue

            messages, assistant = _build_messages(question, wrong_prefix, redirect_text, bucket)
            # (5) LOSS-MASK the wrong prefix (char-index span proxy; real path uses
            # tokenizer lengths). prompt_len=0 because the mask is over the ASSISTANT
            # target only (the SFT tokenizer masks the prompt separately).
            prefix_len = _approx_token_len(wrong_prefix)
            total_len = _approx_token_len(assistant)
            spans = redirect_train_spans(0, prefix_len, total_len)
            loss_mask = build_segment_loss_mask(total_len, spans)
            scenario = BUCKET_REDIRECT
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
            messages, assistant = _build_messages(question, "", verify_text, bucket)
            # verify: no wrong prefix -> train the whole assistant target.
            total_len = _approx_token_len(assistant)
            loss_mask = build_segment_loss_mask(total_len, [(0, total_len)])
            scenario = BUCKET_VERIFY
            summary["kept_verify"] += 1

        tags.setdefault("difficulty", difficulty)
        kept_rows.append(
            {
                "messages": dump_messages(messages),
                "scenario": scenario,
                "confidence_label": float(confidence),
                "loss_mask": loss_mask,
                "split_tags": json.dumps(tags, ensure_ascii=False),
            }
        )

    out_path_p = Path(out_path)
    out_path_p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        kept_rows,
        columns=["messages", "scenario", "confidence_label", "loss_mask", "split_tags"],
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
        "return [(text, is_correct), ...]."
    )


def _real_teacher_fn():  # pragma: no cover - network
    """Build a TRAPI-backed teacher_fn using generator.get_trapi_client (gpt-5.4).

    The system prompt reuses prompt_control_v4.CONTROL_V4_SYSTEM_PROMPT; the user
    body reuses build_control_v4_prompt, augmented with the student's measured
    confidence and (for redirect) the wrong prefix, asking the teacher for the
    redirect / control / verify continuation per ``payload['arm']``.
    """
    raise NotImplementedError(
        "Wire generator.get_trapi_client(model='gpt-5.4_2026-03-05'); system="
        "prompt_control_v4.CONTROL_V4_SYSTEM_PROMPT; user=build_control_v4_prompt(...) "
        "+ measured confidence + wrong prefix; branch on payload['arm']."
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
