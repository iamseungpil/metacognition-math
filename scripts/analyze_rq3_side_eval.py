#!/usr/bin/env python3
"""Compare two RQ3 side-eval summaries against the RQ3-D success gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_json(path: str | Path) -> dict:
    input_path = Path(path)
    if input_path.is_dir():
        input_path = input_path / "rq3_summary.json"
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload: {input_path}")
    return payload


def _load_optional_summary(path: str | None) -> dict | None:
    if not path:
        return None
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _safe_ratio(num: float, den: float) -> float | None:
    if den <= 0:
        return None
    return num / den


def _extract_metrics(summary: dict) -> dict:
    num_cases = float(summary.get("num_cases", 0) or 0)
    triggering = summary.get("triggering", {}) or {}
    curriculum = summary.get("curriculum_retry", {}) or {}
    next_meta = curriculum.get("next_meta", {}) or {}
    winners = summary.get("winner_distribution", {}) or {}

    trigger_rate = float(triggering.get("trigger_rate", 0.0) or 0.0)
    trigger_on_wrong_root_rate = float(triggering.get("trigger_on_wrong_root_rate", 0.0) or 0.0)
    resolved_wins = sum(float(winners.get(label, 0) or 0) for label in ["root", "plain_retry", "retrieval_retry", "mcts_lite"])
    return {
        "num_cases": int(num_cases),
        "root_accuracy": float(summary.get("root_accuracy", 0.0) or 0.0),
        "ood_combined_accuracy": _safe_ratio(resolved_wins, num_cases),
        "trigger_rate": trigger_rate,
        "trigger_on_wrong_root_rate": trigger_on_wrong_root_rate,
        "trigger_precision": _safe_ratio(trigger_on_wrong_root_rate, trigger_rate),
        "retrieval_improvement_rate": float(curriculum.get("improvement_rate_over_root", 0.0) or 0.0),
        "retrieval_beats_plain_retry_rate": float(curriculum.get("beats_plain_retry_rate", 0.0) or 0.0),
        "next_meta_recovery_rate": float(
            next_meta.get("recovery_rate", next_meta.get("correct_with_cleared_trigger_rate", 0.0)) or 0.0
        ),
        "next_meta_trigger_clear_rate": float(next_meta.get("trigger_clear_rate", 0.0) or 0.0),
        "next_meta_confidence_recovery_rate": float(next_meta.get("confidence_recovery_rate", 0.0) or 0.0),
    }


def _delta(candidate: dict, baseline: dict) -> dict:
    out = {}
    for key, value in candidate.items():
        base_value = baseline.get(key)
        if isinstance(value, (int, float)) and isinstance(base_value, (int, float)):
            out[key] = value - base_value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Baseline rq3_summary.json or containing directory")
    parser.add_argument("--candidate", required=True, help="Candidate rq3_summary.json or containing directory")
    parser.add_argument("--candidate-train-summary", default=None, help="Optional self-distill dataset summary JSON")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baseline_raw = _load_json(args.baseline)
    candidate_raw = _load_json(args.candidate)
    baseline = _extract_metrics(baseline_raw)
    candidate = _extract_metrics(candidate_raw)
    delta = _delta(candidate, baseline)
    candidate_train = _load_optional_summary(args.candidate_train_summary)

    feedback_rate = None
    if isinstance(candidate_train, dict):
        rate = candidate_train.get("feedback_available_rate")
        if isinstance(rate, (int, float)):
            feedback_rate = float(rate)

    trigger_precision_drop = None
    if baseline["trigger_precision"] is not None and candidate["trigger_precision"] is not None:
        trigger_precision_drop = baseline["trigger_precision"] - candidate["trigger_precision"]

    gates = {
        "ood_combined_accuracy_delta_ge_2pp": (
            delta.get("ood_combined_accuracy", float("-inf")) >= 0.02
            if candidate.get("ood_combined_accuracy") is not None and baseline.get("ood_combined_accuracy") is not None
            else None
        ),
        "trigger_precision_drop_le_10pp": (
            trigger_precision_drop <= 0.10 if trigger_precision_drop is not None else None
        ),
        "next_meta_recovery_rate_ge_baseline": (
            candidate["next_meta_recovery_rate"] >= baseline["next_meta_recovery_rate"]
        ),
        "teacher_feedback_available_rate_gt_0": (
            feedback_rate > 0.0 if feedback_rate is not None else None
        ),
        "teacher_feedback_available_rate_ge_0_3": (
            feedback_rate >= 0.30 if feedback_rate is not None else None
        ),
    }

    payload = {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
        "candidate_train_summary": candidate_train,
        "derived": {
            "trigger_precision_drop": trigger_precision_drop,
            "teacher_feedback_available_rate": feedback_rate,
        },
        "gates": gates,
        "verdict": (
            "rq3_d2b_success_candidate"
            if all(value is True for key, value in gates.items() if key != "teacher_feedback_available_rate_ge_0_3")
            else "rq3_d2b_not_yet_supported"
        ),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
