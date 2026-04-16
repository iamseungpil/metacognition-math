#!/usr/bin/env python3
"""Render a lightweight working-note markdown from the analysis summary."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path("/home/v-seungplee/metacognition/analysis/behavior_uncertainty_lab")


def main() -> None:
    summary_path = ROOT / "results" / "behavior_uncertainty_summary.csv"
    examples_path = ROOT / "results" / "behavior_uncertainty_examples.md"
    report_path = ROOT / "reports" / "behavior_uncertainty_working_note.md"

    rows = list(csv.DictReader(summary_path.open()))
    table_lines = [
        "| Model | Benchmark | Acc | Conf | Gap | Verify | Redirect | Subgoal | Backward | Uncertainty | Epistemic |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        table_lines.append(
            "| {model} | {benchmark} | {accuracy} | {mean_confidence} | {calibration_gap} | "
            "{verify_rate} | {backtrack_redirect_rate} | {subgoal_rate} | {backward_rate} | "
            "{uncertainty_rate} | {epistemic_behavior_rate} |".format(**row)
        )

    examples = examples_path.read_text(encoding="utf-8") if examples_path.exists() else ""
    report = f"""# Behavior, Uncertainty, and Meta-Control

## Executive Summary

This working note is a placeholder report for a behavior-first interpretation layer over existing
Meta-CoT outputs. The analysis uses the four-behavior taxonomy from the STaR behavior paper and
the epistemic verbalization lens from the uncertainty paper to measure whether current meta traces
look like real control actions or only procedural text.

## Analysis Table

{chr(10).join(table_lines)}

## Preliminary Reading

- `verify` and `redirect` can be measured directly from explicit meta traces.
- The critical distinction is whether the behavior is accompanied by uncertainty language or
  confidence revision.
- This interpretation layer can later drive reward design and curriculum/RAG triggers.

## Example Slice

{examples}
"""
    report_path.write_text(report, encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
