#!/usr/bin/env python3
"""Critic checks for analysis coverage and usefulness."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path("/home/v-seungplee/metacognition/analysis/behavior_uncertainty_lab")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(ROOT / "results" / "behavior_uncertainty_summary.csv"))
    args = parser.parse_args()

    summary_path = Path(args.summary)
    assert summary_path.exists(), "Missing summary CSV"
    rows = list(csv.DictReader(summary_path.open()))
    assert rows, "Summary CSV is empty"

    any_verify = any(float(row["verify_rate"]) > 0 for row in rows)
    any_backtrack = any(float(row["backtrack_redirect_rate"]) > 0 for row in rows)
    any_uncertainty = any(float(row["uncertainty_rate"]) > 0 for row in rows)
    any_epistemic = any(float(row["epistemic_behavior_rate"]) > 0 for row in rows)

    assert any_verify or any_backtrack, "No target behaviors detected"
    assert any_uncertainty, "No uncertainty signal detected"
    assert any_epistemic, "No epistemic behavior detected"

    print("critic_ok")


if __name__ == "__main__":
    main()
