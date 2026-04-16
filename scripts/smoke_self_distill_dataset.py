#!/usr/bin/env python3
"""Smoke test the self-distill dataset builders on tiny synthetic traces."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.self_distill_data import (
    build_self_distill_dataframe,
    summarize_self_distill_dataframe,
)


def check(name: str, condition: bool) -> None:
    if not condition:
        raise RuntimeError(f"Self-distill smoke failed: {name}")
    print(f"PASS: {name}")


def main() -> None:
    rows = [
        {
            "question": "Solve x+3=7.",
            "gold_answer": "4",
            "curriculum_retry": {
                "retry_completion": (
                    "<|meta|>\n"
                    "confidence: 0.32\n"
                    "The issue is that the earlier route was weak because I did not isolate the variable.\n"
                    "study_need: direct isolation\n"
                    "I should recover with a cleaner equation-solving route.\n"
                    "<|/meta|>\n"
                    "Subtract 3 from both sides to get x=4. \\boxed{4}\n"
                    "<|meta|>\n"
                    "confidence: 0.81\n"
                    "The trigger is cleared after the corrected route.\n"
                    "<|/meta|>"
                ),
                "retry_judgment": {"is_correct": True},
                "meta_transition": {"confidence_gain": 0.49, "trigger_cleared": True},
            },
            "root_completion": "I guessed \\boxed{5}",
            "root_analysis": {
                "diagnosis_text": "The earlier route guessed without isolating the variable.",
                "study_need": "direct isolation",
            },
        }
    ]

    naive = build_self_distill_dataframe(rows, mode="naive")
    epistemic = build_self_distill_dataframe(rows, mode="epistemic")
    sdpo_regen = build_self_distill_dataframe(rows, mode="sdpo_regen")

    check("naive should build one row", len(naive) == 1)
    check("epistemic should build one row", len(epistemic) == 1)
    check("sdpo_regen without feedback should skip rows", len(sdpo_regen) == 0)

    naive_msgs = json.loads(naive.iloc[0]["messages"])
    epi_msgs = json.loads(epistemic.iloc[0]["messages"])
    naive_text = naive_msgs[-1]["content"]
    epi_text = epi_msgs[-1]["content"]

    check("naive should remove meta blocks", "<|meta|>" not in naive_text)
    check("epistemic should keep meta blocks", "<|meta|>" in epi_text)
    check("naive should still keep boxed answer", "\\boxed{4}" in naive_text)
    check("epistemic should still keep boxed answer", "\\boxed{4}" in epi_text)

    naive_summary = summarize_self_distill_dataframe(naive)
    epi_summary = summarize_self_distill_dataframe(epistemic)
    check("naive meta emission should be zero", naive_summary["meta_emission_rate"] == 0.0)
    check("epistemic meta emission should be positive", epi_summary["meta_emission_rate"] > 0.0)

    out = {
        "naive": naive_summary,
        "epistemic": epi_summary,
        "sdpo_regen": summarize_self_distill_dataframe(sdpo_regen),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
