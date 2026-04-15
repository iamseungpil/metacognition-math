#!/usr/bin/env python3
"""Compare eval bundles with self-distill-focused epistemic metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_strict_pair_analysis import add_markers, compute_ece

OOD_BENCHMARKS = {"aime2024", "omni_math", "openmath_cot"}
HARD_BENCHMARKS = {"math500", *OOD_BENCHMARKS}


def load_eval_table(path: str) -> pd.DataFrame:
    input_path = Path(path)
    if input_path.suffix == ".parquet":
        df = pd.read_parquet(input_path)
    elif input_path.suffix == ".json":
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            if "results" in payload and isinstance(payload["results"], list):
                rows = payload["results"]
            elif "cases" in payload and isinstance(payload["cases"], list):
                rows = payload["cases"]
            else:
                raise ValueError(
                    f"JSON eval artifact must contain a list or a dict with `results`/`cases`: {input_path}"
                )
        elif isinstance(payload, list):
            rows = payload
        else:
            raise ValueError(f"Unsupported JSON eval payload: {input_path}")
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"Unsupported eval artifact: {input_path}")
    return add_markers(df)


def summarize(df: pd.DataFrame) -> dict:
    mask = df["avg_confidence"].notna()
    hard = df[df["benchmark"].isin(HARD_BENCHMARKS)]
    ood = df[df["benchmark"].isin(OOD_BENCHMARKS)]
    benchmark_breakdown = {}
    for benchmark, bdf in df.groupby("benchmark"):
        bmask = bdf["avg_confidence"].notna()
        benchmark_breakdown[str(benchmark)] = {
            "rows": int(len(bdf)),
            "accuracy": float(bdf["is_correct"].mean()),
            "meta_emission_rate": float((bdf["num_meta_blocks"] > 0).mean()),
            "wrong_high_conf_07": float(((~bdf["is_correct"]) & (bdf["avg_confidence"] >= 0.7)).mean()),
            "ece": compute_ece(
                bdf.loc[bmask, "avg_confidence"].to_numpy(),
                bdf.loc[bmask, "is_correct"].astype(float).to_numpy(),
            ) if bmask.any() else None,
        }
    return {
        "rows": int(len(df)),
        "accuracy": float(df["is_correct"].mean()),
        "hard_accuracy": float(hard["is_correct"].mean()) if len(hard) else None,
        "ood_accuracy": float(ood["is_correct"].mean()) if len(ood) else None,
        "meta_emission_rate": float((df["num_meta_blocks"] > 0).mean()),
        "avg_num_meta_blocks": float(df["num_meta_blocks"].mean()),
        "avg_completion_length_tokens": float(df["completion_length_tokens"].mean()),
        "avg_confidence": float(df["avg_confidence"].dropna().mean()) if mask.any() else None,
        "wrong_high_conf_07": float(((~df["is_correct"]) & (df["avg_confidence"] >= 0.7)).mean()),
        "diagnosis_rate": float(df["has_diagnosis"].mean()),
        "study_need_rate": float(df["completion"].fillna("").str.contains("study_need:", case=False).mean()),
        "epistemic_rate": float(df["has_epistemic"].mean()),
        "ece": compute_ece(
            df.loc[mask, "avg_confidence"].to_numpy(),
            df.loc[mask, "is_correct"].astype(float).to_numpy(),
        ) if mask.any() else None,
        "benchmark_breakdown": benchmark_breakdown,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    baseline = summarize(load_eval_table(args.baseline))
    candidate = summarize(load_eval_table(args.candidate))
    delta = {}
    for key, base_value in baseline.items():
        cand_value = candidate.get(key)
        if isinstance(base_value, (int, float)) and isinstance(cand_value, (int, float)):
            delta[key] = cand_value - base_value

    payload = {
        "baseline": baseline,
        "candidate": candidate,
        "delta": delta,
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
