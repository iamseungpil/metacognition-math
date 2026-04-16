"""Importable eval metrics for self-distill comparisons."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pandas as pd


OOD_BENCHMARKS = {"aime2024", "omni_math", "openmath_cot"}
HARD_BENCHMARKS = {"math500", *OOD_BENCHMARKS}
VERIFY_RE = re.compile(r"\b(verify|check|confirm|validate|double.check|re.?check|sanity check)\b", re.I)
REDIRECT_RE = re.compile(r"\b(instead|alternative|try another|different approach|backtrack|switch|let me try|new plan)\b", re.I)
DIAGNOSIS_RE = re.compile(r"\b(mistake|error|wrong|incorrect|problem is|issue is|not right|missed|overlooked)\b", re.I)
EPISTEMIC_RE = re.compile(r"\b(not sure|uncertain|maybe|perhaps|might be|could be|feels off|hmm|wait)\b", re.I)


def compute_ece(confs: np.ndarray, corrects: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confs >= edges[i]) & (confs < edges[i + 1])
        if i == n_bins - 1:
            mask = mask | (confs == edges[i + 1])
        if not mask.any():
            continue
        ece += mask.mean() * abs(float(confs[mask].mean()) - float(corrects[mask].mean()))
    return float(ece)


def add_markers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    comp = df["completion"].fillna("").astype(str)
    df["has_verify"] = comp.apply(lambda x: bool(VERIFY_RE.search(x)))
    df["has_redirect"] = comp.apply(lambda x: bool(REDIRECT_RE.search(x)))
    df["has_diagnosis"] = comp.apply(lambda x: bool(DIAGNOSIS_RE.search(x)))
    df["has_epistemic"] = comp.apply(lambda x: bool(EPISTEMIC_RE.search(x)))
    df["is_correct"] = df["is_correct"].astype(bool)
    return df


def load_eval_table(path: str | Path) -> pd.DataFrame:
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


def summarize_eval_table(df: pd.DataFrame) -> dict:
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


__all__ = ["HARD_BENCHMARKS", "OOD_BENCHMARKS", "load_eval_table", "summarize_eval_table"]
