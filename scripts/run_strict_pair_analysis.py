#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERIFY_RE = re.compile(r"\b(verify|check|confirm|validate|double.check|re.?check|sanity check)\b", re.I)
REDIRECT_RE = re.compile(r"\b(instead|alternative|try another|different approach|backtrack|switch|let me try|new plan)\b", re.I)
DIAGNOSIS_RE = re.compile(r"\b(mistake|error|wrong|incorrect|problem is|issue is|not right|missed|overlooked)\b", re.I)
EPISTEMIC_RE = re.compile(r"\b(not sure|uncertain|maybe|perhaps|might be|could be|feels off|hmm|wait)\b", re.I)

PAIR_KEYS = ["benchmark", "question", "gold_answer"]


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


def load_first_readable_parquet(root: Path, preferred: str) -> tuple[pd.DataFrame, str]:
    candidates = [preferred]
    if preferred.endswith(".parquet"):
        candidates.append(preferred[:-8] + ".remote.parquet")
    seen: set[str] = set()
    last_error: Exception | None = None
    for rel in candidates:
        if rel in seen:
            continue
        seen.add(rel)
        path = root / rel
        if not path.exists():
            continue
        try:
            return pd.read_parquet(path), rel
        except Exception as exc:  # pragma: no cover - defensive fallback for broken artifacts
            last_error = exc
    if last_error is not None:
        raise last_error
    raise FileNotFoundError(f"No readable parquet found for {preferred}")


def marker_accuracy(sub: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for marker in ["has_verify", "has_redirect", "has_diagnosis", "has_epistemic"]:
        g1 = sub[sub[marker]]
        g0 = sub[~sub[marker]]
        out[marker] = {
            "n_present": int(len(g1)),
            "acc_present": float(g1["is_correct"].mean()) if len(g1) else None,
            "n_absent": int(len(g0)),
            "acc_absent": float(g0["is_correct"].mean()) if len(g0) else None,
        }
    return out


def summarize_meta(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": int(len(df)),
        "accuracy": float(df["is_correct"].mean()),
        "meta_block_count_dist": {str(int(k)): int(v) for k, v in df["num_meta_blocks"].value_counts().sort_index().items()},
        "meta_emission_rate": float((df["num_meta_blocks"] > 0).mean()),
        "avg_num_meta_blocks": float(df["num_meta_blocks"].mean()),
        "avg_confidence": float(df["avg_confidence"].dropna().mean()),
        "wrong_high_conf_07": float(((~df["is_correct"]) & (df["avg_confidence"] >= 0.7)).mean()),
        "wrong_high_conf_08": float(((~df["is_correct"]) & (df["avg_confidence"] >= 0.8)).mean()),
        "low_conf_redirect_rate": float(df.loc[df["avg_confidence"] <= 0.5, "has_redirect"].mean()),
        "high_conf_verify_rate": float(df.loc[df["avg_confidence"] >= 0.7, "has_verify"].mean()),
        "marker_accuracy": {},
        "by_benchmark": {},
        "confidence_bins": [],
    }
    mask = df["avg_confidence"].notna()
    out["ece"] = compute_ece(df.loc[mask, "avg_confidence"].to_numpy(), df.loc[mask, "is_correct"].astype(float).to_numpy())
    slices = {
        "overall": df,
        "hard": df[df["benchmark"].isin(["math500", "aime2024"])],
        "gsm8k": df[df["benchmark"] == "gsm8k"],
    }
    for name, sub in slices.items():
        out["marker_accuracy"][name] = marker_accuracy(sub)
    for bm, g in df.groupby("benchmark"):
        out["by_benchmark"][bm] = {
            "n": int(len(g)),
            "accuracy": float(g["is_correct"].mean()),
            "verify_rate": float(g["has_verify"].mean()),
            "redirect_rate": float(g["has_redirect"].mean()),
            "diagnosis_rate": float(g["has_diagnosis"].mean()),
            "epistemic_rate": float(g["has_epistemic"].mean()),
            "avg_confidence": float(g["avg_confidence"].dropna().mean()),
            "wrong_high_conf_07": float(((~g["is_correct"]) & (g["avg_confidence"] >= 0.7)).mean()),
        }
    for lo, hi in [(0, .3), (.3, .5), (.5, .7), (.7, .9), (.9, 1.01)]:
        g = df[(df["avg_confidence"] >= lo) & (df["avg_confidence"] < hi)]
        out["confidence_bins"].append({
            "bin": f"{lo:.1f}-{min(hi, 1.0):.1f}",
            "n": int(len(g)),
            "accuracy": float(g["is_correct"].mean()) if len(g) else None,
            "verify_rate": float(g["has_verify"].mean()) if len(g) else None,
            "redirect_rate": float(g["has_redirect"].mean()) if len(g) else None,
        })
    return out


def summarize_base(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "rows": int(len(df)),
        "accuracy": float(df["is_correct"].mean()),
        "meta_block_count_dist": {str(int(k)): int(v) for k, v in df["num_meta_blocks"].value_counts().sort_index().items()},
        "meta_emission_rate": float((df["num_meta_blocks"] > 0).mean()),
        "marker_accuracy": {},
        "by_benchmark": {},
    }
    slices = {
        "overall": df,
        "hard": df[df["benchmark"].isin(["math500", "aime2024"])],
        "gsm8k": df[df["benchmark"] == "gsm8k"],
    }
    for name, sub in slices.items():
        out["marker_accuracy"][name] = marker_accuracy(sub)
    for bm, g in df.groupby("benchmark"):
        out["by_benchmark"][bm] = {
            "n": int(len(g)),
            "accuracy": float(g["is_correct"].mean()),
            "verify_rate": float(g["has_verify"].mean()),
            "redirect_rate": float(g["has_redirect"].mean()),
            "diagnosis_rate": float(g["has_diagnosis"].mean()),
            "epistemic_rate": float(g["has_epistemic"].mean()),
        }
    return out


def paired_summary(meta_df: pd.DataFrame, base_df: pd.DataFrame) -> dict[str, Any]:
    meta = meta_df.drop_duplicates(PAIR_KEYS).copy()
    base = base_df.drop_duplicates(PAIR_KEYS).copy()
    merged = meta.merge(base, on=PAIR_KEYS, suffixes=("_meta", "_base"), how="inner")
    if len(merged) != len(meta) or len(merged) != len(base):
        raise ValueError(
            f"paired join mismatch: meta={len(meta)} base={len(base)} merged={len(merged)}"
        )

    def row_counts(frame: pd.DataFrame) -> dict[str, int]:
        return {
            "meta_only_win": int((frame["is_correct_meta"] & ~frame["is_correct_base"]).sum()),
            "base_only_win": int((~frame["is_correct_meta"] & frame["is_correct_base"]).sum()),
            "both_correct": int((frame["is_correct_meta"] & frame["is_correct_base"]).sum()),
            "both_wrong": int((~frame["is_correct_meta"] & ~frame["is_correct_base"]).sum()),
        }

    overall = row_counts(merged)
    per_benchmark = {bm: row_counts(g) for bm, g in merged.groupby("benchmark")}

    behavior_slice = merged[
        [
            "benchmark",
            "question",
            "is_correct_meta",
            "is_correct_base",
            "num_meta_blocks_meta",
            "avg_confidence_meta",
            "completion_meta",
            "completion_base",
            "has_verify_meta",
            "has_redirect_meta",
            "has_diagnosis_meta",
            "has_epistemic_meta",
            "has_verify_base",
            "has_redirect_base",
            "has_diagnosis_base",
            "has_epistemic_base",
        ]
    ].copy()

    return {
        "rows": int(len(merged)),
        "join_keys": PAIR_KEYS,
        "overall": overall,
        "per_benchmark": per_benchmark,
        "meta_win_rate": float(
            (merged["is_correct_meta"] & ~merged["is_correct_base"]).mean()
        ),
        "base_win_rate": float(
            (~merged["is_correct_meta"] & merged["is_correct_base"]).mean()
        ),
        "tie_rate": float(
            (
                (merged["is_correct_meta"] & merged["is_correct_base"])
                | (~merged["is_correct_meta"] & ~merged["is_correct_base"])
            ).mean()
        ),
        "meta_only_examples": behavior_slice[
            behavior_slice["is_correct_meta"] & ~behavior_slice["is_correct_base"]
        ].head(20).to_dict(orient="records"),
        "base_only_examples": behavior_slice[
            ~behavior_slice["is_correct_meta"] & behavior_slice["is_correct_base"]
        ].head(20).to_dict(orient="records"),
    }


def write_aime_dump(df: pd.DataFrame, out_path: Path) -> None:
    aime = df[df["benchmark"] == "aime2024"].copy().reset_index(drop=True)
    lines: list[str] = []
    for label, sub in [("CORRECT", aime[aime["is_correct"]]), ("WRONG", aime[~aime["is_correct"]])]:
        lines.append(label)
        for _, r in sub.iterrows():
            lines.append(f"QUESTION: {str(r['question']).replace(chr(10), ' ')}")
            lines.append(f"IS_CORRECT: {bool(r['is_correct'])}")
            lines.append(f"AVG_CONFIDENCE: {r['avg_confidence']}")
            lines.append(f"NUM_META_BLOCKS: {r['num_meta_blocks']}")
            lines.append("COMPLETION: " + str(r["completion"])[:2000].replace("\n", "\\n"))
            lines.append("---")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_parquet", default="results/eval_v8_meta_inside_strict_sft/eval_v8_meta_inside_strict_sft.parquet")
    parser.add_argument("--base_parquet", default="results/eval_v8_base_matched_strict_sft/eval_v8_base_matched_strict_sft.parquet")
    parser.add_argument("--output_dir", default="results/strict_pair_analysis")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    meta_raw, meta_source = load_first_readable_parquet(root, args.meta_parquet)
    base_raw, base_source = load_first_readable_parquet(root, args.base_parquet)
    meta_df = add_markers(meta_raw)
    base_df = add_markers(base_raw)

    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    meta_summary = summarize_meta(meta_df)
    base_summary = summarize_base(base_df)
    pair_summary = paired_summary(meta_df, base_df)
    meta_summary["artifact_source"] = meta_source
    base_summary["artifact_source"] = base_source
    pair_summary["artifact_sources"] = {"meta": meta_source, "base": base_source}
    (outdir / "meta_strict_behavior.json").write_text(json.dumps(meta_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (outdir / "base_strict_behavior.json").write_text(json.dumps(base_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (outdir / "paired_strict_behavior.json").write_text(json.dumps(pair_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_aime_dump(meta_df, outdir / "meta_strict_aime_examples.txt")
    print(f"saved {outdir}")


if __name__ == "__main__":
    main()
