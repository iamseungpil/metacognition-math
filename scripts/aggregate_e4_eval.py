"""Aggregate a multi-sample (k>1) eval_vllm_1030 output into per-problem metrics.

eval_vllm_1030.py emits ONE row per (problem, sample) — with k=8 that's k rows per
problem. Its built-in summary averages is_correct over ALL rows (= mean per-sample
accuracy). For the E.4 4-arm comparison we also want, grouped BY PROBLEM:

  - mean_accuracy      : overall mean per-sample is_correct  (same as the built-in)
  - pass_at_k          : fraction of problems with >=1 correct sample (best-of-k)
  - self_consistency   : fraction of problems whose MAJORITY-vote answer is correct
  - ece                : Expected Calibration Error (verbalized conf vs correctness),
                         computed at the sample level via compute_ece (the project helper)
  - overconf_rate      : fraction of samples with conf>=0.5 AND wrong (project north-star)

Reported per-benchmark (gsm8k/math500/aime2024) and overall. Reuses the canonical
compute_ece from src.training.self_distill.eval_metrics so ECE matches prior analyses.

Usage:
  python aggregate_e4_eval.py --parquet <model>.parquet --out <model>.e4summary.json [--arm baseline]
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, "/scratch/metacognition")  # node path; harmless if absent
try:
    from src.training.self_distill.eval_metrics import compute_ece
except Exception:
    def compute_ece(confs, corrects, n_bins: int = 10) -> float:
        confs, corrects = np.asarray(confs, float), np.asarray(corrects, float)
        edges = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            m = (confs >= edges[i]) & (confs < edges[i + 1])
            if i == n_bins - 1:
                m = m | (confs == edges[i + 1])
            if m.any():
                ece += m.mean() * abs(confs[m].mean() - corrects[m].mean())
        return float(ece)


def _block(df: pd.DataFrame) -> dict:
    """Metrics for one slice (a benchmark, or the whole set)."""
    n_rows = len(df)
    if n_rows == 0:
        return {"n_problems": 0, "n_samples": 0}
    corr = df["is_correct"].astype(bool).to_numpy()
    # sample-level ECE on verbalized confidence (rows with a confidence only).
    has_conf = df["avg_confidence"].notna().to_numpy()
    ece = (compute_ece(df.loc[has_conf, "avg_confidence"].to_numpy(float),
                       corr[has_conf]) if has_conf.any() else None)
    overconf = float(((df["avg_confidence"].fillna(0).to_numpy(float) >= 0.5) & (~corr)).mean())
    # per-problem aggregation (group by the exact question string).
    pass_k, selfcons, per_prob_acc = [], [], []
    # group on (benchmark, question) so identical question strings across benchmarks
    # are never collapsed into one "problem" (per-benchmark slices have a constant
    # benchmark, so this is equivalent there and correct for the overall block).
    for _, g in df.groupby(["benchmark", "question"], sort=False):
        gc = g["is_correct"].astype(bool).to_numpy()
        per_prob_acc.append(gc.mean())
        pass_k.append(bool(gc.any()))
        # majority-vote answer correctness: most common extracted answer, is it correct?
        ans = [a for a in g["answer_extracted"].tolist() if a not in (None, "")]
        if ans:
            top = Counter(ans).most_common(1)[0][0]
            # a sample with that answer that is_correct ⇒ majority answer is correct.
            selfcons.append(bool(g.loc[g["answer_extracted"] == top, "is_correct"].any()))
        else:
            selfcons.append(False)
    return {
        "n_problems": len(per_prob_acc),
        "n_samples": n_rows,
        "mean_accuracy": float(corr.mean()),                 # mean per-sample acc
        "pass_at_k": float(np.mean(pass_k)),                 # >=1 of k correct
        "self_consistency": float(np.mean(selfcons)),        # majority-vote correct
        "ece": ece,
        "overconf_rate": overconf,
        "mean_confidence": (float(df["avg_confidence"].mean(skipna=True))
                            if df["avg_confidence"].notna().any() else None),
        "meta_emission_rate": float((df["num_meta_blocks"] > 0).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True, help="eval_vllm_1030 per-sample output parquet")
    ap.add_argument("--out", required=True, help="summary JSON path")
    ap.add_argument("--arm", default=None, help="arm label for the summary")
    args = ap.parse_args()

    df = pd.read_parquet(args.parquet)
    out = {"arm": args.arm, "source_parquet": args.parquet,
           "k_samples": int(df.groupby(["benchmark", "question"]).size().max()) if len(df) else 0,
           "overall": _block(df),
           "per_benchmark": {b: _block(g) for b, g in df.groupby("benchmark", sort=True)}}
    json.dump(out, open(args.out, "w"), indent=2)
    print(json.dumps(out, indent=2))
    print(f"[aggregate] wrote {args.out}")


if __name__ == "__main__":
    main()
