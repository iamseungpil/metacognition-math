#!/usr/bin/env python3
"""Paired held-out eval -> T1/T3 result tables (single entry point).

This is the one script that turns the downloaded held-out eval parquets
(from HF model repo iamseungpil/metacot-h200-triobj-dcpo-v3, laid out under
eval/pmishift_1030_v2/ and eval/base_matched_1030_v2/) into the paper's T1
(main accuracy) and T3-style (per benchmark x budget) tables, with a paired
significance test between the pmishift and base arms.

File-name contract (glob `<arm>_gs300_*.parquet` inside --eval-dir):
  pass (a) all benchmarks @4k n=8 seed42   -> <arm>_gs300_4k_n8.parquet
  pass (b) @16k n=8 seed42 SPLIT per bench -> <arm>_gs300_16k_n8_{aime2024,math500,gsm8k}.parquet
  pass (c) aime-only @16k n=8 seed43       -> <arm>_gs300_16k_n8_seed43_aime.parquet
So AIME @16k has 16 samples/problem (8 seed42 from pass b + 8 seed43 from pass c),
combined as avg@16 by unioning the two files (no double count); AIME @4k has 8;
MATH500 / GSM8K have 8 per budget. The script is robust to only SOME passes or
only ONE arm being present (partial-pass / single-arm modes never crash).

Metric rules enforced here (see analysis_common):
  - accuracy = avg@k: per-problem mean over the k samples, then macro over
    problems within a (arm, benchmark, budget)
  - AIME @16k unions seed42+seed43 -> avg@16 by question (union of physical
    samples; each parquet row is one distinct sample, so grouping by qid and
    averaging correctness cannot double-count)
  - grading is RE-DONE offline with the robust math_verify grader via
    analysis_common.resolve_correct; the parquet is_correct column (the
    documented-broken runtime grader) is kept only to print the audit delta
  - truncation rate = mean(finish_reason == "length"); meta emission = mean of
    CLOSED <|meta|>...<|/meta|> blocks > 0

Paired significance (only when BOTH arms present for a benchmark x budget):
  - per-problem paired bootstrap over the SHARED problems (10k resamples,
    resample PROBLEMS, statistic = mean(avg@k_pmishift - avg@k_base)), 95% CI
  - McNemar on per-problem majority-correct (>= ceil(k/2)) 2x2 (exact two-sided
    binomial; uses scipy if available, else a pure-python inline binomial)

Usage:
  python experiments/analysis/paired_eval.py \\
      --eval-dir <dir of downloaded parquets> \\
      [--arms pmishift base] \\
      [--out results/analysis/paired_eval.md]

Writes the markdown to --out (default results/analysis/paired_eval.md) and a
machine-readable sidecar <out>.json with every number for aggregate_tables.py /
the paper to consume.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.analysis.analysis_common import (  # noqa: E402
    BENCH_ORDER,
    Grader,
    fmt_pct,
    load_eval_frame,
    md_table,
    resolve_correct,
)

# Preferred arm ordering: pmishift is the treatment column, base the control.
ARM_ORDER = ["pmishift", "base"]
BUDGET_ORDER = ["4k", "16k"]
# Map every benchmark token that can appear in a filename to its canonical
# name. Pass-c files use the short token `aime` (e.g. `..._seed43_aime.parquet`)
# while the canonical benchmark column is `aime2024`; without this alias the
# file-name pin would be lost (benchmark=None), disabling the per-benchmark
# safety filter and mislabeling coverage. Longer tokens are matched first so
# `aime2024` wins over `aime` when both appear.
BENCH_ALIASES = {
    "aime2024": "aime2024",
    "aime": "aime2024",
    "math500": "math500",
    "gsm8k": "gsm8k",
}
DEFAULT_OUT = "results/analysis/paired_eval.md"
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 12345


# ── filename parsing ─────────────────────────────────────────────────────────

def parse_eval_filename(path: Path) -> dict | None:
    """Parse `<arm>_gs300_<...>.parquet` into {arm, budget, seed, benchmark?}.

    Returns None for files that do not match the `_gs300_` contract. `arm` is
    everything before the first `_gs300`; `budget` is the `\\d+k` token; `seed`
    defaults to 42 unless a `seed<NN>` token is present (pass c). `benchmark`
    is filled when the file name pins a single benchmark (pass b / pass c),
    else None (pass a mixes all benchmarks in one file).
    """
    name = path.name[: -len(".parquet")] if path.name.endswith(".parquet") else path.name
    if "_gs300" not in name:
        return None
    arm, _, rest = name.partition("_gs300")
    if not arm:
        return None
    budget_m = re.search(r"(\d+k)", rest)
    budget = budget_m.group(1) if budget_m else "unknown"
    seed_m = re.search(r"seed(\d+)", rest)
    seed = int(seed_m.group(1)) if seed_m else 42
    bench = None
    # Longest tokens first so `aime2024` is preferred over its `aime` alias.
    for token in sorted(BENCH_ALIASES, key=len, reverse=True):
        if token in rest:
            bench = BENCH_ALIASES[token]
            break
    return {"arm": arm, "budget": budget, "seed": seed, "benchmark": bench}


def discover_frames(eval_dir: Path, arms: list[str] | None, grader: Grader,
                    no_regrade: bool) -> tuple[pd.DataFrame, list[dict]]:
    """Load every `*_gs300_*.parquet` under eval_dir, tag rows, re-grade.

    Returns (long_df, manifest). `manifest` lists what was loaded (one entry
    per file) so partial-pass / single-arm runs report their coverage honestly.
    """
    paths = sorted(Path(eval_dir).glob("*_gs300_*.parquet"))
    frames: list[pd.DataFrame] = []
    manifest: list[dict] = []
    for path in paths:
        meta = parse_eval_filename(path)
        if meta is None:
            continue
        if arms is not None and meta["arm"] not in arms:
            continue
        df = load_eval_frame(path)
        # A per-benchmark file (pass b/c) should only carry its one benchmark;
        # guard against accidental extras so the union stays clean.
        if meta["benchmark"] is not None:
            df = df[df["benchmark"] == meta["benchmark"]].copy()
        df["arm"] = meta["arm"]
        df["budget"] = meta["budget"]
        df["seed"] = meta["seed"]
        df["correct"] = resolve_correct(df, no_regrade, "paired_eval", grader)
        df["runtime_correct"] = df["is_correct"].astype(bool)
        if "finish_reason" in df.columns:
            df["truncated"] = df["finish_reason"].astype(str) == "length"
        else:
            df["truncated"] = False
        df["has_meta"] = df["num_meta_blocks_closed"] > 0
        df["source_file"] = str(path)
        frames.append(df)
        manifest.append({
            "file": path.name, "arm": meta["arm"], "budget": meta["budget"],
            "seed": meta["seed"], "benchmark": meta["benchmark"],
            "n_rows": int(len(df)),
        })
    if not frames:
        return pd.DataFrame(), manifest
    return pd.concat(frames, ignore_index=True), manifest


# ── per-cell metrics ─────────────────────────────────────────────────────────

def per_problem_acc(sub: pd.DataFrame, col: str = "correct") -> pd.Series:
    """Per-problem mean correctness (avg@k for that problem) indexed by qid.

    Grouping by qid unions all physical samples for the problem, so AIME @16k
    (seed42 + seed43 rows in the same slice) becomes avg@16 with no double
    counting.
    """
    return sub.groupby("qid")[col].mean()


def cell_metrics(sub: pd.DataFrame) -> dict:
    """All reported numbers for one (arm, benchmark, budget) slice."""
    if len(sub) == 0:
        return {}
    per_q = per_problem_acc(sub, "correct")
    per_q_rt = per_problem_acc(sub, "runtime_correct")
    counts = sub.groupby("qid").size()
    tokens = sub["completion_length_tokens"].astype(float)
    return {
        "n_problems": int(per_q.shape[0]),
        "n_rows": int(len(sub)),
        "k": int(counts.median()),            # samples per problem (typical)
        "k_min": int(counts.min()),
        "k_max": int(counts.max()),
        "seeds": sorted(int(s) for s in sub["seed"].unique()),
        "acc_robust": float(per_q.mean()),    # macro over problems, avg@k
        "acc_runtime": float(per_q_rt.mean()),
        "acc_delta_rt_minus_robust": float(per_q_rt.mean() - per_q.mean()),
        "truncation_rate": float(sub["truncated"].mean()),
        "mean_tokens": (float(tokens.mean()) if tokens.notna().any()
                        else float("nan")),
        "meta_emission_rate": float(sub["has_meta"].mean()),
    }


def build_cells(df: pd.DataFrame) -> dict:
    """Nested {arm: {benchmark: {budget: cell_metrics}}} over present slices."""
    cells: dict = {}
    for arm in df["arm"].unique():
        cells[arm] = {}
        adf = df[df["arm"] == arm]
        for bench in adf["benchmark"].unique():
            cells[arm][bench] = {}
            bdf = adf[adf["benchmark"] == bench]
            for budget in bdf["budget"].unique():
                sub = bdf[bdf["budget"] == budget]
                cells[arm][bench][budget] = cell_metrics(sub)
    return cells


# ── paired significance ──────────────────────────────────────────────────────

def paired_bootstrap(diff: np.ndarray, n_boot: int, seed: int) -> dict:
    """Bootstrap over PROBLEMS. `diff` = per-problem (avg@k_A - avg@k_B).

    Statistic = mean(diff). Returns the point effect, 95% percentile CI, and a
    two-sided bootstrap p-value (share of resample means on the far side of 0).
    """
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    if n == 0:
        return {"n_problems": 0, "effect": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan"),
                "bootstrap_p": float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = diff[idx].mean(axis=1)
    effect = float(diff.mean())
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    # Two-sided bootstrap p: twice the smaller tail mass across 0.
    frac_le = float(np.mean(boot_means <= 0.0))
    frac_ge = float(np.mean(boot_means >= 0.0))
    p = min(1.0, 2.0 * min(frac_le, frac_ge))
    return {"n_problems": n, "effect": effect,
            "ci95_low": float(lo), "ci95_high": float(hi),
            "bootstrap_p": p}


def _binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial p for k successes of n trials at p=0.5.

    Prefers scipy.stats.binomtest; falls back to a pure-python exact sum of
    all outcomes no more probable than the observed one (symmetric at p=0.5).
    """
    if n == 0:
        return float("nan")
    try:
        from scipy.stats import binomtest  # type: ignore
        return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)
    except Exception:
        # Symmetric at p=0.5: two-sided = 2 * lower tail up to min(k, n-k).
        m = min(k, n - k)
        tail = sum(math.comb(n, i) for i in range(m + 1)) * (0.5 ** n)
        return float(min(1.0, 2.0 * tail))


def mcnemar(major_a: np.ndarray, major_b: np.ndarray) -> dict:
    """McNemar on per-problem majority-correct booleans, arms aligned by qid.

    b = A right & B wrong, c = A wrong & B right. Exact two-sided binomial on
    the discordant pairs (min(b, c) of b + c at p=0.5).
    """
    a = np.asarray(major_a, dtype=bool)
    b_arr = np.asarray(major_b, dtype=bool)
    b = int(np.sum(a & ~b_arr))   # A-only correct
    c = int(np.sum(~a & b_arr))   # B-only correct
    n_disc = b + c
    p = _binom_two_sided_p(min(b, c), n_disc) if n_disc > 0 else float("nan")
    return {"n_pairs": int(len(a)), "b_A_only": b, "c_B_only": c,
            "n_discordant": n_disc, "p_value": p}


def majority_correct(sub: pd.DataFrame) -> pd.Series:
    """Per-problem majority-correct (>= ceil(k/2)) indexed by qid."""
    grp = sub.groupby("qid")["correct"]
    n_correct = grp.sum()
    n_total = grp.size()
    thresh = np.ceil(n_total / 2.0)
    return (n_correct >= thresh)


def paired_stats(df: pd.DataFrame, arm_a: str, arm_b: str,
                 bench: str, budget: str) -> dict | None:
    """Bootstrap + McNemar for one (bench, budget) when both arms are present."""
    sa = df[(df["arm"] == arm_a) & (df["benchmark"] == bench) & (df["budget"] == budget)]
    sb = df[(df["arm"] == arm_b) & (df["benchmark"] == bench) & (df["budget"] == budget)]
    if len(sa) == 0 or len(sb) == 0:
        return None
    # METRIC RULE: both arms must be compared at IDENTICAL (benchmark, budget,
    # k, seed set). A partial-download state can leave one arm at avg@16
    # (both passes present) and the other at avg@8 (only one pass), which are
    # different estimators — comparing them would fabricate a biased effect.
    # Refuse and surface a loud reason instead of a spurious p-value.
    k_a = int(sa.groupby("qid").size().median())
    k_b = int(sb.groupby("qid").size().median())
    seeds_a = sorted(int(s) for s in sa["seed"].unique())
    seeds_b = sorted(int(s) for s in sb["seed"].unique())
    if k_a != k_b or seeds_a != seeds_b:
        return {
            "arm_a": arm_a, "arm_b": arm_b, "benchmark": bench, "budget": budget,
            "skipped_reason": (
                f"k/seed-set mismatch (kA={k_a}, kB={k_b}, "
                f"seedsA={seeds_a}, seedsB={seeds_b})"
            ),
        }
    acc_a = per_problem_acc(sa, "correct")
    acc_b = per_problem_acc(sb, "correct")
    shared = sorted(set(acc_a.index) & set(acc_b.index))
    if not shared:
        return None
    diff = (acc_a.loc[shared] - acc_b.loc[shared]).to_numpy(dtype=float)
    maj_a = majority_correct(sa).reindex(shared).fillna(False)
    maj_b = majority_correct(sb).reindex(shared).fillna(False)
    return {
        "arm_a": arm_a, "arm_b": arm_b, "benchmark": bench, "budget": budget,
        "n_shared_problems": len(shared),
        "bootstrap": paired_bootstrap(diff, N_BOOTSTRAP, BOOTSTRAP_SEED),
        "mcnemar": mcnemar(maj_a.to_numpy(), maj_b.to_numpy()),
    }


# ── ordering helpers ─────────────────────────────────────────────────────────

def order_arms(present: list[str]) -> list[str]:
    """Preferred arms first (pmishift, base), then any extras alphabetically."""
    ranked = [a for a in ARM_ORDER if a in present]
    return ranked + sorted(a for a in present if a not in ranked)


def order_benches(present) -> list[str]:
    ordered = [b for b in BENCH_ORDER if b in present]
    return ordered + sorted(b for b in present if b not in ordered)


def order_budgets(present) -> list[str]:
    ordered = [b for b in BUDGET_ORDER if b in present]
    return ordered + sorted(b for b in present if b not in ordered)


def _get(cells: dict, arm: str, bench: str, budget: str) -> dict:
    return cells.get(arm, {}).get(bench, {}).get(budget, {})


# ── table rendering ──────────────────────────────────────────────────────────

def t1_accuracy_table(cells: dict, arms: list[str], benches: list[str],
                      budgets: list[str]) -> str:
    """T1: arms as columns, (benchmark x budget) rows, robust avg@k + rt delta.

    Each arm gets a robust-accuracy column and a `(rt-Δ)` column showing
    runtime_acc - robust_acc so the grader fix stays auditable in the table.
    """
    header = ["benchmark", "budget", "k"]
    for arm in arms:
        header += [f"{arm} avg@k", f"{arm} rtΔ"]
    rows = []
    for bench in benches:
        for budget in budgets:
            present = [_get(cells, a, bench, budget) for a in arms]
            if not any(present):
                continue
            # Show each present arm's own k; a single shared k would let a
            # reader mistake avg@8 and avg@16 columns for the same estimator.
            ks = [m.get("k") for m in present if m]
            if not ks:
                k_cell = "-"
            elif len(set(ks)) == 1:
                k_cell = ks[0]
            else:
                k_cell = "⚠ " + "/".join(str(k) for k in ks)
            row = [bench, budget, k_cell]
            for m in present:
                if m:
                    row += [fmt_pct(m["acc_robust"]),
                            f"{100.0 * m['acc_delta_rt_minus_robust']:+.1f}pp"]
                else:
                    row += ["—", "—"]
            rows.append(row)
    if not rows:
        return "_No (benchmark x budget) cells present._"
    return md_table(header, rows)


def details_table(cells: dict, arms: list[str], benches: list[str],
                  budgets: list[str]) -> str:
    """Per (arm, benchmark, budget): truncation, mean tokens, meta emission."""
    header = ["arm", "benchmark", "budget", "k", "n prob", "avg@k (robust)",
              "avg@k (runtime)", "trunc%", "mean tokens", "meta%"]
    rows = []
    for arm in arms:
        for bench in benches:
            for budget in budgets:
                m = _get(cells, arm, bench, budget)
                if not m:
                    continue
                rows.append([
                    arm, bench, budget, m["k"], m["n_problems"],
                    fmt_pct(m["acc_robust"]), fmt_pct(m["acc_runtime"]),
                    fmt_pct(m["truncation_rate"]),
                    "n/a" if np.isnan(m["mean_tokens"]) else f"{m['mean_tokens']:.0f}",
                    fmt_pct(m["meta_emission_rate"]),
                ])
    if not rows:
        return "_No cells present._"
    return md_table(header, rows)


def significance_table(pairs: list[dict]) -> str:
    """Paired bootstrap effect + 95% CI and McNemar p, one row per cell."""
    if not pairs:
        return ("_Paired significance needs BOTH arms for a benchmark x budget; "
                "none present._")
    header = ["benchmark", "budget", "n shared", "effect (A-B)", "95% CI",
              "boot p", "McNemar b/c", "McNemar p"]
    rows = []
    for pr in pairs:
        if "skipped_reason" in pr:
            rows.append([
                pr["benchmark"], pr["budget"], "—",
                f"⚠ SKIPPED: {pr['skipped_reason']}", "—", "—", "—", "—",
            ])
            continue
        bs, mc = pr["bootstrap"], pr["mcnemar"]
        rows.append([
            pr["benchmark"], pr["budget"], pr["n_shared_problems"],
            f"{100.0 * bs['effect']:+.1f}pp",
            f"[{100.0 * bs['ci95_low']:+.1f}, {100.0 * bs['ci95_high']:+.1f}]pp",
            f"{bs['bootstrap_p']:.3f}",
            f"{mc['b_A_only']}/{mc['c_B_only']}",
            "n/a" if np.isnan(mc["p_value"]) else f"{mc['p_value']:.3f}",
        ])
    return md_table(header, rows)


def manifest_table(manifest: list[dict]) -> str:
    if not manifest:
        return "_No `*_gs300_*.parquet` files matched in --eval-dir._"
    header = ["file", "arm", "budget", "seed", "benchmark", "rows"]
    rows = [[m["file"], m["arm"], m["budget"], m["seed"],
             m["benchmark"] or "(all)", m["n_rows"]] for m in manifest]
    return md_table(header, rows)


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--eval-dir", required=True,
        help="directory of downloaded <arm>_gs300_*.parquet held-out eval files",
    )
    parser.add_argument(
        "--arms", nargs="*", default=None,
        help="restrict to these arm names (default: every arm discovered; "
             "the paper pair is `pmishift base`)",
    )
    parser.add_argument(
        "--no-regrade", action="store_true",
        help="trust stored is_correct instead of math_verify regrade (NOT for paper)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="output markdown path")
    args = parser.parse_args()

    grader = Grader()
    df, manifest = discover_frames(Path(args.eval_dir), args.arms, grader,
                                   args.no_regrade)

    if df.empty:
        msg = (f"No `*_gs300_*.parquet` eval files found under {args.eval_dir}"
               + (f" for arms {args.arms}" if args.arms else "") + ".")
        print(msg, file=sys.stderr)
        # Still emit an (empty) report/JSON so downstream consumers see the run.
        _write_outputs(args.out, args, manifest, {}, [], [], [], [],
                       note=msg)
        return

    cells = build_cells(df)
    arms = order_arms(list(df["arm"].unique()))
    benches = order_benches(set(df["benchmark"].unique()))
    budgets = order_budgets(set(df["budget"].unique()))

    # Paired stats between the first two arms in preferred order (pmishift vs
    # base), only where both arms cover the (benchmark, budget) cell.
    pairs: list[dict] = []
    if len(arms) >= 2:
        arm_a, arm_b = arms[0], arms[1]
        for bench in benches:
            for budget in budgets:
                pr = paired_stats(df, arm_a, arm_b, bench, budget)
                if pr is not None:
                    pairs.append(pr)

    _write_outputs(args.out, args, manifest, cells, arms, benches, budgets,
                   pairs, note=None)


def _write_outputs(out_path: str, args, manifest, cells, arms, benches,
                   budgets, pairs, note) -> None:
    """Render markdown + JSON sidecar and print the markdown to stdout."""
    grading_note = ("STORED is_correct (--no-regrade, smoke only)"
                    if args.no_regrade else "math_verify robust regrade")
    sections = [
        "# Paired held-out eval — T1 (accuracy) + T3-style tables",
        "",
        f"- eval-dir: {args.eval_dir}",
        f"- grading: {grading_note}; accuracy = avg@k "
        "(per-problem mean over samples, then macro over problems)",
        "- AIME @16k = union of seed42 (pass b) + seed43 (pass c) = avg@16 by "
        "question (no double count)",
        "- meta emission counts CLOSED <|meta|>...<|/meta|> blocks only; "
        "truncation = finish_reason == \"length\"",
        "",
        "## Coverage (what was loaded)",
        "",
        manifest_table(manifest),
        "",
    ]
    if note:
        sections += [f"**{note}**", ""]
    if arms:
        sections += [
            "## T1 — main accuracy (robust avg@k; rtΔ = runtime − robust)",
            "",
            t1_accuracy_table(cells, arms, benches, budgets),
            "",
            "## Per-cell details (truncation, tokens, meta emission)",
            "",
            details_table(cells, arms, benches, budgets),
            "",
            "## Paired significance (pmishift − base, shared problems)",
            "",
            significance_table(pairs),
            "",
        ]
    text = "\n".join(sections)
    print(text)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Wrote markdown: {out}")

    # Machine-readable sidecar: every number aggregate_tables / the paper needs.
    payload = {
        "eval_dir": str(args.eval_dir),
        "grading": "runtime_is_correct" if args.no_regrade else "math_verify_robust",
        "arms": arms,
        "benchmarks": benches,
        "budgets": budgets,
        "manifest": manifest,
        "cells": cells,
        "paired": pairs,
    }
    json_path = out.with_suffix(out.suffix + ".json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote JSON: {json_path}")


if __name__ == "__main__":
    main()
