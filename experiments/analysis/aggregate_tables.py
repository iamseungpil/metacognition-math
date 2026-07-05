#!/usr/bin/env python3
"""Merge per-arm eval outputs into the paper tables T1-T4 (one markdown file).

RQ -> table map (4-RQ structure):
  T1 (RQ1)  does PMI-shift actually improve accuracy: per-benchmark + overall
            avg@k for every arm, mean +/- std over SEEDS. The seed x3 and
            token-budget controls are PART OF THE T1 PROTOCOL (rendered as a
            protocol-control sub-table: per-seed accuracy, mean completion
            tokens, meta emission), not a separate RQ. Raw Qwen3-8B and
            SFT-only arms are REFERENCE ROWS in the same table (selected by
            --reference-arms) so the SFT capability cost is readable in place.
  T2 (RQ2)  what is the effect and where does it come from: Gandhi-arm
            (meta-SFT init + VANILLA_GRPO, --t2-arms) decomposes SFT-priming
            vs RL-reward contributions, plus SAVE/DERAIL flips from flip.py
            JSON summaries (--flip), real vs shuffled-meta placebo side by side.
  T3 (RQ3)  how the meta effect varies with difficulty / problem type / OOD:
            difficulty-quartile stratified accuracy + meta emission (same
            pooled binning as stratify.py) and the OOD-vs-ID accuracy gap
            (aime2024 vs gsm8k+math500).
  T4 (RQ4)  calibration: ECE(15-bin)/Brier/overconfidence per arm, from stated
            meta-block confidence, seeds pooled. Pure calibration only — the
            OOD split lives in T3. Accuracy stays the primary metric.

Script -> table map: flip.py + placebo.py feed T2, stratify.py is the
standalone T3 tool (its quartile binning is imported here), calibration.py is
the standalone T4 tool; this script renders ALL of T1-T4 in one report.

Metric rules enforced here:
  - accuracy is re-graded with math_verify (pass --no-regrade only for smoke
    tests; the stored is_correct comes from the broken check_correctness path)
  - avg@k = per-question mean over samples, then mean over questions
  - meta emission counts CLOSED <|meta|>...<|/meta|> blocks only
  - never compare arms on val-core/reward-style composites — this script only
    reads correctness

Usage:
  python experiments/analysis/aggregate_tables.py \\
      --run arm=base,seed=42,path=results/eval_1030_base_grpo_step300_16k/base_grpo_step300_16k.parquet \\
      --run arm=pmishift,seed=42,path=results/eval_1030_pmishift_gs300/pmishift_gs300.parquet \\
      --run arm=pmishift,seed=43,path=results/eval_1030_pmishift_gs300_s43/pmishift_gs300_s43.parquet \\
      --flip real=results/analysis/flip_meta_gs300.json \\
      --flip placebo=results/analysis/flip_placebo_gs300.json \\
      --out results/analysis/tables_t1_t4.md

Accepts parquet / json / jsonl outputs of scripts/eval_vllm_1030.py and
src/eval/eval_hf.py. Tables whose inputs are not provided yet (e.g. no Gandhi
run exists) render with an explicit "no runs matched/provided" line so the
report stays honest about coverage.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.analysis.analysis_common import (  # noqa: E402
    BENCH_ORDER,
    Grader,
    OOD_BENCHMARKS,
    brier,
    ece_15bin,
    fmt_pct,
    load_eval_frame,
    md_table,
    overconfidence_rate,
    parse_arm_specs,  # noqa: F401  (kept import surface consistent across scripts)
    resolve_correct,
    stated_confidences,
)
from experiments.analysis.stratify import assign_quartiles  # noqa: E402

HIGH_CONF = 0.8
N_QUARTILES = 4


def parse_run_spec(spec: str) -> dict:
    """Parse 'arm=NAME,seed=INT,path=FILE' into a dict."""
    kv = dict(part.split("=", 1) for part in spec.split(",") if "=" in part)
    missing = [k for k in ("arm", "seed", "path") if k not in kv]
    if missing:
        raise ValueError(f"--run '{spec}' missing keys: {missing}")
    kv["seed"] = int(kv["seed"])
    return kv


def bench_list(df: pd.DataFrame) -> list[str]:
    """Benchmarks in canonical order, then any extras alphabetically."""
    present = set(df["benchmark"].unique())
    ordered = [b for b in BENCH_ORDER if b in present]
    return ordered + sorted(present - set(ordered))


def avg_at_k(sub: pd.DataFrame) -> float:
    """Per-question mean over samples, then mean over questions."""
    if len(sub) == 0:
        return float("nan")
    return float(sub.groupby("qid")["correct"].mean().mean())


def mean_std(vals: list[float]) -> str:
    """'71.3% +/- 1.2' over seeds; plain mean when only one seed exists."""
    vals = [v for v in vals if not np.isnan(v)]
    if not vals:
        return "n/a"
    m = float(np.mean(vals))
    if len(vals) < 2:
        return fmt_pct(m)
    return f"{fmt_pct(m)} ±{100.0 * float(np.std(vals, ddof=1)):.1f}"


def accuracy_table(df: pd.DataFrame, arms: list[str]) -> str:
    """Benchmarks x arms accuracy table, mean +/- std across seeds (avg@k)."""
    benches = bench_list(df) + ["overall_1030"]
    header = ["benchmark"] + arms
    rows = []
    for bench in benches:
        row = [bench]
        for arm in arms:
            adf = df[df["arm"] == arm]
            per_seed = []
            for seed in sorted(adf["seed"].unique()):
                sdf = adf[adf["seed"] == seed]
                sub = sdf if bench == "overall_1030" else sdf[sdf["benchmark"] == bench]
                per_seed.append(avg_at_k(sub))
            row.append(mean_std(per_seed))
        rows.append(row)
    return md_table(header, rows)


def flip_table(flip_specs: list[tuple[str, str]]) -> str:
    """T2 mechanism body from flip.py --out JSON files (label=path pairs)."""
    if not flip_specs:
        return "_No flip.py summaries provided (--flip label=path)._"
    header = ["run", "meta rows", "committed", "SAVE", "DERAIL", "net SAVE"]
    rows = []
    for label, path in flip_specs:
        with open(path) as f:
            payload = json.load(f)
        for file_key, summary in payload.items():
            p = summary["pooled"]
            rows.append([
                f"{label} ({Path(file_key).stem})",
                p["n_meta_rows"], p["n_committed_prefix"],
                fmt_pct(p["save_rate"]), fmt_pct(p["derail_rate"]),
                fmt_pct(p["net_save_rate"]),
            ])
    return md_table(header, rows)


def protocol_controls_table(df: pd.DataFrame, arms: list[str]) -> str:
    """T1 protocol-control sub-table: per-seed accuracy, token budget, emission.

    Part of T1 by design: the seed x3 spread and the response-token control
    column are protocol requirements of the main comparison, not a separate RQ.
    """
    header = ["arm", "seeds", "overall acc per seed", "mean±std",
              "mean completion tokens (budget control)", "meta emission"]
    rows = []
    for arm in arms:
        adf = df[df["arm"] == arm]
        seeds = sorted(adf["seed"].unique())
        per_seed = [avg_at_k(adf[adf["seed"] == s]) for s in seeds]
        tokens = adf["completion_length_tokens"].mean()
        rows.append([
            arm,
            ", ".join(str(s) for s in seeds),
            ", ".join(fmt_pct(v) for v in per_seed),
            mean_std(per_seed),
            "n/a" if np.isnan(tokens) else f"{tokens:.0f}",
            fmt_pct((adf["num_meta_blocks_closed"] > 0).mean()),
        ])
    return md_table(header, rows)


def stratified_table(df: pd.DataFrame, arms: list[str]) -> str:
    """T3 body: difficulty-quartile accuracy + meta emission per arm.

    Difficulty = 1 - pooled accuracy over all arms/seeds/samples (same binning
    as stratify.py, imported from there); pooling guards against the
    documented Simpson/selection artifact.
    """
    pooled = df.copy()
    quartile_of = assign_quartiles(pooled, N_QUARTILES)
    pooled["quartile"] = pooled["qid"].map(quartile_of)
    labels = [f"Q{i + 1}" for i in range(N_QUARTILES)]
    header = ["quartile", "n questions"]
    for arm in arms:
        header += [f"{arm} acc", f"{arm} meta%"]
    rows = []
    for label in labels + ["ALL"]:
        sub = pooled if label == "ALL" else pooled[pooled["quartile"] == label]
        row = [label, int(sub["qid"].nunique())]
        for arm in arms:
            asub = sub[sub["arm"] == arm]
            acc = asub["correct"].mean() if len(asub) else float("nan")
            meta = (
                (asub["num_meta_blocks_closed"] > 0).mean()
                if len(asub) else float("nan")
            )
            row += [fmt_pct(acc), fmt_pct(meta)]
        rows.append(row)
    return md_table(header, rows)


def ood_table(df: pd.DataFrame, arms: list[str]) -> str:
    """T3 body: OOD-vs-ID accuracy gap (aime2024 vs gsm8k+math500)."""
    header = ["arm", "ID acc", "OOD acc (aime)", "OOD-ID gap"]
    rows = []
    for arm in arms:
        adf = df[df["arm"] == arm]
        id_df = adf[~adf["benchmark"].isin(OOD_BENCHMARKS)]
        ood_df = adf[adf["benchmark"].isin(OOD_BENCHMARKS)]
        id_acc, ood_acc = avg_at_k(id_df), avg_at_k(ood_df)
        gap = ood_acc - id_acc if not (np.isnan(id_acc) or np.isnan(ood_acc)) else float("nan")
        rows.append([
            arm, fmt_pct(id_acc), fmt_pct(ood_acc),
            "n/a" if np.isnan(gap) else f"{100.0 * gap:+.1f}pp",
        ])
    return md_table(header, rows)


def calibration_table(df: pd.DataFrame, arms: list[str]) -> str:
    """T4 body: pooled calibration only (seeds pooled); OOD split lives in T3."""
    header = ["arm", "coverage", "ECE(15)", "Brier", f"overconf@{HIGH_CONF:g}"]
    rows = []
    for arm in arms:
        adf = df[df["arm"] == arm].copy()
        # last stated confidence per response (closed meta blocks only)
        confs = adf["completion"].map(
            lambda t: (stated_confidences(t) or [None])[-1]
        )
        stated = confs.notna()
        conf = confs[stated].to_numpy(dtype=float)
        corr = adf.loc[stated, "correct"].to_numpy(dtype=bool)
        rows.append([
            arm,
            fmt_pct(stated.mean()),
            "n/a" if len(conf) == 0 else f"{ece_15bin(conf, corr):.3f}",
            "n/a" if len(conf) == 0 else f"{brier(conf, corr):.3f}",
            fmt_pct(overconfidence_rate(conf, corr, HIGH_CONF)),
        ])
    return md_table(header, rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--run", action="append", required=True,
        metavar="arm=NAME,seed=INT,path=FILE",
        help="one eval run (repeatable; several seeds per arm feed the T1 "
             "protocol controls, mean +/- std)",
    )
    parser.add_argument(
        "--flip", action="append", default=[], metavar="LABEL=PATH",
        help="flip.py --out JSON for T2 (repeatable; pass real and placebo)",
    )
    parser.add_argument(
        "--reference-arms", default=r"raw|sft.?only|ref0|nosft",
        help="regex selecting the T1 REFERENCE rows (raw Qwen3-8B / SFT-only / "
             "no-SFT REF-0); they render last in T1 so the SFT capability "
             "cost is readable in the main table",
    )
    parser.add_argument(
        "--t2-arms", default=r"gandhi",
        help="regex selecting the RQ2 decomposition arms (Gandhi arm = "
             "meta-SFT init then VANILLA_GRPO)",
    )
    parser.add_argument(
        "--no-regrade", action="store_true",
        help="trust stored is_correct instead of math_verify regrade (NOT for paper)",
    )
    parser.add_argument("--out", required=True, help="output markdown path")
    args = parser.parse_args()

    grader = Grader()
    frames = []
    for spec in args.run:
        run = parse_run_spec(spec)
        df = load_eval_frame(run["path"])
        df["arm"] = run["arm"]
        df["seed"] = run["seed"]
        df["correct"] = resolve_correct(df, args.no_regrade, "aggregate", grader)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)

    arms = sorted(df["arm"].unique())
    ref_arms = [a for a in arms if re.search(args.reference_arms, a, re.IGNORECASE)]
    main_arms = [a for a in arms if a not in ref_arms]
    t1_arms = main_arms + ref_arms  # reference rows render last
    t2_arms = [a for a in arms if re.search(args.t2_arms, a, re.IGNORECASE)]
    flip_specs = [tuple(s.split("=", 1)) for s in args.flip]

    grading_note = (
        "STORED is_correct (--no-regrade, smoke only)"
        if args.no_regrade else "math_verify regrade"
    )
    ref_note = (
        "reference rows (raw / SFT-only / no-SFT REF-0): " + ", ".join(ref_arms)
        if ref_arms else
        f"no reference rows provided (--reference-arms /{args.reference_arms}/)"
    )
    sections = [
        "# T1-T4 result tables",
        "",
        f"- grading: {grading_note}; accuracy = avg@k "
        "(per-question mean over samples, then mean over questions)",
        "- meta emission counts CLOSED <|meta|>...<|/meta|> blocks only",
        "- runs: " + "; ".join(args.run),
        "",
        "## T1 (RQ1) — main accuracy: does PMI-shift improve accuracy?",
        "",
        f"- {ref_note}",
        "",
        accuracy_table(df, t1_arms),
        "",
        "### T1 protocol controls (seeds x3 mean±std, token budget) — "
        "built into T1, not a separate RQ",
        "",
        protocol_controls_table(df, t1_arms),
        "",
        "## T2 (RQ2) — what is the effect and where does it come from?",
        "",
        "### SFT-priming vs RL-reward decomposition (Gandhi arm)",
        "",
        accuracy_table(df, t2_arms) if t2_arms
        else f"_No runs matched --t2-arms /{args.t2_arms}/._",
        "",
        "### Mechanism: SAVE/DERAIL flips, real vs shuffled-meta placebo",
        "",
        flip_table(flip_specs),
        "",
        "## T3 (RQ3) — difficulty / problem-type / OOD stratification",
        "",
        "### Difficulty-quartile accuracy + meta emission "
        "(pooled bins, Simpson guard)",
        "",
        stratified_table(df, t1_arms),
        "",
        "### OOD robustness (aime2024 vs gsm8k+math500)",
        "",
        ood_table(df, t1_arms),
        "",
        "## T4 (RQ4) — calibration (secondary metric; accuracy is primary)",
        "",
        calibration_table(df, t1_arms),
        "",
    ]
    text = "\n".join(sections)
    print(text)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(f"Wrote markdown: {args.out}")


if __name__ == "__main__":
    main()
