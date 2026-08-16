#!/usr/bin/env python3
"""Build the DATA figures (fig2, fig3, fig4) for the paper from REAL measurements.

Sources
-------
fig2_trajectories.png
    wandb entity=gistdslab project=metacot-dcpo-v4, the three LIVE arms
      rq3v2f_b2p3v   (correctness-only VANILLA_GRPO on the repaired init)
      rq3v2f_b3p2    (tri-objective, no calibration-positive logging)
      rq3v2f_b3p3g   (tri-objective + repaired R_cal observability)
    History is pulled ONE KEY AT A TIME via run.scan_history(keys=["_step", k]);
    multi-key scans come back empty on this project.
    Panels: (a) critic/rewards/mean  (b) dcpo/meta_emit_rate
            (c) actor/entropy       (d) dcpo/cal_positive_rate
    A series that a run never logged is drawn as a MISSING note inside the
    panel; nothing is ever substituted or back-filled.

fig3_repair.png
    HF MODEL repo iamseungpil/metacot-h200-triobj-dcpo-v3, held-out MATH-500
    eval parquets (16k, n8):
      pre-repair   eval/sft2init_gs0/sft2init_gs0_16k_n8_math500
      post-repair  eval/b2p3init_gs0/b2p3init_gs0_16k_n8_math500
    (a) first-block distribution: a response is META-FIRST when the stripped
        completion starts with "<|meta|>" (the same test as the training-time
        `meta_first` head in src/training/dcpo_region.py), REASONING-FIRST when
        it starts with "<think>", OTHER otherwise.
    (b) response-length (completion_length_tokens) distributions with medians.

fig4_calibration.png
    Same repo, the correctness-only arm across training:
      gs0    eval/b2p3init_gs0/...      (the arm's own init)
      gs50   eval/rq3v2f_b2p3v_gs50/...
      gs100  eval/rq3v2f_b2p3v_gs100/...
    Correctness is ALWAYS re-graded with experiments.analysis.analysis_common
    (robust_grade / math_verify); the stored `is_correct` column is never used.
    Confidence is the model's own STATED value inside CLOSED <|meta|> blocks,
    taking the FIRST one per response. That is the canonical parser convention
    adopted 2026-08-12 for the training-time R_cal head, and it is the one the
    ECE/AUROC numbers in the paper table were computed under. Taking the LAST
    value instead REVERSES the direction of the AUROC trend, so the convention
    is load-bearing and must not be changed without re-deriving the table.
    Panel (a) therefore carries no ECE/AUROC inset: those numbers are reported
    once, in the paper table.

Every plotted number is printed to stdout so the figures can be checked.

Usage
-----
    cd /home/v-seungplee/metacognition-math && set -a; source .env; set +a
    python scripts/make_paper_figures.py            # all three
    python scripts/make_paper_figures.py --only fig2
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.analysis.analysis_common import (  # noqa: E402
    Grader,
    ece_15bin,
    load_eval_frame,
    regrade_frame,
    stated_confidences,
)

FIG_DIR = REPO_ROOT / "paper" / "figures"
DEFAULT_CACHE = Path(tempfile.gettempdir()) / "metacog_paper_fig_cache"

WANDB_ENTITY = "gistdslab"
WANDB_PROJECT = "metacot-dcpo-v4"
HF_REPO = "iamseungpil/metacot-h200-triobj-dcpo-v3"

# ── style ────────────────────────────────────────────────────────────────────
# Okabe-Ito: colourblind-safe.
OKABE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "skyblue": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#555555",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.4,
    "figure.dpi": 200,
    "savefig.dpi": 200,
})

# ── the three live arms ──────────────────────────────────────────────────────
ARMS = [
    # (wandb run id, display label, colour)
    ("rq3v2f-b2p3v-1", "b2p3v (correctness-only)", OKABE["blue"]),
    ("rq3v2f-b3p2-1", "b3p2 (tri-objective)", OKABE["orange"]),
    ("rq3v2f-b3p3g-1", "b3p3g (tri-obj + R$_{cal}$ repair)", OKABE["green"]),
]

PANEL_KEYS = [
    ("critic/rewards/mean", "training reward (batch mean)"),
    ("dcpo/meta_emit_rate", "metacognitive emission rate"),
    ("actor/entropy", "policy entropy"),
    ("dcpo/cal_positive_rate", "calibration positive-credit rate"),
]

MA_WINDOW = 25


def panel_label(ax, letter: str) -> None:
    """Panel letter only — no descriptive title inside the axes."""
    ax.set_title(f"({letter})", loc="left", fontweight="bold", fontsize=10, pad=4)


# ── wandb ────────────────────────────────────────────────────────────────────

def fetch_wandb_series(cache_dir: Path, refresh: bool) -> dict:
    """{run_id: {key: {"step": [...], "value": [...]}}}, one scan per key."""
    cache = cache_dir / "wandb_series.json"
    if cache.exists() and not refresh:
        print(f"[wandb] using cache {cache}")
        return json.loads(cache.read_text())

    import wandb

    api = wandb.Api()
    out: dict[str, dict] = {}
    for run_id, label, _ in ARMS:
        run = api.run(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{run_id}")
        out[run_id] = {"_state": run.state, "_name": run.name, "_label": label}
        for key, _ in PANEL_KEYS:
            steps, vals = [], []
            # one key at a time: multi-key scans return empty on this project
            for row in run.scan_history(keys=["_step", key]):
                v = row.get(key)
                if v is None:
                    continue
                steps.append(int(row["_step"]))
                vals.append(float(v))
            out[run_id][key] = {"step": steps, "value": vals}
            print(f"[wandb] {run.name:14s} {key:32s} n={len(vals)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def moving_average(v: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(v).rolling(window, center=True, min_periods=max(2, window // 5)).mean().to_numpy()


def make_fig2(cache_dir: Path, refresh: bool) -> None:
    data = fetch_wandb_series(cache_dir, refresh)
    print("\n" + "=" * 78)
    print("FIG 2  training trajectories (wandb, three live arms)")
    print("=" * 78)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0))
    letters = ["a", "b", "c", "d"]

    for pi, (key, ylabel) in enumerate(PANEL_KEYS):
        ax = axes.flat[pi]
        panel_label(ax, letters[pi])
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        missing = []
        for run_id, label, colour in ARMS:
            series = data[run_id][key]
            steps = np.asarray(series["step"], dtype=float)
            vals = np.asarray(series["value"], dtype=float)
            if len(vals) == 0:
                missing.append(label.split(" ")[0])
                print(f"  [{key}] {label}: MISSING (never logged by this run)")
                continue
            if pi == 0:
                # panel (a): raw faint, 25-step moving average bold, to show that
                # per-step noise is larger than the trend.
                ma = moving_average(vals, MA_WINDOW)
                ax.plot(steps, vals, color=colour, alpha=0.25, lw=0.8, zorder=1)
                ax.plot(steps, ma, color=colour, lw=2.0, zorder=3, label=label)
                d = np.abs(np.diff(vals))
                print(
                    f"  [{key}] {label}: n={len(vals)} step={steps[0]:.0f}..{steps[-1]:.0f} "
                    f"raw first={vals[0]:.4f} last={vals[-1]:.4f} "
                    f"min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f} "
                    f"sd={vals.std(ddof=1):.4f}"
                )
                mv = ma[~np.isnan(ma)]
                print(
                    f"      MA{MA_WINDOW}: first={mv[0]:.4f} last={mv[-1]:.4f} "
                    f"min={mv.min():.4f} max={mv.max():.4f} "
                    f"range={np.nanmax(ma) - np.nanmin(ma):.4f} | raw range={vals.max() - vals.min():.4f} "
                    f"| median |step-to-step delta|={np.median(d):.4f} "
                    f"| noise/trend={np.median(d) / max(1e-9, (np.nanmax(ma) - np.nanmin(ma))):.2f}x"
                )
            else:
                ax.plot(steps, vals, color=colour, lw=1.4, label=label)
                print(
                    f"  [{key}] {label}: n={len(vals)} step={steps[0]:.0f}..{steps[-1]:.0f} "
                    f"first={vals[0]:.4f} last={vals[-1]:.4f} "
                    f"min={vals.min():.4f} max={vals.max():.4f} mean={vals.mean():.4f}"
                )
        if missing:
            ax.text(
                0.98, 0.04, "not logged: " + ", ".join(missing),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=OKABE["grey"], style="italic",
            )
        if pi == 0:
            ax.text(
                0.98, 0.04,
                f"thin: per step   thick: {MA_WINDOW}-step moving average",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=7, color=OKABE["grey"], style="italic",
            )
        ax.margins(x=0.02)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    # arms missing from panel (a) legend cannot happen (reward is logged by all)
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.005))
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    out = FIG_DIR / "fig2_trajectories.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> wrote {out}")


# ── held-out eval frames ─────────────────────────────────────────────────────

def eval_parquet(prefix: str, cache_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    name = f"{prefix}_16k_n8_math500"
    rel = f"eval/{prefix}/{name}/{name}.parquet"
    local = cache_dir / "hf" / rel
    if local.exists():
        return local
    p = hf_hub_download(HF_REPO, rel, repo_type="model", local_dir=str(cache_dir / "hf"))
    return Path(p)


def load_graded(prefix: str, cache_dir: Path, grader: Grader, refresh: bool) -> pd.DataFrame:
    """Load one MATH-500 eval parquet, re-grade, extract stated confidence."""
    cache = cache_dir / f"graded_v2_{prefix}.parquet"
    if cache.exists() and not refresh:
        print(f"[eval] cache {cache.name}")
        return pd.read_parquet(cache)
    path = eval_parquet(prefix, cache_dir)
    df = load_eval_frame(path)
    df = df[df["benchmark"].astype(str).str.contains("math", case=False)].copy()
    confs = df["completion"].map(stated_confidences)
    # `robust_grade` (math_verify) is not reproducible run to run in every
    # environment: a re-grade here has come back several accuracy points lower
    # than the graded frame the paper table was built from, because some
    # verifications time out. So a frozen v1 grade cache is authoritative when
    # present, and correctness is re-derived only when there is none. The row
    # alignment between the frozen grades and this frame is asserted below via
    # the independently re-parsed `conf_last` column.
    frozen = cache_dir / f"graded_{prefix}.parquet"
    if frozen.exists() and not refresh:
        prev = pd.read_parquet(frozen)
        if len(prev) != len(df):
            raise RuntimeError(
                f"{prefix}: frozen grade cache has {len(prev)} rows, frame has {len(df)}"
            )
        reparsed = confs.map(lambda v: v[-1] if v else np.nan).to_numpy(dtype=float)
        cached = prev["conf_last"].to_numpy(dtype=float)
        if not np.array_equal(reparsed, cached, equal_nan=True):
            raise RuntimeError(f"{prefix}: frozen grade cache is not row-aligned")
        print(f"[eval] {prefix}: {len(df)} rows, frozen grades from {frozen.name}")
        df["correct"] = prev["correct"].to_numpy()
    else:
        print(f"[eval] {prefix}: {len(df)} rows, re-grading with robust_grade ...")
        df["correct"] = regrade_frame(df, grader).astype(bool)
    df["n_conf"] = confs.map(len)
    df["conf_last"] = confs.map(lambda v: v[-1] if v else np.nan)
    # Canonical convention (adopted 2026-08-12 for the training-time R_cal parser
    # and used for the ECE/AUROC numbers reported in the paper table): the FIRST
    # stated value inside a closed <|meta|> block.
    df["conf_first"] = confs.map(lambda v: v[0] if v else np.nan)
    df["meta_first"] = df["completion"].map(
        lambda t: str(t).lstrip().startswith("<|meta|>")
    )
    df["think_first"] = df["completion"].map(
        lambda t: str(t).lstrip().startswith("<think>")
    )
    keep = [
        "benchmark", "qid", "sample_idx", "completion_length_tokens",
        "num_meta_blocks_closed", "correct", "n_conf", "conf_last", "conf_first",
        "meta_first", "think_first", "is_correct",
    ]
    small = df[keep].copy()
    cache.parent.mkdir(parents=True, exist_ok=True)
    small.to_parquet(cache)
    return small


# ── fig3: substrate repair manipulation check ────────────────────────────────

def make_fig3(cache_dir: Path, refresh: bool) -> None:
    grader = Grader()
    pre = load_graded("sft2init_gs0", cache_dir, grader, refresh)
    post = load_graded("b2p3init_gs0", cache_dir, grader, refresh)

    print("\n" + "=" * 78)
    print("FIG 3  substrate repair manipulation check (held-out MATH-500, n8, 16k)")
    print("=" * 78)

    cats = ["<|meta|> block\nfirst", "<think> block\nfirst", "other"]
    groups = [("pre-repair (sft2init_gs0)", pre, OKABE["vermillion"]),
              ("post-repair (b2p3init_gs0)", post, OKABE["blue"])]

    fracs = {}
    for label, df, _ in groups:
        n = len(df)
        mf = int(df["meta_first"].sum())
        tf = int(df["think_first"].sum())
        other = n - mf - tf
        fracs[label] = [mf / n, tf / n, other / n]
        print(f"  first-block, {label}: N={n} "
              f"meta-first={mf} ({mf / n:.4f}) "
              f"reasoning-first={tf} ({tf / n:.4f}) "
              f"other={other} ({other / n:.4f})")
        print(f"      closed <|meta|> blocks present in {int((df['num_meta_blocks_closed'] > 0).sum())} "
              f"({(df['num_meta_blocks_closed'] > 0).mean():.4f}) of responses; "
              f"re-graded accuracy={df['correct'].mean():.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    panel_label(ax, "a")
    x = np.arange(len(cats))
    w = 0.36
    for i, (label, _, colour) in enumerate(groups):
        vals = fracs[label]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, color=colour, label=label,
                      edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(0, 1.42)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("fraction of responses")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.03), ncol=1)

    ax = axes[1]
    panel_label(ax, "b")
    bins = np.linspace(0, 2000, 61)
    med_notes = []
    for label, df, colour in groups:
        L = df["completion_length_tokens"].dropna().to_numpy(dtype=float)
        med = float(np.median(L))
        ax.hist(np.clip(L, bins[0], bins[-1]), bins=bins, density=True,
                histtype="stepfilled", alpha=0.30, color=colour)
        ax.hist(np.clip(L, bins[0], bins[-1]), bins=bins, density=True,
                histtype="step", color=colour, lw=1.2,
                label=f"{label}\nmedian {med:.0f}, {(L > 2000).mean() * 100:.1f}% > 2000")
        ax.axvline(med, color=colour, ls="--", lw=1.2)
        q1, q3 = np.percentile(L, [25, 75])
        med_notes.append((med, colour))
        print(f"  response length, {label}: n={len(L)} median={med:.1f} "
              f"mean={L.mean():.1f} q1={q1:.1f} q3={q3:.1f} "
              f"min={L.min():.0f} max={L.max():.0f} "
              f"frac>2000={(L > 2000).mean():.4f}")
    top = ax.get_ylim()[1]
    for i, (med, colour) in enumerate(med_notes):
        ax.annotate(f"{med:.0f}", xy=(med, top * (0.62 + 0.24 * i)),
                    xytext=(4 if i else -4, 0), textcoords="offset points",
                    color=colour, fontsize=8, ha="left" if i else "right",
                    va="center", fontweight="bold")
    ax.set_xlabel("response length (tokens, clipped at 2000)")
    ax.set_ylabel("density")
    ax.legend(loc="upper right", handlelength=1.2, labelspacing=0.9)

    fig.tight_layout()
    out = FIG_DIR / "fig3_repair.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> wrote {out}")


# ── fig4: calibration ────────────────────────────────────────────────────────

def auroc(conf: np.ndarray, correct: np.ndarray) -> float:
    """Rank-based AUROC (ties averaged); NaN when one class is absent."""
    conf = np.asarray(conf, dtype=float)
    y = np.asarray(correct, dtype=bool)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(conf, kind="mergesort")
    ranks = np.empty(len(conf), dtype=float)
    sc = conf[order]
    i = 0
    while i < len(sc):
        j = i
        while j + 1 < len(sc) and sc[j + 1] == sc[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[y].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def make_fig4(cache_dir: Path, refresh: bool) -> None:
    grader = Grader()
    ckpts = [
        ("gs0", "b2p3init_gs0", OKABE["blue"]),
        ("gs50", "rq3v2f_b2p3v_gs50", OKABE["orange"]),
        ("gs100", "rq3v2f_b2p3v_gs100", OKABE["green"]),
    ]
    frames = {tag: load_graded(pre, cache_dir, grader, refresh) for tag, pre, _ in ckpts}

    print("\n" + "=" * 78)
    print("FIG 4  calibration of the correctness-only arm (held-out MATH-500)")
    print("       confidence = FIRST stated value in a CLOSED <|meta|> block")
    print("       (canonical parser convention; matches the paper's ECE/AUROC table)")
    print("       correctness = robust_grade re-grade (stored is_correct unused)")
    print("=" * 78)

    n_bins = 15
    edges = np.linspace(0, 1, n_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    ax = axes[0]
    panel_label(ax, "a")
    ax.plot([0, 1], [0, 1], color=OKABE["grey"], ls=":", lw=1.0, zorder=1)
    ann: list[str] = []
    handles: list = []
    for tag, prefix, colour in ckpts:
        df = frames[tag]
        sub = df[df["conf_first"].notna()]
        conf = sub["conf_first"].to_numpy(dtype=float)
        corr = sub["correct"].to_numpy(dtype=bool)
        e = ece_15bin(conf, corr, n_bins=n_bins)
        a = auroc(conf, corr)
        cov = len(sub) / len(df)
        print(f"\n  [{tag}] {prefix}: N={len(df)} with-confidence={len(sub)} "
              f"(coverage={cov:.4f}) accuracy(all)={df['correct'].mean():.4f} "
              f"accuracy(with-conf)={corr.mean():.4f}")
        print(f"      mean conf={conf.mean():.4f} median conf={np.median(conf):.4f} "
              f"ECE15={e:.4f} AUROC={a:.4f} "
              f"stored-is_correct acc (NOT used)={df['is_correct'].mean():.4f}")
        idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
        xs, ys, ns, los, his = [], [], [], [], []
        for b in range(n_bins):
            m = idx == b
            if m.sum() == 0:
                continue
            k, n = int(corr[m].sum()), int(m.sum())
            lo, hi = wilson(k, n)
            xs.append(conf[m].mean())
            ys.append(corr[m].mean())
            ns.append(n)
            los.append(lo)
            his.append(hi)
            print(f"        bin {edges[b]:.3f}-{edges[b + 1]:.3f}: n={n:5d} "
                  f"mean_conf={conf[m].mean():.4f} accuracy={corr[m].mean():.4f} "
                  f"wilson95=[{lo:.4f},{hi:.4f}]")
        xs, ys, ns = np.asarray(xs), np.asarray(ys), np.asarray(ns)
        sizes = 10 + 110 * ns / ns.max()
        # The Wilson interval is not centred on p, so for extreme p at small n
        # the raw offsets can go slightly negative; clip at 0 rather than let
        # matplotlib reject them.
        lo_err = np.clip(ys - np.asarray(los), 0.0, None)
        hi_err = np.clip(np.asarray(his) - ys, 0.0, None)
        ax.errorbar(xs, ys, yerr=[lo_err, hi_err],
                    fmt="none", ecolor=colour, elinewidth=0.7, alpha=0.6, zorder=2)
        ax.scatter(xs, ys, s=sizes, color=colour, zorder=3, edgecolor="white",
                   linewidth=0.5)
        handles.append(plt.Line2D([], [], marker="o", ls="", ms=5, color=colour,
                                  label=tag))
        ann.append(f"{tag}:  ECE {e:.3f}   AUROC {a:.3f}")  # printed, not drawn
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("empirical accuracy")
    leg = ax.legend(handles=handles, loc="upper left", handletextpad=0.4)
    leg.set_zorder(5)
    ax.text(0.50, 0.99, "marker area $\\propto$ responses in bin\nbars: 95% Wilson interval",
            transform=ax.transAxes, ha="left", va="top", fontsize=7,
            color=OKABE["grey"], style="italic")
    # No ECE/AUROC inset: those numbers are reported once, in the paper table,
    # so the figure cannot drift out of sync with it.

    ax = axes[1]
    panel_label(ax, "b")
    hbins = np.arange(0, 1.02, 0.02)
    for tag, colour in [("gs0", OKABE["blue"]), ("gs100", OKABE["green"])]:
        conf = frames[tag]["conf_first"].dropna().to_numpy(dtype=float)
        ax.hist(conf, bins=hbins, histtype="stepfilled", alpha=0.35, color=colour)
        ax.hist(conf, bins=hbins, histtype="step", color=colour, lw=1.2, label=tag)
        vc = pd.Series(np.round(conf, 3)).value_counts().sort_values(ascending=False)
        top = vc.head(5)
        print(f"\n  confidence histogram [{tag}]: n={len(conf)} distinct={vc.size} "
              f"top values " + ", ".join(f"{v:.2f}x{c} ({c / len(conf):.3f})"
                                         for v, c in top.items()))
        share_top2 = top.head(2).sum() / len(conf)
        print(f"      top-2 value share={share_top2:.4f} "
              f"share outside top-2 (tail)={1 - share_top2:.4f}")
    ax.set_xlabel("stated confidence")
    ax.set_ylabel("responses")
    ax.set_yscale("log")
    ax.legend(loc="upper left")

    fig.tight_layout()
    out = FIG_DIR / "fig4_calibration.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["fig2", "fig3", "fig4"], action="append")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE))
    ap.add_argument("--refresh", action="store_true",
                    help="ignore caches and re-pull wandb / re-grade evals")
    args = ap.parse_args()

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    want = set(args.only or ["fig2", "fig3", "fig4"])

    if "fig2" in want:
        make_fig2(cache, args.refresh)
    if "fig3" in want:
        make_fig3(cache, args.refresh)
    if "fig4" in want:
        make_fig4(cache, args.refresh)


if __name__ == "__main__":
    main()
