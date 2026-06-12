#!/usr/bin/env python3
"""DCPO v4 offline PMI probe — the kill-or-go gate (spec §3).

Scores existing eval rollouts (e8_goldfree 1030x8, ~8k with meta) against the
FROZEN SFT reference: per row, Δ_t = logP_ref(C_t | prefix+meta+C_<t) −
logP_ref(C_t | prefix+C_<t) over the splice-aligned C-span (src.training.dcpo_pmi,
spec C3), teacher-forced at **T=1.0** (spec M1 — logits are never divided).

Report (spec §3 kill criteria):
  (a) Δ-aggregate distribution stats per aggregation method;
  (b) correct-vs-wrong AUC per method, SPLIT by leak-bias population (spec I5):
      rows whose CONTINUATION echoes the v3 signature regex (entangled proxy)
      vs clean, + overall; a split with n < MIN_SPLIT_N is flagged DEGENERATE,
      never silently relabeled;
  (c) PLACEBO control (spec C1, KILL): contentless meta "Let me continue." in
      tags — real Δ must beat placebo Δ (paired one-sided t);
  (d) SHUFFLE control: C from a different rollout of the same problem (random
      other problem as fallback) — Δ must collapse toward 0;
  (e) recommended aggregation + clip constant c (95th pct of |Δ-agg|) + VERDICT.

NO training deps: no verl import anywhere on this path (dcpo_pmi is numpy-pure;
dcpo_region is imported ONLY for the canonical signature regex — Karpathy lock:
that regex is defined exactly once in the repo).

Example:
  python scripts/probe_pmi_offline.py --smoke
  python scripts/probe_pmi_offline.py --max-rows 2000 --out results/probe_pmi_v4.json
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metacot.prompt import META_END, META_START
from src.training.dcpo_pmi import (
    PLACEBO_META,
    PMI_AGG_METHODS,
    SpliceAlignmentError,
    compute_pmi_rows,
    splice_and_align,
    split_first_meta,
)
# Canonical meta-content signature (entangled-population proxy, spec I5) + the
# tag token ids. dcpo_region imports torch/numpy only — NOT verl.
from src.training.dcpo_region import (
    META_CLOSE_DEFAULT,
    META_OPEN_DEFAULT,
    _has_meta_signature,
)

def _default_data() -> str:
    # Resolve whatever snapshot of the eval file is in the local HF cache
    # (pinning a snapshot hash here breaks on cache refresh and trips the
    # release token-leak scan, which flags any 40-hex word).
    import glob

    pattern = (
        "/home/v-seungplee/.cache/huggingface/hub/datasets--iamseungpil--metacot/"
        "snapshots/*/eval/e8_goldfree_1030_16k_k8/e8_goldfree_1030_16k_k8.json"
    )
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else pattern


DEFAULT_DATA = _default_data()
DEFAULT_MODEL = (
    "/home/v-seungplee/sft_v8_strict_local/models/v8_meta_inside_strict_sft/checkpoint-254"
)

# PLACEBO_META now lives in src.training.dcpo_pmi (SSOT): the stage-2
# placebo-corrected reward must subtract the SAME placebo this probe validated.

# Kill-criteria thresholds (spec §3): real-beats-placebo significance, shuffle
# collapse ratio, AUC bar on the entangled split, min n for trusting the split.
PLACEBO_ALPHA = 0.05
SHUFFLE_COLLAPSE_MAX = 0.25
AUC_KILL = 0.6
MIN_SPLIT_N = 20


# ─────────────────────────────────────────────────────────────────────────────
# rollout parsing (text-level, tokenizer-free — unit-testable without GPU)
# ─────────────────────────────────────────────────────────────────────────────
def parse_rollout(record: dict, problem_id: int):
    """Split one eval record into prefix / meta / continuation (spec §2).

    prefix = completion up to the FIRST <|meta|> (chat prompt is prepended later,
    once a tokenizer exists); meta = tag-INCLUSIVE block; C = the model's OWN
    text after <|/meta|> (native position). Returns None for malformed rows:
    no meta, truncated meta (open without close, the 16k-cutoff population), or
    whitespace-only continuation (nothing to score) — the split itself is the
    SHARED dcpo_pmi.split_first_meta (round 2 M-D: one definition, also called
    by verl_sdc's v4 scorer).
    """
    completion = record.get("completion") or ""
    parts = split_first_meta(completion)
    if parts is None:
        return None
    prefix, meta, continuation = parts
    return {
        "problem_id": problem_id,
        "benchmark": record.get("benchmark", ""),
        "question": record.get("question", ""),
        "correct": bool(record.get("is_correct", False)),
        "boxed_answer": record.get("answer_extracted"),
        "completion_prefix": prefix,
        "meta_text": meta,
        "continuation_text": continuation,
        # spec I5 entangled-population proxy — measured on the CONTINUATION,
        # NOT the meta's own inner text: the signature regex (field-label lines
        # confidence:/assessment:/action:) IS the required meta format, so an
        # inner-text flag is a dead constant (100% entangled, clean n=0 on the
        # real probe data — the Agentness-Arena Discovery-metric class). The v3
        # CF leak guard this proxy derives from checks the GENERATED text the
        # same way (dcpo_region cf_txt): a field-label ECHO after <|/meta|>
        # marks the meta<->continuation-entangled rows.
        "entangled": _has_meta_signature(continuation),
    }


def load_rollouts(data_path: str, max_rows: int | None = None) -> list[dict]:
    """Parse the eval json into probe rows; group rollouts into problem_ids.

    problem_id increments whenever sample_idx restarts (records are written as 8
    consecutive rollouts per problem). When `max_rows` truncates, rows from
    MIXED-correctness problems come first (the within-problem-contrast population
    AUC needs; file order is benchmark-sorted so a head-slice would be one-class).
    """
    with open(data_path) as f:
        records = json.load(f)["results"]
    rows, pid, last_idx = [], -1, None
    for rec in records:
        s_idx = int(rec.get("sample_idx", 0))
        if last_idx is None or s_idx <= last_idx:
            pid += 1
        last_idx = s_idx
        row = parse_rollout(rec, pid)
        if row is not None:
            rows.append(row)
    if max_rows is not None and len(rows) > max_rows:
        by_pid: dict[int, set] = {}
        for r in rows:
            by_pid.setdefault(r["problem_id"], set()).add(r["correct"])
        mixed = {p for p, cs in by_pid.items() if len(cs) == 2}
        rows = ([r for r in rows if r["problem_id"] in mixed]
                + [r for r in rows if r["problem_id"] not in mixed])[:max_rows]
    return rows


def pick_shuffle_partners(rows: list[dict], seed: int = 0,
                          mode: str = "same_problem") -> list[int]:
    """Per row, the index whose continuation feeds the SHUFFLE control (spec §3).

    mode='same_problem' (default, spec §3): deterministic cyclic next-rollout
    within the same problem; falls back to a seeded random row from a DIFFERENT
    problem when the row has no usable sibling (singleton problem, or identical
    continuation text).

    mode='cross_problem': seeded random row from a DIFFERENT problem for EVERY
    row. Disambiguates the same-problem control's confound (sibling rollouts of
    one problem share the same solution → near-identical continuations, measured
    mean similarity ~0.49 on e8 — a non-collapsing same-problem shuffle can mean
    EITHER generic-template signal OR legitimate shared-solution content). A
    cross-problem partner removes the shared-solution explanation: surviving
    delta here IS template-generic.
    """
    rng = np.random.default_rng(seed)
    by_pid: dict[int, list[int]] = {}
    for i, r in enumerate(rows):
        by_pid.setdefault(r["problem_id"], []).append(i)
    partners = []
    for i, r in enumerate(rows):
        if mode == "cross_problem":
            others = [k for k in range(len(rows))
                      if rows[k]["problem_id"] != r["problem_id"]]
            partners.append(int(rng.choice(others)) if others else i)
            continue
        group = by_pid[r["problem_id"]]
        j = group[(group.index(i) + 1) % len(group)]
        if j == i or rows[j]["continuation_text"] == r["continuation_text"]:
            others = [k for k in range(len(rows))
                      if rows[k]["problem_id"] != r["problem_id"]]
            j = int(rng.choice(others)) if others else i
        partners.append(j)
    return partners


# ─────────────────────────────────────────────────────────────────────────────
# tokenizer / alignment / scoring (GPU side)
# ─────────────────────────────────────────────────────────────────────────────
def load_probe_tokenizer(model_path: str):
    """Load the checkpoint tokenizer through the verified list->dict workaround.

    The checkpoint was saved by transformers 4.52 where tokenizer_config.json's
    `extra_special_tokens` is a LIST; >=4.53 expects a dict and crashes. Patch a
    temp copy (ids unchanged, verified) and sanity-check the meta tag ids.
    """
    from transformers import AutoTokenizer

    tmp = tempfile.mkdtemp(prefix="probe_pmi_tok_")
    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = Path(model_path) / fname
        if src.exists():
            shutil.copy(src, tmp)
    cfg_path = Path(tmp) / "tokenizer_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.pop("extra_special_tokens", None)
    cfg_path.write_text(json.dumps(cfg))
    tok = AutoTokenizer.from_pretrained(tmp)
    assert tok.encode(META_START, add_special_tokens=False) == [META_OPEN_DEFAULT]
    assert tok.encode(META_END, add_special_tokens=False) == [META_CLOSE_DEFAULT]
    return tok


def render_chat_prompt(tokenizer, question: str) -> str:
    """Exact generation-time prompt (mirrors scripts/eval_vllm_1030.render_chat_prompt)."""
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def align_rows(rows: list[dict], partners: list[int], tokenizer):
    """Build per-row {real, placebo, shuffle} arm alignments + a deduped sequence pool.

    Each arm holds ("with"/"without") -> (sequence index, c-span) referencing the
    pool; arms whose splice has no common C-span (SpliceAlignmentError) are None.
    The real and placebo arms share the without-sequence — dedupe makes that one
    forward, not two.
    """
    sequences: list[list[int]] = []
    index: dict[tuple, int] = {}

    def reg(ids: list[int]) -> int:
        key = tuple(ids)
        if key not in index:
            index[key] = len(sequences)
            sequences.append(ids)
        return index[key]

    for i, row in enumerate(rows):
        prefix = row["prefix_text"]
        arms = {}
        for name, meta, cont in (
            ("real", row["meta_text"], row["continuation_text"]),
            ("placebo", PLACEBO_META, row["continuation_text"]),
            ("shuffle", row["meta_text"], rows[partners[i]]["continuation_text"]),
        ):
            try:
                a = splice_and_align(tokenizer, prefix, meta, cont)
            except SpliceAlignmentError:
                arms[name] = None
                continue
            assert a["c_span_with"][0] >= 1 and a["c_span_without"][0] >= 1
            arms[name] = {
                "meta_text": meta,
                "continuation_text": cont,
                "with": (reg(a["with_ids"]), a["c_span_with"]),
                "without": (reg(a["without_ids"]), a["c_span_without"]),
            }
        row["arms"] = arms
    return sequences


def score_sequences(model, sequences: list[list[int]], pad_id: int,
                    batch_size: int = 4, token_budget: int = 49152,
                    device: str = "cuda") -> list[np.ndarray]:
    """Teacher-forced per-token logprobs at T=1.0 (spec M1: logits NOT divided).

    Returns one float32 array per sequence, lp[t] = logP(ids[t] | ids[:t]) for
    t >= 1 (lp[0] = 0 placeholder, never read — c-spans start past the prefix).
    Sequences are length-sorted and greedily packed under `batch_size` rows AND
    `token_budget` padded tokens per forward (bounds the bf16 logits tensor);
    the float32 log_softmax+gather runs in 1024-position chunks.
    """
    import torch
    import torch.nn.functional as F

    out: list = [None] * len(sequences)
    order = sorted(range(len(sequences)), key=lambda i: -len(sequences[i]))

    def flush(batch: list[int]):
        T = max(len(sequences[i]) for i in batch)
        ids = torch.full((len(batch), T), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), T), dtype=torch.long)
        for r, i in enumerate(batch):
            L = len(sequences[i])
            ids[r, :L] = torch.as_tensor(sequences[i])
            attn[r, :L] = 1
        ids, attn = ids.to(device), attn.to(device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attn, use_cache=False).logits
            lp = torch.empty((len(batch), T - 1), dtype=torch.float32, device=device)
            for s in range(0, T - 1, 1024):
                e = min(s + 1024, T - 1)
                lp[:, s:e] = (
                    F.log_softmax(logits[:, s:e].float(), dim=-1)
                    .gather(-1, ids[:, s + 1:e + 1, None]).squeeze(-1)
                )
        lp = lp.cpu().numpy()
        for r, i in enumerate(batch):
            L = len(sequences[i])
            row_lp = np.zeros(L, dtype=np.float32)
            row_lp[1:] = lp[r, :L - 1]
            out[i] = row_lp

    batch: list[int] = []
    for i in order:
        if batch and (len(batch) >= batch_size
                      or (len(batch) + 1) * len(sequences[batch[0]]) > token_budget):
            flush(batch)
            batch = []
        batch.append(i)
    if batch:
        flush(batch)
    return out


def build_pass_rows(rows: list[dict], arm: str, logps: list[np.ndarray]) -> list[dict]:
    """Materialize compute_pmi_rows inputs for one arm from the scored pool."""
    pass_rows = []
    for row in rows:
        base = {"correct": row["correct"], "boxed_answer": row["boxed_answer"],
                "entangled": row["entangled"]}
        a = row["arms"][arm]
        if a is None:
            pass_rows.append({**base, "meta_text": "", "continuation_text": "",
                              "alignment_failed": True,
                              "logp_with": None, "logp_without": None})
            continue
        (iw, (sw, ew)), (io, (so, eo)) = a["with"], a["without"]
        pass_rows.append({**base, "meta_text": a["meta_text"],
                          "continuation_text": a["continuation_text"],
                          "logp_with": logps[iw][sw:ew],
                          "logp_without": logps[io][so:eo]})
    return pass_rows


# ─────────────────────────────────────────────────────────────────────────────
# statistics (numpy-only; testable with stubbed logprobs)
# ─────────────────────────────────────────────────────────────────────────────
def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return {"n": 0}
    pct = np.percentile(x, [5, 25, 50, 75, 95])
    return {"n": int(x.size), "mean": float(x.mean()), "std": float(x.std()),
            "p05": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]),
            "p75": float(pct[3]), "p95": float(pct[4])}


def rank_auc(scores, labels) -> float:
    """Mann-Whitney AUC (P[score_pos > score_neg], ties = 0.5); NaN if one-class."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < scores.size:  # average ranks over ties
        j = i
        while j + 1 < scores.size and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def paired_t(diffs) -> tuple[float, float]:
    """Paired t-stat + ONE-SIDED p (H1: mean > 0), normal approximation (n large)."""
    d = np.asarray(diffs, dtype=np.float64)
    if d.size < 2:
        return float("nan"), float("nan")
    sd = d.std(ddof=1)
    if sd == 0.0:
        t = math.inf if d.mean() > 0 else (-math.inf if d.mean() < 0 else 0.0)
    else:
        t = float(d.mean() / (sd / math.sqrt(d.size)))
    return t, float(0.5 * math.erfc(t / math.sqrt(2.0)))


# ─────────────────────────────────────────────────────────────────────────────
# report assembly (spec §3 a-e)
# ─────────────────────────────────────────────────────────────────────────────
def assemble_report(real_rows, placebo_rows, shuffle_rows, *, topk_frac=0.25,
                    clip_c_token=2.0, ngram_n=8, ngram_threshold=0.25,
                    smoke=False) -> dict:
    """Turn the three scored passes into the kill-or-go report (spec §3).

    Row dicts follow the compute_pmi_rows contract plus `entangled` (spec I5
    split flag); placebo/shuffle rows are index-paired with real rows.

    Round 2 IMPORTANT-1 (probe-vs-training population mismatch): TRAINING zeroes
    every guard-hit row (member 0), but compute_pmi_rows keeps raw_agg on guard
    rows BY DESIGN (diagnosable). So the VERDICT path — delta_stats, the KILL-3
    AUC, the placebo pairing, and the recommended clip_c=p95(|agg|) — runs on
    the guard-FILTERED population (the rows training actually scores); the
    unfiltered view is reported side by side under report["unfiltered"].
    """
    kw = dict(topk_frac=topk_frac, clip_c_token=clip_c_token,
              ngram_n=ngram_n, ngram_threshold=ngram_threshold)
    _, real_d = compute_pmi_rows(real_rows, **kw)
    _, plac_d = compute_pmi_rows(placebo_rows, **kw)
    _, shuf_d = compute_pmi_rows(shuffle_rows, **kw)
    correct = np.array([r["correct"] for r in real_rows], dtype=bool)
    entangled = np.array([r.get("entangled", False) for r in real_rows], dtype=bool)
    guard = np.asarray(real_d["guard_hits"], dtype=bool)  # per-row, real-arm

    report = {
        "smoke": bool(smoke),
        "n_rows": len(real_rows),
        "n_correct": int(correct.sum()),
        "n_entangled": int(entangled.sum()),
        "guard_hits_real": int(guard.sum()),
        "alignment_failures": {
            "real": int(np.sum(real_d["alignment_failures"])),
            "placebo": int(np.sum(plac_d["alignment_failures"])),
            "shuffle": int(np.sum(shuf_d["alignment_failures"])),
        },
        "nonfinite": {
            "real": int(np.sum(real_d["nonfinite"])),
            "placebo": int(np.sum(plac_d["nonfinite"])),
            "shuffle": int(np.sum(shuf_d["nonfinite"])),
        },
        "delta_stats": {}, "auc": {}, "placebo": {}, "shuffle": {},
        "unfiltered": {"delta_stats": {}, "auc": {}, "placebo": {}, "shuffle": {}},
    }
    _views = ((report, ~guard), (report["unfiltered"], np.ones(len(guard), dtype=bool)))
    for m in PMI_AGG_METHODS:
        ra = np.asarray(real_d["raw_agg"][m], dtype=np.float64)
        pa = np.asarray(plac_d["raw_agg"][m], dtype=np.float64)
        sa = np.asarray(shuf_d["raw_agg"][m], dtype=np.float64)
        for dest, keep in _views:
            v = ~np.isnan(ra) & keep
            # (a) distribution
            dest["delta_stats"][m] = _stats(ra[v])
            # (b) AUC overall + spec-I5 split (entangled = load-bearing population)
            dest["auc"][m] = {
                "overall": rank_auc(ra[v], correct[v]),
                "entangled": rank_auc(ra[v & entangled], correct[v & entangled]),
                "clean": rank_auc(ra[v & ~entangled], correct[v & ~entangled]),
                "n_entangled": int((v & entangled).sum()),
                "n_clean": int((v & ~entangled).sum()),
            }
            # (c) placebo paired comparison (spec C1 KILL)
            both = v & ~np.isnan(pa)
            t, p = paired_t(ra[both] - pa[both])
            dest["placebo"][m] = {
                "n_paired": int(both.sum()),
                "mean_real": float(ra[both].mean()) if both.any() else float("nan"),
                "mean_placebo": float(pa[both].mean()) if both.any() else float("nan"),
                "mean_diff": float((ra[both] - pa[both]).mean()) if both.any() else float("nan"),
                "t_stat": t, "p_one_sided": p,
            }
            # (d) shuffle collapse
            vs = ~np.isnan(sa) & keep
            mean_s = float(sa[vs].mean()) if vs.any() else float("nan")
            mean_r = float(ra[v].mean()) if v.any() else float("nan")
            dest["shuffle"][m] = {
                "n": int(vs.sum()), "mean": mean_s, "mean_real": mean_r,
                "collapse_ratio": float(abs(mean_s) / max(abs(mean_r), 1e-9)),
            }

    # (f) placebo-CORRECTED metric (cross-shuffle finding 2026-06-11): raw
    # delta is dominated by generic text-presence (placebo retains ~86% of the
    # mean-method delta; cross-problem shuffle retains 52%). The trainable
    # content signal is delta' = delta - delta_placebo PER ROW. Grade delta'
    # with the same KILL battery: mean>0 paired-t (delta''s own placebo test),
    # AUC on the entangled split, and shuffle collapse where the shuffle arm is
    # ALSO corrected (delta'_shuffle = delta_shuffle - delta_placebo).
    report["corrected"] = {}
    report["unfiltered"]["corrected"] = {}
    for m in PMI_AGG_METHODS:
        ra = np.asarray(real_d["raw_agg"][m], dtype=np.float64)
        pa = np.asarray(plac_d["raw_agg"][m], dtype=np.float64)
        sa = np.asarray(shuf_d["raw_agg"][m], dtype=np.float64)
        ca, cs = ra - pa, sa - pa
        for dest, keep in _views:
            v = ~np.isnan(ca) & keep
            vs = ~np.isnan(cs) & keep
            t, p = paired_t(ca[v])
            mean_c = float(ca[v].mean()) if v.any() else float("nan")
            mean_cs = float(cs[vs].mean()) if vs.any() else float("nan")
            dest["corrected"][m] = {
                "n": int(v.sum()),
                "delta_stats": _stats(ca[v]),
                "t_stat": t, "p_one_sided": p,
                "auc": {
                    "overall": rank_auc(ca[v], correct[v]),
                    "entangled": rank_auc(ca[v & entangled], correct[v & entangled]),
                    "clean": rank_auc(ca[v & ~entangled], correct[v & ~entangled]),
                    "n_entangled": int((v & entangled).sum()),
                    "n_clean": int((v & ~entangled).sum()),
                },
                "shuffle": {
                    "n": int(vs.sum()), "mean": mean_cs, "mean_real": mean_c,
                    "collapse_ratio": float(abs(mean_cs) / max(abs(mean_c), 1e-9))
                                      if not (math.isnan(mean_cs) or math.isnan(mean_c))
                                      else float("nan"),
                },
            }

    # per-row dump: re-analysis (new metrics, new thresholds) without the
    # ~95min GPU re-scoring pass. ~12 x n float lists (~2 MB at n=8k).
    report["per_row"] = {
        "correct": correct.astype(int).tolist(),
        "entangled": entangled.astype(int).tolist(),
        "guard_hit": guard.astype(int).tolist(),
        "raw_agg": {
            arm: {m: [None if math.isnan(x) else round(float(x), 6)
                      for x in np.asarray(d["raw_agg"][m], dtype=np.float64)]
                  for m in PMI_AGG_METHODS}
            for arm, d in (("real", real_d), ("placebo", plac_d), ("shuffle", shuf_d))
        },
    }

    # (e) recommendation + verdict — ON THE GUARD-FILTERED population (the
    # report top-level), matching what training scores (round 2 IMPORTANT-1).
    # Method = best AUC on the entangled split (the population that dominates
    # training, spec I5); the method CHOICE falls back to overall when that
    # split is too thin — but a thin split is FLAGGED degenerate and KILL 3
    # then FAILS outright: the load-bearing (a)-vs-(b) contrast was never
    # measured, and silently grading the overall AUC under the entangled name
    # is the dead-constant-metric class this report exists to prevent (review
    # round 1).
    n_ent_min = min(report["auc"][m]["n_entangled"] for m in PMI_AGG_METHODS)
    n_clean_min = min(report["auc"][m]["n_clean"] for m in PMI_AGG_METHODS)
    ent_usable = n_ent_min >= MIN_SPLIT_N
    split_degenerate = not ent_usable or n_clean_min < MIN_SPLIT_N
    split = "entangled" if ent_usable else "overall"
    scored = [(m, report["auc"][m][split]) for m in PMI_AGG_METHODS
              if not math.isnan(report["auc"][m][split])]
    method = max(scored, key=lambda x: x[1])[0] if scored else "sum_clip"
    # clip_c = p95(|agg|) over the guard-FILTERED rows: training zeroes guard
    # rows, so a guard-hit outlier in the upper tail must not inflate the clip.
    ra = np.asarray(real_d["raw_agg"][method], dtype=np.float64)
    ra = ra[~np.isnan(ra) & ~guard]
    clip_c = float(np.percentile(np.abs(ra), 95)) if ra.size else float("nan")

    plc, shf = report["placebo"][method], report["shuffle"][method]
    auc_ent = report["auc"][method]["entangled"]
    verdict = {
        "method": method, "auc_split_used": split, "clip_c": clip_c,
        "population": "guard_filtered",
        "split_degenerate": split_degenerate,
        "n_entangled_min": int(n_ent_min), "n_clean_min": int(n_clean_min),
        # KILL 1 (C1): real meta must beat placebo significantly
        "placebo_pass": bool(plc["mean_diff"] > 0
                             and plc["p_one_sided"] < PLACEBO_ALPHA),
        # KILL 2: shuffled-C delta collapses toward 0
        "shuffle_pass": bool(not math.isnan(shf["collapse_ratio"])
                             and shf["collapse_ratio"] < SHUFFLE_COLLAPSE_MAX),
        # KILL 3 (I5): correct/wrong separation must hold ON THE ENTANGLED
        # SPLIT itself — an unmeasurable split (n < MIN_SPLIT_N) fails.
        "auc_entangled_pass": bool(ent_usable and not math.isnan(auc_ent)
                                   and auc_ent > AUC_KILL),
    }
    verdict["overall"] = ("PASS" if (verdict["placebo_pass"] and verdict["shuffle_pass"]
                                     and verdict["auc_entangled_pass"]) else "FAIL")
    report["recommendation"] = {"method": method, "clip_c": clip_c}
    report["verdict"] = verdict

    # corrected verdict — graded on the TRAINING method (the raw recommendation
    # the reward freezes), NOT the corrected-AUC argmax: AUC-shopping picked
    # topk_mean on 2026-06-12, whose corrected shuffle retains +0.50 of the
    # positive signal (wrong content still earns half — gameable); mean's
    # higher per-row variance costs AUC but its wrong-content aggregate is
    # NEGATIVE (signed retention -2.43), which is the anti-gaming property the
    # reward needs.
    c_method = method
    c_ra = (np.asarray(real_d["raw_agg"][c_method], dtype=np.float64)
            - np.asarray(plac_d["raw_agg"][c_method], dtype=np.float64))
    c_ra = c_ra[~np.isnan(c_ra) & ~guard]
    c_clip = float(np.percentile(np.abs(c_ra), 95)) if c_ra.size else float("nan")
    cm = report["corrected"][c_method]
    c_auc_ent = cm["auc"]["entangled"]
    # SIGNED shuffle criterion (2026-06-12 fix): |shuffle|/|real| is
    # direction-blind — it graded mean's wrong-content SIGN FLIP (-0.0456 vs
    # +0.0188, ratio 2.43) as "did not collapse". The criterion's intent is
    # "wrong content must not EARN": pass when the corrected real mean is
    # positive and the corrected shuffle mean retains < SHUFFLE_COLLAPSE_MAX
    # of it IN THE SIGNED sense (negative retention = wrong content punished
    # = better than collapse).
    c_real_mean = cm["shuffle"]["mean_real"]
    c_shuf_mean = cm["shuffle"]["mean"]
    signed_ok = (not math.isnan(c_real_mean) and not math.isnan(c_shuf_mean)
                 and c_real_mean > 0
                 and c_shuf_mean < SHUFFLE_COLLAPSE_MAX * c_real_mean)
    c_verdict = {
        "method": c_method, "auc_split_used": split, "clip_c": c_clip,
        "population": "guard_filtered",
        # delta''s own "beats nothing" test: mean(delta') > 0 significantly
        "mean_gt0_pass": bool(not math.isnan(cm["t_stat"]) and cm["t_stat"] > 0
                              and cm["p_one_sided"] < PLACEBO_ALPHA),
        "shuffle_pass": bool(signed_ok),
        "signed_retention": (float(c_shuf_mean / c_real_mean)
                             if (not math.isnan(c_real_mean) and c_real_mean != 0
                                 and not math.isnan(c_shuf_mean)) else float("nan")),
        "auc_entangled_pass": bool(ent_usable and not math.isnan(c_auc_ent)
                                   and c_auc_ent > AUC_KILL),
    }
    c_verdict["overall"] = ("PASS" if (c_verdict["mean_gt0_pass"]
                                       and c_verdict["shuffle_pass"]
                                       and c_verdict["auc_entangled_pass"]) else "FAIL")
    report["recommendation_corrected"] = {"method": c_method, "clip_c": c_clip}
    report["verdict_corrected"] = c_verdict
    return report


def format_report_text(report: dict) -> str:
    """Human-readable report; ends with the single-line VERDICT.

    Sections (a)-(d) print the guard-FILTERED view (the training population,
    round 2 IMPORTANT-1) with the unfiltered counterpart side by side in
    [unfilt ...] brackets.
    """
    tag = "[SMOKE] " if report["smoke"] else ""
    unf = report["unfiltered"]
    lines = [
        f"{tag}PMI offline probe — n={report['n_rows']} "
        f"(correct {report['n_correct']}, entangled {report['n_entangled']}, "
        f"guard_hits {report['guard_hits_real']}, "
        f"align_fail {report['alignment_failures']}, "
        f"nonfinite {report['nonfinite']})",
        "(a) delta-aggregate distribution (guard-filtered | [unfilt]):",
    ]
    for m in PMI_AGG_METHODS:
        s, su = report["delta_stats"][m], unf["delta_stats"][m]
        if s["n"] or su["n"]:
            base = (f"    {m:10s} mean={s['mean']:+.4f} std={s['std']:.4f} "
                    f"p05={s['p05']:+.4f} p50={s['p50']:+.4f} p95={s['p95']:+.4f}"
                    if s["n"] else f"    {m:10s} n=0")
            lines.append(base + (f" [unfilt n={su['n']} mean={su['mean']:+.4f} "
                                 f"p95={su['p95']:+.4f}]" if su["n"] else " [unfilt n=0]"))
    lines.append("(b) correct-vs-wrong AUC [overall / entangled / clean] "
                 "(guard-filtered | [unfilt overall]):")
    for m in PMI_AGG_METHODS:
        a, au = report["auc"][m], unf["auc"][m]
        lines.append(f"    {m:10s} {a['overall']:.3f} / {a['entangled']:.3f} "
                     f"(n={a['n_entangled']}) / {a['clean']:.3f} (n={a['n_clean']}) "
                     f"[unfilt {au['overall']:.3f}]")
    lines.append("(c) placebo control (real - placebo, paired, guard-filtered "
                 "| [unfilt diff]):")
    for m in PMI_AGG_METHODS:
        p, pu = report["placebo"][m], unf["placebo"][m]
        lines.append(f"    {m:10s} diff={p['mean_diff']:+.4f} t={p['t_stat']:.2f} "
                     f"p={p['p_one_sided']:.4g} (n={p['n_paired']}) "
                     f"[unfilt {pu['mean_diff']:+.4f}]")
    lines.append("(d) shuffle control (should collapse to ~0, guard-filtered):")
    for m in PMI_AGG_METHODS:
        s = report["shuffle"][m]
        lines.append(f"    {m:10s} mean={s['mean']:+.4f} vs real {s['mean_real']:+.4f} "
                     f"ratio={s['collapse_ratio']:.3f}")
    v = report["verdict"]
    lines.append(f"(e) recommended: method={v['method']} clip_c={v['clip_c']:.4f} "
                 f"(auc split={v['auc_split_used']}, population={v['population']})")
    if v["split_degenerate"]:
        lines.append(f"    WARNING: I5 split DEGENERATE "
                     f"(n_entangled={v['n_entangled_min']}, "
                     f"n_clean={v['n_clean_min']}, min={MIN_SPLIT_N}) — the "
                     f"(a)-vs-(b) population contrast is UNMEASURED")
    lines.append(f"{tag}VERDICT: {v['overall']} "
                 f"(placebo={v['placebo_pass']}, shuffle={v['shuffle_pass']}, "
                 f"auc_entangled={v['auc_entangled_pass']}, "
                 f"split_degenerate={v['split_degenerate']})")
    lines.append("(f) placebo-CORRECTED delta' = delta - delta_placebo "
                 "(guard-filtered):")
    for m in PMI_AGG_METHODS:
        c = report["corrected"][m]
        lines.append(f"    {m:10s} mean={c['delta_stats'].get('mean', float('nan')):+.4f} "
                     f"t={c['t_stat']:.2f} p={c['p_one_sided']:.4g} "
                     f"auc={c['auc']['overall']:.3f}/{c['auc']['entangled']:.3f}/"
                     f"{c['auc']['clean']:.3f} "
                     f"shuffle_ratio={c['shuffle']['collapse_ratio']:.3f}")
    cv = report["verdict_corrected"]
    lines.append(f"{tag}VERDICT_CORRECTED: {cv['overall']} "
                 f"(method={cv['method']}, clip_c={cv['clip_c']:.4f}, "
                 f"mean_gt0={cv['mean_gt0_pass']}, shuffle={cv['shuffle_pass']} "
                 f"[signed_retention={cv['signed_retention']:+.3f}], "
                 f"auc_entangled={cv['auc_entangled_pass']})")
    return "\n".join(lines)


def _jsonable(obj):
    """Recursively convert numpy scalars and NaN/inf to JSON-safe values."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="max sequences per forward (token budget also applies)")
    ap.add_argument("--token-budget", type=int, default=49152,
                    help="max padded tokens per forward (bounds the logits tensor)")
    ap.add_argument("--smoke", action="store_true",
                    help="5 rows, asserts the pipeline end-to-end (no statistical meaning)")
    ap.add_argument("--out", default=None, help="JSON report path")
    ap.add_argument("--seed", type=int, default=0, help="shuffle-partner fallback seed")
    ap.add_argument("--shuffle-mode", choices=["same_problem", "cross_problem"],
                    default="same_problem",
                    help="SHUFFLE-control partner pool: same_problem (spec §3 "
                         "default; confounded by sibling solution duplication) or "
                         "cross_problem (disambiguation re-test: surviving delta "
                         "is template-generic)")
    args = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM

    max_rows = 5 if args.smoke else args.max_rows
    rows = load_rollouts(args.data, max_rows)
    print(f"[probe] {len(rows)} parseable meta rollouts from {args.data}")

    tokenizer = load_probe_tokenizer(args.model)
    for row in rows:
        row["prefix_text"] = (render_chat_prompt(tokenizer, row["question"])
                              + row["completion_prefix"])
    partners = pick_shuffle_partners(rows, seed=args.seed, mode=args.shuffle_mode)
    sequences = align_rows(rows, partners, tokenizer)
    print(f"[probe] {len(sequences)} unique sequences to score "
          f"(max len {max(map(len, sequences))})")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16).to("cuda").eval()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
        else tokenizer.eos_token_id
    logps = score_sequences(model, sequences, pad_id,
                            batch_size=args.batch_size,
                            token_budget=args.token_budget)

    report = assemble_report(
        build_pass_rows(rows, "real", logps),
        build_pass_rows(rows, "placebo", logps),
        build_pass_rows(rows, "shuffle", logps),
        smoke=args.smoke,
    )
    print(format_report_text(report))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(_jsonable(report), f, indent=2)
        print(f"[probe] report written to {args.out}")

    if args.smoke:
        # pipeline assertions only — 5 rows carry no statistical meaning
        assert report["n_rows"] == len(rows) > 0
        assert report["alignment_failures"]["real"] < report["n_rows"], \
            "smoke: every real-arm alignment failed"
        # unfiltered: a guard hit on the tiny 5-row smoke set must not fail the
        # pipeline assertion (the filtered view may legitimately be thinner).
        assert all(report["unfiltered"]["delta_stats"][m]["n"] > 0
                   for m in PMI_AGG_METHODS)
        assert report["verdict"]["overall"] in ("PASS", "FAIL")
        print("[probe] SMOKE OK")


if __name__ == "__main__":
    main()
