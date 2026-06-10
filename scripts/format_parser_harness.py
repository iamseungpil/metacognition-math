#!/usr/bin/env python
"""Offline harness for the v3k format parser (classify_dcpo_format) on REAL data.

Proves the parser on real rollouts BEFORE any training-code wiring (v3k spec
2026-06-10-dcpo-v3-format-tier-design.md §2):
  1. Pulls ALL dcpo/rollouts wandb tables from the last 3 runs named
     triobj_dcpo_v3_h100_4x4k (project gistdslab/skilldiscovery2).
  2. The tables store TEXT tails (main_tail, last DCPO_WANDB_TEXT_CHARS chars) —
     tokenizes them with the LOCAL SFT tokenizer (extra_special_tokens stripped
     from a temp copy of tokenizer_config.json: known transformers load bug).
  3. Classifies every rollout with the ONE pure parser; for tier-1 rows applies
     the replacement_plan INDEPENDENTLY and re-classifies — the round-trip MUST
     come back 'wellformed' (the §2.2 guarantee, proven here from the outside).
  4. Emits /tmp/format_harness_report.md: class histogram vs the measured
     taxonomy, round-trip pass rate, 5 rendered samples per class with
     token-level region boundaries, and 10 discard (severe) samples for
     over/under-trigger eyeballing.

Run:  /home/v-seungplee/miniconda3/envs/metaprobe/bin/python scripts/format_parser_harness.py
Secrets: reads WANDB_API_KEY from the repo .env; NEVER prints token values.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from collections import Counter, defaultdict

REPO = "/home/v-seungplee/metacognition-math"
sys.path.insert(0, REPO)

from src.training.dcpo_region import (  # noqa: E402 — the ONE parser, no local regex
    META_CLOSE_DEFAULT,
    META_OPEN_DEFAULT,
    THINK_CLOSE_DEFAULT,
    classify_dcpo_format,
)

ENTITY_PROJECT = "gistdslab/skilldiscovery2"
RUN_NAME = "triobj_dcpo_v3_h100_4x4k"
N_RUNS = 3
TOKENIZER_SRC = "/home/v-seungplee/sft_v8_strict_local/models/v8_meta_inside_strict_sft/checkpoint-254"
REPORT_PATH = "/tmp/format_harness_report.md"

# Measured taxonomy (512 real rollouts, run ...ab03684-7068) — the comparison
# baseline. NOTE: the measured "drift 19%" INCLUDED the dup-open x2 variant
# (8%); the parser splits them (drift ~11% + dup_open ~8%).
MEASURED = {
    "no_meta": 34.0,
    "wellformed": 17.0,
    "swapped": 25.0,
    "drift": 11.0,
    "dup_open": 8.0,
    "reversed": 5.0,
}

CLASS_ORDER = [
    "wellformed", "no_meta", "swapped", "dup_open", "reversed",
    "drift", "truncation", "discard",
]
TIER1 = {"swapped", "dup_open", "reversed"}


def _load_env(path):
    """Export KEY=VALUE pairs from .env (values never printed)."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _prep_tokenizer():
    """Temp copy of the SFT tokenizer with extra_special_tokens stripped
    (transformers chokes on that key when loading this checkpoint)."""
    tmp = tempfile.mkdtemp(prefix="dcpo_tok_")
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "special_tokens_map.json"):
        p = os.path.join(TOKENIZER_SRC, f)
        if os.path.exists(p):
            shutil.copy(p, tmp)
    cfg_p = os.path.join(tmp, "tokenizer_config.json")
    with open(cfg_p) as f:
        cfg = json.load(f)
    cfg.pop("extra_special_tokens", None)
    with open(cfg_p, "w") as f:
        json.dump(cfg, f)
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(tmp)


def _pull_tables():
    """Download every dcpo/rollouts table json from the last N_RUNS matching runs.
    Returns list of (run_id, step, main_tail) rows."""
    import wandb
    api = wandb.Api(timeout=120)
    runs = list(api.runs(ENTITY_PROJECT,
                         filters={"display_name": RUN_NAME},
                         order="-created_at"))[:N_RUNS]
    if not runs:
        sys.exit(f"no runs named {RUN_NAME} in {ENTITY_PROJECT}")
    rows = []
    dl = tempfile.mkdtemp(prefix="dcpo_tables_")
    for r in runs:
        files = [f for f in r.files()
                 if "rollouts" in f.name and f.name.endswith(".table.json")]
        print(f"run {r.id} ({r.state}, {r.created_at}): {len(files)} table files")
        for f in files:
            f.download(root=dl, replace=True)
            with open(os.path.join(dl, f.name)) as fh:
                tbl = json.load(fh)
            cols = tbl["columns"]
            i_step = cols.index("step")
            i_tail = cols.index("main_tail")
            for row in tbl["data"]:
                rows.append((r.id, int(row[i_step]), row[i_tail] or ""))
    return rows


def _render(ids, res, tok, max_chars=700):
    """Token-level region rendering: walk positions, insert boundary markers.
    [OPEN]/[/CLOSE] = meta delimiter tokens, [META]...[/META] = content span,
    [VIOL:tok] = FORMAT_VIOLATION position, [OK] = FORMAT_OK closer,
    [ANS] = everything else. Plan replacements annotated inline."""
    span = res["meta_content_span"]
    lo, hi = span if span else (-1, -1)
    viol = set(res["violation_positions"])
    ok = set(res["format_ok_positions"])
    plan = {p: (o, n) for (p, o, n) in res["replacement_plan"]}
    parts, buf, mode = [], [], "[ANS]"

    def flush():
        if buf:
            parts.append(tok.decode(buf))
            buf.clear()

    for i, t in enumerate(ids):
        if i == lo:
            flush(); parts.append("⟦META⟧"); mode = "META"
        if i == hi:
            flush(); parts.append("⟦/META⟧"); mode = "[ANS]"
        if i in plan:
            flush()
            o, n = plan[i]
            parts.append(f"⟦REPLACE {tok.decode([o])!r}->{tok.decode([n])!r}⟧")
            continue
        if i in viol:
            flush(); parts.append(f"⟦VIOL:{tok.decode([t])!r}⟧"); continue
        if i in ok:
            flush(); parts.append(f"⟦OK:{tok.decode([t])!r}⟧"); continue
        buf.append(t)
    flush()
    s = "".join(parts)
    if len(s) > max_chars:  # keep the marker-dense region (the tail end has the answer)
        cut = s.find("⟦")
        start = max(0, cut - 120) if cut != -1 else len(s) - max_chars
        s = ("…" if start else "") + s[start:start + max_chars] + ("…" if start + max_chars < len(s) else "")
    return s.replace("\n", "⏎")


def main():
    _load_env(os.path.join(REPO, ".env"))
    rows = _pull_tables()
    print(f"total rollout rows: {len(rows)}")
    tok = _prep_tokenizer()
    decode_fn = tok.decode

    results = []          # (run_id, step, ids, res, head_cut)
    rt_attempted = rt_passed = 0
    rt_failures = []      # (idx, fmt_class, re_class)
    for (rid, step, tail) in rows:
        # MEASUREMENT ARTIFACT FLAG: the table stores the LAST 1500 chars of the
        # rollout. Every full response begins with "<think>"; a tail that does
        # not was HEAD-CUT — early delimiters may be missing, so its class is a
        # tail artifact, not the true full-sequence class. The taxonomy
        # comparison therefore uses the FULL-response subset.
        head_cut = not tail.lstrip().startswith("<think>")
        ids = tok.encode(tail, add_special_tokens=False)
        res = classify_dcpo_format(ids, None, decode_fn)
        # Round-trip proof (independent of the parser's internal §2.2 check):
        # apply the plan OURSELVES and re-classify — must come back wellformed.
        if res["fmt_class"] in TIER1:
            rt_attempted += 1
            fixed = list(ids)
            plan_ok = True
            for (pos, old_id, new_id) in res["replacement_plan"]:
                if fixed[pos] != old_id:
                    plan_ok = False
                    break
                fixed[pos] = new_id
            assert len(fixed) == len(ids), "1:1 plan changed length"
            re_cls = classify_dcpo_format(fixed, None, decode_fn)["fmt_class"] if plan_ok else "PLAN_MISMATCH"
            if plan_ok and re_cls == "wellformed":
                rt_passed += 1
            else:
                rt_failures.append((len(results), res["fmt_class"], re_cls))
        results.append((rid, step, ids, res, head_cut))

    hist = Counter(r["fmt_class"] for (_, _, _, r, _) in results)
    full = [(rid, s, ids, r, hc) for (rid, s, ids, r, hc) in results if not hc]
    hist_full = Counter(r["fmt_class"] for (_, _, _, r, _) in full)
    N, Nf = len(results), len(full)

    # ── report ──
    by_class = defaultdict(list)
    for i, (rid, step, ids, res, head_cut) in enumerate(results):
        by_class[res["fmt_class"]].append(i)

    L = []
    L.append("# DCPO v3k format-parser harness report\n")
    L.append(f"- source: last {N_RUNS} wandb runs named `{RUN_NAME}` in `{ENTITY_PROJECT}`")
    runs_seen = sorted({rid for (rid, _, _, _, _) in results})
    L.append(f"- runs: {', '.join(runs_seen)}")
    L.append(f"- rollouts classified: **{N}** (tokenized text TAILS, local SFT tokenizer); "
             f"**{Nf}** are FULL responses (tail starts with `<think>`), "
             f"{N - Nf} are HEAD-CUT by the 1500-char tail window (artifact subset)")
    L.append(f"- parser: `src.training.dcpo_region.classify_dcpo_format` "
             f"(ids {META_OPEN_DEFAULT}/{META_CLOSE_DEFAULT}/{THINK_CLOSE_DEFAULT})\n")

    L.append("## 1. Class histogram vs measured taxonomy\n")
    L.append("Training-time inputs are FULL sequences, so the **full-response column** is "
             "the honest taxonomy comparison; the all-rows column includes head-cut tails "
             "(missing early delimiters inflate close-only/discard counts).\n")
    L.append("| fmt_class | all rows | all % | full-resp | full % | measured (512-rollout baseline) |")
    L.append("|---|---|---|---|---|---|")
    for c in CLASS_ORDER:
        n, nf = hist.get(c, 0), hist_full.get(c, 0)
        m = MEASURED.get(c)
        L.append(f"| {c} | {n} | {100.0 * n / N:.1f}% | {nf} | {100.0 * nf / max(1, Nf):.1f}% "
                 f"| {f'{m:.0f}%' if m is not None else '—'} |")
    L.append("")

    L.append("## 1b. Discard sub-shapes (delimiter sequence O=<|meta|> C=<|/meta|> K=</think>)\n")
    sub = Counter()
    for i in by_class.get("discard", []):
        _, _, ids, _, hc = results[i]
        seq = "".join(
            "O" if t == META_OPEN_DEFAULT else
            "C" if t == META_CLOSE_DEFAULT else
            "K" if t == THINK_CLOSE_DEFAULT else "" for t in ids)
        sub[(seq[:12], hc)] += 1
    L.append("| shape | head_cut | n |")
    L.append("|---|---|---|")
    for (seq, hc), n in sub.most_common(12):
        L.append(f"| `{seq or '(none)'}` | {hc} | {n} |")
    L.append("")

    L.append("## 2. Round-trip validation (tier-1 replacement plans)\n")
    rate = (100.0 * rt_passed / rt_attempted) if rt_attempted else float("nan")
    L.append(f"- attempted: **{rt_attempted}** · passed: **{rt_passed}** · rate: **{rate:.1f}%**")
    if rt_failures:
        L.append(f"- FAILURES ({len(rt_failures)}):")
        for (i, fc, rc) in rt_failures:
            L.append(f"  - row {i}: {fc} -> re-classified {rc}")
    else:
        L.append("- failures: none (every emitted plan re-classifies wellformed; "
                 "plans that would fail are demoted to discard INSIDE the parser, §2.2)")
    L.append("")

    L.append("## 3. Rendered samples (5 per class, token-level region boundaries)\n")
    L.append("Markers: `⟦META⟧…⟦/META⟧` = META_CONTENT span · `⟦REPLACE 'old'->'new'⟧` = "
             "tier-1 plan position · `⟦VIOL:'tok'⟧` = FORMAT_VIOLATION · `⟦OK:'tok'⟧` = "
             "FORMAT_OK closer · `⏎` = newline.\n")
    for c in CLASS_ORDER:
        idxs = by_class.get(c, [])
        # Prefer FULL-response samples (training-realistic); pad with head-cut.
        idxs = sorted(idxs, key=lambda i: results[i][4])
        L.append(f"### {c} ({len(idxs)} rows)\n")
        for i in idxs[:5]:
            rid, step, ids, res, hc = results[i]
            L.append(f"- run `{rid[-9:]}` step {step} head_cut={hc} · plan={res['replacement_plan']} "
                     f"viol={res['violation_positions']} ok={res['format_ok_positions']} "
                     f"span={res['meta_content_span']} sig={res['has_signature']}")
            L.append(f"  ```\n  {_render(ids, res, tok)}\n  ```")
        L.append("")

    L.append("## 4. Severe (discard) eyeball — over/under-trigger check (10 samples)\n")
    L.append("Per sample: the delimiter inventory (counts/positions of "
             "`<|meta|>`/`<|/meta|>`/`</think>`) + the rendered tail. A discard is "
             "JUSTIFIED iff no 1:1 replacement or content-anchored recovery exists. "
             "FULL-response samples shown first (head-cut discards are window artifacts).\n")
    disc = sorted(by_class.get("discard", []), key=lambda i: results[i][4])
    for i in disc[:10]:
        rid, step, ids, res, hc = results[i]
        O = [j for j, t in enumerate(ids) if t == META_OPEN_DEFAULT]
        C = [j for j, t in enumerate(ids) if t == META_CLOSE_DEFAULT]
        K = [j for j, t in enumerate(ids) if t == THINK_CLOSE_DEFAULT]
        L.append(f"- run `{rid[-9:]}` step {step} head_cut={hc} · open×{len(O)}@{O} "
                 f"close×{len(C)}@{C} think_close×{len(K)}@{K} · viol={res['violation_positions']}")
        L.append(f"  ```\n  {_render(ids, res, tok, max_chars=900)}\n  ```")
    L.append("")

    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(L))
    print(f"\nreport: {REPORT_PATH}")
    print("histogram:", {c: hist.get(c, 0) for c in CLASS_ORDER})
    print(f"round-trip: {rt_passed}/{rt_attempted}")


if __name__ == "__main__":
    main()
