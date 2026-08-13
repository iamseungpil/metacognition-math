#!/usr/bin/env python
"""Build rvfull_verify_clean.parquet: repair the VERIFY prefixes in place.

Context (EXP-0812c follow-up)
---------------------------
`src/training/sft.py:_should_mask_prefix` returns True only for
`scenario in ("", "redirect")`, so a VERIFY row can never be prefix-masked no
matter what `wrong_prefix` holds. Blanking `wrong_prefix` therefore unmasked the
verify prefixes for the loss WITHOUT removing the junk text from `messages`:
the assistant content still averages ~2010 chars in both the pre- and
post-repair corpora, and no row is truncated by max_length=4096 (max assistant
= 2951 tokens), so the junk-repetition rows are trained IN FULL.

This script does the only repair that actually removes the defect: it rewrites
the assistant content inside `messages`.

Cleaning spec (VERIFY rows only; REDIRECT rows are copied byte-identically):
  Step 1  strip meta-shaped content from the prefix P, in order
          1a closed   <|meta|> ... <|/meta|>   spans
          1b unclosed <|meta|>  -> cut to end of P
          1c          <|im_start|> ... (<|im_end|> | end of P)  spans
          1d contiguous runs of bare `confidence:/assessment:/action:` lines
  Step 2  truncate at repetition onset (60-char window sampled every 10 chars,
          appearing >= 5 times -> cut at its first occurrence)
  Step 3  rstrip, then re-append exactly "\\n\\n" (builder convention)
  Step 4  drop the row if the cleaned prefix is shorter than 40 chars

  A_new = P_clean + A[len(P_original):]     # teacher-written remainder untouched
  wrong_prefix = ""   prefix_split_char = 0 (audit-only column, kept consistent)

`\\boxed` is NOT stripped: it is legitimate in the verify scenario (the student
answers, the meta block then checks that answer).

Read-only w.r.t. the source; no network; no HF upload.
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_SRC = (
    "/tmp/claude-587327809/-home-v-seungplee/41a99d3b-d246-48cd-b893-68375dc4e059/"
    "scratchpad/pq/data/rv_redirect_verify_functional.parquet"
)
DEFAULT_OUT = (
    "/tmp/claude-587327809/-home-v-seungplee/41a99d3b-d246-48cd-b893-68375dc4e059/"
    "scratchpad/rvfull_verify_clean.parquet"
)

# ---------------------------------------------------------------- constants --
META_OPEN = "<|meta|>"
META_CLOSE = "<|/meta|>"
IM_START = "<|im_start|>"

CLOSED_META_RE = re.compile(re.escape(META_OPEN) + r".*?" + re.escape(META_CLOSE), re.DOTALL)
IM_SPAN_RE = re.compile(re.escape(IM_START) + r".*?(?:" + re.escape("<|im_end|>") + r"|\Z)", re.DOTALL)
TRIPLE_LINE_RE = re.compile(r"^\s*(?:confidence|assessment|action)\s*:", re.IGNORECASE)

WINDOW = 60      # repetition window width, chars
STRIDE = 10      # window sampling stride, chars
MIN_REPEATS = 5  # >= this many identical sampled windows == a junk loop
MIN_PREFIX_CHARS = 40  # step-4 drop guard


# ------------------------------------------------------------ step functions --
def step1a_strip_closed_meta(text: str) -> str:
    """Remove every closed <|meta|> ... <|/meta|> span."""
    return CLOSED_META_RE.sub("", text)


def step1b_cut_unclosed_meta(text: str) -> str:
    """Remove every UNCLOSED <|meta|> from its position to the end of the text."""
    idx = text.find(META_OPEN)
    return text if idx < 0 else text[:idx]


def step1c_strip_im_start(text: str) -> str:
    """Remove every <|im_start|> ... (<|im_end|> or end of text) span."""
    return IM_SPAN_RE.sub("", text)


def step1d_strip_triple_runs(text: str) -> str:
    """Remove every contiguous run of bare confidence:/assessment:/action: lines."""
    kept = [ln for ln in text.split("\n") if not TRIPLE_LINE_RE.match(ln)]
    return "\n".join(kept)


def find_repetition_onset(text: str) -> int | None:
    """Index at which a junk repetition loop starts, or None.

    Slide a WINDOW-char window every STRIDE chars. If any window is seen
    >= MIN_REPEATS times, the onset is its first occurrence. When several
    windows qualify we take the EARLIEST onset (most conservative cut).
    """
    if len(text) < WINDOW:
        return None
    first_at: dict[str, int] = {}
    counts: Counter[str] = Counter()
    for i in range(0, len(text) - WINDOW + 1, STRIDE):
        w = text[i : i + WINDOW]
        counts[w] += 1
        first_at.setdefault(w, i)
    onsets = [first_at[w] for w, c in counts.items() if c >= MIN_REPEATS]
    return min(onsets) if onsets else None


def step2_truncate_repetition(text: str) -> str:
    onset = find_repetition_onset(text)
    return text if onset is None else text[:onset]


def step3_normalize(text: str) -> str:
    """rstrip, then ensure the prefix ends with exactly one blank line."""
    return text.rstrip() + "\n\n"


def clean_prefix(prefix: str) -> tuple[str, dict[str, bool]]:
    """Apply the full spec to one prefix; report which steps changed it."""
    touched: dict[str, bool] = {}
    p = prefix
    for name, fn in (
        ("1a", step1a_strip_closed_meta),
        ("1b", step1b_cut_unclosed_meta),
        ("1c", step1c_strip_im_start),
        ("1d", step1d_strip_triple_runs),
        ("2", step2_truncate_repetition),
    ):
        after = fn(p)
        touched[name] = after != p
        p = after
    return step3_normalize(p), touched


# ------------------------------------------------------------------ helpers --
def load_messages(raw: str) -> list[dict]:
    return json.loads(raw)


def dump_messages(msgs: list[dict]) -> str:
    """Same json.dumps conventions as the source corpus (verified: all 1763 rows
    round-trip exactly under ensure_ascii=False with default separators)."""
    return json.dumps(msgs, ensure_ascii=False)


def assistant_index(msgs: list[dict]) -> int:
    for i, m in enumerate(msgs):
        if m["role"] == "assistant":
            return i
    raise ValueError("row has no assistant turn")


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "  n/a"


def quantiles(values: list[int]) -> str:
    if not values:
        return "empty"
    s = sorted(values)

    def q(p: float) -> int:
        return s[min(len(s) - 1, int(round(p * (len(s) - 1))))]

    return f"p50={q(0.50):6d}  p90={q(0.90):6d}  p99={q(0.99):6d}  max={s[-1]:6d}"


def snippet(text: str, budget: int = 400) -> str:
    """<= budget chars: head + tail so the defect region stays visible."""
    if len(text) <= budget:
        return repr(text)
    half = budget // 2
    return (
        repr(text[:half])
        + f"\n        ...[{len(text) - budget} chars omitted]...\n        "
        + repr(text[-half:])
    )


# --------------------------------------------------------------------- main --
def main(src_path: str, out_path: str) -> int:
    src_table = pq.read_table(src_path)
    src_schema = src_table.schema
    df = src_table.to_pandas()
    cols = list(df.columns)

    out_rows: list[dict] = []
    step_counts = Counter()
    dropped_short = 0
    len_before: list[int] = []
    len_after: list[int] = []
    kept_prefixes: list[str] = []  # cleaned prefix of every surviving verify row
    samples_loop: list[tuple[str, str]] = []
    samples_triple: list[tuple[str, str]] = []
    in_by_scen = Counter(df["scenario"])

    for _, row in df.iterrows():
        rec = {c: row[c] for c in cols}
        if row["scenario"] != "verify":
            out_rows.append(rec)  # redirect: byte-identical pass-through
            continue

        p_old = row["wrong_prefix"]
        msgs = load_messages(row["messages"])
        ai = assistant_index(msgs)
        a_old = msgs[ai]["content"]
        assert a_old.startswith(p_old), "assistant content does not start with wrong_prefix"

        p_new, touched = clean_prefix(p_old)
        for k, v in touched.items():
            if v:
                step_counts[k] += 1

        if len(p_new) < MIN_PREFIX_CHARS:
            dropped_short += 1
            continue

        msgs[ai] = dict(msgs[ai])
        msgs[ai]["content"] = p_new + a_old[len(p_old) :]
        rec["messages"] = dump_messages(msgs)
        rec["wrong_prefix"] = ""
        rec["prefix_split_char"] = 0

        kept_prefixes.append(p_new)
        len_before.append(len(p_old))
        len_after.append(len(p_new))

        if touched["2"] and len(samples_loop) < 2:
            samples_loop.append((p_old, p_new))
        elif touched["1d"] and not touched["2"] and len(samples_triple) < 2:
            samples_triple.append((p_old, p_new))

        out_rows.append(rec)

    out_df = pd.DataFrame(out_rows, columns=cols)
    out_df["prefix_split_char"] = out_df["prefix_split_char"].astype("int64")
    out_df["confidence_label"] = out_df["confidence_label"].astype("float64")
    out_table = pa.Table.from_pandas(
        out_df, schema=pa.schema(list(src_schema)), preserve_index=False
    )
    pq.write_table(out_table, out_path)

    # ------------------------------------------------------------ reporting --
    out_by_scen = Counter(out_df["scenario"])
    n_v_in = in_by_scen["verify"]
    print("=" * 78)
    print("STATS REPORT  build_rvfull_verify_clean.py")
    print("=" * 78)
    print(f"source : {src_path}")
    print(f"output : {out_path}")
    print()
    print("ROWS BY SCENARIO")
    for scen in sorted(set(in_by_scen) | set(out_by_scen)):
        print(f"  {scen:10s} in={in_by_scen[scen]:5d}  out={out_by_scen[scen]:5d}"
              f"  delta={out_by_scen[scen] - in_by_scen[scen]:+d}")
    print(f"  {'TOTAL':10s} in={len(df):5d}  out={len(out_df):5d}"
          f"  delta={len(out_df) - len(df):+d}")
    print()
    print(f"VERIFY ROWS TOUCHED BY EACH STEP (of {n_v_in} verify rows in)")
    labels = {
        "1a": "1a closed <|meta|>...<|/meta|> spans removed",
        "1b": "1b unclosed <|meta|> cut-to-end",
        "1c": "1c <|im_start|>... spans removed",
        "1d": "1d bare confidence/assessment/action line runs removed",
        "2": "2  truncated at repetition onset",
    }
    for k in ("1a", "1b", "1c", "1d", "2"):
        print(f"  {labels[k]:56s} {step_counts[k]:5d}  ({pct(step_counts[k], n_v_in)})")
    print()
    print("STEP 4 DROP GUARD")
    print(f"  verify rows dropped (clean prefix < {MIN_PREFIX_CHARS} chars): {dropped_short}")
    print()
    print("VERIFY PREFIX LENGTH (chars), surviving rows only")
    print(f"  before : {quantiles(len_before)}")
    print(f"  after  : {quantiles(len_after)}")
    print(f"  total chars removed: {sum(len_before) - sum(len_after):,}")
    print()

    # --------------------------------------------------------------- samples --
    print("=" * 78)
    print("VERBATIM SAMPLES (<=400 chars per side; head+tail when longer)")
    print("=" * 78)
    tagged = [("LOOP", a, b) for a, b in samples_loop] + [
        ("META-TRIPLE", a, b) for a, b in samples_triple
    ]
    for i, (tag, before, after) in enumerate(tagged, 1):
        print(f"\n--- SAMPLE {i} [{tag}]  before={len(before)} chars  after={len(after)} chars ---")
        print("  BEFORE: " + snippet(before))
        print("  AFTER : " + snippet(after))
    print()

    # ------------------------------------------------------------ assertions --
    print("=" * 78)
    print("ASSERTIONS")
    print("=" * 78)
    failures: list[str] = []

    # (i) redirect rows byte-identical to source
    src_red = df[df["scenario"] == "redirect"].reset_index(drop=True)
    out_red = out_df[out_df["scenario"] == "redirect"].reset_index(drop=True)
    if len(src_red) != len(out_red):
        failures.append(f"(i) redirect row count {len(out_red)} != source {len(src_red)}")
    else:
        diffs = [c for c in cols if not src_red[c].equals(out_red[c])]
        if diffs:
            failures.append(f"(i) redirect columns differ from source: {diffs}")
    print(f"  (i)   redirect byte-identical .............. {len(out_red)} rows compared, "
          f"{len(cols)} columns")

    # (ii) no surviving verify assistant content carries a junk loop
    loop_left = 0
    for _, row in out_df[out_df["scenario"] == "verify"].iterrows():
        msgs = load_messages(row["messages"])
        content = msgs[assistant_index(msgs)]["content"]
        if find_repetition_onset(content) is not None:
            loop_left += 1
    if loop_left:
        failures.append(f"(ii) {loop_left} verify assistant contents still loop")
    print(f"  (ii)  no repeating {WINDOW}-char window (>={MIN_REPEATS}x) "
          f"in verify assistant content ... violations={loop_left}")

    # (iii) no surviving verify prefix carries meta-shaped content.
    # `wrong_prefix` is blanked on output, so the check runs on the cleaned
    # prefixes retained from the build pass (one per surviving verify row).
    assert len(kept_prefixes) == int((out_df["scenario"] == "verify").sum()), (
        "prefix bookkeeping mismatch"
    )
    meta_left = im_left = triple_left = 0
    for p in kept_prefixes:
        if META_OPEN in p:
            meta_left += 1
        if IM_START in p:
            im_left += 1
        if any(TRIPLE_LINE_RE.match(ln) for ln in p.split("\n")):
            triple_left += 1
    if meta_left or im_left or triple_left:
        failures.append(
            f"(iii) meta residue in cleaned prefixes: <|meta|>={meta_left} "
            f"<|im_start|>={im_left} triple-lines={triple_left}"
        )
    print(f"  (iii) no <|meta|> / <|im_start|> / confidence-assessment-action line "
          f"in cleaned prefix ... {meta_left}/{im_left}/{triple_left}")

    print()
    if failures:
        print("RESULT: FAIL")
        for f in failures:
            print("  " + f)
        return 1
    print("RESULT: ALL ASSERTIONS PASS")
    print(f"WROTE: {out_path}")
    return 0


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    out = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    raise SystemExit(main(src, out))
