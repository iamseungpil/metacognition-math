#!/usr/bin/env python
"""Build rvfull_verify_clean.parquet: repair the VERIFY prefixes in place.

Context (EXP-0812c follow-up, revised 2026-08-13)
-------------------------------------------------
`src/training/sft.py:_should_mask_prefix` returns True only for
`scenario in ("", "redirect")`, so a VERIFY row can NEVER be prefix-masked no
matter what `wrong_prefix` holds.  The 2026-08-12 repair blanked `wrong_prefix`
on verify rows, which did not remove one byte from `messages`; it only promoted
the student-rollout head from "prefix" to TRAINING TARGET.

The regression that followed is a CHAT-TEMPLATE GRAMMAR bug, not a junk-loop bug:

  the Qwen3 template splits the assistant content into a reasoning slot only when
  the content contains "</think>"; otherwise it emits "<think>\\n\\n</think>\\n\\n"
  and then appends the content verbatim.  The verify assistant content BEGINS with
  the student rollout, which starts with "<think>" and in 47% of rows never closes
  it, so the rendered training target became

      <think></think><think> ...(never closed)

  for 568 / 1209 verify rows.  Trained on that, the new init opens <think> and
  never closes it in 59.6% of held-out generations, and 8.47% of those truncate.

  Measured on the defective corpus:
    <think>-unclosed generation rate  old init 0.0000  ->  new init 0.5960
    P(truncate | unclosed) 8.47%   vs   P(truncate | closed) 1.30%

Cleaning spec (C1).  VERIFY rows only; REDIRECT rows are copied byte-identically.
Let P = row['wrong_prefix'] and A = the assistant content inside row['messages']
(A.startswith(P) holds for all 1209 verify rows).

  step1   remove meta-shaped content from P by LINE-ORIENTED SPAN DELETION.
          Anchor a span at a standalone `confidence: <float in [0,1]>` line;
          extend forward while lines stay inside the meta grammar (key lines,
          the meta prose that belongs to the block, or an opening delimiter);
          end the span at the first line that leaves the grammar.  Delete ONLY
          that span.  Then delete stray lone delimiter lines
          (<|meta|>, <|/meta|>, <|im_start|>, <|im_end|>).
          NEVER truncate-to-end: `\\boxed` sits at ~97% of the prefix (p50 char
          offset 757) and `</think>` often follows the meta block, so the
          first-draft truncate-to-end steps destroyed exactly the two things
          that must survive.
  step1e  delete lines that are entirely foreign-script decoder garbage
          (Arabic / Thai / CJK / Hangul / Cyrillic ... ).  Greek and Latin
          letters are KEPT so legitimate math unicode survives.
  step2   repetition truncation with a NON-OVERLAPPING CONSECUTIVE definition:
          a 60-char window repeated >= 5 times back-to-back.  Cut at the start
          of the run, then rewind to the preceding newline.  (The first draft
          used a sampled/sliding definition that fired on 99 rows, 43 of which
          were not real loops.)
  step5   THE LOAD-BEARING STEP.  Normalize think grammar: strip every <think>
          and </think> from the cleaned P, then re-wrap
              P_final = "<think>\\n" + body.strip() + "\\n</think>\\n\\n"
          The rendered training target now contains </think>, so the template
          splits it into the reasoning slot and the rendered text is balanced.
  step6   cap the cleaned P at 2000 chars, rewinding to the preceding newline,
          then re-apply step5's wrap so the tags stay balanced.

  A_new = P_final + A[len(P):]      # teacher-written remainder untouched
  messages: only the assistant content is replaced; the user turn is byte-identical
  wrong_prefix = ""   prefix_split_char = 0   (audit-only column, kept consistent)
  scenario / confidence_label / split_tags unchanged; column order + dtypes preserved

NO minimum-length drop, NO row drops, NO mask restore (F1: restoring the mask is
an inert lever for verify rows).  All 1763 rows survive.

Read-only w.r.t. the source; no network; no HF upload; no amlt submission.
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from collections import Counter

import numpy as np
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

# candidate local locations of the b2p2_rvfull_sft tokenizer (real render if present)
TOKENIZER_CANDIDATES = [
    "models/b2p2_rvfull_sft",
    "/home/v-seungplee/metacognition-math/models/b2p2_rvfull_sft",
]
# fall-back Qwen3 chat template for a jinja cross-check of the simulated rule
JINJA_CANDIDATES = [
    "/home/v-seungplee/sft_e20a_local/chat_template.jinja",
]

# ---------------------------------------------------------------- constants --
META_OPEN = "<|meta|>"
META_CLOSE = "<|/meta|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
DELIMS = (META_OPEN, META_CLOSE, IM_START, IM_END)
OPEN_DELIMS = (META_OPEN, IM_START)
BOXED = "\\boxed"

# a standalone `confidence: <float in [0,1]>` line -- the meta-block anchor
CONF_ANCHOR_RE = re.compile(r"^[ \t]*confidence:[ \t]*(?:0?\.\d+|1(?:\.0+)?|0(?:\.0+)?)[ \t]*$")
# key lines of the meta grammar
KEY_LINE_RE = re.compile(
    r"^[ \t]*(?:confidence|assessment|action|inaction|decision|study_need|dualcheck|verification)"
    r"[ \t]*:",
    re.IGNORECASE,
)
# lines that must never be swallowed by a meta span (real content / structure)
HARD_STOP_RE = re.compile(r"(?:\\boxed|</think>|<think>|" + re.escape(META_CLOSE) + r"|" + re.escape(IM_END) + r")")

WINDOW = 60       # legacy constant, kept for callers; unused by the period scan
MIN_REPEATS = 5   # >= this many CONSECUTIVE NON-OVERLAPPING copies == a junk loop
PREFIX_CAP = 2000  # step6 cap, chars
# Period-agnostic loop scan (2026-08-13 fix). The first cut fixed the window at 60
# chars and tested text[i:i+60] == text[i:i+60]*5, which can only see loops whose
# period DIVIDES 60. Real junk loops in this corpus have periods 9, 14, 19, 21, 34
# and 67, so 49 rows (11.9% of the trained <think> body) sailed through — in a
# corpus whose whole purpose is to stop degenerate generation. P3 missed it too
# because the probe reused the same 60-char detector it was auditing.
MIN_PERIOD = 4     # shorter than this is not a "loop", it is punctuation
MAX_PERIOD = 300   # observed junk periods are <= 67; 300 is generous headroom
MIN_RUN_CHARS = 200  # a periodic span must cover at least this many chars to count.
                     # Guards legitimate math ("1 + 1 + 1 + 1 + 1", ~20 chars) while
                     # the real loops span 1,462-1,904 chars.

BOXED_FLOOR = 340
EXPECTED_ROWS = 1763
TARGET_P99_CAP = 3000


# ---------------------------------------------------------- step 1: meta spans --
def _is_blank(line: str) -> bool:
    return line.strip() == ""


def _is_lone_delim(line: str) -> bool:
    return line.strip() in DELIMS


def _is_open_delim(line: str) -> bool:
    return line.strip() in OPEN_DELIMS


def _in_meta_grammar(line: str) -> bool:
    """True if `line` may be swallowed while extending a meta span forward.

    A line stays inside the block when it is a key line, an opening delimiter,
    or meta prose (any non-blank line that carries no real content markers).
    A blank line ENDS the block, which is what separates the meta triple from
    the reasoning that follows it in every observed row.
    """
    if _is_blank(line):
        return False
    if HARD_STOP_RE.search(line):
        return False
    if KEY_LINE_RE.match(line):
        return True
    if _is_open_delim(line):
        return True
    return True  # meta prose belonging to the block


def step1_delete_meta_spans(text: str) -> str:
    """Delete every meta block as a LINE SPAN anchored at a confidence line,
    then delete stray lone delimiter lines."""
    lines = text.split("\n")
    drop = [False] * len(lines)
    i = 0
    while i < len(lines):
        if CONF_ANCHOR_RE.match(lines[i]):
            j = i
            drop[j] = True
            j += 1
            while j < len(lines) and _in_meta_grammar(lines[j]):
                drop[j] = True
                j += 1
            i = j
            continue
        i += 1
    kept: list[str] = []
    for k, ln in enumerate(lines):
        if drop[k]:
            continue
        # stray lone delimiters left behind by the span deletion (or never inside one)
        if _is_lone_delim(ln):
            continue
        if any(d in ln for d in DELIMS):
            # a delimiter fused with decoder garbage, e.g. '<|im_start|>:right:...'.
            # Delimiters are never legitimate prefix content: strip the token and
            # drop the line when nothing substantive is left.
            for d in DELIMS:
                ln = ln.replace(d, "")
            if not ln.strip():
                continue
        kept.append(ln)
    return "\n".join(kept)


# ------------------------------------------------- step 1e: decoder garbage --
def _is_foreign_letter(ch: str) -> bool:
    """A letter/mark from a script that never appears in this corpus' math."""
    if ord(ch) < 128:
        return False
    cat = unicodedata.category(ch)
    if cat[0] != "L" and cat not in ("Mn", "Mc"):
        return False  # math symbols (Sm/So), punctuation, dashes -> keep
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return True
    return not name.startswith(("GREEK", "LATIN"))


def _is_garbage_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    foreign = sum(1 for ch in s if _is_foreign_letter(ch))
    return foreign / len(s) > 0.5


def step1e_drop_garbage_lines(text: str) -> str:
    return "\n".join(ln for ln in text.split("\n") if not _is_garbage_line(ln))


# --------------------------------------------------- step 2: repetition loop --
def find_consecutive_repeat_onset(text: str, min_repeats: int = MIN_REPEATS,
                                  min_period: int = MIN_PERIOD,
                                  max_period: int = MAX_PERIOD,
                                  min_run_chars: int = MIN_RUN_CHARS) -> int | None:
    """Earliest index where some substring repeats back-to-back and degenerately.

    Period-agnostic: scans every period p in [min_period, max_period] instead of
    assuming the period divides a fixed 60-char window. A hit needs BOTH
      - at least `min_repeats` consecutive non-overlapping copies, and
      - a total periodic span of at least `min_run_chars` characters,
    so short legitimate repetition ("1 + 1 + 1 + 1 + 1") is not a loop while the
    corpus's 1.4k-1.9k char junk runs are. Returns the earliest onset, or None.

    text[i:i+p] repeats k times exactly when text[j] == text[j+p] for every j in
    [i, i + p*(k-1)), so each period reduces to finding a long run of True in the
    shifted-equality array — one vectorized pass per period.
    """
    n = len(text)
    if n < max(min_period * min_repeats, min_run_chars):
        return None
    a = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    best: int | None = None
    hi = min(max_period, n // min_repeats)
    for p in range(min_period, hi + 1):
        need = p * (min_repeats - 1)          # matching positions required
        if n - p < need:
            continue
        eq = a[:-p] == a[p:]
        if not eq.any():
            continue
        false_at = np.flatnonzero(~eq)
        starts = np.concatenate(([0], false_at + 1))
        ends = np.concatenate((false_at, [eq.size]))
        lens = ends - starts
        # span covered by the repeat is run_len + p chars
        ok = np.flatnonzero((lens >= need) & (lens + p >= min_run_chars))
        if ok.size:
            i = int(starts[ok[0]])
            if best is None or i < best:
                best = i
    return best


def step2_truncate_repetition(text: str) -> str:
    onset = find_consecutive_repeat_onset(text)
    if onset is None:
        return text
    nl = text.rfind("\n", 0, onset)
    return text[:nl] if nl > 0 else text[:onset]


# ------------------------------------------- step 5: think-grammar normalize --
THINK_TAG_RE = re.compile(r"</?think>")
MULTI_NL_RE = re.compile(r"\n{3,}")


def _wrap_think(body: str) -> str:
    body = MULTI_NL_RE.sub("\n\n", body.strip())
    return "<think>\n" + body + "\n</think>\n\n"


def step5_normalize_think(text: str) -> str:
    return _wrap_think(THINK_TAG_RE.sub("", text))


# ------------------------------------------------------------ step 6: cap --
def step6_cap(p_final: str, cap: int = PREFIX_CAP) -> tuple[str, bool]:
    if len(p_final) <= cap:
        return p_final, False
    cut = p_final[:cap]
    nl = cut.rfind("\n")
    if nl > 0:
        cut = cut[:nl]
    body = THINK_TAG_RE.sub("", cut)
    body = re.sub(r"<[^<>\n]*$", "", body)  # a tag chopped in half by the cap
    return _wrap_think(body), True


# ------------------------------------------------------- full clean pipeline --
def clean_prefix(prefix: str) -> tuple[str, dict[str, bool]]:
    touched: dict[str, bool] = {}
    p = prefix
    for name, fn in (("1", step1_delete_meta_spans),
                     ("1e", step1e_drop_garbage_lines),
                     ("2", step2_truncate_repetition)):
        after = fn(p)
        touched[name] = after != p
        p = after
    p5 = step5_normalize_think(p)
    touched["5"] = p5 != p
    p6, capped = step6_cap(p5)
    touched["6"] = capped
    return p6, touched


# ----------------------------------------------- chat-template render check --
def simulate_render(content: str) -> str:
    """The documented Qwen3 assistant rule, transcribed from the jinja source:

        if '</think>' in content:
            reasoning = content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n')
            content   = content.split('</think>')[-1].lstrip('\\n')
        emit '<think>\\n' + reasoning.strip('\\n') + '\\n</think>\\n\\n' + content.lstrip('\\n')

    When the content has no '</think>' the reasoning slot is empty, so the
    template emits '<think>\\n\\n</think>\\n\\n' and then the content verbatim --
    which is how an unclosed student <think> ended up doubled in the target.
    """
    reasoning = ""
    body = content
    if "</think>" in content:
        reasoning = content.split("</think>")[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        body = content.split("</think>")[-1].lstrip("\n")
    return "<think>\n" + reasoning.strip("\n") + "\n</think>\n\n" + body.lstrip("\n")


def load_real_renderer():
    """Return (fn, label) rendering the assistant target with the real chat
    template when one is available locally, else (None, reason)."""
    try:
        from transformers import AutoTokenizer  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover
        return None, f"transformers unavailable ({exc.__class__.__name__})"
    import os
    for path in TOKENIZER_CANDIDATES:
        if os.path.isdir(path):
            try:
                tok = AutoTokenizer.from_pretrained(path)
            except Exception as exc:
                return None, f"{path}: load failed ({exc.__class__.__name__})"

            def _render(msgs, _tok=tok):
                return _tok.apply_chat_template(msgs, tokenize=False)

            return _render, f"AutoTokenizer({path})"
    return None, "models/b2p2_rvfull_sft not present locally"


def load_jinja_crosscheck():
    """Optional: render with a local Qwen3 chat_template.jinja to prove the
    simulated rule agrees with a real template of the same family."""
    import os
    try:
        from jinja2 import Environment  # noqa: PLC0415
    except Exception:
        return None, "jinja2 unavailable"
    for path in JINJA_CANDIDATES:
        if os.path.isfile(path):
            src = open(path, encoding="utf-8").read()
            env = Environment(trim_blocks=False, lstrip_blocks=False)
            env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
            tpl = env.from_string(src)

            def _render(msgs, _tpl=tpl):
                return _tpl.render(messages=msgs, add_generation_prompt=False,
                                   tools=None, enable_thinking=True)

            return _render, path
    return None, "no local chat_template.jinja"


def think_unclosed(rendered_assistant: str) -> bool:
    """True when the rendered text has a <think> with no matching </think>."""
    depth = 0
    for m in re.finditer(r"</?think>", rendered_assistant):
        if m.group(0) == "<think>":
            depth += 1
        else:
            depth -= 1
            if depth < 0:
                return True  # stray close == malformed too
    return depth != 0


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


def assistant_content(raw: str) -> str:
    msgs = load_messages(raw)
    return msgs[assistant_index(msgs)]["content"]


def pct(n: int, d: int) -> str:
    return f"{100.0 * n / d:5.1f}%" if d else "  n/a"


def quantiles(values: list[int]) -> str:
    if not values:
        return "empty"
    a = np.asarray(values)
    return (f"p50={int(np.percentile(a, 50)):6d}  p90={int(np.percentile(a, 90)):6d}  "
            f"p99={int(np.percentile(a, 99)):6d}  max={int(a.max()):6d}")


def snippet(text: str, budget: int = 500) -> str:
    """<= budget chars: head + tail so the defect region stays visible."""
    if len(text) <= budget:
        return repr(text)
    half = budget // 2
    return (repr(text[:half])
            + f"\n            ...[{len(text) - budget} chars omitted]...\n            "
            + repr(text[-half:]))


# --------------------------------------------------------------------- main --
def main(src_path: str, out_path: str) -> int:
    src_table = pq.read_table(src_path)
    src_schema = src_table.schema
    df = src_table.to_pandas()
    cols = list(df.columns)

    real_render, render_label = load_real_renderer()
    jinja_render, jinja_label = load_jinja_crosscheck()

    def render_target(msgs: list[dict]) -> str:
        """Rendered assistant TARGET (the text after '<|im_start|>assistant\\n')."""
        if real_render is not None:
            full = real_render(msgs)
            head = "<|im_start|>assistant\n"
            k = full.rfind(head)
            body = full[k + len(head):] if k >= 0 else full
            return body.split("<|im_end|>")[0]
        return simulate_render(msgs[assistant_index(msgs)]["content"])

    out_rows: list[dict] = []
    step_counts: Counter[str] = Counter()
    len_before: list[int] = []
    len_after: list[int] = []
    tgt_before: list[int] = []
    tgt_after: list[int] = []
    kept_prefixes: list[str] = []
    unclosed_before = unclosed_after = 0
    boxed_before = boxed_after = 0
    samples: dict[str, list[tuple[int, str, str]]] = {"LOOP": [], "META": [], "UNCLOSED": []}
    in_by_scen = Counter(df["scenario"])
    jinja_agree = jinja_disagree = 0

    for ridx, row in df.iterrows():
        rec = {c: row[c] for c in cols}
        if row["scenario"] != "verify":
            out_rows.append(rec)  # redirect: byte-identical pass-through
            continue

        p_old = row["wrong_prefix"]
        msgs = load_messages(row["messages"])
        ai = assistant_index(msgs)
        a_old = msgs[ai]["content"]
        assert a_old.startswith(p_old), f"row {ridx}: assistant content does not start with wrong_prefix"

        had_unclosed_prefix = "</think>" not in p_old
        if think_unclosed(render_target(msgs)):
            unclosed_before += 1
        tgt_before.append(len(render_target(msgs)))
        if BOXED in p_old:
            boxed_before += 1

        p_new, touched = clean_prefix(p_old)
        for k, v in touched.items():
            if v:
                step_counts[k] += 1

        new_msgs = [dict(m) for m in msgs]
        new_msgs[ai]["content"] = p_new + a_old[len(p_old):]
        rec["messages"] = dump_messages(new_msgs)
        rec["wrong_prefix"] = ""
        rec["prefix_split_char"] = 0

        rendered = render_target(new_msgs)
        if think_unclosed(rendered):
            unclosed_after += 1
        tgt_after.append(len(rendered))
        if BOXED in p_new:
            boxed_after += 1
        if jinja_render is not None:
            full = jinja_render(new_msgs)
            head = "<|im_start|>assistant\n"
            k = full.rfind(head)
            got = (full[k + len(head):] if k >= 0 else full).split("<|im_end|>")[0]
            if got.rstrip("\n") == simulate_render(new_msgs[ai]["content"]).rstrip("\n"):
                jinja_agree += 1
            else:
                jinja_disagree += 1

        kept_prefixes.append(p_new)
        len_before.append(len(p_old))
        len_after.append(len(p_new))

        if touched["2"] and len(samples["LOOP"]) < 2:
            samples["LOOP"].append((ridx, p_old, p_new))
        elif touched["1"] and not touched["2"] and len(samples["META"]) < 2:
            samples["META"].append((ridx, p_old, p_new))
        elif had_unclosed_prefix and not touched["2"] and len(samples["UNCLOSED"]) < 1:
            samples["UNCLOSED"].append((ridx, p_old, p_new))

        out_rows.append(rec)

    out_df = pd.DataFrame(out_rows, columns=cols)
    out_df["prefix_split_char"] = out_df["prefix_split_char"].astype("int64")
    out_df["confidence_label"] = out_df["confidence_label"].astype("float64")
    out_table = pa.Table.from_pandas(out_df, schema=pa.schema(list(src_schema)),
                                     preserve_index=False)
    pq.write_table(out_table, out_path)

    # ------------------------------------------------------------ reporting --
    out_by_scen = Counter(out_df["scenario"])
    n_v = in_by_scen["verify"]
    print("=" * 78)
    print("STATS REPORT  build_rvfull_verify_clean.py   (spec C1)")
    print("=" * 78)
    print(f"source   : {src_path}")
    print(f"output   : {out_path}")
    print(f"renderer : {render_label if real_render else 'SIMULATED documented rule (' + render_label + ')'}")
    print(f"jinja x-check : {jinja_label}  agree={jinja_agree}  disagree={jinja_disagree}")
    print()
    print("ROWS BY SCENARIO")
    for scen in sorted(set(in_by_scen) | set(out_by_scen)):
        print(f"  {scen:10s} in={in_by_scen[scen]:5d}  out={out_by_scen[scen]:5d}"
              f"  delta={out_by_scen[scen] - in_by_scen[scen]:+d}")
    print(f"  {'TOTAL':10s} in={len(df):5d}  out={len(out_df):5d}"
          f"  delta={len(out_df) - len(df):+d}")
    print()
    print(f"VERIFY ROWS TOUCHED BY EACH STEP (of {n_v} verify rows)")
    labels = {
        "1": "1   meta-block line spans + stray delimiters deleted",
        "1e": "1e  foreign-script decoder-garbage lines deleted",
        "2": "2   truncated at consecutive 60-char x5 repeat run",
        "5": "5   think grammar normalized (strip tags + re-wrap)",
        "6": f"6   cleaned prefix capped at {PREFIX_CAP} chars",
    }
    for k in ("1", "1e", "2", "5", "6"):
        print(f"  {labels[k]:56s} {step_counts[k]:5d}  ({pct(step_counts[k], n_v)})")
    print()
    print("VERIFY PREFIX LENGTH (chars)")
    print(f"  before : {quantiles(len_before)}")
    print(f"  after  : {quantiles(len_after)}")
    print(f"  total chars removed from prefixes: {sum(len_before) - sum(len_after):,}")
    print()
    print("TRAINED TARGET (rendered assistant text) LENGTH (chars)")
    print(f"  before : {quantiles(tgt_before)}")
    print(f"  after  : {quantiles(tgt_after)}")
    print()
    print("GROUNDING / GRAMMAR")
    print(f"  \\boxed-bearing verify prefixes : before={boxed_before:4d}   after={boxed_after:4d}"
          f"   (floor {BOXED_FLOOR})")
    print(f"  rendered-unclosed <think> rows  : before={unclosed_before:4d}   after={unclosed_after:4d}"
          f"   (target 0)")
    print()

    # --------------------------------------------------------------- samples --
    print("=" * 78)
    print("VERBATIM SAMPLES (<=500 chars per side; head+tail when longer)")
    print("=" * 78)
    tagged = ([("FORMER LOOP", *s) for s in samples["LOOP"]]
              + [("FORMER META-TRIPLE", *s) for s in samples["META"]]
              + [("UNCLOSED <think>", *s) for s in samples["UNCLOSED"]])
    for i, (tag, ridx, before, after) in enumerate(tagged, 1):
        print(f"\n--- SAMPLE {i} [{tag}]  src_row={ridx}  before={len(before)} chars  "
              f"after={len(after)} chars ---")
        print("  BEFORE: " + snippet(before))
        print("  AFTER : " + snippet(after))
    print()

    # ------------------------------------------------------------ assertions --
    print("=" * 78)
    print("ASSERTIONS")
    print("=" * 78)
    failures: list[str] = []

    ok = unclosed_after == 0
    failures += [] if ok else [f"(A) rendered-unclosed <think> = {unclosed_after} != 0"]
    print(f"  (A) rendered-unclosed <think> == 0 ................... {unclosed_after:5d}   "
          f"{'PASS' if ok else 'FAIL'}")

    ok = boxed_after >= BOXED_FLOOR
    failures += [] if ok else [f"(B) \\boxed-bearing prefixes {boxed_after} < {BOXED_FLOOR}"]
    print(f"  (B) \\boxed-bearing prefixes >= {BOXED_FLOOR} ............. {boxed_after:5d}   "
          f"{'PASS' if ok else 'FAIL'}")

    ok = len(out_df) == EXPECTED_ROWS
    failures += [] if ok else [f"(C) row count {len(out_df)} != {EXPECTED_ROWS}"]
    print(f"  (C) row count == {EXPECTED_ROWS} ............................ {len(out_df):5d}   "
          f"{'PASS' if ok else 'FAIL'}")

    src_red = df[df["scenario"] == "redirect"].reset_index(drop=True)
    out_red = out_df[out_df["scenario"] == "redirect"].reset_index(drop=True)
    if len(src_red) != len(out_red):
        diffs = [f"row count {len(out_red)} != {len(src_red)}"]
    else:
        diffs = [c for c in cols if not src_red[c].equals(out_red[c])]
    ok = not diffs
    failures += [] if ok else [f"(D) redirect rows differ from source: {diffs}"]
    print(f"  (D) redirect rows byte-identical ({len(out_red)} rows x {len(cols)} cols) . "
          f"{'PASS' if ok else 'FAIL ' + str(diffs)}")

    meta_left = im_left = conf_left = 0
    for p in kept_prefixes:
        if META_OPEN in p:
            meta_left += 1
        if IM_START in p:
            im_left += 1
        if any(CONF_ANCHOR_RE.match(ln) for ln in p.split("\n")):
            conf_left += 1
    ok = not (meta_left or im_left or conf_left)
    failures += [] if ok else [f"(E) residue: <|meta|>={meta_left} <|im_start|>={im_left} "
                               f"standalone-confidence={conf_left}"]
    print(f"  (E) no <|meta|> / <|im_start|> / standalone confidence line "
          f".. {meta_left}/{im_left}/{conf_left}   {'PASS' if ok else 'FAIL'}")

    loop_left = 0
    for raw in out_df.loc[out_df["scenario"] == "verify", "messages"]:
        if find_consecutive_repeat_onset(assistant_content(raw)) is not None:
            loop_left += 1
    ok = loop_left == 0
    failures += [] if ok else [f"(F) {loop_left} verify assistant contents still loop"]
    print(f"  (F) no 5x consecutive non-overlapping {WINDOW}-char repeat ... {loop_left:5d}   "
          f"{'PASS' if ok else 'FAIL'}")

    p99_tgt = int(np.percentile(np.asarray(tgt_after), 99))
    ok = p99_tgt <= TARGET_P99_CAP
    failures += [] if ok else [f"(G) trained-target p99 {p99_tgt} > {TARGET_P99_CAP}"]
    print(f"  (G) trained-target p99 <= {TARGET_P99_CAP} chars ................ {p99_tgt:5d}   "
          f"{'PASS' if ok else 'FAIL'}")

    ok = jinja_disagree == 0
    failures += [] if ok else [f"(H) jinja/simulation disagree on {jinja_disagree} rows"]
    print(f"  (H) real-jinja render == simulated rule .............. {jinja_disagree:5d}   "
          f"{'PASS' if ok else 'FAIL'}  (informational when no template found)")

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
