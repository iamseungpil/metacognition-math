#!/usr/bin/env python3
"""Build the META-REMOVED TWIN of the rv_functional SFT corpus (base SFT-2 data).

The matched base must differ from the meta models ONLY in the meta mechanism, so
we take the EXACT meta SFT-2 data (rv_redirect_verify_functional.parquet) and strip
just the metacognitive blocks:

  assistant = [wrong_prefix][<|meta|>...<|/meta|>][recovery]   (meta)
            -> [wrong_prefix][recovery]                        (base twin)

We keep the rv-native schema (messages / scenario / confidence_label / wrong_prefix /
prefix_split_char / split_tags). The wrong_prefix + prefix_split_char columns are
RETAINED on purpose: src/training/sft.py loss-masks the wrong_prefix tokens, so the
base trains on ONLY the recovery (no meta) with the flawed/attempt prefix masked --
byte-for-byte the same masking the meta model gets, just with the meta block removed.
Dropping wrong_prefix (to mirror v8_base_matched_strict's clean-solution schema) would
make the base ALSO learn to emit the flawed prefix = a NEW confound, breaking the
"differ only in meta" invariant. sft.py consumes this schema identically (it reads
messages + wrong_prefix, exactly as for the meta rv data).

Strip is scoped to <|meta|>...<|/meta|> ONLY (per spec). Stray <|im_start|> tokens
inside the (masked) wrong_prefix are left as-is and reported.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

META_BLOCK_RE = re.compile(r"<\|meta\|>.*?<\|/meta\|>", re.DOTALL)
# Orphan delimiters: the rv source carries student hollow-meta with MISMATCHED
# tags (open rendered as <|im_start|>, close as <|/meta|>), so a balanced-block
# strip leaves stray <|/meta|> (314 tokens / ~290 rows). Delete the bare meta
# delimiter tokens too -> ZERO <|meta|>/<|/meta|> substrings (spec invariant).
ORPHAN_DELIM_RE = re.compile(r"<\|/?meta\|>")
HF_REPO = "iamseungpil/metacot-rv"
HF_FILE = "data/rv_redirect_verify_functional.parquet"
OUT_COLS = ["messages", "scenario", "confidence_label",
            "wrong_prefix", "prefix_split_char", "split_tags"]


def strip_meta(text: str) -> str:
    """Remove every <|meta|>...<|/meta|> block (DOTALL) then any orphan meta
    delimiter token. Pure string op. Distributes over the wrong_prefix|recovery
    concatenation (no balanced block spans that boundary), so applying it to the
    assistant and to wrong_prefix keeps wrong_prefix an exact prefix."""
    return ORPHAN_DELIM_RE.sub("", META_BLOCK_RE.sub("", text or ""))


def _last_boxed(text: str) -> str:
    """Inner arg of the LAST brace-balanced \\boxed{...}, or '' if none."""
    last, i, n = "", 0, len(text)
    while i < n:
        if text.startswith("\\boxed", i):
            j = i + 6
            while j < n and text[j] in " \t":
                j += 1
            if j < n and text[j] == "{":
                depth, k = 0, j
                while k < n:
                    if text[k] == "{":
                        depth += 1
                    elif text[k] == "}":
                        depth -= 1
                        if depth == 0:
                            last = text[j + 1:k]
                            break
                    k += 1
                i = k + 1
                continue
        i += 1
    return last.strip()


def strip_row(messages: list[dict]) -> tuple[list[dict], bool]:
    """Strip meta from every assistant turn; drop assistant turns that go
    empty/whitespace-only after stripping. User turns verbatim.

    Returns (new_messages, meta_was_present). NOTE: this whole-content strip is
    used only by the self-check / for rows WITHOUT a wrong_prefix split; the build
    path uses ``strip_at_boundary`` so the masked prefix stays an exact prefix."""
    out: list[dict] = []
    meta_present = False
    for m in messages:
        if m.get("role") != "assistant":
            out.append(m)
            continue
        content = m.get("content", "")
        if "<|meta|>" in content or "<|/meta|>" in content:
            meta_present = True
        stripped = strip_meta(content)
        if not stripped.strip():
            continue  # conf/verify-only turn -> drop
        out.append({**m, "content": stripped})
    return out, meta_present


def strip_at_boundary(asst: str, wrong_prefix: str):
    """Strip meta by CLEANING the masked-prefix part and the trained-recovery part
    SEPARATELY at the original wrong_prefix boundary.

    The rv source has student hollow-meta with an UNCLOSED <|meta|> open inside the
    wrong_prefix (246 rows); a whole-content balanced strip would wrongly pair that
    open with the teacher block's close ACROSS the boundary. Splitting at the boundary
    first (wrong_prefix is an exact char-prefix of the assistant in all source rows)
    guarantees new_asst == new_wp + new_recovery, so new_wp stays an exact prefix and
    the trained recovery is identical to meta-minus-the-meta-block.

    Returns (new_asst, new_wp)."""
    cut = len(wrong_prefix)
    new_wp = strip_meta(asst[:cut])
    new_recovery = strip_meta(asst[cut:])
    return new_wp + new_recovery, new_wp


def build(df):
    """Map the meta rv DataFrame to its meta-removed twin (same schema)."""
    rows, stats = [], {
        "n_in": len(df), "n_meta_stripped": 0, "n_broken": 0,
        "n_answer_preserved": 0, "n_answer_total": 0,
        "n_imstart_residual": 0,
    }
    for _, r in df.iterrows():
        messages = json.loads(r["messages"])
        asst_turns = [i for i, m in enumerate(messages) if m.get("role") == "assistant"]
        # rv rows are single-assistant-turn; the wrong_prefix indexes that turn.
        assert len(asst_turns) == 1, "expected exactly one assistant turn"
        ai = asst_turns[0]
        orig_asst = messages[ai]["content"]
        wp_orig = str(r["wrong_prefix"])
        assert orig_asst.startswith(wp_orig), "source wrong_prefix not an exact prefix"

        if "<|meta|>" in orig_asst or "<|/meta|>" in orig_asst:
            stats["n_meta_stripped"] += 1

        new_asst, new_wp = strip_at_boundary(orig_asst, wp_orig)
        if not new_asst.strip():  # nothing left to train -> broken, drop
            stats["n_broken"] += 1
            continue

        # answer preservation: original last \boxed survives in stripped assistant.
        orig_ans = _last_boxed(orig_asst)
        if orig_ans:
            stats["n_answer_total"] += 1
            if _last_boxed(new_asst) == orig_ans:
                stats["n_answer_preserved"] += 1
            else:
                stats["n_broken"] += 1
                continue  # final answer lost -> broken, drop
        if "<|im_start|>" in new_asst:
            stats["n_imstart_residual"] += 1

        new_msgs = list(messages)
        new_msgs[ai] = {**messages[ai], "content": new_asst}

        rows.append({
            "messages": json.dumps(new_msgs, ensure_ascii=False),
            "scenario": r["scenario"],
            "confidence_label": float(r["confidence_label"]),
            "wrong_prefix": new_wp,
            "prefix_split_char": len(new_wp),
            "split_tags": r["split_tags"],
        })
    stats["n_out"] = len(rows)
    return rows, stats


def main():
    import pandas as pd

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data" / "v8_base_rv_sft.parquet"))
    ap.add_argument("--src", default=None,
                    help="local parquet override (default: pull from HF)")
    args = ap.parse_args()

    if args.src:
        df = pd.read_parquet(args.src)
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(HF_REPO, HF_FILE, repo_type="dataset",
                               token=os.environ.get("HF_TOKEN"))
        df = pd.read_parquet(path)

    rows, stats = build(df)
    out = pd.DataFrame(rows, columns=OUT_COLS)

    # hard invariant: ZERO meta delimiters anywhere in output.
    blob = out["messages"].str.cat(sep="")
    assert "<|meta|>" not in blob and "<|/meta|>" not in blob, "meta delimiter leaked!"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    stats["out_path"] = str(out_path)
    stats["answer_preservation_pct"] = round(
        100.0 * stats["n_answer_preserved"] / max(1, stats["n_answer_total"]), 2)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def _selfcheck():
    """GPU/network-free unit check of the strip + drop + preservation logic."""
    import pandas as pd

    # 1) redirect-style row: flawed prefix + meta + recovery w/ boxed.
    red = [
        {"role": "user", "content": "Q?"},
        {"role": "assistant",
         "content": "flawed work\n<|meta|>\nconfidence: 0.1\ndecision: redirect\n<|/meta|>\n"
                    "correct work\n\\boxed{28}"},
    ]
    new, present = strip_row(red)
    assert present and len(new) == 2
    asst = new[1]["content"]
    assert "<|meta|>" not in asst and "<|/meta|>" not in asst
    assert "flawed work" in asst and "\\boxed{28}" in asst
    assert _last_boxed(asst) == "28"

    # 2) verify row whose boxed lives INSIDE the meta block but ALSO in recovery.
    ver = [
        {"role": "user", "content": "Q?"},
        {"role": "assistant",
         "content": "attempt \\boxed{3}\n<|meta|>\nconfidence: 0.9 \\boxed{3}\n<|/meta|>\n"
                    "verify: substitute back \\boxed{3}"},
    ]
    nv, _ = strip_row(ver)
    assert _last_boxed("".join(m["content"] for m in nv if m["role"] == "assistant")) == "3"

    # 3) meta-only assistant turn -> dropped.
    only = [
        {"role": "user", "content": "Q?"},
        {"role": "assistant", "content": "<|meta|>\nconfidence: 0.5\n<|/meta|>"},
    ]
    no_asst, _ = strip_row(only)
    assert all(m["role"] != "assistant" for m in no_asst)

    # 3b) UNCLOSED hollow <|meta|> open in the prefix + teacher block in recovery
    #     (the boundary-spanning case): by-part strip keeps the split exact, leaves
    #     ZERO delimiters, preserves the recovery boxed.
    asst = ("attempt \\boxed{5}\n<|meta|>\nconfidence: 0.6 hollow\n\n"  # unclosed open
            "<|meta|>\nconfidence: 0.9\ndecision: verify\n<|/meta|>\n"   # teacher block
            "verify: recompute \\boxed{5}")
    wp = "attempt \\boxed{5}\n<|meta|>\nconfidence: 0.6 hollow\n\n"
    new_a, new_wp = strip_at_boundary(asst, wp)
    assert new_a.startswith(new_wp)
    assert "<|meta|>" not in new_a and "<|/meta|>" not in new_a
    assert _last_boxed(new_a) == "5" and "verify: recompute" in new_a

    # 4) build() end-to-end on a tiny frame: schema + zero-meta invariant.
    df = pd.DataFrame([{
        "messages": json.dumps(red), "scenario": "redirect",
        "confidence_label": 0.1, "wrong_prefix": "flawed work\n",
        "prefix_split_char": 11, "split_tags": json.dumps({"difficulty": "easy"}),
    }])
    rows, stats = build(df)
    assert stats["n_out"] == 1 and stats["n_meta_stripped"] == 1 and stats["n_broken"] == 0
    assert stats["n_answer_preserved"] == 1
    assert list(rows[0].keys()) == OUT_COLS
    assert "<|meta|>" not in rows[0]["messages"]
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        main()
