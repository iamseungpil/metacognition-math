#!/usr/bin/env python3
"""Build the G5-gated variant of the RV SFT-2 corpus (attempt (C') of E-095).

THE GATE
--------
``sft.py`` masks ``[prompt] + [wrong_prefix]`` and trains only what follows
(sft.py:111-116). A row is therefore useful for teaching meta emission only if
its whole ``<|meta|>...<|/meta|>`` block lands at or after the wrong_prefix
boundary. Rows that fail this teach recovery prose and say nothing about
emitting meta at all — they silently starve the very behaviour the reward later
scores.

Measured on ``rv_redirect_verify_functional.parquet`` (1763 rows) on 2026-07-26:
524 rows (29.7%) fail, and they are EXACTLY the rows whose meta delimiters are
unbalanced. Keeping the rest leaves 1239 rows with the redirect share rising
from 31.4% to 36.5% — closer to the meta arm's need without any scenario-based
selection, because the gate never looks at ``scenario``.

WHY THIS IS THE RIGHT REPLACEMENT FOR THE OLD FILTER
----------------------------------------------------
The previous SFT-2 filter required a closed thinking block. E-093 showed that
condition is satisfied only when the force-fed prefix happens to contain the
closing tag (the model never emits it in its own continuation), and the redirect
scenario cuts that prefix mid-thought by construction — so the filter was a
covert scenario selector, dropping redirect 554 -> 67. This gate is scenario
neutral by construction: it is a statement about the loss mask, not about form.

ORDERING NOTE
-------------
This is the SECOND attempt by design. The first (``sft_b2p2_rvfull``) trains on
the RAW 1763 rows because that is byte-for-byte what T1 trained on, dead rows
included, so a negative base result cannot be blamed on a deviation we invented.
Only once that literal replication has run does dropping the dead rows become an
interpretable improvement rather than another unvalidated change.

Verify the output with:
    python scripts/audit_sft2_render_mask.py --data <out> --tokenizer <tok>
G5 must read 100% on both scenarios.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "iamseungpil/metacot-rv"
HF_FILE = "data/rv_redirect_verify_functional.parquet"
META_OPEN = "<|meta|>"
META_CLOSE = "<|/meta|>"


def meta_in_trained_region(assistant: str, prefix_split_char: int) -> bool:
    """True iff the whole meta block starts at/after the loss-mask boundary.

    ``prefix_split_char`` is the character length of ``wrong_prefix``, and
    ``wrong_prefix`` is an exact character prefix of ``assistant`` in every
    source row, so a char-index comparison is the faithful analogue of the
    token-index mask that ``sft.py`` actually applies. The render-layer audit
    (``audit_sft2_render_mask.py``) re-checks the same property on final token
    ids, which is what settles it.
    """
    open_at = assistant.find(META_OPEN)
    close_at = assistant.find(META_CLOSE)
    if open_at < 0 or close_at < 0:
        return False
    return open_at >= prefix_split_char and close_at >= prefix_split_char


def build(df):
    import pandas as pd  # noqa: F401  (imported by caller; kept for clarity)

    keep, drop = [], []
    for idx, r in df.iterrows():
        messages = json.loads(r["messages"]) if isinstance(r["messages"], str) else r["messages"]
        assistant = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant"), ""
        )
        cut = int(r.get("prefix_split_char", 0) or 0)
        (keep if meta_in_trained_region(assistant, cut) else drop).append(idx)
    return keep, drop


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "data" / "rv_g5gated_sft2.parquet"))
    ap.add_argument("--src", default=None, help="local parquet override (default: pull from HF)")
    args = ap.parse_args()

    import pandas as pd

    if args.src:
        df = pd.read_parquet(args.src)
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(
            HF_REPO, HF_FILE, repo_type="dataset", token=os.environ.get("HF_TOKEN")
        )
        df = pd.read_parquet(path)

    keep, drop = build(df)
    out = df.loc[keep].reset_index(drop=True)

    # Hard invariant: every kept row must have a meta block that is trained.
    assert len(keep) + len(drop) == len(df), "row accounting lost rows"
    blob_ok = all(
        META_OPEN in (json.loads(m)[-1]["content"] if isinstance(m, str) else m[-1]["content"])
        for m in out["messages"]
    )
    assert blob_ok, "a kept row has no meta block"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)

    def share(frame):
        vc = frame["scenario"].value_counts()
        tot = int(vc.sum())
        red = int(vc.get("redirect", 0))
        return tot, red, (red / tot * 100 if tot else 0.0)

    st, sr, sp = share(df)
    kt, kr, kp = share(out)
    dropped = df.loc[drop]
    dt, dr, dp = share(dropped) if len(dropped) else (0, 0, 0.0)

    print(f"source  : {st:5d} rows  redirect {sr:4d} ({sp:4.1f}%)")
    print(f"dropped : {dt:5d} rows  redirect {dr:4d} ({dp:4.1f}%)   <- meta block inside masked prefix")
    print(f"kept    : {kt:5d} rows  redirect {kr:4d} ({kp:4.1f}%)")
    print(f"wrote {out_path}")
    print()
    print("Now verify at the render layer (G5 must be 100% on both scenarios):")
    print(f"  python scripts/audit_sft2_render_mask.py --data {out_path} --tokenizer <tok_dir>")


if __name__ == "__main__":
    main()
