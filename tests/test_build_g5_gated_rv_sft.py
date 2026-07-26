"""Unit tests for the G5 gate used to build the RV SFT-2 (C') corpus.

The gate decides whether a row can teach meta emission at all: ``sft.py`` masks
``[prompt] + [wrong_prefix]``, so a row is only useful if its whole
``<|meta|>...<|/meta|>`` block sits at or after that boundary. Getting this
predicate wrong is silent — the corpus still trains, it just stops teaching the
behaviour the reward later scores — so it is worth pinning down.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from build_g5_gated_rv_sft import meta_in_trained_region  # noqa: E402


def test_block_after_boundary_is_trained():
    asst = "wrong reasoning here<|meta|>check<|/meta|>recovery"
    cut = len("wrong reasoning here")
    assert meta_in_trained_region(asst, cut) is True


def test_block_starting_exactly_at_boundary_is_trained():
    """The boundary is half-open: training starts AT ``prefix_split_char``."""
    asst = "<|meta|>check<|/meta|>recovery"
    assert meta_in_trained_region(asst, 0) is True


def test_block_entirely_inside_masked_prefix_is_rejected():
    asst = "<|meta|>hollow<|/meta|>flawed prefix continues|RECOVERY"
    cut = asst.index("|RECOVERY") + 1
    assert meta_in_trained_region(asst, cut) is False


def test_block_straddling_the_boundary_is_rejected():
    """Open masked, close trained: the model never sees the opening tag as a
    target, so the row cannot teach emission even though a close survives."""
    asst = "prefix<|meta|>spans the cut<|/meta|>tail"
    cut = asst.index("spans")
    assert meta_in_trained_region(asst, cut) is False


def test_row_without_meta_is_rejected():
    assert meta_in_trained_region("no meta at all", 0) is False


def test_row_with_open_but_no_close_is_rejected():
    assert meta_in_trained_region("x<|meta|>unclosed forever", 1) is False


def test_gate_never_reads_the_scenario():
    """Scenario neutrality is the whole point of replacing the old filter, so
    the predicate must depend on nothing but the text and the boundary."""
    import inspect

    src = inspect.getsource(meta_in_trained_region)
    assert "scenario" not in src
    assert "think" not in src


def test_real_corpus_gate_matches_measured_counts():
    """Regression pin on the 0726 measurement: 1763 -> 1239 kept, 524 dropped,
    redirect share 31.4% -> 36.5%. Skips when the corpus is not staged."""
    import pytest

    src = ROOT / "data" / "rv_redirect_verify_functional.parquet"
    if not src.exists():
        pytest.skip("rv_redirect_verify_functional.parquet not staged locally")

    import json

    import pandas as pd

    df = pd.read_parquet(src)
    kept = []
    for _, r in df.iterrows():
        messages = json.loads(r["messages"])
        asst = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "assistant"), ""
        )
        if meta_in_trained_region(asst, int(r["prefix_split_char"])):
            kept.append(r["scenario"])

    assert len(df) == 1763
    assert len(kept) == 1239
    assert kept.count("redirect") == 452
