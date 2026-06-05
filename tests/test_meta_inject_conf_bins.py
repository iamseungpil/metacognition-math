"""E.9 BCI-RLVR unit tests for the pure binned-confidence helpers in
src/training/meta_inject.py. No torch/verl import — numpy-only module.

The tokenizer round-trip / equal-length check uses a real local tokenizer if one
with the <|meta|> tokens is discoverable; otherwise that part is SKIPPED.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.training.meta_inject import (  # noqa: E402
    build_conf_seed_ids,
    conf_seed_template,
    default_conf_bins,
)


def test_default_conf_bins_n4():
    bins = default_conf_bins(4)
    assert bins == [0.2, 0.4, 0.6, 0.8], bins


def test_default_conf_bins_formula():
    # center i = (i+1)/(n+1)
    for n in (1, 2, 3, 5, 8):
        bins = default_conf_bins(n)
        assert len(bins) == n
        for i, b in enumerate(bins):
            assert abs(b - (i + 1) / (n + 1)) < 1e-12
        # strictly increasing, all strictly inside (0, 1)
        assert all(0.0 < b < 1.0 for b in bins)
        assert all(bins[i] < bins[i + 1] for i in range(n - 1))


def test_default_conf_bins_invalid():
    with pytest.raises(ValueError):
        default_conf_bins(0)


def test_conf_seed_template_format():
    assert conf_seed_template(0.2) == "\n<|meta|>\nconfidence: 0.20\n<|/meta|>\n"
    assert conf_seed_template(0.8) == "\n<|meta|>\nconfidence: 0.80\n<|/meta|>\n"
    # 2-decimal formatting (so all bins are equal text length)
    for c in default_conf_bins(4):
        s = conf_seed_template(c)
        assert "confidence: " in s
        assert s.startswith("\n<|meta|>\n")
        assert s.endswith("<|/meta|>\n")
        # the numeric field is always exactly 4 chars "0.XY"
        num = s.split("confidence: ")[1].split("\n")[0]
        assert len(num) == 4, num


def _find_local_tokenizer():
    """A local HF tokenizer dir that contains the <|meta|>/<|/meta|> tokens."""
    import glob
    import json

    roots = [
        "/scratch/models",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..")),
    ]
    for root in roots:
        for cfg in glob.glob(os.path.join(root, "**", "tokenizer.json"), recursive=True):
            try:
                d = json.load(open(cfg))
                toks = [a.get("content") for a in d.get("added_tokens", [])]
                if "<|meta|>" in toks and "<|/meta|>" in toks:
                    return os.path.dirname(cfg)
            except Exception:
                continue
    return None


def test_build_conf_seed_ids_roundtrip_and_equal_length():
    try:
        from transformers import AutoTokenizer
    except Exception as e:  # numpy/transformers missing in this env
        pytest.skip(f"transformers unavailable: {e}")

    tdir = _find_local_tokenizer()
    if tdir is None:
        pytest.skip("no local tokenizer with <|meta|> tokens found")

    tok = AutoTokenizer.from_pretrained(tdir)
    bins = default_conf_bins(4)
    ids = [build_conf_seed_ids(tok, c) for c in bins]

    # round-trip: decoding the ids reproduces the (stripped) seed text
    for c, seed in zip(bins, ids):
        decoded = tok.decode(seed)
        assert "<|meta|>" in decoded
        assert "<|/meta|>" in decoded
        assert f"confidence: {c:.2f}" in decoded

    # all 4 bins tokenize to EQUAL length (the wrap relies on this fixed width)
    lens = [len(x) for x in ids]
    assert len(set(lens)) == 1, f"unequal seed token lengths {lens} for bins {bins}"
