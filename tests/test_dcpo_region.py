"""Unit tests for build_dcpo_region_masks (TRIOBJ_DCPO_V2, spec §4 / §6.1).

PURE PYTHON. No transformers — a fake token->surface vocab provides decode_fn so
cumulative char offsets are exact. Runnable under the metaprobe env.
"""
import numpy as np

from src.training.dcpo_region import build_dcpo_region_masks

META_OPEN = 151669
META_CLOSE = 151670

# Fake token vocab: id -> surface string. decode = concatenation, so the
# cumulative char-offset table in Pass B is exact (no tokenizer merges).
VOCAB = {
    META_OPEN: "<|meta|>",
    META_CLOSE: "<|/meta|>",
    1: "The answer ",
    2: "is wrong. ",
    3: "\\boxed{5}",
    4: "review ",
    5: "confidence:",
    6: " 0",
    7: ".",
    8: "88",
    9: "8",
    10: " redirect ",
    11: "probability: ",
    12: "0.",
    13: "more text ",
    14: "final ",
    15: "\\boxed{7}",
}


def decode_fn(ids):
    return "".join(VOCAB.get(int(t), "?") for t in ids)


def _masks(ids, rmask=None):
    if rmask is None:
        rmask = [True] * len(ids)
    return build_dcpo_region_masks(ids, rmask, decode_fn, META_OPEN, META_CLOSE)


def _assert_invariants(ids, rmask, m):
    rm = np.asarray(rmask, dtype=bool)
    META = m["META_REGION"]
    CONTENT = m["META_CONTENT"]
    CONF = m["CONF"]
    ANS = m["ANSWER_REGION"]
    # CONF ⊆ META_CONTENT ⊆ META_REGION ⊆ response_mask
    assert np.all(CONF <= CONTENT)
    assert np.all(CONTENT <= META)
    assert np.all(META <= rm)
    # META_CONTENT and ANSWER disjoint
    assert not np.any(CONTENT & ANS)
    # ANSWER ∪ META_REGION == response_mask
    assert np.array_equal(ANS | META, rm)
    # tag tokens that belong to a valid block ∈ META_REGION \ META_CONTENT \ ANSWER.
    # (A STRAY close with no matching open is not a block delimiter; it is simply
    # ignored — not in META_REGION, not in CONTENT. It then falls into ANSWER as an
    # ordinary token. The invariant only constrains tags that ARE block delimiters.)
    for i, t in enumerate(ids):
        if i < len(rm) and rm[i] and t in (META_OPEN, META_CLOSE) and META[i]:
            assert not CONTENT[i] and not ANS[i]


def test_no_meta():
    ids = [1, 2, 3]
    m = _masks(ids)
    assert not m["META_REGION"].any()
    assert not m["META_CONTENT"].any()
    assert not m["CONF"].any()
    assert np.all(m["ANSWER_REGION"])  # answer == response
    _assert_invariants(ids, [True] * 3, m)


def test_single_block_with_conf():
    # [reason] <|meta|> review confidence: 0 . 88 <|/meta|> final
    ids = [1, 3, META_OPEN, 4, 5, 6, 7, 8, META_CLOSE, 14, 15]
    m = _masks(ids)
    _assert_invariants(ids, [True] * len(ids), m)
    # tags at idx 2 and 8
    assert m["META_REGION"][2] and m["META_REGION"][8]
    assert not m["META_CONTENT"][2] and not m["META_CONTENT"][8]
    # content = idx 3..7
    assert np.array_equal(np.where(m["META_CONTENT"])[0], np.array([3, 4, 5, 6, 7]))
    # CONF tokens: the " 0" "." "88" run = idx 5,6,7
    assert np.array_equal(np.where(m["CONF"])[0], np.array([5, 6, 7]))
    # answer = everything outside the block
    assert np.array_equal(np.where(m["ANSWER_REGION"])[0], np.array([0, 1, 9, 10]))


def test_missing_close_truncation():
    # <|meta|> opened, never closed, NO </think> after (true truncation at max
    # length). v3 format-fix semantics: META_REGION still runs open..end, but the
    # block is GATED — NO META_CONTENT, NO conf span (a truncated CF is useless),
    # and it is NOT a violation (length problem, not a format habit).
    ids = [1, META_OPEN, 4, 5, 6, 7, 8]
    m = _masks(ids)
    _assert_invariants(ids, [True] * len(ids), m)
    assert m["META_REGION"][1]  # open tag
    assert np.array_equal(np.where(m["META_REGION"])[0], np.array([1, 2, 3, 4, 5, 6]))
    assert not m["META_CONTENT"].any()
    assert not m["CONF"].any()
    assert not m["FORMAT_VIOLATION"].any()
    assert m["meta_unclosed"] is True
    assert m["meta_drift"] is False


def test_double_block():
    # two meta blocks; CONF only in the FIRST block that has a conf.
    ids = [META_OPEN, 5, 6, 7, 8, META_CLOSE, 1, META_OPEN, 4, META_CLOSE, 15]
    m = _masks(ids)
    _assert_invariants(ids, [True] * len(ids), m)
    # content = union of both inner spans: idx 1..4 and idx 8
    assert np.array_equal(np.where(m["META_CONTENT"])[0], np.array([1, 2, 3, 4, 8]))
    # CONF only in first block (idx 2,3,4 = "0",".", "88")
    assert np.array_equal(np.where(m["CONF"])[0], np.array([2, 3, 4]))


def test_nested_dup_open():
    # second open before close → force-close the first span, start fresh.
    ids = [META_OPEN, 4, META_OPEN, 5, 6, 7, 8, META_CLOSE, 15]
    m = _masks(ids)
    _assert_invariants(ids, [True] * len(ids), m)
    # first span content = idx 1 (between first open and second open)
    # second span content = idx 3..6
    assert m["META_CONTENT"][1]
    assert np.all(m["META_CONTENT"][[3, 4, 5, 6]])
    assert not m["META_CONTENT"][2]  # the second open tag itself
    # CONF in the SECOND block (first block has no conf)
    assert np.array_equal(np.where(m["CONF"])[0], np.array([4, 5, 6]))


def test_stray_close_ignored():
    ids = [1, META_CLOSE, 3]
    m = _masks(ids)
    assert not m["META_REGION"].any()
    assert np.all(m["ANSWER_REGION"])
    _assert_invariants(ids, [True] * 3, m)


def test_pad_interleaved():
    # pad (rmask False) inside what would be a block → block closes at last valid.
    ids = [META_OPEN, 5, 6, 7, 8, 0, 0]
    rmask = [True, True, True, True, True, False, False]
    m = _masks(ids, rmask)
    _assert_invariants(ids, rmask, m)
    # pad positions are 0 in every mask
    assert not m["META_REGION"][5] and not m["META_REGION"][6]
    assert not m["ANSWER_REGION"][5] and not m["ANSWER_REGION"][6]


def test_conf_alt_tokenization():
    # "0." "8" "8" tokenization of 0.88 inside probability: block.
    ids = [META_OPEN, 11, 12, 9, 9, META_CLOSE]
    m = _masks(ids)
    _assert_invariants(ids, [True] * len(ids), m)
    # content = idx 1..4 (probability:, 0., 8, 8)
    assert np.array_equal(np.where(m["META_CONTENT"])[0], np.array([1, 2, 3, 4]))
    # CONF = the number run "0." "8" "8" = idx 2,3,4
    assert np.array_equal(np.where(m["CONF"])[0], np.array([2, 3, 4]))


def test_free_text_conf_outside_meta_not_marked():
    # confidence stated OUTSIDE any meta block → CONF must be empty (CONF ⊆ CONTENT).
    ids = [5, 6, 7, 8, 15]  # confidence: 0 . 88 \boxed, no meta tags
    m = _masks(ids)
    assert not m["CONF"].any()
    assert not m["META_CONTENT"].any()
    _assert_invariants(ids, [True] * len(ids), m)


def test_multi_open_recovers_first_pair_when_flag_on():
    from src.training.dcpo_region import classify_dcpo_format
    # ids: <|meta|> sig <|/meta|> ... <|meta|> stray   (a stray 2nd open after a valid pair)
    O, C = 151669, 151670
    ids = [O, 1, 2, C, 9, 9, O, 3]   # first pair valid, trailing stray open
    rm = [True] * len(ids)
    dec = lambda xs: "confidence: 0.5"   # signature present
    out = classify_dcpo_format(ids, rm, dec, recover_first_pair=True)
    assert out["fmt_class"] in ("wellformed", "dup_open", "swapped", "reversed")  # recovered, not discard
