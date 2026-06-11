"""dcpo_pmi pure core — splice-align (C3), aggregations, sign-gate (M3), guard (C2).

Alignment tests run on TWO tokenizers: a char-level fake with a controllable
cross-boundary merge (deterministic reproduction of the C3 bug class) and, when the
local SFT checkpoint tokenizer is present, the REAL Qwen3 tokenizer (loaded via the
verified extra_special_tokens list->dict workaround for transformers >= 4.53).
"""

import json
import math
import os
import shutil
import tempfile

import numpy as np
import pytest

from src.training.dcpo_pmi import (
    PMI_AGG_METHODS,
    SpliceAlignmentError,
    compute_pmi_rows,
    ngram_overlap_guard,
    pmi_aggregate,
    sign_gate,
    splice_and_align,
)


# ── fake tokenizer: char-level with greedy 2-char merges ─────────────────────
class FakeMergeTokenizer:
    """Greedy longest-match over MERGES (2-char strings) else single chars.

    Mimics byte-level BPE's C3 footgun: deleting the meta block can create a NEW
    merge across the prefix|continuation boundary, so token indices do NOT
    correspond between the with/without arms.
    """

    def __init__(self, merges=("ab",)):
        self.merges = tuple(merges)
        self._tokens = []
        self._ids = {}

    def _intern(self, piece):
        if piece not in self._ids:
            self._ids[piece] = len(self._tokens)
            self._tokens.append(piece)
        return self._ids[piece]

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text[i:i + 2] in self.merges:
                ids.append(self._intern(text[i:i + 2]))
                i += 2
            else:
                ids.append(self._intern(text[i]))
                i += 1
        return ids

    def decode(self, ids):
        return "".join(self._tokens[i] for i in ids)


# ── optional REAL tokenizer (Qwen3 + meta tokens, non-special) ───────────────
_CKPT = "/home/v-seungplee/sft_v8_strict_local/models/v8_meta_inside_strict_sft/checkpoint-254"


def _load_real_tokenizer():
    if not os.path.isdir(_CKPT):
        return None
    try:
        from transformers import AutoTokenizer
    except Exception:
        # ImportError when transformers is absent; AttributeError when collected
        # AFTER test_dcpo_v3_cf.py (its verl/ray auto-stub finder breaks the real
        # torch.distributed import transformers pulls in). Skip gracefully either way.
        return None
    tmp = tempfile.mkdtemp(prefix="qwen_tok_patch_")
    for fname in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        src = os.path.join(_CKPT, fname)
        if os.path.exists(src):
            shutil.copy(src, tmp)
    cfg_path = os.path.join(tmp, "tokenizer_config.json")
    with open(cfg_path) as f:
        cfg = json.load(f)
    # checkpoint saved by transformers 4.52 stores extra_special_tokens as a LIST;
    # >=4.53 expects a dict -> drop the key (ids unchanged, verified).
    cfg.pop("extra_special_tokens", None)
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)
    try:
        return AutoTokenizer.from_pretrained(tmp)
    except Exception:
        return None


REAL_TOK = _load_real_tokenizer()
needs_real_tok = pytest.mark.skipif(REAL_TOK is None, reason="local Qwen3 tokenizer unavailable")


# ═══════════════════════════════════════════════════════════════════════════
# splice_and_align (spec C3)
# ═══════════════════════════════════════════════════════════════════════════
def test_splice_align_clean_boundary_full_span():
    # no merge fires at the boundary -> the whole continuation is the common span
    tok = FakeMergeTokenizer(merges=("ab",))
    out = splice_and_align(tok, "xy", "Mz", "cd")
    s, e = out["c_span_with"]
    sw, ew = out["c_span_without"]
    assert out["c_text"] == "cd"
    assert out["with_ids"][s:e] == out["without_ids"][sw:ew]
    assert tok.decode(out["without_ids"][sw:ew]) == "cd"


def test_splice_align_boundary_merge_excludes_swallowed_token():
    # without-arm "xabcd" merges "ab" ACROSS the deleted-meta boundary: the
    # continuation's leading "b" is swallowed into a token that also carries a
    # prefix byte. The common span must EXCLUDE it (not misalign onto it).
    tok = FakeMergeTokenizer(merges=("ab",))
    out = splice_and_align(tok, "xa", "Mz", "bcd")
    s, e = out["c_span_with"]
    sw, ew = out["c_span_without"]
    assert out["c_text"] == "cd"                      # "b" correctly dropped
    assert "bcd".endswith(out["c_text"])              # byte-identity, within C
    assert out["with_ids"][s:e] == out["without_ids"][sw:ew]   # id-identical spans
    assert tok.decode(out["with_ids"][s:e]) == "cd"


def test_splice_align_token_indices_differ_between_arms():
    # the C3 footgun itself: same C-text, DIFFERENT start indices per arm
    tok = FakeMergeTokenizer(merges=("ab",))
    out = splice_and_align(tok, "xa", "Mz", "bcd")
    assert out["c_span_with"][0] != out["c_span_without"][0]
    # equal span LENGTH though (id-identical tails) -> per-token deltas align
    s, e = out["c_span_with"]
    sw, ew = out["c_span_without"]
    assert (e - s) == (ew - sw)


def test_splice_align_empty_meta_arms_identical():
    tok = FakeMergeTokenizer(merges=("ab",))
    out = splice_and_align(tok, "xy", "", "cd")
    assert out["with_ids"] == out["without_ids"]
    assert out["c_span_with"] == out["c_span_without"]
    assert out["c_text"] == "cd"


def test_splice_align_raises_when_continuation_fully_merged():
    # without-arm "ab" is ONE token spanning the boundary -> no common span left
    tok = FakeMergeTokenizer(merges=("ab",))
    with pytest.raises(SpliceAlignmentError):
        splice_and_align(tok, "a", "M", "b")


def test_splice_align_raises_on_empty_continuation():
    tok = FakeMergeTokenizer()
    with pytest.raises(SpliceAlignmentError):
        splice_and_align(tok, "xy", "Mz", "")


@needs_real_tok
def test_splice_align_real_qwen_meta_block():
    # realistic shape: think-prefix + tag-inclusive meta block + native continuation
    prefix = "<think>\nLet me work through this. The total is 6 * 3."
    meta = "\n<|meta|>\nconfidence: 0.7\nassessment: arithmetic is simple\n<|/meta|>\n"
    cont = "So the total is 18.\n</think>\n\nThe answer is \\boxed{18}."
    out = splice_and_align(REAL_TOK, prefix, meta, cont)
    s, e = out["c_span_with"]
    sw, ew = out["c_span_without"]
    assert out["with_ids"][s:e] == out["without_ids"][sw:ew]
    assert cont.endswith(out["c_text"]) and len(out["c_text"]) > 0
    # meta tag ids live in the with-arm but NEVER inside the scored span
    assert 151669 in out["with_ids"] and 151670 in out["with_ids"]
    assert 151669 not in out["with_ids"][s:e] and 151670 not in out["with_ids"][s:e]
    assert 151669 not in out["without_ids"]


@needs_real_tok
def test_splice_align_real_qwen_bpe_merge_at_boundary():
    # prefix ends "hello " and C starts "world": without-arm BPE merges " world"
    # (verified: "hello world" -> [14990, 1879]) so C's first word is swallowed —
    # the aligned span must start strictly inside the continuation.
    prefix = "hello "
    meta = "<|meta|>\nconfidence: 0.5\n<|/meta|>"
    cont = "world is round and the sky is blue today"
    out = splice_and_align(REAL_TOK, prefix, meta, cont)
    s, e = out["c_span_with"]
    sw, ew = out["c_span_without"]
    assert out["with_ids"][s:e] == out["without_ids"][sw:ew]
    assert cont.endswith(out["c_text"])
    assert len(out["c_text"]) < len(cont)            # "world" was swallowed
    assert REAL_TOK.decode(out["without_ids"][sw:ew]) == out["c_text"]


@needs_real_tok
def test_splice_align_real_qwen_raises_when_whole_c_merges():
    # single-word continuation fully merged into " world" in the without-arm
    with pytest.raises(SpliceAlignmentError):
        splice_and_align(REAL_TOK, "hello ", "<|meta|>x<|/meta|>", "world")


# ═══════════════════════════════════════════════════════════════════════════
# pmi_aggregate (spec §2 menu)
# ═══════════════════════════════════════════════════════════════════════════
_DELTA = [1.0, -0.5, 3.0, 0.5]


def test_aggregate_mean_and_max():
    assert pmi_aggregate(_DELTA, "mean") == pytest.approx(1.0)
    assert pmi_aggregate(_DELTA, "max") == pytest.approx(3.0)


def test_aggregate_sum_clip_clips_per_token():
    # clip_c=2: [1, -0.5, 2, 0.5] -> 3.0 (the 3.0 outlier is bounded, not the sum)
    assert pmi_aggregate(_DELTA, "sum_clip", clip_c=2.0) == pytest.approx(3.0)
    assert pmi_aggregate(_DELTA, "sum_clip", clip_c=10.0) == pytest.approx(4.0)
    # symmetric on the negative side
    assert pmi_aggregate([-5.0, 1.0], "sum_clip", clip_c=2.0) == pytest.approx(-1.0)


def test_aggregate_topk_mean_fraction_knob():
    # frac 0.5 of 4 tokens -> k=2 -> mean(top2) = (3 + 1) / 2
    assert pmi_aggregate(_DELTA, "topk_mean", topk_frac=0.5) == pytest.approx(2.0)
    # tiny fraction still keeps k >= 1 -> max
    assert pmi_aggregate(_DELTA, "topk_mean", topk_frac=0.01) == pytest.approx(3.0)
    # frac 1.0 -> plain mean
    assert pmi_aggregate(_DELTA, "topk_mean", topk_frac=1.0) == pytest.approx(1.0)


def test_aggregate_max_minus_min_rejected():
    with pytest.raises(ValueError, match="direction-blind"):
        pmi_aggregate(_DELTA, "max_minus_min")


def test_aggregate_unknown_method_and_empty_raise():
    with pytest.raises(ValueError):
        pmi_aggregate(_DELTA, "median")
    with pytest.raises(ValueError):
        pmi_aggregate([], "mean")


def test_aggregate_accepts_numpy_input():
    assert pmi_aggregate(np.asarray(_DELTA, dtype=np.float32), "max") == pytest.approx(3.0)


# ═══════════════════════════════════════════════════════════════════════════
# sign_gate (review M3)
# ═══════════════════════════════════════════════════════════════════════════
def test_sign_gate_table():
    assert sign_gate(2.0, True, 3.0) == pytest.approx(2.0)    # correct, in range
    assert sign_gate(5.0, True, 3.0) == pytest.approx(3.0)    # correct, clipped at c
    assert sign_gate(-2.0, True, 3.0) == 0.0                  # harmful meta: no credit
    assert sign_gate(2.0, False, 3.0) == pytest.approx(-2.0)  # wrong: mirrored penalty
    assert sign_gate(5.0, False, 3.0) == pytest.approx(-3.0)  # wrong, clipped at -c
    assert sign_gate(-2.0, False, 3.0) == 0.0                 # wrong + harmful: still 0


def test_sign_gate_invariant_correct_nonneg_wrong_nonpos():
    for agg in np.linspace(-4.0, 4.0, 17):
        assert sign_gate(float(agg), True, 1.5) >= 0.0
        assert sign_gate(float(agg), False, 1.5) <= 0.0


# ═══════════════════════════════════════════════════════════════════════════
# ngram_overlap_guard (review C2)
# ═══════════════════════════════════════════════════════════════════════════
_CONT = ("we compute the sum of the first ten terms directly and then "
         "subtract the overlap to obtain the final count")


def test_guard_flags_copied_phrase():
    # meta verbatim-copies an 8+-word run of the continuation -> every meta 8-gram
    # is also a continuation 8-gram -> ratio 1.0 -> invalid
    meta = "we compute the sum of the first ten terms directly"
    assert ngram_overlap_guard(meta, _CONT, n=8, threshold=0.25) is True


def test_guard_passes_benign_meta():
    meta = "checking my assumptions once more before committing to a final answer here"
    assert ngram_overlap_guard(meta, _CONT, n=8, threshold=0.25) is False


def test_guard_threshold_and_n_knobs():
    # meta shares only a 5-word run: invisible at n=8, caught at n=4
    meta = "note that the final count matters so I should double check everything again"
    cont = "subtract the overlap to obtain the final count matters not"
    assert ngram_overlap_guard(meta, cont, n=8, threshold=0.25) is False
    assert ngram_overlap_guard(meta, cont, n=4, threshold=0.1) is True
    # same overlap, stricter threshold -> valid again
    assert ngram_overlap_guard(meta, cont, n=4, threshold=0.9) is False


def test_guard_short_meta_has_no_ngrams():
    assert ngram_overlap_guard("looks right", _CONT, n=8, threshold=0.25) is False


def test_guard_boxed_answer_leak():
    meta = "confidence: 0.9 I am sure the result is 42 here"
    assert ngram_overlap_guard(meta, _CONT, boxed_answer="42") is True
    assert ngram_overlap_guard(meta, _CONT, boxed_answer="17") is False
    # leak check fires even on short metas (length-independent guard)
    assert ngram_overlap_guard("answer 42", _CONT, boxed_answer="42") is True
    # empty/None answers never trip it
    assert ngram_overlap_guard(meta, _CONT, boxed_answer="") is False
    assert ngram_overlap_guard(meta, _CONT, boxed_answer=None) is False


# ═══════════════════════════════════════════════════════════════════════════
# compute_pmi_rows (probe orchestrator)
# ═══════════════════════════════════════════════════════════════════════════
def _row(logp_with, logp_without, correct=True, meta="thinking it over once more",
         cont=_CONT, **extra):
    return dict(meta_text=meta, continuation_text=cont, correct=correct,
                logp_with=logp_with, logp_without=logp_without, **extra)


def test_compute_rows_end_to_end_signs_and_values():
    rows = [
        _row([-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0], correct=True),    # delta +1/tok
        _row([-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0], correct=False),   # same, wrong
        _row([-3.0, -3.0], [-1.0, -1.0], correct=True),                # harmful meta
    ]
    r, diag = compute_pmi_rows(rows, method="mean", clip_c_gate=2.0)
    assert r[0] == pytest.approx(1.0)      # correct + helpful -> +mean(delta)
    assert r[1] == pytest.approx(-1.0)     # wrong + helpful -> mirrored
    assert r[2] == 0.0                     # correct + harmful -> clipped to 0
    assert diag["guard_hits"] == [False, False, False]
    assert diag["alignment_failures"] == [False, False, False]


def test_compute_rows_gate_clip_applies():
    rows = [_row([0.0] * 4, [-3.0] * 4, correct=True)]                 # mean delta = 3
    r, _ = compute_pmi_rows(rows, method="mean", clip_c_gate=2.0)
    assert r[0] == pytest.approx(2.0)


def test_compute_rows_guard_hit_zeroes_but_keeps_raw_agg():
    leak = _row([-1.0] * 3, [-2.0] * 3, correct=True, boxed_answer="42",
                meta="surely the answer is 42")
    r, diag = compute_pmi_rows([leak], method="mean")
    assert r[0] == 0.0
    assert diag["guard_hits"] == [True]
    # raw aggregate still recorded for probe analysis (guard hits are diagnosable)
    assert diag["raw_agg"]["mean"][0] == pytest.approx(1.0)


def test_compute_rows_alignment_failure_scores_zero_nan_diag():
    rows = [
        _row(None, None, alignment_failed=True),
        _row([], []),                                   # empty spans count as failed
        _row([-1.0], [-2.0], correct=True),
    ]
    r, diag = compute_pmi_rows(rows, method="mean")
    assert r[0] == 0.0 and r[1] == 0.0
    assert r[2] == pytest.approx(1.0)
    assert diag["alignment_failures"] == [True, True, False]
    assert math.isnan(diag["raw_agg"]["sum_clip"][0])
    assert math.isnan(diag["raw_agg"]["max"][1])


def test_compute_rows_records_all_methods():
    r, diag = compute_pmi_rows([_row([-1.0, -1.0], [-2.0, -4.0], correct=True)],
                               method="sum_clip", clip_c_token=2.0)
    assert set(diag["raw_agg"]) == set(PMI_AGG_METHODS)
    assert diag["raw_agg"]["max"][0] == pytest.approx(3.0)
    assert diag["raw_agg"]["mean"][0] == pytest.approx(2.0)
    assert diag["raw_agg"]["sum_clip"][0] == pytest.approx(3.0)   # 1 + clip(3 -> 2)
    assert r[0] == pytest.approx(2.0)                             # gate clip at 2.0


def test_compute_rows_misaligned_arm_lengths_raise():
    with pytest.raises(ValueError, match="span-aligned"):
        compute_pmi_rows([_row([-1.0, -1.0], [-2.0])])


def test_compute_rows_empty_input():
    r, diag = compute_pmi_rows([])
    assert len(r) == 0 and diag["guard_hits"] == [] and diag["alignment_failures"] == []
