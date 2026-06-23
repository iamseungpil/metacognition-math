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
    _MAX_SPLICE_BOUNDARY_DROPS,
    PMI_AGG_METHODS,
    SpliceAlignmentError,
    compute_pmi_rows,
    ngram_overlap_guard,
    pmi_aggregate,
    sign_gate,
    splice_and_align,
    split_first_meta,
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


class ParityTokenizer(FakeMergeTokenizer):
    """Pathological divergence (round 2 M-B repro): tokenizes in 2-char pieces
    when len(text) is EVEN, single chars when ODD — the with/without arms get
    incompatible granularity over the WHOLE shared continuation, so the
    common-tail refinement would otherwise walk every token (O(L^2))."""

    def encode(self, text, add_special_tokens=False):
        if len(text) % 2 == 0:
            pieces = [text[i:i + 2] for i in range(0, len(text), 2)]
        else:
            pieces = list(text)
        return [self._intern(p) for p in pieces]


def test_splice_align_caps_boundary_drops_on_pathological_divergence():
    # with-arm "x"+"y"+cont has EVEN length (2-char tokens), without-arm "x"+cont
    # ODD (1-char tokens): tails never id-match before the cap. Continuation is
    # long enough that uncapped refinement would need ~3x the cap in drops.
    cont = "abcdefghij" * (_MAX_SPLICE_BOUNDARY_DROPS // 5)   # 512 chars
    tok = ParityTokenizer(merges=())
    with pytest.raises(SpliceAlignmentError, match="boundary"):
        splice_and_align(tok, "x", "y", cont)
    # sanity: a clean small case on the same tokenizer class still aligns
    out = splice_and_align(ParityTokenizer(merges=()), "xy", "Mz", "cdef")
    assert out["c_text"]  # non-empty common span, no cap trip


# ═══════════════════════════════════════════════════════════════════════════
# split_first_meta (round 2 M-D: ONE definition for probe + verl_sdc)
# ═══════════════════════════════════════════════════════════════════════════
def test_split_first_meta_normal_and_first_block_only():
    text = "pre<|meta|>check<|/meta|>mid<|meta|>again<|/meta|>tail"
    prefix, meta, cont = split_first_meta(text)
    assert prefix == "pre"
    assert meta == "<|meta|>check<|/meta|>"
    assert cont == "mid<|meta|>again<|/meta|>tail"   # FIRST block only
    assert prefix + meta + cont == text              # lossless 3-way split


def test_split_first_meta_rejects_malformed():
    assert split_first_meta("no tags at all") is None
    assert split_first_meta("work <|meta|>truncated at 16k cut") is None  # no close
    assert split_first_meta(None) is None
    assert split_first_meta("") is None


def test_split_first_meta_whitespace_only_continuation_is_none():
    # the STRICTER probe semantics, unified (round 2 M-D): nothing to score.
    assert split_first_meta("p<|meta|>m<|/meta|>") is None
    assert split_first_meta("p<|meta|>m<|/meta|>  \n\t") is None
    assert split_first_meta("p<|meta|>m<|/meta|> x") is not None


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


def test_aggregate_mean_min_alpha_zero_is_clipped_mean():
    # RLT-faithful mean + alpha*min, on the per-token-CLIPPED deltas. alpha=0
    # reduces to the clipped mean (NOT the unclipped "mean" method): clip_c=2
    # bounds the 3.0 outlier -> [1, -0.5, 2, 0.5], mean = 0.75.
    assert pmi_aggregate(_DELTA, "mean_min", clip_c=2.0, alpha=0.0) == pytest.approx(0.75)
    # alpha weights the worst (clipped) token: 0.75 + 0.5 * min([1,-0.5,2,0.5]) =
    # 0.75 + 0.5*(-0.5) = 0.5.
    assert pmi_aggregate(_DELTA, "mean_min", clip_c=2.0, alpha=0.5) == pytest.approx(0.5)


def test_aggregate_mean_min_penalizes_worst_token_at_equal_mean():
    # SELECTIVITY: two metas with the SAME mean lift, but the "tanked" one drops
    # a single token. mean+alpha*min must rank the uniform one strictly higher —
    # this is what punishes generic verify that fails the hard token (Gandhi).
    uniform = [0.4, 0.4, 0.4, 0.4]  # mean 0.4, min 0.4
    tanked = [0.6, 0.6, 0.6, -0.2]  # mean 0.4, min -0.2
    u = pmi_aggregate(uniform, "mean_min", clip_c=2.0, alpha=0.5)
    t = pmi_aggregate(tanked, "mean_min", clip_c=2.0, alpha=0.5)
    assert u == pytest.approx(0.6)
    assert t == pytest.approx(0.3)
    assert u > t


def test_aggregate_mean_min_clip_bounds_outlier_min():
    # A single catastrophic token must NOT swamp the aggregate: clip_c saturates
    # the min so -100 and -1000 give the IDENTICAL result (the swamping fix).
    a = pmi_aggregate([0.5, 0.5, 0.5, -100.0], "mean_min", clip_c=2.0, alpha=0.5)
    b = pmi_aggregate([0.5, 0.5, 0.5, -1000.0], "mean_min", clip_c=2.0, alpha=0.5)
    assert a == pytest.approx(b)
    # value: clip -> [0.5,0.5,0.5,-2], mean=-0.125, min=-2 -> -0.125 + 0.5*(-2) = -1.125
    assert a == pytest.approx(-1.125)


def test_mean_min_in_agg_methods():
    assert "mean_min" in PMI_AGG_METHODS


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


def test_guard_boxed_answer_boundary_aware():
    # review round 1: bare `ans in meta_text` zeroed ~37% of single-digit-answer
    # rows (GSM8K-skewed) — the confidence decimal and step numbering are NOT
    # answer leaks. The two mandated cases first:
    assert ngram_overlap_guard("confidence: 0.7", _CONT, boxed_answer="7") is False
    assert ngram_overlap_guard("the answer is 7", _CONT, boxed_answer="7") is True
    # decimal fragments / digit-inside-number / step numbering never trip it
    assert ngram_overlap_guard("I am about 0.75 sure here", _CONT, boxed_answer="7") is False
    assert ngram_overlap_guard("rechecking step 27 above", _CONT, boxed_answer="7") is False
    # round 2: a SENTENCE-FINAL dot after the bare answer digit now flags (the
    # decimal-aware trailing lookaround) — step numbering "2." is textually
    # indistinguishable from "the answer is 2." and the guard fails CLOSED
    # (member 0 = conservative under-credit, never a leak pass-through).
    assert ngram_overlap_guard("as shown in step 2. we proceed", _CONT, boxed_answer="2") is True
    # genuine standalone statements still fire, mid-sentence or punctuated
    assert ngram_overlap_guard("so 7 must be the result", _CONT, boxed_answer="7") is True
    assert ngram_overlap_guard("I get 7, then verify", _CONT, boxed_answer="7") is True
    # regex metachars in the boxed answer are escaped, not interpreted
    assert ngram_overlap_guard("the value (x+1) appears", _CONT, boxed_answer="(x+1)") is True
    assert ngram_overlap_guard("plain text only", _CONT, boxed_answer="(x+1)") is False


def test_guard_boxed_answer_sentence_final_punctuation():
    # Round 2 IMPORTANT-2 (verified by execution): the round-1 lookarounds
    # (?<![\w.])ans(?![\w.]) blocked '.' in EVERY trailing context, so
    # "the answer is 7." passed the guard. The four mandated cases:
    # 1. trailing SENTENCE period -> HIT
    assert ngram_overlap_guard("the answer is 7.", _CONT, boxed_answer="7") is True
    # 2. digit inside a decimal -> no hit ("8" inside "0.85")
    assert ngram_overlap_guard("confidence: 0.85", _CONT, boxed_answer="8") is False
    # 3. decimal followed by punctuation -> no hit ("7" inside "0.7,")
    assert ngram_overlap_guard("maybe 0.7, maybe less", _CONT, boxed_answer="7") is False
    # 4. non-numeric answer with trailing period -> HIT
    assert ngram_overlap_guard("answer is 1/2.", _CONT, boxed_answer="1/2") is True
    # supporting edges: "answer: 42." (the other executed repro) + the decimal
    # CONTINUATION "7.5" must still be clean for ans="7" (dot-then-digit).
    assert ngram_overlap_guard("answer: 42.", _CONT, boxed_answer="42") is True
    assert ngram_overlap_guard("roughly 7.5 maybe", _CONT, boxed_answer="7") is False


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


def test_compute_rows_nonfinite_arm_logprob_fails_row_not_poisons():
    # Round 2 IMPORTANT-3: one NaN/inf in either arm must fail the ROW (R 0,
    # 'nonfinite' counter), never propagate — a NaN r_meta with member=1 NaNs
    # every sibling's centered A_meta in group_mean_subtract downstream.
    rows = [
        _row([-1.0, np.nan, -1.0], [-2.0, -2.0, -2.0], correct=True),   # NaN with-arm
        _row([-1.0, -1.0], [-np.inf, -2.0], correct=True),              # inf without-arm
        _row([-1.0, -1.0, -1.0], [-2.0, -2.0, -2.0], correct=True),     # healthy sibling
    ]
    r, diag = compute_pmi_rows(rows, method="mean")
    assert r[0] == 0.0 and r[1] == 0.0                  # never NaN, exactly 0
    assert np.isfinite(r).all()
    assert r[2] == pytest.approx(1.0)                   # sibling unaffected
    assert diag["nonfinite"] == [True, True, False]
    assert diag["alignment_failures"] == [False, False, False]
    assert diag["guard_hits"] == [False, False, False]
    # raw_agg is NaN'd for the failed rows (excluded from probe stats), and the
    # per-row diagnostics lists stay index-aligned across all four counters.
    assert math.isnan(diag["raw_agg"]["mean"][0])
    assert math.isnan(diag["raw_agg"]["sum_clip"][1])
    assert len(diag["nonfinite"]) == len(diag["guard_hits"]) == 3


def test_compute_rows_alignment_failure_nonfinite_counter_stays_false():
    # failed-alignment rows are counted under alignment_failures, NOT nonfinite.
    r, diag = compute_pmi_rows([_row(None, None, alignment_failed=True)])
    assert diag["alignment_failures"] == [True]
    assert diag["nonfinite"] == [False]


def test_compute_rows_misaligned_arm_lengths_raise():
    with pytest.raises(ValueError, match="span-aligned"):
        compute_pmi_rows([_row([-1.0, -1.0], [-2.0])])


def test_compute_rows_empty_input():
    r, diag = compute_pmi_rows([])
    assert len(r) == 0 and diag["guard_hits"] == [] and diag["alignment_failures"] == []


# ═══════════════════════════════════════════════════════════════════════════
# placebo-corrected delta' (cross-shuffle amendment 2026-06-11, report §4.1)
# ═══════════════════════════════════════════════════════════════════════════
def test_placebo_correct_cancels_generic_component():
    # generic text-presence: placebo lifts the continuation EXACTLY as much as
    # the real meta -> corrected aggregate 0 -> R_meta 0 (emitting filler must
    # not out-earn silence).
    rows = [_row([-1.0] * 3, [-2.0] * 3, correct=True,
                 logp_placebo=[-1.0] * 3)]
    r, diag = compute_pmi_rows(rows, method="mean", placebo_correct=True)
    assert r[0] == pytest.approx(0.0)
    assert diag["placebo_failures"] == [False]


def test_placebo_correct_rewards_only_the_content_increment():
    # real meta lifts by 2/tok, placebo by 0.5/tok -> corrected = 1.5; wrong row
    # mirrors through the sign gate.
    rows = [
        _row([-1.0] * 4, [-3.0] * 4, correct=True, logp_placebo=[-2.5] * 4),
        _row([-1.0] * 4, [-3.0] * 4, correct=False, logp_placebo=[-2.5] * 4),
    ]
    r, _ = compute_pmi_rows(rows, method="mean", clip_c_gate=2.0,
                            placebo_correct=True)
    assert r[0] == pytest.approx(1.5)
    assert r[1] == pytest.approx(-1.5)


def test_placebo_correct_wrong_content_clips_to_zero_not_negative_credit():
    # meta WORSE than placebo on a correct row: corrected < 0 -> clip(., 0, c)
    # -> 0 (a correct rollout never pays for a bad meta beyond losing credit).
    rows = [_row([-2.0] * 3, [-2.5] * 3, correct=True,
                 logp_placebo=[-1.0] * 3)]
    r, _ = compute_pmi_rows(rows, method="mean", placebo_correct=True)
    assert r[0] == pytest.approx(0.0)


def test_placebo_correct_missing_arm_fails_closed():
    # No logp_placebo / placebo_alignment_failed / nonfinite placebo: the row
    # must fail closed (R 0 + placebo_failures True -> caller member 0), never
    # silently fall back to the raw delta (mixed reward definitions inside one
    # centering group).
    rows = [
        _row([-1.0] * 2, [-2.0] * 2, correct=True),                       # missing
        _row([-1.0] * 2, [-2.0] * 2, correct=True, logp_placebo=[-1.0] * 2,
             placebo_alignment_failed=True),                               # flagged
        _row([-1.0] * 2, [-2.0] * 2, correct=True,
             logp_placebo=[np.nan, -1.0]),                                 # nonfinite
        _row([-1.0] * 2, [-2.0] * 2, correct=True, logp_placebo=[-1.5] * 2),  # healthy
    ]
    r, diag = compute_pmi_rows(rows, method="mean", placebo_correct=True)
    assert r[0] == 0.0 and r[1] == 0.0 and r[2] == 0.0
    assert r[3] == pytest.approx(0.5)
    assert diag["placebo_failures"] == [True, True, True, False]
    # failed rows are excluded from probe stats (raw_agg NaN), index-aligned
    assert math.isnan(diag["raw_agg"]["mean"][0])
    assert len(diag["placebo_failures"]) == len(diag["guard_hits"]) == 4


def test_placebo_correct_separate_without_arm_and_length_mismatch():
    # explicit logp_placebo_without is honored; a length mismatch between the
    # placebo arms fails closed instead of raising mid-training.
    ok = _row([-1.0] * 3, [-2.0] * 3, correct=True,
              logp_placebo=[-1.2] * 3, logp_placebo_without=[-2.0] * 3)
    bad = _row([-1.0] * 3, [-2.0] * 3, correct=True,
               logp_placebo=[-1.2] * 2)  # 2 tokens vs without's 3
    r, diag = compute_pmi_rows([ok, bad], method="mean", placebo_correct=True)
    # real delta 1.0/tok, placebo delta 0.8/tok -> corrected content increment 0.2
    assert r[0] == pytest.approx(0.2)
    assert r[1] == 0.0
    assert diag["placebo_failures"] == [False, True]


def test_placebo_correct_off_ignores_placebo_keys_and_logs_no_failures():
    # default path unchanged: placebo keys ignored, placebo_failures all False.
    rows = [_row([-1.0] * 3, [-2.0] * 3, correct=True, logp_placebo=[-1.0] * 3)]
    r, diag = compute_pmi_rows(rows, method="mean")
    assert r[0] == pytest.approx(1.0)       # raw delta, no subtraction
    assert diag["placebo_failures"] == [False]


def test_placebo_meta_constant_is_tag_wrapped_ssot():
    from src.training.dcpo_pmi import PLACEBO_META
    from src.metacot.prompt import META_END, META_START
    assert PLACEBO_META.startswith(META_START) and PLACEBO_META.endswith(META_END)
    assert "Let me continue." in PLACEBO_META
