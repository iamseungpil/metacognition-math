"""Unit tests for the pure force-inject core (no GPU / tokenizer needed)."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))  # repo root
from src.training.meta_inject import (
    find_inject_position, meta_mask, splice_prefix, plan_inject_prefixes,
)

OPEN, CLOSE = 151669, 151670


class FakeTok:
    """Minimal tokenizer stub: id 7 decodes to '\\boxed', everything else ''.
    encode returns a fixed 3-token segment."""
    def decode(self, ids):
        return r"\boxed" if ids == [7] else ""
    def encode(self, text, add_special_tokens=False):
        return [OPEN, 42, CLOSE]


def test_argmax_before_answer():
    # entropy spike at index 60; answer at 80 → inject at 60
    ent = np.array([0.1] * 60 + [9.0] + [0.2] * 40)
    resp = [1] * 101
    assert find_inject_position(ent, resp, OPEN, CLOSE, answer_cap=80, min_tok=50) == 60


def test_spike_after_answer_excluded():
    # only spike is at 90, but answer_cap=80 → that spike is excluded → next best <80
    ent = np.array([0.1] * 90 + [9.0] + [0.1] * 10)
    ent[55] = 1.0  # modest pre-answer peak
    resp = [1] * 101
    assert find_inject_position(ent, resp, OPEN, CLOSE, answer_cap=80, min_tok=50) == 55


def test_spike_inside_meta_excluded():
    # high entropy at 60 but 60 is inside a meta span → excluded
    ent = np.array([0.1] * 60 + [9.0] + [0.1] * 40)
    resp = [1] * 58 + [OPEN, 5, 5, CLOSE] + [1] * 39  # meta span covers ~58-61
    pos = find_inject_position(ent, resp, OPEN, CLOSE, answer_cap=100, min_tok=50)
    assert pos != 60 and pos != -1


def test_no_valid_position_returns_minus_one():
    ent = np.array([0.1] * 40)
    resp = [1] * 40
    assert find_inject_position(ent, resp, OPEN, CLOSE, answer_cap=40, min_tok=50) == -1


def test_meta_mask_marks_span_and_markers():
    resp = [1, 1, OPEN, 5, 5, CLOSE, 1]
    m = meta_mask(resp, OPEN, CLOSE, len(resp))
    assert m[2] and m[3] and m[4] and m[5]  # markers + content
    assert not m[0] and not m[1] and not m[6]


def test_splice_prefix_order():
    out = splice_prefix([9, 9], [1, 2, 3, 4], pos=2, segment_ids=[7, 8])
    assert out == [9, 9, 1, 2, 7, 8]


def test_unclosed_meta_span_masked_to_end():
    # <|meta|> opened at idx 2, never closed → idxs 2..end all masked
    resp = [1, 1, OPEN, 5, 5, 5]
    m = meta_mask(resp, OPEN, CLOSE, len(resp))
    assert m[2] and m[3] and m[4] and m[5]
    assert not m[0] and not m[1]


def test_plan_inject_prefixes_length_mismatch_raises():
    tok = FakeTok()
    import numpy as _np
    try:
        plan_inject_prefixes([[1]], [[1, 2]], [_np.array([0.0]), _np.array([0.0])],
                             tok, OPEN, CLOSE)
        assert False, "expected AssertionError"
    except AssertionError:
        pass


def test_plan_inject_prefixes_batch():
    tok = FakeTok()
    # sample 0: spike at 60, answer (id 7) at 90 → inject at 60
    ent0 = np.array([0.1] * 60 + [9.0] + [0.1] * 40)
    r0 = [1] * 90 + [7] + [1] * 10
    # sample 1: too short, no valid position → None
    ent1 = np.array([0.1] * 30)
    r1 = [1] * 30
    prefixes = plan_inject_prefixes([[9, 9], [9, 9]], [r0, r1], [ent0, ent1],
                                    tok, OPEN, CLOSE, min_tok=50)
    assert prefixes[0] == [9, 9] + [1] * 60 + [OPEN, 42, CLOSE]  # spliced at 60
    assert prefixes[1] is None  # skipped


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"ok  {fn.__name__}")
    print(f"\n{len(fns)} passed")
