import torch
from src.training.meta_close_processor import MetaCloseLogitsProcessor


def test_forces_close_after_budget():
    p = MetaCloseLogitsProcessor(meta_open=12, meta_close=13, max_open_tokens=3)
    V = 20
    # before any open: no-op
    lg = torch.zeros(V); out = p([1, 2, 3], lg.clone()); assert torch.allclose(out, lg)
    # after open, within budget: forbid a 2nd open (id12 -> -inf), close not forced yet
    p2 = MetaCloseLogitsProcessor(12, 13, max_open_tokens=3)
    o = p2([12, 5], torch.zeros(V)); assert o[12] == float("-inf")
    # at budget: force close (only id13 finite)
    p3 = MetaCloseLogitsProcessor(12, 13, max_open_tokens=2)
    o = p3([12, 5, 6], torch.zeros(V))
    assert o[13] > -1e30 and o[0] == float("-inf")
