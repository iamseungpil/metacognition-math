import torch

from src.training.switch_ban_processor import SwitchBanLogitsProcessor


def test_switch_logit_forced_to_neg_inf():
    p = SwitchBanLogitsProcessor(ban_ids=[151670])
    logits = torch.zeros(151700)
    logits[151670] = 50.0  # primed model: switch token leads by a wide margin
    out = p([1, 2, 3], logits)
    assert out[151670] == float("-inf")
    assert out[5] == 0.0  # other tokens untouched


def test_multiple_ban_ids():
    p = SwitchBanLogitsProcessor(ban_ids=[10, 20])
    logits = torch.ones(50)
    out = p([], logits)
    assert out[10] == float("-inf") and out[20] == float("-inf")
    assert out[0] == 1.0


def test_picklable_for_ray():
    import pickle
    p = SwitchBanLogitsProcessor(ban_ids=[7])
    p2 = pickle.loads(pickle.dumps(p))
    assert p2.ban_ids == [7]
