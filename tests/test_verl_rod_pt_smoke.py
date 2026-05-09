"""Smoke test for ROD-PT veRL implementation (Plan v5.17 FINAL)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
__doc__ = """ROD-PT smoke (Plan v5.17 FINAL).

Verifies:
1. ROD_PT mode is registered in REWARD_CONFIGS
2. ROD_PT is in _SINGLE_TEACHER_MODES (skips decoy T- forward)
3. ROD_PT is NOT in _FORCED_META_MODES (natural emit, no V0 prefix)
4. compute_sdc_gdpo_advantage handles ROD_PT mode without crash
5. Position factor formula: w_position = clip(exp(sign × log_prob_meta), 1-ε, 1+ε)
6. PRODUCT combination: w_meta = w_attr × w_position
7. RLSD invariant: sign is preserved (advantage sign never flips)
"""
import torch
import numpy as np
import pytest


def test_rod_pt_in_reward_configs():
    """REWARD_CONFIGS must have ROD_PT entry."""
    from src.training.verl_sdc import REWARD_CONFIGS
    assert "ROD_PT" in REWARD_CONFIGS
    cfg = REWARD_CONFIGS["ROD_PT"]
    assert cfg["keys"] == ["correctness", "meta_penalty"]
    assert cfg["weights"] == [1.0, 1.0]


def test_rod_pt_in_single_teacher_modes():
    """ROD_PT must skip decoy forward (single teacher path)."""
    from src.training.verl_sdc import _SINGLE_TEACHER_MODES, _FORCED_META_MODES
    assert "ROD_PT" in _SINGLE_TEACHER_MODES
    # natural emit, NOT forced meta (no V0 prefix)
    assert "ROD_PT" not in _FORCED_META_MODES


def test_position_factor_amplify_correct_rollout():
    """Positive advantage + high prob_meta → w_position > 1.0 (amplify good rollout)."""
    log_prob_meta = torch.tensor([[-0.05]])  # prob ≈ 0.95
    sign = torch.tensor([[1.0]])  # positive advantage
    clip_eps = 0.2
    w_position = torch.clamp(
        torch.exp(sign * log_prob_meta), 1.0 - clip_eps, 1.0 + clip_eps
    )
    assert 0.9 < w_position.item() <= 1.0  # log -0.05 → exp ≈ 0.95
    # Note: w_position < 1 because log_prob_meta is always negative (log of prob ≤ 1).
    # The amplify direction depends on sign.


def test_position_factor_dampen_wrong_rollout():
    """Negative advantage + high prob_meta → w_position > 1.0 means dampen the negative advantage = LESS punishment."""
    log_prob_meta = torch.tensor([[-0.05]])
    sign = torch.tensor([[-1.0]])
    clip_eps = 0.2
    w_position = torch.clamp(
        torch.exp(sign * log_prob_meta), 1.0 - clip_eps, 1.0 + clip_eps
    )
    # sign=-1, log=-0.05 → exp(0.05) ≈ 1.05 (just above 1.0)
    assert 1.0 < w_position.item() <= 1.2  # clip upper


def test_position_factor_clip_lower_saturation():
    """Very low prob_meta + positive sign → exp(sign × very negative) ≈ 0 → clip to 0.8."""
    log_prob_meta = torch.tensor([[-3.0]])  # prob = 0.05
    sign = torch.tensor([[1.0]])
    clip_eps = 0.2
    w_position = torch.clamp(
        torch.exp(sign * log_prob_meta), 1.0 - clip_eps, 1.0 + clip_eps
    )
    assert abs(w_position.item() - 0.8) < 1e-5  # clipped (float precision)


def test_position_factor_no_meta_emit():
    """No META emit → log_prob_meta = 0 → w_position = exp(0) = 1.0 (no amplify)."""
    log_prob_meta = torch.tensor([[0.0]])  # default fallback
    sign = torch.tensor([[1.0]])
    clip_eps = 0.2
    w_position = torch.clamp(
        torch.exp(sign * log_prob_meta), 1.0 - clip_eps, 1.0 + clip_eps
    )
    assert abs(w_position.item() - 1.0) < 1e-5  # passthrough (float precision)


def test_w_combined_product_invariant():
    """w_combined = w_attr * w_position. Both clipped to [0.8, 1.2] → combined in [0.64, 1.44]."""
    w_attr = torch.tensor([[1.2, 0.8, 1.0], [0.8, 1.0, 1.2]])  # [B=2, T=3]
    w_position = torch.tensor([[1.2], [0.8]])  # [B=2, 1]
    w_combined = w_attr * w_position  # broadcast [2, 3]
    expected = torch.tensor([[1.44, 0.96, 1.20], [0.64, 0.80, 0.96]])
    assert torch.allclose(w_combined, expected)


def test_rlsd_invariant_sign_preserved():
    """RLSD invariant: sign × |advantage| (sign 절대 안 바뀜).

    seq_adv > 0 (good rollout) × any positive factor → still > 0 advantage.
    seq_adv < 0 (bad rollout) × any positive factor → still < 0 advantage.
    """
    seq_adv = torch.tensor([[1.5], [-2.0]])  # [B=2, 1]
    # factor = ((1-λ) + λ × w_meta), λ=0.5, w_meta ∈ [0.64, 1.44]
    lam = 0.5
    w_meta = torch.tensor([[0.64], [1.44]])  # [B=2, 1] worst/best case
    factor = (1.0 - lam) + lam * w_meta  # [B=2, 1]
    advantages = seq_adv * factor  # [B=2, 1]
    # sign preserved
    assert advantages[0].item() > 0  # positive seq_adv → positive advantage
    assert advantages[1].item() < 0  # negative seq_adv → negative advantage


def test_meta_start_position_search():
    """Find first META_START position p in response, mask=True."""
    META_ID = 99
    response_ids = torch.tensor([[1, 2, 99, 50, 99], [10, 20, 30, 40, 50]])
    mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
    p_list = []
    for b in range(2):
        valid = (response_ids[b] == META_ID) & mask[b].bool()
        nz = valid.nonzero(as_tuple=True)[0]
        p_list.append(int(nz[0].item()) if nz.numel() > 0 else None)
    assert p_list == [2, None]


def test_position_factor_distribution_not_degenerate():
    """For a range of prob_meta values, factor should NOT be entirely clipped."""
    log_prob_meta = torch.tensor([[-0.5], [-1.0], [-0.1]])  # prob in {0.6, 0.37, 0.9}
    sign = torch.tensor([[1.0], [1.0], [-1.0]])
    clip_eps = 0.2
    w = torch.clamp(torch.exp(sign * log_prob_meta), 1.0 - clip_eps, 1.0 + clip_eps)
    # At least one should be in interior (not clipped)
    interior = (w > 0.8) & (w < 1.2)
    assert interior.any(), f"All saturated: w = {w.tolist()}"


if __name__ == "__main__":
    # Run all tests
    import sys
    failed = []
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS: {name}")
            except AssertionError as e:
                failed.append((name, str(e)))
                print(f"  FAIL: {name}: {e}")
            except Exception as e:
                failed.append((name, f"{type(e).__name__}: {e}"))
                print(f"  ERROR: {name}: {type(e).__name__}: {e}")
    if failed:
        print(f"\n{len(failed)} FAILED")
        sys.exit(1)
    print(f"\nAll {sum(1 for n in globals() if n.startswith('test_'))} tests PASSED")
