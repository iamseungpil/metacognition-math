"""GDPO advantage computation for veRL.

Replaces the monkey-patch approach in grpo_v2.py with a clean function that
plugs into veRL's compute_advantage dispatch.

GDPO (Group-wise Distributional Policy Optimization):
  GRPO: sum(rewards) -> group_normalize              (collapses distinct reward combos)
  GDPO: group_normalize(each_reward) -> sum -> batch_normalize  (preserves signal)

Reference: arXiv:2601.05242 (NVIDIA NVlabs)

This module is a standalone addition. It does NOT modify veRL's core_algos.py.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List

import torch


def compute_gdpo_outcome_advantage(
    per_reward_token_level_rewards: List[torch.Tensor],
    reward_weights: List[float],
    eos_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GDPO advantages from per-reward token-level reward tensors.

    Unlike GRPO which sums rewards first then normalizes within the group,
    GDPO normalizes each reward independently within the group, then takes
    a weighted sum, then batch-normalizes the result.

    Args:
        per_reward_token_level_rewards: List of tensors, each shape (bs, response_length).
            Each tensor has the scalar reward placed at the EOS token position
            (matching veRL's reward format).
        reward_weights: Weight for each reward function.
        eos_mask: (bs, response_length) mask for valid response tokens.
        index: (bs,) prompt group indices (uid). Responses from the same prompt
            share the same index for within-group normalization.
        epsilon: Small constant for numerical stability.

    Returns:
        advantages: (bs, response_length) GDPO advantages broadcast to all tokens.
        returns: Same as advantages (no value function in outcome-only setting).
    """
    num_rewards = len(per_reward_token_level_rewards)
    assert num_rewards == len(reward_weights), (
        f"Mismatch: {num_rewards} reward tensors vs {len(reward_weights)} weights"
    )

    bs, response_length = per_reward_token_level_rewards[0].shape
    device = per_reward_token_level_rewards[0].device

    # Extract scalar scores from each reward tensor
    # (rewards are placed at EOS position, sum over response_length to get scalar)
    per_reward_scores = []
    for r_tensor in per_reward_token_level_rewards:
        non_zero_mask = (r_tensor != 0)
        scores = (r_tensor * non_zero_mask).sum(dim=-1)  # (bs,)
        per_reward_scores.append(scores)

    # Build group index mapping
    id2indices: dict[str, list[int]] = defaultdict(list)
    with torch.no_grad():
        for i in range(bs):
            id2indices[index[i]].append(i)

        # Normalize each reward independently within groups
        all_normalized = []
        for r_idx in range(num_rewards):
            scores = per_reward_scores[r_idx].clone()
            for uid, indices in id2indices.items():
                group_scores = scores[indices]
                if len(indices) == 1:
                    # Single sample in group: zero advantage
                    scores[indices[0]] = 0.0
                else:
                    mean = group_scores.mean()
                    std = group_scores.std()
                    for idx in indices:
                        scores[idx] = (scores[idx] - mean) / (std + epsilon)
            all_normalized.append(scores)

        # Weighted sum of per-reward normalized advantages
        weights = torch.tensor(reward_weights, dtype=torch.float32, device=device)
        stacked = torch.stack(all_normalized, dim=1)  # (bs, num_rewards)
        pre_bn = (stacked * weights.unsqueeze(0)).sum(dim=1)  # (bs,)

        # Batch normalization (across all samples in the batch)
        advantages_scalar = (pre_bn - pre_bn.mean()) / (pre_bn.std() + epsilon)

        # Broadcast to response token positions (same as GRPO: scalar -> all tokens via eos_mask)
        advantages = advantages_scalar.unsqueeze(-1).expand(-1, response_length) * eos_mask

    return advantages, advantages
