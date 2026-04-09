"""veRL-compatible reward function for Meta-CoT GDPO.

veRL 0.7.1 calls compute_score(solution_str, ground_truth, **kwargs) → dict
The dict keys must match algorithm.gdpo_reward_keys in the config.

Usage in veRL config:
  reward:
    custom_reward_function:
      path: src/training/verl_reward.py
      name: compute_score
  algorithm:
    adv_estimator: gdpo
    gdpo_reward_keys: [correctness, switch_v2, verify_v2, conf_traj, meta_floor]
"""

import re
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.environ.get('PYTHONPATH', '/scratch/metacognition'))

from src.training.rewards import (
    correctness_reward,
    structural_switch_reward_v2,
    verify_outcome_v2,
    confidence_trajectory_reward,
    confidence_omission_floor,
)


def compute_score(solution_str, ground_truth, **kwargs):
    """Compute multi-dimensional reward scores for GDPO.

    Args:
        solution_str: Full model output (prompt + response decoded)
        ground_truth: Ground truth answer string

    Returns:
        dict: {reward_name: float} for each GDPO dimension
    """
    # Format as expected by our reward functions
    completion = [{"content": solution_str}]
    gt = [ground_truth]

    try:
        corr = correctness_reward(completion, gt)[0]
    except Exception:
        corr = 0.0

    try:
        switch = structural_switch_reward_v2(completion, gt)[0]
    except Exception:
        switch = 0.0

    try:
        verify = verify_outcome_v2(completion, gt)[0]
    except Exception:
        verify = 0.0

    try:
        conf = confidence_trajectory_reward(completion, gt)[0]
    except Exception:
        conf = 0.0

    try:
        floor = confidence_omission_floor(completion, gt)[0]
    except Exception:
        floor = 0.0

    return {
        "correctness": corr,
        "switch_v2": switch,
        "verify_v2": verify,
        "conf_traj": conf,
        "meta_floor": floor,
    }


def compute_score_base(solution_str, ground_truth, **kwargs):
    """Correctness-only reward for base GRPO baseline."""
    completion = [{"content": solution_str}]
    gt = [ground_truth]

    try:
        corr = correctness_reward(completion, gt)[0]
    except Exception:
        corr = 0.0

    return corr  # Single float for standard GRPO
