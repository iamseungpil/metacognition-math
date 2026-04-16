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
    gdpo_reward_keys: [correctness, switch_v2, verify_v2, conf_traj, meta_floor, meta_count_bonus]

For the confidence-centered redirect controller use:
  reward:
    custom_reward_function:
      path: src/training/verl_reward.py
      name: compute_score_confidence_centered
  algorithm:
    adv_estimator: gdpo
    gdpo_reward_keys: [correctness, confidence_revision, redirect_execution, verify_execution, meta_floor, meta_count_bonus]
"""

import re
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.environ.get('PYTHONPATH', '/scratch/metacognition'))

from src.training.rewards import (
    correctness_reward,
    outcome_calibration_reward,
    meta_structure_reward,
    meta_commit_shape_reward,
    structural_switch_reward_v2,
    verify_outcome_v2,
    confidence_trajectory_reward,
    confidence_omission_floor,
    confidence_revision_reward_v2,
    redirect_execution_reward_v2,
    verify_execution_reward_v2,
    meta_count_bonus,
)


def compute_score(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    """Compute multi-dimensional reward scores for veRL 0.7.1 GDPO.

    veRL's GDPORewardManager calls:
        compute_score(data_source=..., solution_str=..., ground_truth=..., extra_info=...)

    Must return dict with "score" key (combined) + per-dimension keys matching
    algorithm.gdpo_reward_keys in config.
    """
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

    try:
        meta_count = meta_count_bonus(completion, gt)[0]
    except Exception:
        meta_count = 0.0

    # Weighted combined score (same weights as E21 config)
    combined = corr * 1.0 + switch * 0.15 + verify * 0.3 + conf * 0.15 + floor * 0.5 + meta_count

    return {
        "score": combined,  # required by GDPORewardManager
        "correctness": corr,
        "switch_v2": switch,
        "verify_v2": verify,
        "conf_traj": conf,
        "meta_floor": floor,
        "meta_count_bonus": meta_count,
    }


def compute_score_confidence_centered(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    """Confidence-centered GDPO reward for redirect-focused experiments."""
    completion = [{"content": solution_str}]
    gt = [ground_truth]

    try:
        corr = correctness_reward(completion, gt)[0]
    except Exception:
        corr = 0.0

    try:
        conf_rev = confidence_revision_reward_v2(completion, gt)[0]
    except Exception:
        conf_rev = 0.0

    try:
        redirect_exec = redirect_execution_reward_v2(completion, gt)[0]
    except Exception:
        redirect_exec = 0.0

    try:
        verify_exec = verify_execution_reward_v2(completion, gt)[0]
    except Exception:
        verify_exec = 0.0

    try:
        floor = confidence_omission_floor(completion, gt)[0]
    except Exception:
        floor = 0.0

    try:
        meta_count = meta_count_bonus(completion, gt)[0]
    except Exception:
        meta_count = 0.0

    combined = corr * 1.0 + conf_rev * 0.35 + redirect_exec * 0.30 + verify_exec * 0.15 + floor * 0.5 + meta_count

    return {
        "score": combined,
        "correctness": corr,
        "confidence_revision": conf_rev,
        "redirect_execution": redirect_exec,
        "verify_execution": verify_exec,
        "meta_floor": floor,
        "meta_count_bonus": meta_count,
    }


def compute_score_e21r_v2(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    """E21R-v2: 2-head GDPO (correctness + outcome_calibration).

    GDPO heads: correctness, outcome_calibration
    meta_floor is added to combined score only (not a GDPO head).
    """
    completion = [{"content": solution_str}]
    gt = [ground_truth]

    try:
        corr = correctness_reward(completion, gt)[0]
    except Exception:
        corr = 0.0

    try:
        cal = outcome_calibration_reward(completion, gt)[0]
    except Exception:
        cal = 0.0

    try:
        floor = confidence_omission_floor(completion, gt)[0]
    except Exception:
        floor = 0.0

    combined = corr * 1.0 + cal * 1.0 + floor * 0.3

    return {
        "score": combined,
        "correctness": corr,
        "outcome_calibration": cal,
    }


def compute_score_e21r_v3_smoke(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    """Isolated smoke reward for controller-aware RL experiments.

    This entrypoint is intentionally separate from the claim-bearing E21R-v2 path.
    It adds controller execution terms without paying style-only bonuses such as
    `meta_count_bonus`, so it can be used for future smoke runs without mutating
    the mainline reward contract.
    """
    completion = [{"content": solution_str}]
    gt = [ground_truth]

    try:
        corr = correctness_reward(completion, gt)[0]
    except Exception:
        corr = 0.0

    try:
        cal = outcome_calibration_reward(completion, gt)[0]
    except Exception:
        cal = 0.0

    try:
        conf_rev = confidence_revision_reward_v2(completion, gt)[0]
    except Exception:
        conf_rev = 0.0

    try:
        redirect_exec = redirect_execution_reward_v2(completion, gt)[0]
    except Exception:
        redirect_exec = 0.0

    try:
        verify_exec = verify_execution_reward_v2(completion, gt)[0]
    except Exception:
        verify_exec = 0.0

    try:
        floor = confidence_omission_floor(completion, gt)[0]
    except Exception:
        floor = 0.0

    try:
        structure = meta_structure_reward(completion, gt)[0]
    except Exception:
        structure = 0.0

    combined = (
        corr * 1.0
        + cal * 0.6
        + conf_rev * 0.30
        + redirect_exec * 0.25
        + verify_exec * 0.10
        + floor * 0.25
        + structure * 0.20
    )

    return {
        "score": combined,
        "correctness": corr,
        "outcome_calibration": cal,
        "confidence_revision": conf_rev,
        "redirect_execution": redirect_exec,
        "verify_execution": verify_exec,
        "meta_floor": floor,
        "meta_structure": structure,
    }


def compute_score_e21r_v4_smoke(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    """Smoke reward that incorporates commit/decoherence findings from analysis.

    Intended for future RLSD-lite / controller-smoke experiments only.
    It keeps correctness dominant, then adds:
      - structured calibration / revision
      - controller execution terms
      - commit-shape reward based on good-meta vs bad-meta findings
    """
    completion = [{"content": solution_str}]
    gt = [ground_truth]

    try:
        corr = correctness_reward(completion, gt)[0]
    except Exception:
        corr = 0.0

    try:
        cal = outcome_calibration_reward(completion, gt)[0]
    except Exception:
        cal = 0.0

    try:
        conf_rev = confidence_revision_reward_v2(completion, gt)[0]
    except Exception:
        conf_rev = 0.0

    try:
        redirect_exec = redirect_execution_reward_v2(completion, gt)[0]
    except Exception:
        redirect_exec = 0.0

    try:
        verify_exec = verify_execution_reward_v2(completion, gt)[0]
    except Exception:
        verify_exec = 0.0

    try:
        floor = confidence_omission_floor(completion, gt)[0]
    except Exception:
        floor = 0.0

    try:
        structure = meta_structure_reward(completion, gt)[0]
    except Exception:
        structure = 0.0

    try:
        commit_shape = meta_commit_shape_reward(completion, gt)[0]
    except Exception:
        commit_shape = 0.0

    combined = (
        corr * 1.0
        + cal * 0.45
        + conf_rev * 0.25
        + redirect_exec * 0.20
        + verify_exec * 0.10
        + floor * 0.20
        + structure * 0.15
        + commit_shape * 0.35
    )

    return {
        "score": combined,
        "correctness": corr,
        "outcome_calibration": cal,
        "confidence_revision": conf_rev,
        "redirect_execution": redirect_exec,
        "verify_execution": verify_exec,
        "meta_floor": floor,
        "meta_structure": structure,
        "meta_commit_shape": commit_shape,
    }


def compute_score_base(data_source=None, solution_str="", ground_truth="", extra_info=None, **kwargs):
    """Correctness-only reward for base GRPO baseline."""
    completion = [{"content": solution_str}]
    gt = [ground_truth]

    try:
        corr = correctness_reward(completion, gt)[0]
    except Exception:
        corr = 0.0

    return corr  # Single float for standard GRPO
