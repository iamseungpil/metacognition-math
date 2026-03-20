"""Metacognitive reward functions for GRPO training."""
import re
from typing import Optional


def compute_r_correct(is_correct: bool) -> float:
    """R_correct: binary correctness reward."""
    return 1.0 if is_correct else 0.0


def compute_r_calib(
    c_text: Optional[float],
    p_hat: float,
) -> float:
    """R_calib: calibration reward.

    Measures alignment between model's self-reported confidence (from Meta-CoT
    Stage 2) and Gnosis probe's hidden-state-based prediction.

    Both overconfidence (c_text >> p_hat) and underconfidence (c_text << p_hat)
    are penalized.
    """
    if c_text is None:
        return 0.0
    return max(0.0, 1.0 - abs(c_text - p_hat))


def compute_r_strat(
    strategy_category: Optional[str],
    weak_categories: list,
) -> float:
    """R_strat: strategy quality reward.

    Rewards when model's Stage 3 learning strategy targets an actual weakness.
    """
    if strategy_category is None:
        return 0.0
    return 1.0 if strategy_category in weak_categories else 0.0


def compute_r_meta(
    is_correct: bool,
    c_text: Optional[float],
    p_hat: float,
    strategy_category: Optional[str],
    weak_categories: list,
    lambda1: float = 0.5,
    lambda2: float = 0.0,
) -> dict:
    """Combined metacognitive reward R_meta.

    R_meta = R_correct + lambda1 * R_calib + lambda2 * R_strat
    Returns dict with total and individual components for logging.
    """
    r_correct = compute_r_correct(is_correct)
    r_calib = compute_r_calib(c_text, p_hat)
    r_strat = compute_r_strat(strategy_category, weak_categories)
    total = r_correct + lambda1 * r_calib + lambda2 * r_strat
    return {
        "total": total,
        "r_correct": r_correct,
        "r_calib": r_calib,
        "r_strat": r_strat,
    }


def extract_confidence_from_chain(chain_text: str) -> Optional[float]:
    """Extract numeric confidence from Meta-CoT Stage 2 output."""
    from src.metacot.prompt import parse_metacot_stages
    parsed = parse_metacot_stages(chain_text)
    return parsed.get("confidence")


def extract_strategy_target(chain_text: str) -> Optional[str]:
    """Extract the target category from Meta-CoT Stage 3."""
    # Match categories used in dataset_loader.py, longest first
    categories = [
        "intermediate_algebra", "counting_probability", "number_theory",
        "precalculus", "prealgebra", "geometry", "algebra",
        "competition", "olympiad",
    ]
    chain_lower = chain_text.lower()
    for cat in categories:
        if cat.replace("_", " ") in chain_lower or cat in chain_lower:
            return cat
    return None


def compute_gnosis_temporal_difference(
    gnosis_scores: list,
    p0: float = 0.5,
) -> list:
    """Compute Gnosis Temporal Difference (GTD) rewards.

    Given T checkpoint scores from Gnosis, compute per-segment rewards:
        r_k = p_hat(t_k) - p_hat(t_{k-1}), with p_hat(t_0) = p0

    Segments where Gnosis probability increases get positive reward;
    segments where it decreases get negative reward.
    """
    scores = [p0] + list(gnosis_scores)
    rewards = []
    for k in range(1, len(scores)):
        rewards.append(scores[k] - scores[k - 1])
    return rewards


def compute_grpo_advantages(
    rewards: list,
    group_size: int = 8,
) -> list:
    """Compute GRPO normalized advantages within a group.

    rewards: list of R_meta values for G rollouts of the same problem.
    Returns: normalized advantages (zero-mean, unit-variance within group).
    """
    if len(rewards) < 2:
        return [0.0] * len(rewards)

    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = max(var_r ** 0.5, 1e-8)

    return [(r - mean_r) / std_r for r in rewards]
