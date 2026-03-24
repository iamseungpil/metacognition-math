"""Metacognitive reward functions for stepwise GRPO.

Three rewards:
- R_correct: strong reward for solving the problem (+2.0)
- R_calib: per-step calibration (model confidence vs probe p̂)
- R_penalty: penalty for not using <|meta|> self-reflection
"""
import re
from typing import Optional


def compute_r_correct(is_correct: bool) -> float:
    """Strong reward for correct answer."""
    return 2.0 if is_correct else 0.0


def compute_r_calib(c_text: Optional[float], p_hat: float) -> float:
    """Per-step calibration: how well model's stated confidence matches probe."""
    if c_text is None:
        return 0.0
    return max(0.0, 1.0 - abs(c_text - p_hat))


def compute_r_penalty(num_meta_blocks: int) -> float:
    """Penalty for not using <|meta|> self-reflection."""
    if num_meta_blocks >= 2:
        return 0.0
    elif num_meta_blocks == 1:
        return -0.3
    else:
        return -0.5


def compute_reward(
    is_correct: bool,
    chain_text: str,
    gnosis_scores: list,
    model_confidences: list,
    num_meta_blocks: int,
    lambda_calib: float = 1.0,
) -> dict:
    """Compute total reward with all 3 components.

    Total = R_correct + λ · mean(R_calib_per_step) + R_penalty
    """
    r_correct = compute_r_correct(is_correct)
    r_penalty = compute_r_penalty(num_meta_blocks)

    # Per-step calibration (average across all meta steps)
    step_calibs = []
    n_steps = max(len(gnosis_scores), len(model_confidences))
    for k in range(n_steps):
        p_hat = gnosis_scores[k] if k < len(gnosis_scores) else 0.5
        c_text = model_confidences[k] if k < len(model_confidences) else None
        step_calibs.append(compute_r_calib(c_text, p_hat))

    avg_calib = sum(step_calibs) / max(len(step_calibs), 1) if step_calibs else 0.0

    total = r_correct + lambda_calib * avg_calib + r_penalty

    return {
        "total": total,
        "r_correct": r_correct,
        "r_calib_avg": avg_calib,
        "r_calib_per_step": step_calibs,
        "r_penalty": r_penalty,
        "num_meta_blocks": num_meta_blocks,
    }


def extract_confidence_from_chain(chain_text: str) -> Optional[float]:
    """Extract numeric confidence from model output."""
    from src.metacot.prompt import parse_meta_blocks
    parsed = parse_meta_blocks(chain_text)
    confs = parsed.get("confidences", [])
    return confs[0] if confs else None


def compute_grpo_advantages(rewards: list) -> list:
    """GRPO normalized advantages within a group."""
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = max(var_r ** 0.5, 1e-8)
    return [(r - mean_r) / std_r for r in rewards]
