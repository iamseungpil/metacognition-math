"""Custom reward function for verl GRPO training.

This module is registered as a custom reward scorer in verl.
It computes R_correct + R_calib + R_penalty for metacognitive reasoning.
"""
import re
import sys
sys.path.insert(0, "/scratch/metacognition")

from src.metacot.prompt import META_START, META_END, parse_meta_blocks
from src.rollout.vllm_rollout import check_correctness
from src.training.rewards import compute_r_correct, compute_r_calib, compute_r_penalty


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Custom reward for metacognitive math reasoning.

    Called by verl for each generated response.

    Args:
        data_source: dataset name (e.g., "metacot_math")
        solution_str: model's generated response
        ground_truth: correct answer string
        extra_info: dict with additional task info

    Returns:
        float: total reward
    """
    # Parse <|meta|> blocks
    parsed = parse_meta_blocks(solution_str)
    num_meta = parsed["num_blocks"]
    confidences = parsed["confidences"]

    # R_correct
    is_correct = check_correctness(solution_str, str(ground_truth))
    r_correct = compute_r_correct(is_correct)  # +2.0 if correct

    # R_penalty
    r_penalty = compute_r_penalty(num_meta)  # -0.5 if no meta

    # R_calib (based on model's stated confidence)
    r_calib = 0.0
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        if is_correct and avg_conf > 0.5:
            r_calib = avg_conf
        elif not is_correct and avg_conf < 0.5:
            r_calib = 1.0 - avg_conf
        else:
            r_calib = 0.1

    total = r_correct + r_calib + r_penalty
    return total
