"""Metacognitive reward functions for GRPO training.

Reward components:
- R_correct: Did the model solve the problem?
- R_calibration: Does the model's stated confidence match probe's p̂?
- R_epistemic: When the model expresses uncertainty + changes approach, does it help?
"""
import re
from typing import Optional


def compute_r_correct(is_correct: bool) -> float:
    return 1.0 if is_correct else 0.0


def compute_r_calibration(c_text: Optional[float], p_hat: float) -> float:
    """How accurately does the model know its own ability?

    c_text: model's self-stated confidence (from generated text)
    p_hat: probe's prediction from hidden states (ground truth)
    """
    if c_text is None:
        return 0.0
    return max(0.0, 1.0 - abs(c_text - p_hat))


def compute_r_epistemic(chain_text: str, is_correct: bool) -> float:
    """Does self-correction behavior actually help?

    Rewards:
    - Model expresses uncertainty + changes approach + correct → 1.0
    - Model expresses uncertainty + changes approach + wrong → 0.2
    - Model expresses uncertainty but doesn't change approach → 0.0
    - No uncertainty expression → 0.0
    """
    text_lower = chain_text.lower()

    # Detect epistemic expressions (uncertainty markers)
    epistemic_phrases = [
        "wait", "let me verify", "let me check", "is this correct",
        "i'm not sure", "not confident", "double-check", "let me reconsider",
        "hmm", "on second thought", "actually", "let me re-examine",
        "this doesn't seem right",
    ]
    has_epistemic = any(p in text_lower for p in epistemic_phrases)
    if not has_epistemic:
        return 0.0

    # Detect approach change (model actually redirected reasoning)
    redirect_phrases = [
        "alternatively", "instead", "let me try", "different approach",
        "another way", "let me redo", "starting over", "reconsider",
        "better approach", "try again",
    ]
    has_redirect = any(p in text_lower for p in redirect_phrases)
    if not has_redirect:
        return 0.0

    return 1.0 if is_correct else 0.2


def extract_confidence_from_chain(chain_text: str) -> Optional[float]:
    """Extract numeric confidence from model's output."""
    patterns = [
        r'confidence[:\s]+([0-9]+\.?[0-9]*)',
        r'probability[:\s]+([0-9]+\.?[0-9]*)\s*%?',
        r'([0-9]+\.?[0-9]*)\s*%\s*(?:confidence|chance|probability)',
        r'estimated.*?([0-9]+\.?[0-9]*)\s*%',
    ]
    for pattern in patterns:
        match = re.search(pattern, chain_text, re.IGNORECASE)
        if match:
            val = float(match.group(1))
            if val > 1.0:
                val = val / 100.0  # Convert percentage
            return max(0.0, min(1.0, val))
    return None


def compute_reward(
    is_correct: bool,
    chain_text: str,
    c_text: Optional[float],
    p_hat: float,
    reward_mode: str = "RL-B",
    lambda1: float = 0.5,
    lambda2: float = 0.3,
) -> dict:
    """Compute reward based on experiment mode.

    Modes:
    - RL-A: R_correct only
    - RL-B: R_correct + λ₁·R_calibration
    - RL-C: R_correct + λ₁·R_calibration + λ₂·R_epistemic
    - RL-D: R_calibration only
    """
    r_correct = compute_r_correct(is_correct)
    r_calib = compute_r_calibration(c_text, p_hat)
    r_epistemic = compute_r_epistemic(chain_text, is_correct)

    if reward_mode == "RL-A":
        total = r_correct
    elif reward_mode == "RL-B":
        total = r_correct + lambda1 * r_calib
    elif reward_mode == "RL-C":
        total = r_correct + lambda1 * r_calib + lambda2 * r_epistemic
    elif reward_mode == "RL-D":
        total = r_calib
    else:
        total = r_correct + lambda1 * r_calib

    return {
        "total": total,
        "r_correct": r_correct,
        "r_calib": r_calib,
        "r_epistemic": r_epistemic,
        "reward_mode": reward_mode,
    }


def compute_grpo_advantages(rewards: list, group_size: int = 8) -> list:
    """Compute GRPO normalized advantages within a group."""
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = max(var_r ** 0.5, 1e-8)
    return [(r - mean_r) / std_r for r in rewards]
