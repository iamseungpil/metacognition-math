"""Custom reward for verl GRPO — metacognitive math reasoning.

3 rewards:
- R_correct = 2.0 if answer correct, 0.0 otherwise
- R_calib = reward for accurate self-assessment in <|meta|> blocks
  (model's stated confidence matches actual correctness)
- R_penalty = -0.5 if no <|meta|> blocks, -0.3 if only 1

Gnosis probe is NOT used during training (verl doesn't expose hidden states).
Instead, model's text-based confidence from <|meta|> blocks is used for R_calib.
Gnosis is used as eval metric after training.
"""
import re
import sys
sys.path.insert(0, "/scratch/metacognition")

from src.metacot.prompt import parse_meta_blocks


def _extract_answer(text):
    """Extract answer from \\boxed{} or other formats."""
    # \boxed{}
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    # #### format
    m = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    # "The answer is"
    m = re.search(r'(?:the answer is|answer:\s*)\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def check_correctness(model_answer, gold_answer):
    model_final = _extract_answer(model_answer)
    gold_str = str(gold_answer).strip()
    # Also try extracting from gold (in case it has \boxed{} etc)
    gold_final = _extract_answer(gold_str)
    if not gold_final:
        gold_final = gold_str  # Use raw gold if extraction fails

    if not model_final:
        return False
    if model_final == gold_final:
        return True
    try:
        if abs(float(model_final) - float(gold_final)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    if model_final.lower().strip() == gold_final.lower().strip():
        return True
    return False


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Called by verl for each generated response."""
    if data_source != "metacot_math":
        return 0.0

    # Parse <|meta|> blocks
    parsed = parse_meta_blocks(solution_str)
    num_meta = parsed["num_blocks"]
    confidences = parsed["confidences"]

    # R_correct: strong reward for correct answer
    is_correct = check_correctness(solution_str, str(ground_truth))
    r_correct = 2.0 if is_correct else 0.0

    # R_penalty: penalize not using metacognitive reflection
    if num_meta >= 2:
        r_penalty = 0.0
    elif num_meta == 1:
        r_penalty = -0.3
    else:
        r_penalty = -0.5

    # R_calib: reward accurate self-assessment
    # If model says "confidence 0.9" and is correct → good
    # If model says "confidence 0.3" and is wrong → good (knows it's uncertain)
    # If model says "confidence 0.9" and is wrong → bad (overconfident)
    r_calib = 0.0
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        if is_correct:
            r_calib = avg_conf  # correct + confident = good
        else:
            r_calib = 1.0 - avg_conf  # wrong + uncertain = good self-awareness

    total = r_correct + r_calib + r_penalty
    return total
