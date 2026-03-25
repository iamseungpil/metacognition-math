"""Reward function for veRL GRPO — metacognitive math reasoning.

3 rewards:
- R_correct = 2.0 if answer correct, 0.0 otherwise
- R_calib = calibration reward (encourages accurate self-assessment)
- R_penalty = -0.5 if no <|meta|> blocks, -0.3 if only 1

Registered as 'metacot_math' data_source in veRL's reward system.
"""
import re
import math


# Inline parse_meta_blocks to avoid import issues across processes
META_START = "<|meta|>"
META_END = "<|/meta|>"


def _parse_meta_blocks(text):
    start_esc = re.escape(META_START)
    end_esc = re.escape(META_END)
    pattern = rf'{start_esc}(.*?){end_esc}'
    blocks = re.findall(pattern, text, re.DOTALL)

    confidences = []
    for block in blocks:
        conf_pattern = r'(?:probability|confidence|확률|확신)[:\s]*(\d+(?:\.\d+)?%?)'
        matches = re.findall(conf_pattern, block, re.IGNORECASE)
        for m in matches:
            val = m.rstrip('%')
            try:
                c = float(val)
                if c > 1.0:
                    c /= 100.0
                confidences.append(max(0.0, min(1.0, c)))
            except ValueError:
                pass

    return {"num_blocks": len(blocks), "confidences": confidences}


def _extract_answer(text):
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    m = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'(?:the answer is|answer:\s*)\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return ""


def _check_correctness(model_answer, gold_answer):
    model_final = _extract_answer(model_answer)
    gold_str = str(gold_answer).strip()
    gold_final = _extract_answer(gold_str) or gold_str

    if not model_final:
        return False
    if model_final == gold_final:
        return True
    try:
        if abs(float(model_final) - float(gold_final)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    return model_final.lower().strip() == gold_final.lower().strip()


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """Called by veRL for each generated response.

    R_calib without probe: rewards calibrated self-assessment.
    - correct + high confidence → good
    - wrong + low confidence → good (self-aware)
    - wrong + high confidence → bad (overconfident)
    """
    parsed = _parse_meta_blocks(solution_str)
    num_meta = parsed["num_blocks"]
    confidences = parsed["confidences"]

    # R_correct
    is_correct = _check_correctness(solution_str, str(ground_truth))
    r_correct = 2.0 if is_correct else 0.0

    # R_penalty
    if num_meta >= 2:
        r_penalty = 0.0
    elif num_meta == 1:
        r_penalty = -0.3
    else:
        r_penalty = -0.5

    # R_calib (text-based, no probe)
    r_calib = 0.0
    if confidences:
        avg_conf = sum(confidences) / len(confidences)
        actual = 1.0 if is_correct else 0.0
        r_calib = max(0.0, 1.0 - abs(avg_conf - actual))

    total = r_correct + r_calib + r_penalty
    return total
