"""Reward functions for Meta-CoT GRPO.

Four independent reward functions, each usable with vanilla GRPOTrainer.
GDPO (per-reward normalization) prevents strong rewards from drowning weak ones.

R1: correctness_reward   — binary +1/-1
R2: meta_quality_reward  — meta block presence, length, Q&A structure
R3: calibration_reward   — Rewarding Doubt log scoring rule, summation across blocks
R4: uncertainty_meta_reward — (1-conf) weighted meta quality, summation
"""
import math
import re


def _extract_answer(text):
    """Extract boxed or #### answer."""
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    m = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    return ""


def _check_correctness(pred_text, gold):
    p = _extract_answer(pred_text)
    g = _extract_answer(str(gold))
    # If gold has no boxed/####, try extracting the last number as answer
    if not g:
        # Try "the answer is X" pattern
        m = re.search(r'(?:answer|result)\s+(?:is|=)\s+[\\$]*(-?[\d,.]+(?:/\d+)?)', str(gold), re.IGNORECASE)
        if m:
            g = m.group(1).strip()
        else:
            # Last resort: extract last number from gold
            nums = re.findall(r'(-?\d+(?:\.\d+)?(?:/\d+)?)', str(gold))
            if nums:
                g = nums[-1]
            else:
                g = str(gold).strip()
    if not p:
        return False
    if p == g:
        return True
    # Normalize: strip $, commas, whitespace
    p_norm = re.sub(r'[\$,\s]', '', p)
    g_norm = re.sub(r'[\$,\s]', '', g)
    if p_norm == g_norm:
        return True
    try:
        if abs(float(p_norm) - float(g_norm)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    return p_norm.lower() == g_norm.lower()


def _get_text(completion):
    """Extract text from TRL completion format."""
    if isinstance(completion, list):
        return completion[0]["content"] if completion else ""
    return str(completion)


def _parse_meta_blocks(text):
    """Parse <|meta|> blocks from text. Works even when tokens are stripped.

    Returns list of dicts: [{text, confidence, length}, ...]
    """
    blocks = []
    # Try special token boundaries first
    parts = re.split(r'<\|meta\|>', text)
    for i, part in enumerate(parts[1:], 1):
        end_idx = part.find('<|/meta|>')
        block_text = part[:end_idx] if end_idx != -1 else part[:200]
        conf = _parse_confidence(block_text)
        blocks.append({"text": block_text, "confidence": conf, "length": len(block_text.split())})

    # Fallback: detect meta-like patterns in stripped text
    if not blocks:
        conf_matches = re.findall(
            r'(?:probability|confidence|확률|확신)[:\s\w]*?(\d+\.\d+|\d+)\s*%?',
            text, re.IGNORECASE
        )
        for m in conf_matches:
            v = float(m)
            if v > 1:
                v /= 100
            v = max(0.0, min(1.0, v))
            blocks.append({"text": "", "confidence": v, "length": 0})

    return blocks


def _parse_confidence(text):
    """Extract confidence value from a meta block."""
    matches = re.findall(
        r'(?:probability|confidence|확률|확신)[:\s\w]*?(\d+\.\d+|\d+)\s*%?',
        text, re.IGNORECASE
    )
    for m in matches:
        v = float(m)
        if v > 1:
            v /= 100
        return max(0.01, min(0.99, v))
    return None


# ─── R1: Correctness ───

def correctness_reward(completions, ground_truth=None, **kwargs):
    """Binary: +1 correct, -1 incorrect."""
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        rewards.append(1.0 if _check_correctness(text, gt) else -1.0)
    return rewards


# ─── R2: Meta Quality ───

def meta_quality_reward(completions, **kwargs):
    """Reward for meta reasoning presence and quality.

    -0.5 if no meta blocks
    +0.1 per meta block (up to 3)
    +0.1 bonus if block has Q&A structure
    +0.1 bonus if block has 10+ words
    Max per block: 0.3, max total: ~1.0
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        blocks = _parse_meta_blocks(text)

        if not blocks:
            rewards.append(-0.5)
            continue

        r = 0.0
        for block in blocks[:4]:  # cap at 4 blocks
            r += 0.1  # existence bonus
            if block["length"] >= 10:
                r += 0.1  # length bonus
            if re.search(r'[?？]', block["text"]):
                r += 0.1  # Q&A structure bonus
        rewards.append(min(r, 1.2))
    return rewards


# ─── R3: Calibration (Rewarding Doubt) ───

def calibration_reward(completions, ground_truth=None, **kwargs):
    """Log scoring rule (Rewarding Doubt), summed across meta blocks.

    correct + high conf → small negative (good)
    correct + low conf  → large negative (bad, underconfident)
    wrong + low conf    → small negative (good, knows it doesn't know)
    wrong + high conf   → large negative (bad, overconfident)

    Summation: more meta blocks = stronger signal.
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        blocks = _parse_meta_blocks(text)

        if not blocks or all(b["confidence"] is None for b in blocks):
            rewards.append(0.0)
            continue

        r = 0.0
        n = 0
        for block in blocks:
            conf = block["confidence"]
            if conf is None:
                continue
            if is_correct:
                r += math.log(max(conf, 0.01))
            else:
                r += math.log(max(1.0 - conf, 0.01))
            n += 1

        # Normalize to reasonable range: log(0.01)=-4.6, log(0.99)=-0.01
        # Scale so typical values are in [-2, 0]
        if n > 0:
            r = r / n  # average per block, then scale by count
            r = r * min(n, 3)  # summation effect capped at 3
        rewards.append(max(r, -5.0))  # floor
    return rewards


# ─── R4: Uncertainty-Aware Meta ───

def uncertainty_meta_reward(completions, ground_truth=None, **kwargs):
    """Higher reward when meta blocks appear at uncertain moments.

    (1 - confidence) × meta_quality → summation
    Uncertain + good meta = high reward
    Confident + any meta = low reward (unnecessary reflection)
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        blocks = _parse_meta_blocks(text)

        if not blocks:
            rewards.append(0.0)
            continue

        r = 0.0
        for block in blocks[:4]:
            conf = block["confidence"]
            if conf is None:
                conf = 0.5  # unknown confidence treated as moderate

            uncertainty = 1.0 - conf  # high when unsure

            # Quality component
            quality = 0.0
            if block["length"] >= 10:
                quality += 0.5
            if re.search(r'[?？]', block["text"]):
                quality += 0.5
            quality = min(quality, 1.0)

            r += uncertainty * quality  # uncertain + quality meta = high

        rewards.append(r)
    return rewards
