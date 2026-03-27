"""Reward functions for Meta-CoT GRPO.

Four independent reward functions, each usable with vanilla GRPOTrainer.
GDPO (per-reward normalization) prevents strong rewards from drowning weak ones.

R1: correctness_reward   — sympy-based math verification (Open-R1 style)
R2: meta_quality_reward  — meta block presence, length, Q&A structure
R3: calibration_reward   — Rewarding Doubt log scoring rule, summation across blocks
R4: uncertainty_meta_reward — (1-conf) weighted meta quality, summation

Reference: Open-R1 (https://github.com/huggingface/open-r1)
"""
import math
import re

# Math verification via sympy (same as Open-R1)
try:
    from math_verify import parse, verify
    from latex2sympy2_extended import NormalizationConfig
    HAS_MATH_VERIFY = True
except ImportError:
    HAS_MATH_VERIFY = False


def _check_correctness(pred_text, gold):
    """Verify math answer using sympy (Open-R1 style).

    Handles: fractions, decimals, LaTeX expressions, boxed answers, etc.
    Fallback to string matching if math_verify not available.
    """
    if HAS_MATH_VERIFY:
        try:
            gold_parsed = parse(str(gold), extraction_mode="first_match")
            pred_parsed = parse(str(pred_text), extraction_mode="first_match")
            return bool(verify(gold_parsed, pred_parsed))
        except Exception:
            pass  # fallback to string matching

    # Fallback: string matching
    p = _extract_answer_fallback(pred_text)
    g = _extract_answer_fallback(str(gold))
    if not p or not g:
        return False
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return p.lower().strip() == g.lower().strip()


def _extract_answer_fallback(text):
    """Fallback answer extraction (string-based)."""
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    m = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    nums = re.findall(r'(-?\d+(?:\.\d+)?)', text)
    return nums[-1] if nums else ""


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


# ─── R1b: Format Reward (Open-R1 style) ───

def format_reward(completions, **kwargs):
    """Reward for using \\boxed{} answer format.

    +1 if completion contains \\boxed{...}
    0 otherwise
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        has_boxed = bool(re.search(r'\\boxed\{', text))
        rewards.append(1.0 if has_boxed else 0.0)
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


# ─── R3: Calibration (Group-based Doubt, CoCA style) ───

def calibration_reward(completions, ground_truth=None, **kwargs):
    """Group-based Brier score calibration (CoCA/Rewarding Doubt hybrid).

    Uses group empirical accuracy as target (not binary per-sample):
      p̂ = fraction of group that got correct answer
      r = -(stated_confidence - p̂)² per meta block, summed

    This gives smoother calibration signal than binary correct/wrong.
    """
    # First pass: compute group accuracy
    correct_flags = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        correct_flags.append(1.0 if _check_correctness(text, gt) else 0.0)
    group_accuracy = sum(correct_flags) / max(len(correct_flags), 1)

    # Second pass: compute calibration reward per completion
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
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
            # Brier score: -(conf - p̂)²
            r += -(conf - group_accuracy) ** 2
            n += 1

        # Also add log scoring (Rewarding Doubt) for individual correctness
        is_correct = bool(correct_flags[i])
        last_conf = None
        for block in reversed(blocks):
            if block["confidence"] is not None:
                last_conf = block["confidence"]
                break
        if last_conf is not None:
            if is_correct:
                r += math.log(max(last_conf, 0.01))
            else:
                r += math.log(max(1.0 - last_conf, 0.01))
            n += 1

        if n > 0:
            r = r / n  # average
        rewards.append(max(r, -5.0))
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
