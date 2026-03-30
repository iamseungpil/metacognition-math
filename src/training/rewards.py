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


# ─── R5: Stepwise Confidence Trajectory ───

def stepwise_trajectory_reward(completions, ground_truth=None, **kwargs):
    """Reward confidence trajectory: start low, end accurate.

    Ideal: pre-meta conf ~0.3 → mid-meta conf ~0.6 → post-meta conf matches accuracy

    Components:
    1. Pre-meta should be uncertain (conf < 0.7 → bonus)
    2. Confidence should generally increase (monotonic bonus)
    3. Final confidence should match group accuracy (Brier)
    4. If wrong, final conf should be low (Rewarding Doubt)
    """
    # Compute group accuracy
    correct_flags = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        correct_flags.append(1.0 if _check_correctness(text, gt) else 0.0)
    group_accuracy = sum(correct_flags) / max(len(correct_flags), 1)

    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        blocks = _parse_meta_blocks(text)
        is_correct = bool(correct_flags[i])

        if not blocks or all(b["confidence"] is None for b in blocks):
            rewards.append(0.0)
            continue

        confs = [b["confidence"] for b in blocks if b["confidence"] is not None]
        if not confs:
            rewards.append(0.0)
            continue

        r = 0.0

        # 1. Pre-meta uncertainty bonus: first conf should be low
        if confs[0] < 0.5:
            r += 0.3  # "starts uncertain = good"
        elif confs[0] > 0.9:
            r -= 0.3  # "starts overconfident = bad"

        # 2. Monotonic increase bonus
        increases = sum(1 for j in range(1, len(confs)) if confs[j] >= confs[j-1])
        if len(confs) > 1:
            r += 0.2 * (increases / (len(confs) - 1))  # fraction of increasing steps

        # 3. Final confidence accuracy (Brier + Doubt)
        final_conf = confs[-1]
        r += -(final_conf - group_accuracy) ** 2  # Brier
        if is_correct:
            r += 0.5 * math.log(max(final_conf, 0.01))  # Doubt
        else:
            r += 0.5 * math.log(max(1.0 - final_conf, 0.01))

        rewards.append(max(r, -3.0))
    return rewards


# ─── R6: Probe Calibration Reward ───

# Global probe state (loaded once, reused)
_probe_model = None
_probe_head = None
_probe_tokenizer = None


def _load_probe(model_path="checkpoints/qwen3_meta_sft",
                probe_path="checkpoints/simple_probe_qwen3/best_probe.pt"):
    """Load probe model and head once (lazy init)."""
    global _probe_model, _probe_head, _probe_tokenizer
    if _probe_model is not None:
        return

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading probe model (frozen)...")
    _probe_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    _probe_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    _probe_model.eval()
    # Don't put on GPU yet — will use training model's device

    print("Loading probe head...")
    _probe_head = torch.load(probe_path, map_location="cpu")
    _probe_head.eval()
    print(f"Probe loaded: model={model_path}, head={probe_path}")


def stepwise_probe_reward(completions, ground_truth=None, **kwargs):
    """Stepwise credit assignment: independent reward per meta block position.

    Each completion's meta blocks are classified by position (pre/mid/post)
    and scored independently based on confidence accuracy at that stage.

    Scoring:
      Pre-meta (first block):
        base = -(conf - group_accuracy)^2
        +0.3 if conf < 0.5 (starts uncertain = good)
        -0.3 if conf > 0.9 (starts overconfident = bad)

      Mid-meta (middle blocks):
        +0.5 if error-correction pattern detected ("wait", "wrong", "actually", "fix")
        -0.2 if conf > 0.95 (overconfident mid-stream)

      Post-meta (last block before \\boxed{}):
        Brier: -(conf - correct)^2 where correct in {0,1}
        Log (Rewarding Doubt):
          correct: +0.5 * log(conf)
          wrong:   +0.5 * log(1-conf)

    Final: R = 0.2 * mean(pre) + 0.3 * mean(mid) + 0.5 * post
    Returns: list[float], one per completion (TRL GRPOTrainer compatible).
    """
    # Error-correction keywords regex (compiled once)
    _error_correction_re = re.compile(
        r'\b(wait|wrong|fix|actually|mistake|no,|let me re|hold on|incorrect|error)\b',
        re.IGNORECASE,
    )

    # First pass: compute group accuracy for pre-meta target
    correct_flags = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        correct_flags.append(1.0 if _check_correctness(text, gt) else 0.0)
    raw_accuracy = sum(correct_flags) / max(len(correct_flags), 1)
    # With small batches, group_accuracy is degenerate (0 or 1).
    # Smooth toward 0.5 prior to give meaningful pre-meta calibration signal.
    # At batch>=4 (GRPO standard), smoothing effect is small.
    _PRIOR = 0.5
    _PRIOR_WEIGHT = 2  # equivalent to 2 pseudo-observations
    n = len(correct_flags)
    group_accuracy = (raw_accuracy * n + _PRIOR * _PRIOR_WEIGHT) / (n + _PRIOR_WEIGHT)

    # Second pass: stepwise scoring
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        blocks = _parse_meta_blocks(text)
        is_correct = bool(correct_flags[i])

        # Filter to blocks with confidence values
        conf_blocks = [b for b in blocks if b["confidence"] is not None]
        if not conf_blocks:
            rewards.append(0.0)
            continue

        pre_scores = []
        mid_scores = []
        post_score = None

        n_blocks = len(conf_blocks)
        for idx, block in enumerate(conf_blocks):
            conf = block["confidence"]
            block_text = block["text"]

            if idx == 0 and n_blocks >= 2:
                # --- Pre-meta (first block, only when >=2 blocks) ---
                score = -(conf - group_accuracy) ** 2
                if conf < 0.5:
                    score += 0.3
                elif conf > 0.9:
                    score -= 0.3
                pre_scores.append(score)

            elif idx == n_blocks - 1:
                # --- Post-meta (last block) ---
                target = 1.0 if is_correct else 0.0
                score = -(conf - target) ** 2  # Brier
                # Rewarding Doubt log scoring
                conf_clamped = max(0.01, min(0.99, conf))
                if is_correct:
                    score += 0.5 * math.log(conf_clamped)
                else:
                    score += 0.5 * math.log(1.0 - conf_clamped)
                post_score = score

            else:
                # --- Mid-meta (middle blocks) ---
                score = 0.0
                if _error_correction_re.search(block_text):
                    score += 0.5
                if conf > 0.95:
                    score -= 0.2
                mid_scores.append(score)

        # Handle edge case: single block => treat it as post
        if n_blocks == 1:
            conf = conf_blocks[0]["confidence"]
            target = 1.0 if is_correct else 0.0
            score = -(conf - target) ** 2
            conf_clamped = max(0.01, min(0.99, conf))
            if is_correct:
                score += 0.5 * math.log(conf_clamped)
            else:
                score += 0.5 * math.log(1.0 - conf_clamped)
            post_score = score

        # Weighted combination: 0.2 pre + 0.3 mid + 0.5 post
        r = 0.0
        if pre_scores:
            r += 0.2 * (sum(pre_scores) / len(pre_scores))
        if mid_scores:
            r += 0.3 * (sum(mid_scores) / len(mid_scores))
        if post_score is not None:
            r += 0.5 * post_score

        # Clamp to avoid extreme negatives destabilizing training
        rewards.append(max(r, -3.0))

    return rewards


def probe_calibration_reward(completions, ground_truth=None,
                              model=None, tokenizer=None, **kwargs):
    """Calibration reward using hidden state probe.

    Uses the TRAINING model's hidden states (no separate model needed).
    Requires model and tokenizer to be passed via closure.

    R = -(stated_confidence - probe_p_hat)²
    Forces model's stated confidence to match its internal belief.
    """
    import torch

    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        blocks = _parse_meta_blocks(text)

        # Get stated confidence
        confs = [b["confidence"] for b in blocks if b["confidence"] is not None]
        if not confs:
            rewards.append(0.0)
            continue
        stated_conf = confs[-1]  # last meta block confidence

        # TODO: When model/tokenizer are available via closure,
        # compute probe p_hat from hidden states:
        #   inputs = tokenizer(text, return_tensors="pt").to(model.device)
        #   with torch.no_grad():
        #       outputs = model(**inputs, output_hidden_states=True)
        #       hidden = outputs.hidden_states[-1][:, -1, :].float()
        #       p_hat = probe_head(hidden).sigmoid().item()
        #   r = -(stated_conf - p_hat) ** 2

        # For now: use group accuracy as proxy for p_hat
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        p_hat = 1.0 if is_correct else 0.0

        # Brier score against binary outcome (like Rewarding Doubt)
        r = -(stated_conf - p_hat) ** 2
        rewards.append(r)

    return rewards


# ─── R7: Length Penalty ───

def length_penalty_reward(completions, **kwargs):
    """Penalize verbose responses to prevent reward hacking via length.

    No penalty below 500 tokens. Linear penalty above 500.
    -0.001 per token over 500 (max penalty -1.5 at 2000 tokens).

    This prevents GRPO from gaming meta_quality/format rewards
    by generating long, low-quality responses.
    """
    rewards = []
    for c in completions:
        text = _get_text(c)
        n_tokens = len(text.split())  # word-level approximation
        if n_tokens <= 500:
            rewards.append(0.0)
        else:
            penalty = -0.001 * (n_tokens - 500)
            rewards.append(max(penalty, -1.5))
    return rewards


# ─── R8: Correctness-Conditional Meta ───

def correct_meta_reward(completions, ground_truth=None, **kwargs):
    """Only reward meta quality when the answer is correct.

    This prevents the model from generating verbose meta blocks
    on wrong answers just to collect meta_quality reward.

    Correct + good meta → +0.5
    Correct + no meta → 0.0
    Wrong + any meta → -0.3 (penalty for wasting tokens on wrong answer)
    Wrong + no meta → 0.0
    """
    rewards = []
    for i, c in enumerate(completions):
        text = _get_text(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = _check_correctness(text, gt)
        blocks = _parse_meta_blocks(text)

        if is_correct:
            if blocks:
                # Reward proportional to meta quality
                q = 0.0
                for b in blocks[:3]:
                    if b["confidence"] is not None:
                        q += 0.1
                    if b["length"] >= 10:
                        q += 0.05
                rewards.append(min(q, 0.5))
            else:
                rewards.append(0.0)
        else:
            if blocks:
                rewards.append(-0.3)  # Penalize meta on wrong answers
            else:
                rewards.append(0.0)
    return rewards
