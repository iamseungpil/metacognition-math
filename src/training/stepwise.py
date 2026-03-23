"""Stepwise RL training with <|meta|> token boundaries.

Each <|meta|>...<|/meta|> block defines a step boundary.
Gnosis probe provides p̂ at each step for:
  - R_calibration: does the model's stated confidence match p̂?
  - R_progress: is p̂ increasing (reasoning going in right direction)?
  - R_correct: final answer correctness (last step only)
"""
import re
import torch
import torch.nn.functional as F
from typing import Optional

from src.metacot.prompt import META_START, META_END, parse_meta_blocks


def find_meta_token_positions(input_ids: torch.Tensor, tokenizer) -> list:
    """Find positions of <|meta|> and <|/meta|> tokens in the sequence.

    Returns list of (start_pos, end_pos) tuples for each meta block.
    """
    meta_start_id = tokenizer.convert_tokens_to_ids(META_START)
    meta_end_id = tokenizer.convert_tokens_to_ids(META_END)

    ids = input_ids.squeeze().tolist()
    blocks = []
    i = 0
    while i < len(ids):
        if ids[i] == meta_start_id:
            # Find matching end
            for j in range(i + 1, len(ids)):
                if ids[j] == meta_end_id:
                    blocks.append((i, j))
                    i = j + 1
                    break
            else:
                i += 1
        else:
            i += 1

    return blocks


def get_hidden_states_at_meta(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    meta_positions: list,
) -> list:
    """Extract hidden states at each <|/meta|> position (end of each meta block).

    These represent the model's internal state after each self-reflection.
    """
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (1, S, D)

    # Extract hidden state at end of each meta block
    hidden_states = []
    for start_pos, end_pos in meta_positions:
        if end_pos < last_hidden.shape[1]:
            hs = last_hidden[0, end_pos]  # (D,)
            hidden_states.append(hs)

    return hidden_states


def compute_gnosis_scores(probe, hidden_states: list) -> list:
    """Run Gnosis probe on hidden states at each meta position.

    Returns list of p̂ values (probability of correctness at each step).
    """
    if not hidden_states:
        return []

    scores = []
    probe.eval()
    with torch.no_grad():
        for hs in hidden_states:
            # Probe expects (B, S, D) — wrap single vector
            hs_input = hs.unsqueeze(0).unsqueeze(0)  # (1, 1, D)
            p_hat = probe(hs_input).item()
            scores.append(p_hat)

    return scores


def compute_stepwise_rewards(
    chain_text: str,
    is_correct: bool,
    gnosis_scores: list,
    lambda1: float = 0.5,
    lambda2: float = 0.3,
) -> list:
    """Compute per-step rewards using 3 signals.

    For each <|meta|> block:
      R_calibration = 1 - |model_confidence - gnosis_p_hat|
      R_progress = gnosis_p_hat[k] - gnosis_p_hat[k-1]
    For last step only:
      R_correct = 1.0 if correct else 0.0

    Returns list of reward dicts per step.
    """
    parsed = parse_meta_blocks(chain_text)
    model_confidences = parsed["confidences"]

    # Pad to match lengths
    n_steps = max(len(gnosis_scores), len(model_confidences), 1)

    step_rewards = []
    prev_p_hat = 0.5  # prior before any reasoning

    for k in range(n_steps):
        p_hat = gnosis_scores[k] if k < len(gnosis_scores) else 0.5
        c_text = model_confidences[k] if k < len(model_confidences) else None

        # R_calibration
        r_calib = (1.0 - abs(c_text - p_hat)) if c_text is not None else 0.0

        # R_progress (Gnosis Temporal Difference)
        r_progress = p_hat - prev_p_hat

        # R_correct (last step only)
        r_correct = 0.0
        if k == n_steps - 1:
            r_correct = 1.0 if is_correct else 0.0

        total = r_correct + lambda1 * r_calib + lambda2 * r_progress

        step_rewards.append({
            "step": k,
            "r_correct": r_correct,
            "r_calib": r_calib,
            "r_progress": r_progress,
            "total": total,
            "p_hat": p_hat,
            "c_text": c_text,
        })

        prev_p_hat = p_hat

    return step_rewards


def split_token_ids_by_meta(
    input_ids: torch.Tensor,
    meta_positions: list,
    prompt_len: int,
) -> list:
    """Split generated token IDs into segments between <|meta|> blocks.

    Each segment gets assigned the reward of the following meta block.
    Returns list of (start_idx, end_idx) tuples in the generated portion.
    """
    ids = input_ids.squeeze()
    gen_start = prompt_len
    gen_end = len(ids)

    if not meta_positions:
        return [(gen_start, gen_end)]

    segments = []
    prev_end = gen_start

    for start_pos, end_pos in meta_positions:
        if start_pos >= gen_start:
            # Segment before this meta block
            if start_pos > prev_end:
                segments.append((prev_end, start_pos))
            # The meta block itself
            segments.append((start_pos, end_pos + 1))
            prev_end = end_pos + 1

    # Remaining tokens after last meta block
    if prev_end < gen_end:
        segments.append((prev_end, gen_end))

    return segments


def compute_step_level_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    meta_positions: list,
    step_rewards: list,
    prompt_len: int,
) -> torch.Tensor:
    """Compute policy gradient loss with step-level rewards.

    Each segment between <|meta|> blocks gets the reward of its
    corresponding step. This gives finer credit assignment than
    whole-rollout reward.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits

    # Compute per-token log probs for generated tokens
    gen_logits = logits[:, prompt_len - 1:-1]  # (1, gen_len, V)
    gen_targets = input_ids[:, prompt_len:]     # (1, gen_len)

    log_probs = F.log_softmax(gen_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1)  # (1, gen_len)

    # Assign step rewards to token ranges
    segments = split_token_ids_by_meta(input_ids, meta_positions, prompt_len)

    # Map each segment to a step reward
    total_loss = torch.tensor(0.0, device=input_ids.device)
    reward_idx = 0

    for seg_start, seg_end in segments:
        # Adjust to gen_logits indexing (offset by prompt_len)
        tok_start = max(0, seg_start - prompt_len)
        tok_end = min(token_log_probs.shape[1], seg_end - prompt_len)

        if tok_start >= tok_end:
            continue

        # Get reward for this segment
        if reward_idx < len(step_rewards):
            advantage = step_rewards[reward_idx]["total"]
        else:
            advantage = 0.0

        # Policy gradient: -advantage * mean(log_probs)
        segment_log_probs = token_log_probs[0, tok_start:tok_end]
        if len(segment_log_probs) > 0:
            total_loss += -(advantage * segment_log_probs.mean())

        # Advance reward index at meta block boundaries
        if any(seg_start == pos[0] for pos in meta_positions):
            reward_idx += 1

    return total_loss
