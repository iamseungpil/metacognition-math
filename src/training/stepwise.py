"""Stepwise RL training with <|meta|> token boundaries.

Each <|meta|>...<|/meta|> block defines a step boundary.
Gnosis probe provides p̂ at each step for:
  - R_calibration: does the model's stated confidence match p̂?
  - R_progress: is p̂ increasing (reasoning going in right direction)?
  - R_correct: final answer correctness (last step only)
"""
import torch
import torch.nn.functional as F

from src.metacot.prompt import META_START, META_END, parse_meta_blocks


def find_meta_token_positions(input_ids: torch.Tensor, tokenizer) -> list:
    """Find positions of <|meta|> and <|/meta|> tokens in the sequence.

    Returns list of (start_pos, end_pos) tuples for each meta block.
    """
    meta_start_id = tokenizer.convert_tokens_to_ids(META_START)
    meta_end_id = tokenizer.convert_tokens_to_ids(META_END)

    # Guard: ensure special tokens were actually added
    unk_id = getattr(tokenizer, 'unk_token_id', None)
    if meta_start_id == unk_id or meta_end_id == unk_id:
        return []  # tokens not in vocab, return empty

    if isinstance(input_ids, torch.Tensor):
        ids = input_ids.squeeze(0).tolist() if input_ids.dim() > 1 else input_ids.tolist()
    else:
        ids = list(input_ids)
    blocks = []
    i = 0
    while i < len(ids):
        if ids[i] == meta_start_id:
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
    """Extract hidden states at each <|/meta|> position (end of each meta block)."""
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        last_hidden = outputs.hidden_states[-1]  # (1, S, D)

    hidden_states = []
    for _, end_pos in meta_positions:
        if end_pos < last_hidden.shape[1]:
            hidden_states.append(last_hidden[0, end_pos].clone())

    # Free memory immediately
    del outputs, last_hidden
    torch.cuda.empty_cache()

    return hidden_states


def compute_gnosis_scores(probe, hidden_states: list) -> list:
    """Run Gnosis probe on hidden states at each meta position."""
    if not hidden_states:
        return []

    scores = []
    probe.eval()
    with torch.no_grad():
        for hs in hidden_states:
            hs_input = hs.float().unsqueeze(0).unsqueeze(0)  # (1, 1, D) in float32
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
    """Compute per-step rewards using 3 signals."""
    parsed = parse_meta_blocks(chain_text)
    model_confidences = parsed["confidences"]

    n_steps = max(len(gnosis_scores), len(model_confidences), 1)

    step_rewards = []
    prev_p_hat = 0.5

    for k in range(n_steps):
        p_hat = gnosis_scores[k] if k < len(gnosis_scores) else 0.5
        c_text = model_confidences[k] if k < len(model_confidences) else None

        r_calib = (1.0 - abs(c_text - p_hat)) if c_text is not None else 0.0
        r_progress = p_hat - prev_p_hat
        r_correct = (1.0 if is_correct else 0.0) if k == n_steps - 1 else 0.0

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


def compute_step_level_loss(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    meta_positions: list,
    step_rewards: list,
    prompt_len: int,
) -> torch.Tensor:
    """Compute policy gradient loss with step-level rewards.

    Strategy: each meta block k and the text AFTER it (until next meta block)
    share step_rewards[k]. The text before the first meta block gets step_rewards[0].
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits

    gen_logits = logits[:, prompt_len - 1:-1]
    gen_targets = input_ids[:, prompt_len:]
    log_probs = F.log_softmax(gen_logits, dim=-1)
    token_log_probs = log_probs.gather(-1, gen_targets.unsqueeze(-1)).squeeze(-1)  # (1, gen_len)

    gen_len = token_log_probs.shape[1]
    if gen_len == 0:
        return torch.tensor(0.0, device=input_ids.device, requires_grad=True)

    # Build per-token reward assignment based on meta block positions
    # Each token gets the reward of its enclosing/preceding meta block
    token_rewards = torch.zeros(gen_len, device=input_ids.device)

    if not meta_positions or not step_rewards:
        # No meta blocks: apply average reward to all tokens
        avg_reward = sum(sr["total"] for sr in step_rewards) / max(len(step_rewards), 1)
        token_rewards[:] = avg_reward
    else:
        # Convert meta positions to gen-relative indices
        gen_boundaries = []
        for start_pos, end_pos in meta_positions:
            gen_start = start_pos - prompt_len
            gen_end = end_pos - prompt_len
            if gen_start >= 0 and gen_end < gen_len:
                gen_boundaries.append((gen_start, gen_end))

        # Assign rewards: tokens between meta block k-1 end and meta block k end
        # get step_rewards[k]
        reward_idx = 0
        for t in range(gen_len):
            # Check if we've passed a meta block end → advance to next reward
            while (reward_idx < len(gen_boundaries) - 1 and
                   t > gen_boundaries[reward_idx][1]):
                reward_idx += 1

            if reward_idx < len(step_rewards):
                token_rewards[t] = step_rewards[reward_idx]["total"]

    # Policy gradient: -reward * log_prob per token
    total_loss = -(token_rewards * token_log_probs[0]).mean()

    del outputs, logits
    return total_loss
