"""GRPO + Full Gnosis + Stepwise Credit Assignment for Qwen3-8B Meta-CoT.

Key architecture:
- TRL GRPOTrainer subclass (MetaCotGRPOTrainer)
- vLLM colocate for fast generation
- Full Gnosis correctness head (attention + hidden + confidence extractors)
- <|meta|> stepwise credit assignment: per-token reward weighting
- Combined loss: GRPO_loss + lambda_gnosis * Gnosis_correctness_loss
- LoRA for memory efficiency on 4x A100 80GB

Loss design:
  L_total = L_grpo(stepwise) + λ_gnosis * L_correctness(BCE)

  L_grpo: standard clipped policy gradient, but with per-token reward weights
          from <|meta|> step boundaries. Each step k gets reward:
            r_k = R_calib_k + R_progress_k + R_correct_k(last step only)

  L_correctness: BCE(Gnosis_probe_output, is_correct_label)
          Trains the Gnosis head to predict answer correctness
          from attention maps + hidden states + token probabilities
"""
import argparse
import json
import math
import os
import re
from contextlib import nullcontext

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset

# gnosis_repo's transformers and TRL must be on PYTHONPATH
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from trl.trainer.grpo_trainer import selective_log_softmax
from peft import LoraConfig


# ─── Constants ───
META_START = "<|meta|>"
META_END = "<|/meta|>"
LAMBDA_GNOSIS = 0.5  # Weight for Gnosis correctness loss


# ─── Meta-CoT parsing ───
def parse_meta_blocks(text):
    start_esc = re.escape(META_START)
    end_esc = re.escape(META_END)
    blocks = re.findall(rf'{start_esc}(.*?){end_esc}', text, re.DOTALL)
    confidences = []
    for block in blocks:
        matches = re.findall(
            r'(?:probability|confidence|확률|확신)[:\s]*(\d+(?:\.\d+)?%?)',
            block, re.IGNORECASE
        )
        for m in matches:
            try:
                c = float(m.rstrip('%'))
                if c > 1.0:
                    c /= 100.0
                confidences.append(max(0.0, min(1.0, c)))
            except ValueError:
                pass
    return {"num_blocks": len(blocks), "confidences": confidences, "block_texts": blocks}


def extract_answer(text):
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


def check_correctness(model_answer, gold_answer):
    model_final = extract_answer(model_answer)
    gold_str = str(gold_answer).strip()
    gold_final = extract_answer(gold_str) or gold_str
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


def find_meta_positions_in_ids(token_ids, tokenizer):
    """Find <|meta|>...<|/meta|> block positions in token ID sequence."""
    meta_start_id = tokenizer.convert_tokens_to_ids(META_START)
    meta_end_id = tokenizer.convert_tokens_to_ids(META_END)
    unk_id = getattr(tokenizer, 'unk_token_id', None)
    if meta_start_id == unk_id or meta_end_id == unk_id:
        return []

    ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
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


def compute_stepwise_reward_weights(
    completion_ids,
    completion_text,
    is_correct,
    tokenizer,
    prompt_len=0,
    lambda_calib=1.0,
    lambda_progress=0.3,
):
    """Compute per-token reward weights based on <|meta|> step boundaries.

    Returns:
        step_weights: (completion_len,) tensor of per-token reward weights
        step_info: dict with per-step details for logging
    """
    parsed = parse_meta_blocks(completion_text)
    num_meta = parsed["num_blocks"]
    confidences = parsed["confidences"]

    comp_len = len(completion_ids) if isinstance(completion_ids, list) else completion_ids.shape[-1]

    # Find meta positions in completion (relative to completion start)
    meta_positions = find_meta_positions_in_ids(completion_ids, tokenizer)

    if not meta_positions or num_meta == 0:
        # No meta blocks: uniform weight with penalty
        r_penalty = -0.5
        r_correct = 2.0 if is_correct else 0.0
        total = r_correct + r_penalty
        weights = torch.ones(comp_len) * max(total, 0.01)  # Avoid zero weights
        return weights, {"num_steps": 0, "total_reward": total, "has_meta": False}

    # Compute per-step rewards
    n_steps = len(meta_positions)
    step_rewards = []
    prev_conf = 0.5

    for k in range(n_steps):
        c_text = confidences[k] if k < len(confidences) else None

        # R_calib: how well does stated confidence match reality
        if c_text is not None:
            actual = 1.0 if is_correct else 0.0
            r_calib = max(0.0, 1.0 - abs(c_text - actual))
        else:
            r_calib = 0.0

        # R_progress: is confidence moving in the right direction?
        if c_text is not None:
            r_progress = (c_text - prev_conf) if is_correct else (prev_conf - c_text)
            prev_conf = c_text
        else:
            r_progress = 0.0

        # R_correct: only for last step
        r_correct = (2.0 if is_correct else 0.0) if k == n_steps - 1 else 0.0

        # R_penalty: reward for using meta blocks (positive here since we have them)
        r_meta = 0.1 if num_meta >= 2 else 0.0

        total_k = r_correct + lambda_calib * r_calib + lambda_progress * r_progress + r_meta
        step_rewards.append(total_k)

    # Assign per-token weights based on step boundaries
    weights = torch.zeros(comp_len)

    # Tokens before first meta block → step 0 reward
    # Tokens in/after meta block k → step k reward
    # Tokens after last meta block → last step reward
    current_step = 0
    for t in range(comp_len):
        # Advance step if we've passed a meta block end
        while (current_step < n_steps - 1 and
               current_step < len(meta_positions) and
               t > meta_positions[current_step][1]):
            current_step += 1

        if current_step < len(step_rewards):
            weights[t] = step_rewards[current_step]

    # Normalize weights to mean 1.0 (so total gradient magnitude is preserved)
    if weights.abs().sum() > 0:
        weights = weights / weights.abs().mean().clamp(min=1e-8)

    total_reward = sum(step_rewards) / len(step_rewards)
    return weights, {
        "num_steps": n_steps,
        "step_rewards": step_rewards,
        "total_reward": total_reward,
        "has_meta": True,
        "confidences": confidences,
    }


# ─── Reward function for TRL ───
def metacot_reward_fn(completions, ground_truth=None, **kwargs):
    """Compute sequence-level rewards (used for GRPO advantage computation)."""
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = check_correctness(text, str(gt))

        parsed = parse_meta_blocks(text)
        num_meta = parsed["num_blocks"]
        confidences = parsed["confidences"]

        r_correct = 2.0 if is_correct else 0.0
        r_penalty = 0.0 if num_meta >= 2 else (-0.3 if num_meta == 1 else -0.5)

        r_calib = 0.0
        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            actual = 1.0 if is_correct else 0.0
            r_calib = max(0.0, 1.0 - abs(avg_conf - actual))

        rewards.append(r_correct + r_calib + r_penalty)
    return rewards


# ─── Custom GRPO Trainer ───
class MetaCotGRPOTrainer(GRPOTrainer):
    """GRPO + Full Gnosis + Stepwise Credit Assignment.

    Overrides:
    1. compute_loss(): combines GRPO loss (stepwise) + Gnosis correctness loss
    2. _compute_loss(): adds per-token reward weighting from <|meta|> boundaries
    """

    def __init__(self, *args, lambda_gnosis=LAMBDA_GNOSIS, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_gnosis = lambda_gnosis
        # Store stepwise weights computed during generation scoring
        self._stepwise_cache = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("MetaCotGRPOTrainer does not support returning outputs")

        # 1. Standard GRPO loss (with stepwise per-token weighting)
        grpo_loss = self._compute_loss_stepwise(model, inputs)

        # 2. Gnosis correctness loss (if model has _should_stop)
        gnosis_loss = torch.tensor(0.0, device=grpo_loss.device)
        unwrapped = self.accelerator.unwrap_model(model)
        if hasattr(unwrapped, '_should_stop'):
            try:
                gnosis_loss = self._compute_correctness_loss(model, inputs)
            except Exception as e:
                # Gnosis head may not be initialized yet
                if self.state.global_step < 5:
                    pass  # Silently skip for first few steps
                else:
                    print(f"Gnosis loss error at step {self.state.global_step}: {e}")

        total_loss = grpo_loss + self.lambda_gnosis * gnosis_loss

        # Log
        mode = "train" if model.training else "eval"
        self._metrics[mode]["grpo_loss"].append(grpo_loss.detach().item())
        if gnosis_loss.item() > 0:
            self._metrics[mode]["gnosis_loss"].append(gnosis_loss.detach().item())

        return total_loss

    def _compute_loss_stepwise(self, model, inputs):
        """Standard GRPO loss with per-token reward weighting from <|meta|> steps."""
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, compute_entropy=True,
        )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps
        coef_1 = torch.exp(log_ratio)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # ─── Stepwise reward weighting ───
        # Apply per-token weights from <|meta|> step boundaries
        stepwise_weights = inputs.get("stepwise_weights")
        if stepwise_weights is not None:
            per_token_loss = per_token_loss * stepwise_weights

        # Standard GRPO aggregation
        loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        loss = loss / self.current_gradient_accumulation_steps

        # Metrics
        mode = "train" if self.model.training else "eval"
        completion_token_count = completion_mask.sum().clamp(min=1.0)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            mean_kl = (per_token_kl * completion_mask).sum() / completion_token_count
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        clip_ratio = (coef_1 > 1 + self.epsilon_high) | (coef_1 < 1 - self.epsilon_low)
        clip_ratio = ((clip_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather(clip_ratio).nanmean().item())

        return loss

    def _generate_and_score_completions(self, inputs):
        """Override to compute stepwise reward weights and correctness labels."""
        outputs = super()._generate_and_score_completions(inputs)

        # Add stepwise weights and correctness labels
        B = outputs["completion_ids"].shape[0]
        comp_len = outputs["completion_ids"].shape[1]
        device = outputs["completion_ids"].device

        stepwise_weights = torch.ones(B, comp_len, device=device)
        correctness_labels = torch.full((B,), -1.0, device=device)

        tokenizer = self.tokenizer

        for i in range(B):
            comp_ids = outputs["completion_ids"][i]
            comp_text = tokenizer.decode(comp_ids, skip_special_tokens=False)

            # Get ground truth from non_tensor_batch if available
            gt = ""
            if "ground_truth" in outputs.get("non_tensor_batch", {}):
                gt = outputs["non_tensor_batch"]["ground_truth"][i]

            is_correct = check_correctness(comp_text, str(gt))
            correctness_labels[i] = 1.0 if is_correct else 0.0

            # Compute stepwise weights
            weights, info = compute_stepwise_reward_weights(
                comp_ids, comp_text, is_correct, tokenizer,
            )
            # Pad/truncate to match completion length
            if len(weights) >= comp_len:
                stepwise_weights[i] = weights[:comp_len].to(device)
            else:
                stepwise_weights[i, :len(weights)] = weights.to(device)

        outputs["stepwise_weights"] = stepwise_weights
        outputs["correctness_labels"] = correctness_labels

        return outputs


# ─── Dataset preparation ───
def prepare_dataset(data_path):
    df = pd.read_parquet(data_path)
    records = []
    for _, row in df.iterrows():
        prompt = row["prompt"]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)
        gt = row.get("reward_model", {})
        if isinstance(gt, str):
            gt = json.loads(gt)
        records.append({
            "prompt": prompt,
            "ground_truth": gt.get("ground_truth", ""),
        })
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--train_data", required=True)
    parser.add_argument("--output_dir", default="checkpoints/qwen3_grpo_gnosis")
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=2048)
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lambda_gnosis", type=float, default=0.5)
    args = parser.parse_args()

    os.environ["WANDB_PROJECT"] = "metacot-math"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    meta_tokens = [META_START, META_END]
    existing = set(tokenizer.additional_special_tokens or [])
    to_add = [t for t in meta_tokens if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": list(existing) + to_add})

    # Load Qwen3 model (gnosis_repo's transformers has Gnosis integrated)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",  # Required for attention extraction (Gnosis)
        trust_remote_code=True,
        use_cache=False,
    )
    if to_add:
        model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=2048,
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        beta=0.0,
        # Gnosis — handled by our custom compute_loss, not TRL's built-in
        enable_correctness_head=False,
        logging_steps=1,
        save_steps=200,
        save_total_limit=3,
        report_to="wandb",
        run_name="qwen3-grpo-gnosis-stepwise",
        remove_unused_columns=False,
    )

    train_dataset = prepare_dataset(args.train_data)
    print(f"Training dataset: {len(train_dataset)} problems")
    print(f"Lambda Gnosis: {args.lambda_gnosis}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Stepwise credit assignment: enabled (<|meta|> boundaries)")

    trainer = MetaCotGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=metacot_reward_fn,
        tokenizer=tokenizer,
        peft_config=lora_config,
        lambda_gnosis=args.lambda_gnosis,
    )

    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"GRPO + Full Gnosis + Stepwise training complete.")
    print(f"Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
