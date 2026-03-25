"""GRPO + Stepwise Credit Assignment (Per-Token Step Rewards).

Design:
- Generate full completion, find <|meta|> step boundaries
- Each step gets its own reward: R_correct(same) + R_calib_k(per-step) + R_penalty(same)
- Per-token advantages: expand sequence-level advantage by step-level reward ratios
- Keeps batch size constant (DDP-compatible, no step splitting)
- Full Gnosis: patched Qwen3 model with attention+hidden+confidence extractors

Reward structure:
  Step k advantage = sequence_advantage * (step_k_reward / mean_step_reward)
  → Steps with better calibration get amplified gradient signal
  → Steps with poor calibration get reduced gradient signal
  → R_correct is the SAME for all steps (final correctness)
"""
import argparse
import json
import os
import re

import pandas as pd
import torch
import torch.nn.functional as F

# Monkey-patch FSDPModule for PyTorch 2.5 compat with TRL 0.19
import torch.distributed.fsdp as _fsdp_mod
if not hasattr(_fsdp_mod, "FSDPModule"):
    _fsdp_mod.FSDPModule = type("FSDPModule", (), {})

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig

from src.metacot.prompt import META_START, META_END, parse_meta_blocks
from src.training.stepwise import find_meta_token_positions as find_meta_positions_in_ids


def _extract_answer(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
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


def compute_per_token_step_advantages(
    completion_ids, completion_text, is_correct, sequence_advantage, tokenizer
):
    """Compute per-token advantages from step-level rewards.

    Each <|meta|> step gets: R_correct(same) + R_calib_k(per-step) + R_penalty(same)
    The per-token advantage = sequence_advantage * (step_reward / mean_step_reward)

    This amplifies gradient for well-calibrated steps and reduces it for poorly calibrated ones.
    """
    parsed = parse_meta_blocks(completion_text)
    num_meta = parsed["num_blocks"]
    confidences = parsed["confidences"]
    comp_len = completion_ids.shape[-1] if isinstance(completion_ids, torch.Tensor) else len(completion_ids)

    # Base rewards (same for all steps)
    r_correct = 2.0 if is_correct else 0.0
    r_penalty = 0.0 if num_meta >= 2 else (-0.3 if num_meta == 1 else -0.5)

    meta_positions = find_meta_positions_in_ids(completion_ids, tokenizer)

    if not meta_positions or num_meta == 0:
        # No meta blocks: uniform advantage
        return torch.ones(comp_len)

    # Compute per-step rewards
    n_steps = max(len(meta_positions), 1)
    step_rewards = []
    for k in range(n_steps):
        conf_k = confidences[k] if k < len(confidences) else None
        r_calib_k = 0.0
        if conf_k is not None:
            actual = 1.0 if is_correct else 0.0
            r_calib_k = max(0.0, 1.0 - abs(conf_k - actual))
        step_reward = r_correct + r_calib_k + r_penalty
        step_rewards.append(max(step_reward, 0.01))  # clamp positive

    mean_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 1.0
    mean_reward = max(mean_reward, 0.01)

    # Build per-token multiplier
    multiplier = torch.ones(comp_len)
    current_step = 0
    for t in range(comp_len):
        while (current_step < n_steps - 1 and
               current_step < len(meta_positions) and
               t > meta_positions[current_step][1]):
            current_step += 1
        if current_step < len(step_rewards):
            multiplier[t] = step_rewards[current_step] / mean_reward

    return multiplier


def metacot_reward_fn(completions, ground_truth=None, **kwargs):
    """Sequence-level reward for GRPO advantage computation."""
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


class MetaCotGRPOTrainer(GRPOTrainer):
    """GRPO with per-token step-level advantages (DDP-compatible).

    Instead of splitting steps into separate examples (breaks DDP),
    we expand the sequence-level advantage into per-token advantages
    using step-level reward ratios from <|meta|> boundaries.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_ground_truths = []
        self._step_multipliers = {}  # cache per-token multipliers

    def _generate_and_score_completions(self, inputs):
        """Cache ground truths and compute per-token step multipliers."""
        if isinstance(inputs, list):
            self._cached_ground_truths = [x.get("ground_truth", "") for x in inputs]
        elif isinstance(inputs, dict):
            self._cached_ground_truths = inputs.get("ground_truth", [])

        outputs = super()._generate_and_score_completions(inputs)

        B = outputs["completion_ids"].shape[0]
        comp_len = outputs["completion_ids"].shape[1]
        device = outputs["completion_ids"].device
        tokenizer = self.processing_class
        num_gens = getattr(self.args, 'num_generations', 1)
        n_prompts = len(self._cached_ground_truths)

        step_multipliers = torch.ones(B, comp_len, device=device)
        n_steps_total = 0
        n_with_meta = 0

        for i in range(B):
            comp_ids = outputs["completion_ids"][i]
            comp_text = tokenizer.decode(comp_ids, skip_special_tokens=False)

            prompt_idx = i // num_gens if num_gens > 0 else i
            gt = self._cached_ground_truths[prompt_idx] if prompt_idx < n_prompts else ""
            is_correct = check_correctness(comp_text, str(gt))

            seq_adv = outputs["advantages"][i].item()
            multiplier = compute_per_token_step_advantages(
                comp_ids, comp_text, is_correct, seq_adv, tokenizer,
            )

            if len(multiplier) >= comp_len:
                step_multipliers[i] = multiplier[:comp_len].to(device)
            else:
                step_multipliers[i, :len(multiplier)] = multiplier.to(device)

            meta_pos = find_meta_positions_in_ids(comp_ids, tokenizer)
            n_steps_total += max(len(meta_pos), 1)
            if meta_pos:
                n_with_meta += 1

        outputs["step_multipliers"] = step_multipliers

        mode = "train" if self.model.training else "eval"
        self._metrics[mode]["meta_block_ratio"].append(n_with_meta / max(B, 1))
        self._metrics[mode]["avg_steps_per_completion"].append(n_steps_total / max(B, 1))

        return outputs

    def _compute_loss(self, model, inputs):
        """GRPO loss with per-token step advantages.

        per_token_advantage = sequence_advantage * step_multiplier
        step_multiplier = step_reward / mean_step_reward (always > 0)

        This is DDP-safe because batch size doesn't change.
        """
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps) - 1
            )

        advantages = inputs["advantages"]
        old_per_token_logps = (
            per_token_logps.detach() if inputs["old_per_token_logps"] is None
            else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        # Per-token step advantages: expand sequence advantage by step multiplier
        step_multipliers = inputs.get("step_multipliers")
        if step_multipliers is not None:
            per_token_advantages = advantages.unsqueeze(1) * step_multipliers
        else:
            per_token_advantages = advantages.unsqueeze(1)

        per_token_loss1 = coef_1 * per_token_advantages
        per_token_loss2 = coef_2 * per_token_advantages
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Metrics
        mode = "train" if self.model.training else "eval"
        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather(mean_kl).nanmean().item())

        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (per_token_advantages < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (per_token_advantages > 0)
        low_clip = ((is_low_clipped * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        high_clip = ((is_high_clipped * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
        self._metrics[mode]["clip_ratio/low_mean"].append(self.accelerator.gather(low_clip).nanmean().item())
        self._metrics[mode]["clip_ratio/high_mean"].append(self.accelerator.gather(high_clip).nanmean().item())

        return loss


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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        use_cache=False,
    )
    if to_add:
        model.resize_token_embeddings(len(tokenizer))

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        modules_to_save=["stop_head", "attn_extractor", "hid_extractor", "conf_extractor"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=2048,
        use_vllm=False,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=0.0,
        logging_steps=1,
        save_steps=200,
        save_total_limit=3,
        report_to="wandb",
        run_name="qwen3-grpo-stepwise-v2",
        remove_unused_columns=False,
    )

    train_dataset = prepare_dataset(args.train_data)
    print(f"=== MetaCot GRPO: Per-Token Step Advantages ===")
    print(f"Dataset: {len(train_dataset)} problems")
    print(f"Step rewards: R_correct(same) + R_calib_k(per-step) + R_penalty(same)")
    print(f"Per-token advantage = seq_advantage * (step_reward / mean_step_reward)")
    print(f"DDP-compatible: batch size constant, no step splitting")

    trainer = MetaCotGRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=metacot_reward_fn,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"Training complete. Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
