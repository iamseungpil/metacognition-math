"""GRPO + Stepwise Credit Assignment (Agent Lightning Transition Mode).

Key design:
- Generate full completion, split by <|meta|> boundaries into steps
- Each step becomes a SEPARATE training example
- R_correct: SAME for all steps (final correctness)
- R_calib: PER-STEP (each step's confidence vs actual outcome)
- R_penalty: SAME for all steps
- GRPO normalizes across all steps from all rollouts of same prompt
  → each step competes with the same step from other rollouts

Critic-reviewed and fixed (v4):
- FIX #1: ground_truth cached before super()
- FIX #2: skip_special_tokens=False for <|meta|>
- FIX #3: stepwise rewards non-negative
- FIX #7: Agent Lightning transition mode (step splitting)
"""
import argparse
import json
import math
import os
import re

import pandas as pd
import torch
import torch.nn.functional as F

# Monkey-patch FSDPModule for PyTorch 2.5 compatibility with TRL 0.19+
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


def compute_step_reward(is_correct, confidence, num_meta_blocks):
    """Compute reward for a single step.

    R_correct: same for ALL steps (2.0 if correct, 0.0 if not)
    R_calib: per-step (1 - |confidence - actual|)
    R_penalty: same for ALL steps (based on total meta block count)
    """
    r_correct = 2.0 if is_correct else 0.0

    if num_meta_blocks >= 2:
        r_penalty = 0.0
    elif num_meta_blocks == 1:
        r_penalty = -0.3
    else:
        r_penalty = -0.5

    r_calib = 0.0
    if confidence is not None:
        actual = 1.0 if is_correct else 0.0
        r_calib = max(0.0, 1.0 - abs(confidence - actual))

    return r_correct + r_calib + r_penalty


# ─── Reward function for TRL (sequence-level, called by parent) ───
def metacot_reward_fn(completions, ground_truth=None, **kwargs):
    """Sequence-level reward for initial GRPO advantage computation.
    These advantages are REPLACED by step-level advantages in our override.
    """
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


def _compute_grpo_advantages(rewards):
    """GRPO group normalization: z-score within group."""
    if len(rewards) < 2:
        return [0.0] * len(rewards)
    mean_r = sum(rewards) / len(rewards)
    var_r = sum((r - mean_r) ** 2 for r in rewards) / len(rewards)
    std_r = max(var_r ** 0.5, 1e-8)
    return [(r - mean_r) / std_r for r in rewards]


# ─── Custom GRPO Trainer: Agent Lightning Transition Mode ───
class MetaCotGRPOTrainer(GRPOTrainer):
    """GRPO with step-level splitting (Agent Lightning transition mode).

    Each <|meta|> step becomes a separate training example:
    - Step k prompt = original_prompt + all tokens before step k
    - Step k completion = tokens of step k (until next <|meta|> or end)
    - Step k reward = R_correct(same) + R_calib_k(per-step) + R_penalty(same)
    - GRPO advantages computed across all steps from all rollouts
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_ground_truths = []

    def _generate_and_score_completions(self, inputs):
        """Generate full completions, then split into per-step training examples."""
        # Cache ground truths before super() consumes inputs
        if isinstance(inputs, list):
            self._cached_ground_truths = [
                x.get("ground_truth", "") for x in inputs
            ]
        elif isinstance(inputs, dict):
            self._cached_ground_truths = inputs.get("ground_truth", [])

        # Get full completions from parent
        outputs = super()._generate_and_score_completions(inputs)

        B = outputs["completion_ids"].shape[0]
        device = outputs["completion_ids"].device
        tokenizer = self.processing_class
        num_gens = getattr(self.args, 'num_generations', 1)
        n_prompts = len(self._cached_ground_truths)

        # Collect per-step data
        all_step_prompt_ids = []
        all_step_completion_ids = []
        all_step_rewards = []
        all_step_old_logps = []  # will be None, recomputed by parent

        for i in range(B):
            prompt_ids_i = outputs["prompt_ids"][i]
            comp_ids_i = outputs["completion_ids"][i]
            full_ids_i = torch.cat([prompt_ids_i, comp_ids_i])

            # Decode with special tokens preserved
            comp_text = tokenizer.decode(comp_ids_i, skip_special_tokens=False)
            full_text = tokenizer.decode(full_ids_i, skip_special_tokens=False)

            # Get ground truth
            prompt_idx = i // num_gens if num_gens > 0 else i
            gt = self._cached_ground_truths[prompt_idx] if prompt_idx < n_prompts else ""
            is_correct = check_correctness(full_text, str(gt))

            # Find <|meta|> boundaries in completion
            meta_positions = find_meta_positions_in_ids(comp_ids_i, tokenizer)
            parsed = parse_meta_blocks(comp_text)
            num_meta = parsed["num_blocks"]
            confidences = parsed["confidences"]

            if not meta_positions:
                # No meta blocks: keep as single example
                reward = compute_step_reward(is_correct, None, 0)
                all_step_prompt_ids.append(prompt_ids_i)
                all_step_completion_ids.append(comp_ids_i)
                all_step_rewards.append(reward)
            else:
                # Split by <|meta|> boundaries
                prompt_len = prompt_ids_i.shape[0]
                step_boundaries = []

                for k, (start, end) in enumerate(meta_positions):
                    step_start = 0 if k == 0 else meta_positions[k - 1][1] + 1
                    step_end = end + 1  # include the </meta> token
                    step_boundaries.append((step_start, step_end))

                # Last segment: from last meta end to end of completion
                if meta_positions:
                    last_end = meta_positions[-1][1] + 1
                    if last_end < comp_ids_i.shape[0]:
                        step_boundaries.append((last_end, comp_ids_i.shape[0]))

                for k, (s_start, s_end) in enumerate(step_boundaries):
                    # Step k prompt = original prompt + all tokens before this step
                    step_prompt = torch.cat([prompt_ids_i, comp_ids_i[:s_start]])
                    # Step k completion = this step's tokens
                    step_completion = comp_ids_i[s_start:s_end]

                    if step_completion.shape[0] == 0:
                        continue

                    # Per-step reward: R_correct(same) + R_calib_k + R_penalty
                    conf_k = confidences[k] if k < len(confidences) else None
                    reward_k = compute_step_reward(is_correct, conf_k, num_meta)

                    all_step_prompt_ids.append(step_prompt)
                    all_step_completion_ids.append(step_completion)
                    all_step_rewards.append(reward_k)

        if not all_step_prompt_ids:
            # Fallback: return original outputs unchanged
            return outputs

        # Pad to uniform lengths
        max_prompt_len = max(p.shape[0] for p in all_step_prompt_ids)
        max_comp_len = max(c.shape[0] for c in all_step_completion_ids)
        B_new = len(all_step_prompt_ids)

        pad_id = tokenizer.pad_token_id or 0

        padded_prompt_ids = torch.full((B_new, max_prompt_len), pad_id, device=device, dtype=torch.long)
        padded_prompt_mask = torch.zeros(B_new, max_prompt_len, device=device, dtype=torch.long)
        padded_comp_ids = torch.full((B_new, max_comp_len), pad_id, device=device, dtype=torch.long)
        padded_comp_mask = torch.zeros(B_new, max_comp_len, device=device, dtype=torch.long)

        for j in range(B_new):
            p_len = all_step_prompt_ids[j].shape[0]
            c_len = all_step_completion_ids[j].shape[0]
            # Right-align prompt (pad left)
            padded_prompt_ids[j, max_prompt_len - p_len:] = all_step_prompt_ids[j]
            padded_prompt_mask[j, max_prompt_len - p_len:] = 1
            # Left-align completion (pad right)
            padded_comp_ids[j, :c_len] = all_step_completion_ids[j]
            padded_comp_mask[j, :c_len] = 1

        # Compute GRPO advantages from step-level rewards
        advantages = torch.tensor(
            _compute_grpo_advantages(all_step_rewards),
            device=device, dtype=torch.float32
        )

        # Build new outputs
        new_outputs = {
            "prompt_ids": padded_prompt_ids,
            "prompt_mask": padded_prompt_mask,
            "completion_ids": padded_comp_ids,
            "completion_mask": padded_comp_mask,
            "advantages": advantages,
            "old_per_token_logps": None,  # will be recomputed
        }

        # Copy any other keys from original outputs
        for key in outputs:
            if key not in new_outputs:
                new_outputs[key] = outputs[key]

        # Metrics
        mode = "train" if self.model.training else "eval"
        n_steps_total = B_new
        n_with_meta = sum(1 for r in all_step_rewards if r > 0)
        avg_reward = sum(all_step_rewards) / max(len(all_step_rewards), 1)
        self._metrics[mode]["step_count"].append(float(n_steps_total))
        self._metrics[mode]["avg_step_reward"].append(avg_reward)
        self._metrics[mode]["reward"].append(avg_reward)

        print(f"[Step split] {B} completions → {B_new} step examples, "
              f"avg_reward={avg_reward:.3f}, avg_steps={B_new/max(B,1):.1f}")

        return new_outputs


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
        # Gnosis heads trained with full params (not LoRA)
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
        run_name="qwen3-grpo-stepwise-transition",
        remove_unused_columns=False,
    )

    train_dataset = prepare_dataset(args.train_data)
    print(f"=== MetaCot GRPO: Agent Lightning Transition Mode ===")
    print(f"Dataset: {len(train_dataset)} problems")
    print(f"Step splitting: each <|meta|> step = separate training example")
    print(f"R_correct: same for ALL steps | R_calib: per-step | R_penalty: same")
    print(f"GRPO: steps compete across rollouts")

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
