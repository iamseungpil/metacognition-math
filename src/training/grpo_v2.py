"""Meta-CoT GRPO v2: Full FT + GDPO + modular rewards.

Key design:
  - Full fine-tuning (NO LoRA)
  - GDPO monkey-patch: per-reward normalization before summing
  - 4 experiments via --mode: E1, E2, E3, E4
  - Response samples saved every 50 steps
  - Token entropy logged to wandb

Usage:
  accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_v2.py --mode E3 --max_steps 200
"""
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch

# Prevent FSDP import error
import torch.distributed.fsdp as _fsdp_mod
if not hasattr(_fsdp_mod, "FSDPModule"):
    _fsdp_mod.FSDPModule = type("FSDPModule", (), {})

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from src.training.rewards import (
    correctness_reward, format_reward, meta_quality_reward,
    calibration_reward, uncertainty_meta_reward,
    stepwise_trajectory_reward, probe_calibration_reward,
    stepwise_probe_reward,
)


# ─── GDPO Monkey-Patch ───

def _apply_gdpo_patch():
    """Patch GRPOTrainer to use GDPO advantage computation.

    GRPO: sum(rewards) → group_normalize  (collapses distinct reward combos)
    GDPO: group_normalize(each_reward) → sum → batch_normalize  (preserves signal)

    Reference: arXiv:2601.05242 (NVIDIA NVlabs)
    """
    import trl.trainer.grpo_trainer as grpo_module

    original_method = GRPOTrainer._generate_and_score_completions

    def patched_method(self, inputs):
        # Call original to get all data
        result = original_method(self, inputs)

        # Only patch if we have multiple reward functions
        if not hasattr(self, '_gdpo_enabled') or not self._gdpo_enabled:
            return result

        # Re-compute advantages with GDPO
        # Access rewards_per_func from the stored attribute
        if hasattr(self, '_last_rewards_per_func') and self._last_rewards_per_func is not None:
            rewards_per_func = self._last_rewards_per_func
            device = rewards_per_func.device
            num_gen = self.num_generations

            all_adv = []
            for i in range(rewards_per_func.shape[1]):
                r_i = torch.nan_to_num(rewards_per_func[:, i])
                mean_i = r_i.view(-1, num_gen).mean(dim=1)
                std_i = r_i.view(-1, num_gen).std(dim=1)
                mean_i = mean_i.repeat_interleave(num_gen, dim=0)
                std_i = std_i.repeat_interleave(num_gen, dim=0)
                adv_i = (r_i - mean_i) / (std_i + 1e-4)
                all_adv.append(adv_i)

            combined = torch.stack(all_adv, dim=1)
            weights = self.reward_weights.to(device).unsqueeze(0)
            pre_bn = (combined * weights).nansum(dim=1)
            advantages = (pre_bn - pre_bn.mean()) / (pre_bn.std() + 1e-4)

            # Replace advantages in result
            process_slice = slice(
                self.accelerator.process_index * (len(advantages) // self.accelerator.num_processes),
                (self.accelerator.process_index + 1) * (len(advantages) // self.accelerator.num_processes),
            )
            result["advantages"] = advantages[process_slice]

        return result

    # Also patch _calculate_rewards to store rewards_per_func
    original_calc = GRPOTrainer._calculate_rewards

    def patched_calc(self, inputs, prompts, completions, completion_ids_list):
        result = original_calc(self, inputs, prompts, completions, completion_ids_list)
        self._last_rewards_per_func = result.clone()
        return result

    GRPOTrainer._generate_and_score_completions = patched_method
    GRPOTrainer._calculate_rewards = patched_calc


# ─── Data Loading ───

def load_filtered(path):
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        prompt = json.loads(row["prompt"]) if isinstance(row["prompt"], str) else row["prompt"]
        gt = json.loads(row["reward_model"]) if isinstance(row.get("reward_model"), str) else row.get("reward_model", {})
        records.append({"prompt": prompt, "ground_truth": gt.get("ground_truth", "")})
    return Dataset.from_list(records)


def load_gsm8k(max_n=500):
    from datasets import load_dataset as hf_load
    ds = hf_load("openai/gsm8k", "main", split="train")
    records = []
    for row in ds:
        if len(records) >= max_n:
            break
        ans = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
        records.append({"prompt": [{"role": "user", "content": row["question"]}], "ground_truth": ans})
    return Dataset.from_list(records)


# ─── Sample Saving Callback ───

class SampleSaver:
    """Save completion samples every N steps for qualitative analysis."""

    def __init__(self, output_dir, every_n=50):
        self.output_dir = output_dir
        self.every_n = every_n
        self.samples = []
        os.makedirs(os.path.join(output_dir, "samples"), exist_ok=True)

    def maybe_save(self, step, completions, prompts, rewards):
        if step % self.every_n != 0 or step == 0:
            return
        samples = []
        for i in range(min(5, len(completions))):
            text = completions[i][0]["content"] if isinstance(completions[i], list) else str(completions[i])
            samples.append({
                "step": step,
                "prompt": str(prompts[i])[:200] if i < len(prompts) else "",
                "completion": text[:1000],
                "reward": float(rewards[i]) if i < len(rewards) else None,
            })
        path = os.path.join(self.output_dir, "samples", f"step_{step:04d}.json")
        with open(path, "w") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)


# ─── Main ───

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["E1", "E2", "E3", "E4", "E5", "E6", "E7"], default="E1")
    parser.add_argument("--model_path", default="checkpoints/qwen3_meta_sft")
    parser.add_argument("--data", choices=["gsm8k", "filtered"], default="filtered")
    parser.add_argument("--data_path", default="verl_train_filtered.parquet")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--num_generations", type=int, default=4)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"checkpoints/grpo_v2_{args.mode}"

    os.environ["WANDB_PROJECT"] = "metacot-math"

    # ─── Select rewards by mode ───
    reward_configs = {
        "E1": ([correctness_reward, format_reward], [1.0, 0.5]),
        "E2": ([correctness_reward, format_reward, meta_quality_reward], [1.0, 0.5, 1.0]),
        "E3": ([correctness_reward, format_reward, meta_quality_reward, calibration_reward], [1.0, 0.5, 1.0, 0.5]),
        "E4": ([correctness_reward, format_reward, meta_quality_reward, calibration_reward, uncertainty_meta_reward],
               [1.0, 0.5, 1.0, 0.5, 0.5]),
        "E5": ([correctness_reward, format_reward, meta_quality_reward, stepwise_trajectory_reward],
               [1.0, 0.5, 0.5, 1.0]),  # stepwise gets highest weight
        "E6": ([correctness_reward, format_reward, meta_quality_reward, probe_calibration_reward],
               [1.0, 0.5, 0.5, 1.5]),  # probe gets highest weight
        "E7": ([correctness_reward, format_reward, meta_quality_reward, stepwise_probe_reward],
               [1.0, 0.5, 0.5, 1.5]),  # stepwise probe gets highest weight
    }
    reward_funcs, reward_weights = reward_configs[args.mode]
    use_gdpo = args.mode in ("E3", "E4", "E5", "E6", "E7")  # GDPO when 3+ rewards

    if use_gdpo:
        _apply_gdpo_patch()
        print("GDPO patch applied (per-reward normalization)")

    # ─── Model (Full FT, NO LoRA) ───
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from src.metacot.prompt import META_START, META_END
    existing = set(tokenizer.additional_special_tokens or [])
    to_add = [t for t in [META_START, META_END] if t not in existing]
    if to_add:
        tokenizer.add_special_tokens({"additional_special_tokens": list(existing) + to_add})

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=True, use_cache=False,
    )
    if to_add:
        model.resize_token_embeddings(len(tokenizer))

    # ─── Data ───
    if args.data == "gsm8k":
        dataset = load_gsm8k()
    else:
        dataset = load_filtered(args.data_path)

    # ─── Config ───
    run_name = f"grpo-v2-{args.mode}-{args.max_steps}s"
    print(f"=== GRPO v2: {args.mode} ===")
    print(f"Rewards: {[f.__name__ for f in reward_funcs]} × {reward_weights}")
    print(f"GDPO: {use_gdpo}")
    print(f"Full FT (no LoRA)")
    print(f"Dataset: {len(dataset)} problems")

    # Config based on Open-R1 patterns, adapted for 4xA100 80GB
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=2048,
        max_prompt_length=512,
        temperature=0.9,
        # HF generate (no vLLM, no veRL — just TRL)
        use_vllm=False,
        # Batch: 4 GPU × 1 batch × 4 accum = 16, 16/4 gen = 4 unique prompts
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        # Loss: dr_grpo (Open-R1 default, no length bias)
        loss_type="dr_grpo",
        beta=0.04,
        scale_rewards=False,
        num_iterations=2,  # >1 for non-zero loss display
        # Logging
        logging_steps=1,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
        reward_weights=reward_weights,
        log_completions=True,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        processing_class=tokenizer,
        # NO peft_config → Full FT
    )

    # Enable GDPO flag
    if use_gdpo:
        trainer._gdpo_enabled = True
        trainer._last_rewards_per_func = None

    # Response logging callback
    from transformers import TrainerCallback
    class ResponseLogger(TrainerCallback):
        def __init__(self, output_dir):
            self.output_dir = output_dir
            os.makedirs(os.path.join(output_dir, "responses"), exist_ok=True)

        def on_log(self, args, state, control, logs=None, **kwargs):
            step = state.global_step
            if step % 10 == 0 and logs:
                log_path = os.path.join(self.output_dir, "responses", f"step_{step:04d}.json")
                with open(log_path, "w") as f:
                    json.dump({"step": step, "logs": {k: str(v) for k, v in logs.items()}}, f, indent=2)
                print(f"  [Step {step}] reward={logs.get('reward', '?')}, loss={logs.get('loss', '?')}")

    trainer.add_callback(ResponseLogger(args.output_dir))

    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"Done. Saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
