"""Baseline GRPO — no probe, no meta, just correctness reward.

Purpose: verify that standard TRL GRPO works on our setup.
If this shows a reward curve but meta GRPO doesn't,
the problem is in our additions (probe, meta parsing, etc.)

Usage:
  # Baseline (GSM8K, binary reward only)
  accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_baseline.py --mode baseline

  # Meta (our full pipeline)
  accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_baseline.py --mode meta
"""
import argparse
import json
import os
import re

import pandas as pd
import torch

import torch.distributed.fsdp as _fsdp_mod
if not hasattr(_fsdp_mod, "FSDPModule"):
    _fsdp_mod.FSDPModule = type("FSDPModule", (), {})

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig


# ─── Reward functions ───

def extract_answer(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    m = re.search(r'####\s*(.+?)(?:\n|$)', text)
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


def baseline_reward_fn(completions, ground_truth=None, **kwargs):
    """Simplest possible: +1 correct, -1 incorrect. No meta, no probe."""
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = check_correctness(text, str(gt))
        rewards.append(1.0 if is_correct else -1.0)
    return rewards


def meta_reward_fn(completions, ground_truth=None, **kwargs):
    """Our reward: R_correct + R_penalty. No probe (text-based only)."""
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else str(completion)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = check_correctness(text, str(gt))
        r_correct = 1.0 if is_correct else -1.0
        # Note: TRL strips <|meta|> via skip_special_tokens=True
        # So we can't parse meta blocks here. Just use correctness.
        rewards.append(r_correct)
    return rewards


# ─── Dataset ───

def load_gsm8k_grpo(max_problems=500):
    """Load GSM8K test as GRPO training data."""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    records = []
    for row in ds:
        if len(records) >= max_problems:
            break
        answer = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
        records.append({
            "prompt": [{"role": "user", "content": row["question"]}],
            "ground_truth": answer,
        })
    return Dataset.from_list(records)


def load_filtered_data(data_path):
    """Load our filtered parquet data."""
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
    parser.add_argument("--mode", choices=["baseline", "meta"], required=True)
    parser.add_argument("--model_path", default="checkpoints/qwen3_meta_sft")
    parser.add_argument("--data_path", default="verl_train_filtered.parquet")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--num_generations", type=int, default=32)
    parser.add_argument("--max_problems", type=int, default=500)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"checkpoints/grpo_{args.mode}"

    os.environ["WANDB_PROJECT"] = "metacot-math"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        use_cache=False,
    )

    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # Choose reward function and data based on mode
    if args.mode == "baseline":
        reward_fn = baseline_reward_fn
        dataset = load_gsm8k_grpo(args.max_problems)
        run_name = f"grpo-baseline-gsm8k-{args.max_steps}steps"
        print(f"=== BASELINE GRPO (GSM8K, binary reward) ===")
    else:
        reward_fn = meta_reward_fn
        dataset = load_filtered_data(args.data_path)
        run_name = f"grpo-meta-filtered-{args.max_steps}steps"
        print(f"=== META GRPO (filtered data, meta reward) ===")

    print(f"Dataset: {len(dataset)} problems")
    print(f"Reward: {reward_fn.__name__}")
    print(f"num_generations: {args.num_generations}")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=1024,
        max_prompt_length=512,
        temperature=1.0,
        use_vllm=False,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        beta=0.001,
        num_iterations=1,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    print(f"Done. Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()
