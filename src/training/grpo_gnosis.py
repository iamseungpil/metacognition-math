"""GRPO + Full Gnosis training for Qwen3-8B Meta-CoT.

Uses gnosis_repo's TRL GRPOTrainer with:
- vLLM colocate mode for fast generation
- Full Gnosis correctness head (attention + hidden + confidence extractors)
- 3 rewards: R_correct, R_calib (probe-based), R_penalty (meta block usage)
- LoRA for memory efficiency on 4x A100 80GB
"""
import argparse
import json
import math
import os
import re
import sys

import pandas as pd
import torch
from datasets import Dataset

# gnosis_repo's transformers and TRL must be on PYTHONPATH
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model


# ─── Meta-CoT parsing (inline for self-containment) ───
META_START = "<|meta|>"
META_END = "<|/meta|>"


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
    return {"num_blocks": len(blocks), "confidences": confidences}


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


# ─── Reward function ───
def metacot_reward_fn(completions, prompts=None, ground_truth=None, **kwargs):
    """Compute 3-component reward for each completion.

    Returns list of floats (one per completion).
    """
    rewards = []
    for i, completion in enumerate(completions):
        text = completion[0]["content"] if isinstance(completion, list) else completion

        # R_correct
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = check_correctness(text, str(gt))
        r_correct = 2.0 if is_correct else 0.0

        # R_penalty (meta block usage)
        parsed = parse_meta_blocks(text)
        num_meta = parsed["num_blocks"]
        if num_meta >= 2:
            r_penalty = 0.0
        elif num_meta == 1:
            r_penalty = -0.3
        else:
            r_penalty = -0.5

        # R_calib (text-based for now; Gnosis probe supplements via correctness_loss)
        r_calib = 0.0
        if parsed["confidences"]:
            avg_conf = sum(parsed["confidences"]) / len(parsed["confidences"])
            actual = 1.0 if is_correct else 0.0
            r_calib = max(0.0, 1.0 - abs(avg_conf - actual))

        total = r_correct + r_calib + r_penalty
        rewards.append(total)

    return rewards


# ─── Dataset preparation ───
def prepare_dataset(data_path):
    """Load veRL-format parquet and convert to TRL GRPO format."""
    df = pd.read_parquet(data_path)

    records = []
    for _, row in df.iterrows():
        prompt = row["prompt"]  # list of dicts [{"role": "user", "content": ...}]
        if isinstance(prompt, str):
            prompt = json.loads(prompt)

        gt = row.get("reward_model", {})
        if isinstance(gt, str):
            gt = json.loads(gt)
        ground_truth = gt.get("ground_truth", "")

        records.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
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

    # ─── Load model with Gnosis-modified transformers ───
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add meta tokens if not present
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

    # ─── LoRA for memory efficiency ───
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    # ─── GRPO Config ───
    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=2048,

        # vLLM for fast generation
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=0.4,

        # Training
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        gradient_checkpointing=True,

        # GRPO
        beta=0.0,  # No KL penalty (GRPO uses group advantages)

        # Gnosis correctness head
        enable_correctness_head=True,
        freeze_except_stop_head=False,  # Train both base model + Gnosis head
        correctness_last_k=10,

        # Logging
        logging_steps=10,
        save_steps=200,
        save_total_limit=3,
        report_to="wandb",
        run_name="qwen3-grpo-gnosis",

        # Misc
        remove_unused_columns=False,
    )

    # ─── Dataset ───
    train_dataset = prepare_dataset(args.train_data)
    print(f"Training dataset: {len(train_dataset)} problems")

    # ─── Trainer ───
    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=metacot_reward_fn,
        tokenizer=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()

    # Save final model
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"GRPO + Gnosis training done. Model saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
