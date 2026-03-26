"""Clean GRPO for Meta-CoT — no trainer override, just reward_funcs.

Proven: vanilla GRPOTrainer produces loss ≠ 0.
Our MetaCotGRPOTrainer override broke loss computation.
Fix: use vanilla GRPOTrainer + custom reward functions only.

Modes:
  --mode baseline: binary correctness only (+1/-1)
  --mode meta:     correctness + calibration from meta blocks
  --mode gsm8k:    GSM8K data (easy, for validation)
  --mode filtered: pass-rate filtered data (harder)

Usage:
  accelerate launch --num_processes 4 --multi_gpu \
    src/training/grpo_clean.py --mode meta --data filtered
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


# ─── Answer extraction ───
def extract_answer(text):
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    m = re.search(r'####\s*(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()
    return ""


def check_correctness(pred, gold):
    p = extract_answer(pred)
    g = extract_answer(str(gold)) or str(gold).strip()
    if not p:
        return False
    if p == g:
        return True
    try:
        if abs(float(p) - float(g)) < 1e-6:
            return True
    except (ValueError, TypeError):
        pass
    return p.lower().strip() == g.lower().strip()


# ─── Reward functions (plain callables, no trainer override) ───

def correctness_reward(completions, ground_truth=None, **kwargs):
    """Binary: +1 correct, -1 incorrect."""
    rewards = []
    for i, c in enumerate(completions):
        text = c[0]["content"] if isinstance(c, list) else str(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        rewards.append(1.0 if check_correctness(text, gt) else -1.0)
    return rewards


def meta_calibration_reward(completions, ground_truth=None, completion_ids=None, **kwargs):
    """Calibration reward based on <|meta|> confidence.

    TRL strips special tokens, but we can detect confidence patterns
    even in stripped text (e.g., "probability of solving it correctly is about 0.85").
    """
    rewards = []
    for i, c in enumerate(completions):
        text = c[0]["content"] if isinstance(c, list) else str(c)
        gt = ground_truth[i] if ground_truth is not None else ""
        is_correct = check_correctness(text, gt)

        # Parse confidence from text (works even without <|meta|> tokens)
        # The model outputs "probability ... is about X.XX" or "confidence: X.XX"
        confs = re.findall(
            r'(?:probability|confidence)[:\s]*(?:about\s+)?(\d+(?:\.\d+)?)',
            text, re.IGNORECASE
        )
        confidences = []
        for m in confs:
            try:
                v = float(m)
                if v > 1:
                    v /= 100
                confidences.append(max(0.0, min(1.0, v)))
            except ValueError:
                pass

        if confidences:
            avg_conf = sum(confidences) / len(confidences)
            # Logarithmic scoring rule (Rewarding Doubt style):
            # correct + high confidence = good
            # incorrect + low confidence = good (knows it doesn't know)
            if is_correct:
                r_calib = max(-2.0, 0.5 * (avg_conf - 0.5))  # reward high confidence
            else:
                r_calib = max(-2.0, 0.5 * (0.5 - avg_conf))  # reward low confidence
        else:
            r_calib = 0.0

        rewards.append(r_calib)
    return rewards


def meta_format_reward(completions, **kwargs):
    """Bonus for having metacognitive structure in output.

    Detects self-assessment patterns even when <|meta|> is stripped.
    """
    rewards = []
    for c in completions:
        text = c[0]["content"] if isinstance(c, list) else str(c)
        # Check for self-assessment patterns
        has_assessment = bool(re.search(
            r'(?:can I solve|probability|confidence|watch out|verify|check)',
            text, re.IGNORECASE
        ))
        rewards.append(0.2 if has_assessment else 0.0)
    return rewards


# ─── Dataset loading ───

def load_gsm8k(max_n=500):
    ds = load_dataset("openai/gsm8k", "main", split="train")
    records = []
    for row in ds:
        if len(records) >= max_n:
            break
        ans = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]
        records.append({"prompt": [{"role": "user", "content": row["question"]}], "ground_truth": ans})
    return Dataset.from_list(records)


def load_filtered(path):
    df = pd.read_parquet(path)
    records = []
    for _, row in df.iterrows():
        prompt = json.loads(row["prompt"]) if isinstance(row["prompt"], str) else row["prompt"]
        gt = json.loads(row["reward_model"]) if isinstance(row.get("reward_model"), str) else row.get("reward_model", {})
        records.append({"prompt": prompt, "ground_truth": gt.get("ground_truth", "")})
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "meta"], default="meta")
    parser.add_argument("--data", choices=["gsm8k", "filtered"], default="filtered")
    parser.add_argument("--model_path", default="checkpoints/qwen3_meta_sft")
    parser.add_argument("--data_path", default="verl_train_filtered.parquet")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--max_steps", type=int, default=200)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"checkpoints/grpo_clean_{args.mode}_{args.data}"

    os.environ["WANDB_PROJECT"] = "metacot-math"

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add meta tokens
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

    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, task_type="CAUSAL_LM",
    )

    # Select reward functions
    if args.mode == "baseline":
        reward_funcs = [correctness_reward]
        reward_weights = [1.0]
    else:  # meta
        reward_funcs = [correctness_reward, meta_calibration_reward, meta_format_reward]
        reward_weights = [1.0, 0.5, 0.2]

    # Select data
    if args.data == "gsm8k":
        dataset = load_gsm8k()
    else:
        dataset = load_filtered(args.data_path)

    run_name = f"grpo-clean-{args.mode}-{args.data}-{args.max_steps}s"

    print(f"=== GRPO Clean: {args.mode} mode, {args.data} data ===")
    print(f"Dataset: {len(dataset)} problems")
    print(f"Rewards: {[f.__name__ for f in reward_funcs]} × {reward_weights}")
    print(f"Run: {run_name}")

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_generations=8,
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
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        reward_funcs=reward_funcs,
        reward_weights=reward_weights,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir + "/final")
    tokenizer.save_pretrained(args.output_dir + "/final")
    print(f"Done. Saved to {args.output_dir}/final")


if __name__ == "__main__":
    main()
