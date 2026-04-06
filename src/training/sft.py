"""Meta-CoT SFT training using Gnosis-compatible TRL."""
import inspect
import json
import os
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


def _normalize_token_ids(encoded) -> list[int]:
    """Convert tokenizer outputs into a plain list[int] for Arrow compatibility."""
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict) and "input_ids" in encoded:
        encoded = encoded["input_ids"]

    if isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()

    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]

    return [int(x) for x in encoded]


def prepare_sft_dataset(data_path: str, tokenizer, max_length: int = 4096) -> Dataset:
    """Load and tokenize Meta-CoT SFT data."""
    df = pd.read_parquet(data_path)

    def tokenize_row(row):
        messages = json.loads(row["messages"])

        # Tokenize prompt (all messages except the last assistant) to find boundary
        # Messages can be [user, assistant] or [system, user, assistant]
        prompt_messages = messages[:-1]  # everything except assistant response
        prompt_ids = _normalize_token_ids(tokenizer.apply_chat_template(
            prompt_messages, tokenize=True, add_generation_prompt=True
        ))
        prompt_len = len(prompt_ids)

        # Tokenize full conversation (system + user + assistant)
        full_ids = _normalize_token_ids(tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        ))

        max_len = max_length
        if len(full_ids) > max_len:
            full_ids = full_ids[:max_len]

        # Mask prompt tokens with -100 so model only learns assistant output
        labels = full_ids.copy()
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        attention_mask = [1] * len(full_ids)
        return {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    ds = Dataset.from_pandas(df)
    ds = ds.map(tokenize_row, remove_columns=df.columns.tolist())
    return ds


def run_sft(config_path: str):
    """Run Meta-CoT SFT training."""
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model_name_or_path"]
    data_path = config["dataset_path"]
    output_dir = config["output_dir"]
    # Let HuggingFace Trainer handle wandb init via report_to="wandb"
    os.environ["WANDB_PROJECT"] = config.get("wandb_project", "metacot-math")
    os.environ["WANDB_NAME"] = config.get("run_name", "metacot-sft")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add <|meta|> as regular tokens (NOT special tokens) so they survive
    # skip_special_tokens=True during GRPO reward computation.
    from src.metacot.prompt import META_START, META_END
    to_add = [t for t in [META_START, META_END] if t not in tokenizer.get_vocab()]
    num_added = tokenizer.add_tokens(to_add)
    print(f"Added {num_added} tokens: {META_START}, {META_END}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))
        print(f"Resized embeddings to {len(tokenizer)}")

    full_dataset = prepare_sft_dataset(data_path, tokenizer, max_length=config.get("max_length", 4096))
    split = full_dataset.train_test_split(test_size=0.05, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]

    from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 2e-5),
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        bf16=True,
        logging_steps=10,
        save_steps=config.get("save_steps", 500),
        save_total_limit=3,
        report_to="wandb",
        eval_strategy="no",  # Disable eval to prevent OOM on long Meta-CoT chains
        deepspeed=config.get("deepspeed", None),
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, padding=True, return_tensors="pt"
    )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }
    trainer_signature = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"SFT model saved to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_sft(args.config)
