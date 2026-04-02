"""Single-example adaptation helpers for curriculum experiments."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from datasets import Dataset
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from src.curriculum.control_rag import build_model_inputs, generate_from_messages


def _tokenize_example(messages: list[dict[str, str]], tokenizer, max_length: int) -> dict[str, Any]:
    prompt_messages = messages[:-1]
    prompt_text, prompt_inputs = build_model_inputs(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
        max_prompt_tokens=max_length,
    )
    full_text, full_inputs = build_model_inputs(
        tokenizer,
        messages,
        add_generation_prompt=False,
        max_prompt_tokens=max_length,
    )
    prompt_ids = prompt_inputs["input_ids"][0].tolist()
    full_ids = full_inputs["input_ids"][0].tolist()
    labels = full_ids.copy()
    for i in range(min(len(prompt_ids), len(labels))):
        labels[i] = -100
    return {
        "prompt_text": prompt_text,
        "full_text": full_text,
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }


def run_one_example_adaptation(
    *,
    model_name_or_path: str,
    example_messages: list[dict[str, str]],
    target_question: str,
    output_dir: str,
    max_steps: int = 1,
    learning_rate: float = 5e-5,
    max_length: int = 512,
    max_new_tokens: int = 96,
    device: str = "cpu",
) -> dict[str, Any]:
    """Train on one retrieved example, then generate on a target question."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch_dtype)
    if device != "cpu":
        model = model.to(device)

    tokenized = _tokenize_example(example_messages, tokenizer, max_length=max_length)
    train_ds = Dataset.from_list([
        {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["labels"],
        }
        for _ in range(max(1, max_steps))
    ])

    args = TrainingArguments(
        output_dir=str(output_path),
        max_steps=max_steps,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        logging_steps=1,
        save_steps=max_steps,
        save_total_limit=1,
        report_to="none",
        remove_unused_columns=False,
        use_cpu=(device == "cpu"),
        bf16=(device != "cpu"),
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt"),
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    prompt_messages = [{"role": "user", "content": target_question}]
    completion, prompt_text, _, _ = generate_from_messages(
        model,
        tokenizer,
        prompt_messages,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        max_prompt_tokens=max_length,
    )
    summary = {
        "model_name_or_path": model_name_or_path,
        "output_dir": str(output_path),
        "max_steps": max_steps,
        "target_question": target_question,
        "example_messages": example_messages,
        "prompt_text": prompt_text,
        "completion": completion,
    }
    with open(output_path / "one_example_summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
