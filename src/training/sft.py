"""Meta-CoT SFT training using Gnosis-compatible TRL."""
import inspect
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from datasets import Dataset

from src.training.segment_loss_mask import build_segment_loss_mask, redirect_train_spans

try:
    import torch
except ImportError:  # pragma: no cover - dataset prep should still be testable without torch
    torch = None


def _normalize_token_ids(encoded) -> list[int]:
    """Convert tokenizer outputs into a plain list[int] for Arrow compatibility."""
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict) and "input_ids" in encoded:
        encoded = encoded["input_ids"]

    if torch is not None and isinstance(encoded, torch.Tensor):
        encoded = encoded.tolist()

    if encoded and isinstance(encoded[0], (list, tuple)):
        encoded = encoded[0]

    return [int(x) for x in encoded]


def _load_messages(raw_messages):
    if isinstance(raw_messages, list):
        return raw_messages
    if isinstance(raw_messages, str):
        return json.loads(raw_messages)
    raise TypeError(f"Unsupported messages payload type: {type(raw_messages)!r}")


def _load_teacher_kl_config(config: dict[str, Any]) -> dict[str, Any]:
    teacher = dict(config.get("teacher_kl", {}) or {})
    teacher.setdefault("enabled", False)
    teacher.setdefault("coef", 0.0)
    teacher.setdefault("require_targets", True)
    teacher.setdefault("mask_mode", "control_spans")
    teacher.setdefault("meta_weight", 1.0)
    teacher.setdefault("diagnosis_weight", 1.25)
    teacher.setdefault("study_need_weight", 1.4)
    teacher.setdefault("recovery_weight", 0.7)
    teacher.setdefault("verify_weight", 1.1)
    teacher.setdefault("row_quality_scaling", True)
    teacher.setdefault("entropy_beta", 0.0)
    teacher.setdefault("entropy_weight_floor", 0.65)
    teacher.setdefault("entropy_weight_ceil", 1.10)
    teacher.setdefault("entropy_penalty_strength", 0.35)
    return teacher


def prepare_sft_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 4096,
    *,
    teacher_kl: dict[str, Any] | None = None,
) -> Dataset:
    """Load and tokenize Meta-CoT SFT data."""
    df = pd.read_parquet(data_path)
    teacher_kl = dict(teacher_kl or {})
    teacher_kl_enabled = bool(teacher_kl.get("enabled", False))

    from src.training.self_distill.kl import (
        build_control_span_weights,
        load_teacher_topk_payload,
        trim_teacher_payload,
    )
    from src.training.meta_quality import (
        apply_entropy_weighting,
        compute_teacher_kl_row_scale,
    )

    def tokenize_row(row):
        messages = _load_messages(row["messages"])

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

        # Mask prompt tokens with -100 so the model only learns the assistant output.
        # REDIRECT rows additionally carry the student's wrong_prefix at the HEAD of
        # the assistant target — mask it too (train ONLY the meta block + recovery,
        # never teach the model to PRODUCE the flawed prefix). VERIFY / plain rows
        # have an empty wrong_prefix and keep the prompt-only boundary mask.
        labels = full_ids.copy()
        wrong_prefix = str(row.get("wrong_prefix", "") or "")
        if wrong_prefix:
            prefix_len = len(tokenizer.encode(wrong_prefix, add_special_tokens=False))
            spans = redirect_train_spans(prompt_len, prefix_len, len(full_ids))
            keep = build_segment_loss_mask(len(full_ids), spans)
            labels = [tok if k == 1 else -100 for tok, k in zip(full_ids, keep)]
        else:
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100

        attention_mask = [1] * len(full_ids)
        num_target_tokens = sum(1 for token in labels if token != -100)
        built = {
            "input_ids": full_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "num_target_tokens": num_target_tokens,
        }
        if teacher_kl_enabled:
            teacher_payload = load_teacher_topk_payload(row)
            if teacher_payload is not None and num_target_tokens > 0:
                trimmed = trim_teacher_payload(teacher_payload, target_length=num_target_tokens)
                if trimmed.assistant_token_ids:
                    assistant_text = str(messages[-1]["content"])
                    weights = build_control_span_weights(
                        tokenizer=tokenizer,
                        assistant_text=assistant_text,
                        expected_length=len(trimmed.assistant_token_ids),
                        mask_mode=str(teacher_kl.get("mask_mode", "control_spans")),
                        diagnosis_text=str(row.get("diagnosis_text", "") or ""),
                        study_need=str(row.get("study_need", "") or ""),
                        meta_weight=float(teacher_kl.get("meta_weight", 1.0)),
                        diagnosis_weight=float(teacher_kl.get("diagnosis_weight", 1.25)),
                        study_need_weight=float(teacher_kl.get("study_need_weight", 1.4)),
                        recovery_weight=float(teacher_kl.get("recovery_weight", 0.7)),
                        verify_weight=float(teacher_kl.get("verify_weight", 1.1)),
                    )
                    if bool(teacher_kl.get("row_quality_scaling", True)):
                        row_scale = compute_teacher_kl_row_scale(
                            row,
                            entropy_penalty_strength=float(teacher_kl.get("entropy_penalty_strength", 0.35)),
                        )
                        weights = [float(weight) * float(row_scale) for weight in weights]
                    entropy_payload = row.get("teacher_token_entropy_json")
                    if entropy_payload:
                        if isinstance(entropy_payload, str):
                            entropy_values = json.loads(entropy_payload)
                        else:
                            entropy_values = entropy_payload
                        weights = apply_entropy_weighting(
                            position_weights=weights,
                            entropy_values=entropy_values,
                            beta=float(teacher_kl.get("entropy_beta", 0.0)),
                            floor=float(teacher_kl.get("entropy_weight_floor", 0.65)),
                            ceil=float(teacher_kl.get("entropy_weight_ceil", 1.10)),
                        )
                    built.update({
                        "teacher_topk_token_ids": trimmed.token_ids,
                        "teacher_topk_logprobs": trimmed.logprobs,
                        "teacher_target_logprobs": trimmed.target_logprobs,
                        "teacher_assistant_token_ids": trimmed.assistant_token_ids,
                        "teacher_position_weights": weights,
                        "teacher_kl_active": int(any(float(weight) > 0.0 for weight in weights)),
                    })
                else:
                    built["teacher_kl_active"] = 0
            else:
                built["teacher_kl_active"] = 0
        return built

    ds = Dataset.from_pandas(df)
    ds = ds.map(tokenize_row, remove_columns=df.columns.tolist())
    ds = ds.filter(lambda row: row["num_target_tokens"] > 0)
    if len(ds) == 0:
        raise ValueError("SFT dataset has zero trainable rows after truncation/masking.")
    if teacher_kl_enabled:
        active_rows = sum(int(x) for x in ds["teacher_kl_active"]) if "teacher_kl_active" in ds.column_names else 0
        if active_rows == 0 and bool(teacher_kl.get("require_targets", True)):
            raise ValueError(
                "teacher_kl is enabled, but no rows contained teacher top-k targets. "
                "Build teacher_topk_targets.parquet first or disable teacher_kl."
            )
        if "teacher_kl_active" in ds.column_names:
            ds = ds.remove_columns(["teacher_kl_active"])
    ds = ds.remove_columns(["num_target_tokens"])
    return ds


class DistillDataCollator:
    """Pad normal SFT fields and optional teacher-KL annotations."""

    EXTRA_KEYS = {
        "teacher_topk_token_ids",
        "teacher_topk_logprobs",
        "teacher_target_logprobs",
        "teacher_assistant_token_ids",
        "teacher_position_weights",
    }

    def __init__(self, tokenizer):
        from transformers import DataCollatorForSeq2Seq

        self.base = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, return_tensors="pt")

    def __call__(self, features):
        import torch

        extra_payloads = [{k: feature.pop(k) for k in list(feature.keys()) if k in self.EXTRA_KEYS} for feature in features]
        batch = self.base(features)

        if not any(payload for payload in extra_payloads):
            return batch

        max_positions = max((len(payload.get("teacher_assistant_token_ids", [])) for payload in extra_payloads), default=0)
        max_top_k = max(
            (
                len(position)
                for payload in extra_payloads
                for position in payload.get("teacher_topk_token_ids", [])
            ),
            default=0,
        )
        if max_positions == 0 or max_top_k == 0:
            return batch

        bsz = len(extra_payloads)
        token_ids = torch.full((bsz, max_positions, max_top_k), -1, dtype=torch.long)
        logprobs = torch.full((bsz, max_positions, max_top_k), -1e9, dtype=torch.float32)
        target_logprobs = torch.full((bsz, max_positions), -1e9, dtype=torch.float32)
        assistant_token_ids = torch.full((bsz, max_positions), -1, dtype=torch.long)
        position_weights = torch.zeros((bsz, max_positions), dtype=torch.float32)

        for batch_idx, payload in enumerate(extra_payloads):
            positions = min(
                len(payload.get("teacher_assistant_token_ids", [])),
                len(payload.get("teacher_topk_token_ids", [])),
                len(payload.get("teacher_topk_logprobs", [])),
                len(payload.get("teacher_target_logprobs", [])),
                len(payload.get("teacher_position_weights", [])),
            )
            for pos_idx in range(positions):
                ids = [int(x) for x in payload["teacher_topk_token_ids"][pos_idx][:max_top_k]]
                lps = [float(x) for x in payload["teacher_topk_logprobs"][pos_idx][:max_top_k]]
                size = min(len(ids), len(lps), max_top_k)
                if size:
                    token_ids[batch_idx, pos_idx, :size] = torch.tensor(ids[:size], dtype=torch.long)
                    logprobs[batch_idx, pos_idx, :size] = torch.tensor(lps[:size], dtype=torch.float32)
                target_logprobs[batch_idx, pos_idx] = float(payload["teacher_target_logprobs"][pos_idx])
                assistant_token_ids[batch_idx, pos_idx] = int(payload["teacher_assistant_token_ids"][pos_idx])
                position_weights[batch_idx, pos_idx] = float(payload["teacher_position_weights"][pos_idx])

        batch["teacher_topk_token_ids"] = token_ids
        batch["teacher_topk_logprobs"] = logprobs
        batch["teacher_target_logprobs"] = target_logprobs
        batch["teacher_assistant_token_ids"] = assistant_token_ids
        batch["teacher_position_weights"] = position_weights
        return batch


class DistillTrainerMixin:
    """Add optional control-span teacher KL on top of standard CE/SFT."""

    EXTRA_KEYS = DistillDataCollator.EXTRA_KEYS

    def __init__(self, *args, teacher_kl_coef: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher_kl_coef = float(teacher_kl_coef)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        import torch

        teacher_inputs = {key: inputs.pop(key) for key in list(inputs.keys()) if key in self.EXTRA_KEYS}
        outputs = model(**inputs)
        loss = outputs.loss

        if self.teacher_kl_coef > 0.0 and teacher_inputs:
            kl_loss = self._compute_teacher_kl(outputs.logits, inputs["labels"], teacher_inputs)
            if kl_loss is not None:
                loss = loss + self.teacher_kl_coef * kl_loss

        return (loss, outputs) if return_outputs else loss

    def _compute_teacher_kl(self, logits, labels, teacher_inputs):
        import torch

        teacher_token_ids = teacher_inputs.get("teacher_topk_token_ids")
        teacher_logprobs = teacher_inputs.get("teacher_topk_logprobs")
        teacher_target_logprobs = teacher_inputs.get("teacher_target_logprobs")
        teacher_assistant_token_ids = teacher_inputs.get("teacher_assistant_token_ids")
        position_weights = teacher_inputs.get("teacher_position_weights")
        if any(x is None for x in (teacher_token_ids, teacher_logprobs, teacher_target_logprobs, teacher_assistant_token_ids, position_weights)):
            return None

        shifted_logits = logits[:, :-1, :]
        shifted_labels = labels[:, 1:]
        total = shifted_logits.new_zeros(())
        total_weight = shifted_logits.new_zeros(())

        for batch_idx in range(shifted_logits.shape[0]):
            assistant_positions = torch.nonzero(shifted_labels[batch_idx] != -100, as_tuple=False).squeeze(-1)
            if assistant_positions.numel() == 0:
                continue
            usable = min(
                assistant_positions.numel(),
                teacher_assistant_token_ids.shape[1],
                position_weights.shape[1],
            )
            if usable == 0:
                continue
            student_logits = shifted_logits[batch_idx, assistant_positions[:usable], :]
            for pos_idx in range(usable):
                weight = position_weights[batch_idx, pos_idx]
                if weight <= 0:
                    continue
                token_row = teacher_token_ids[batch_idx, pos_idx]
                logprob_row = teacher_logprobs[batch_idx, pos_idx]
                valid = token_row >= 0
                if not torch.any(valid):
                    continue

                ids = token_row[valid]
                teacher_lps = logprob_row[valid]
                target_id = teacher_assistant_token_ids[batch_idx, pos_idx]
                target_lp = teacher_target_logprobs[batch_idx, pos_idx]
                label_token = shifted_labels[batch_idx, assistant_positions[pos_idx]]
                if target_id >= 0 and label_token >= 0 and int(target_id.item()) != int(label_token.item()):
                    continue
                if target_id >= 0 and not torch.any(ids == target_id):
                    ids = torch.cat([ids, target_id.view(1)])
                    teacher_lps = torch.cat([teacher_lps, target_lp.view(1)])

                teacher_probs = torch.softmax(teacher_lps, dim=-1)
                student_log_probs = torch.log_softmax(student_logits[pos_idx, ids], dim=-1)
                teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-12))
                kl = torch.sum(teacher_probs * (teacher_log_probs - student_log_probs))
                total = total + weight * kl
                total_weight = total_weight + weight

        if total_weight.item() <= 0:
            return None
        return total / total_weight


def run_sft(config_path: str):
    """Run Meta-CoT SFT training."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = config["model_name_or_path"]
    data_path = config["dataset_path"]
    output_dir = config["output_dir"]
    teacher_kl = _load_teacher_kl_config(config)
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

    import os as _os
    if _os.environ.get("S3B_META_EMB_TRANSPLANT", "0") == "1":
        from src.training.meta_token_init import transplant_meta_embeddings_from_think
        transplant_meta_embeddings_from_think(model, tokenizer)
        print("[s3b] transplanted meta-token embeddings from think tokens")

    full_dataset = prepare_sft_dataset(
        data_path,
        tokenizer,
        max_length=config.get("max_length", 4096),
        teacher_kl=teacher_kl,
    )
    if len(full_dataset) < 2:
        train_dataset = full_dataset
        eval_dataset = full_dataset
    else:
        n_eval = max(1, int(round(len(full_dataset) * 0.05)))
        n_eval = min(n_eval, len(full_dataset) - 1)
        split = full_dataset.train_test_split(test_size=n_eval, seed=42)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    from transformers import Trainer, TrainingArguments

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
        save_strategy=config.get("save_strategy", "steps"),
        save_steps=config.get("save_steps", 500),
        save_total_limit=3,
        report_to="wandb",
        eval_strategy=config.get("eval_strategy", "no"),
        deepspeed=config.get("deepspeed", None),
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    data_collator = DistillDataCollator(tokenizer=tokenizer)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
    }

    TrainerClass = Trainer
    if teacher_kl.get("enabled", False) and float(teacher_kl.get("coef", 0.0)) > 0.0:
        class DistillTrainer(DistillTrainerMixin, Trainer):
            pass
        TrainerClass = DistillTrainer

    trainer_signature = inspect.signature(TrainerClass.__init__)
    if "processing_class" in trainer_signature.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    if issubclass(TrainerClass, DistillTrainerMixin):
        trainer_kwargs["teacher_kl_coef"] = float(teacher_kl.get("coef", 0.0))

    trainer = TrainerClass(**trainer_kwargs)

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
