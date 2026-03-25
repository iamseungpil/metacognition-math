"""Gnosis Head SFT Training (Phase 2).

Trains the Gnosis correctness head on Qwen3 Meta SFT model.
Backbone is FROZEN — only Gnosis head params (~5M) are trained.

Loss: BCE(P(correct), is_correct_label)
Data: rollout completions with is_correct labels

Usage:
  accelerate launch --config_file configs/accelerate_ds3.yaml \
    src/training/gnosis_sft.py \
    --model_path checkpoints/qwen3_meta_sft \
    --data_path rollouts/rollouts_final.parquet \
    --output_dir checkpoints/gnosis_head
"""
import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb


class RolloutCorrectnessDataset(Dataset):
    """Dataset of (prompt+completion, is_correct) pairs from rollouts."""

    def __init__(self, df, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        for _, row in df.iterrows():
            question = str(row.get("question", ""))
            completion = str(row.get("completion", ""))
            is_correct = bool(row.get("is_correct", False))

            # Build full text: question + completion
            text = f"Question: {question}\n\nAnswer: {completion}"
            self.data.append({"text": text, "label": 1.0 if is_correct else 0.0})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )
        input_ids = enc["input_ids"]
        # Create LM labels (shifted input_ids) — needed for token_probs in Gnosis forward
        labels = input_ids[1:] + [self.tokenizer.pad_token_id or 0]
        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"],
            "labels_lm": labels,  # token-level LM labels for token_probs
            "correctness_label": item["label"],  # sequence-level correctness
        }


def collate_fn(batch):
    """Pad batch to uniform length."""
    max_len = max(len(b["input_ids"]) for b in batch)
    pad_id = 0

    input_ids = []
    attention_mask = []
    labels_lm = []
    correctness_labels = []

    for b in batch:
        pad_len = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad_len)
        attention_mask.append(b["attention_mask"] + [0] * pad_len)
        labels_lm.append(b["labels_lm"] + [-100] * pad_len)  # -100 = ignore in CE loss
        correctness_labels.append(b["correctness_label"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels_lm, dtype=torch.long),  # LM labels for token_probs
        "correctness_label": torch.tensor(correctness_labels, dtype=torch.float32),
    }


def train_gnosis(args):
    # No mixed precision — BCE loss is unsafe with autocast
    # Backbone is frozen anyway, so bf16 autocast not needed for memory
    accelerator = Accelerator()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (gnosis-patched Qwen3)
    # NO gradient_checkpointing — conflicts with output_attentions for Gnosis
    # Memory is OK because backbone is frozen (no grad storage needed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        use_cache=False,
    )

    # Verify model has Gnosis head
    has_gnosis = hasattr(model, "_should_stop")
    if accelerator.is_main_process:
        if has_gnosis:
            print("[Gnosis SFT] Model has _should_stop method — Full Gnosis training")
        else:
            print("[Gnosis SFT] WARNING: Model lacks _should_stop — cannot train Full Gnosis")
            print("[Gnosis SFT] Training SimpleCorrectnessProbe instead")

    # Freeze backbone, unfreeze Gnosis heads
    gnosis_prefixes = ("stop_head", "attn_extractor", "hid_extractor", "conf_extractor")
    n_trainable = 0
    n_frozen = 0
    for name, param in model.named_parameters():
        if any(p in name for p in gnosis_prefixes):
            param.requires_grad_(True)
            n_trainable += param.numel()
        else:
            param.requires_grad_(False)
            n_frozen += param.numel()

    if accelerator.is_main_process:
        print(f"[Gnosis SFT] Trainable: {n_trainable:,} params ({n_trainable/1e6:.1f}M)")
        print(f"[Gnosis SFT] Frozen: {n_frozen:,} params ({n_frozen/1e6:.1f}M)")

    # Load data
    df = pd.read_parquet(args.data_path)
    if accelerator.is_main_process:
        n_correct = df["is_correct"].sum()
        print(f"[Gnosis SFT] Data: {len(df)} rollouts, {n_correct} correct ({n_correct/len(df)*100:.1f}%)")

    # Train/val split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_size = min(int(len(df) * 0.05), 1000)
    train_df = df.iloc[val_size:]
    val_df = df.iloc[:val_size]

    train_ds = RolloutCorrectnessDataset(train_df, tokenizer, max_length=args.max_length)
    val_ds = RolloutCorrectnessDataset(val_df, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

    # wandb
    if accelerator.is_main_process:
        wandb.init(project="metacot-math", name="gnosis-sft-qwen3", config=vars(args))

    # Training loop
    best_auroc = 0.0
    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            labels_lm = batch["labels"]  # (B, S) token-level LM labels
            correctness_label = batch["correctness_label"]  # (B,) correctness

            # Forward with both labels and correctness_label
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels_lm,  # needed for token_probs computation
                correctness_label=correctness_label,  # for Gnosis BCE loss
                use_cache=False,
            )
            loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=input_ids.device)

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            n_batches += 1

            if accelerator.is_main_process and batch_idx % 50 == 0:
                avg_loss = total_loss / max(n_batches, 1)
                print(f"Epoch {epoch} Step {batch_idx}/{len(train_loader)} loss={avg_loss:.4f}")
                wandb.log({"gnosis/train_loss": avg_loss, "gnosis/step": epoch * len(train_loader) + batch_idx})

        # Validation
        model.eval()
        all_probs = []
        all_labels = []
        val_loss = 0
        n_val = 0

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels_lm = batch["labels"]
                correctness_label = batch["correctness_label"]

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_lm,
                    correctness_label=correctness_label,
                    use_cache=False,
                )
                loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0)
                val_loss += loss.item()
                n_val += 1

        if accelerator.is_main_process:
            avg_val_loss = val_loss / max(n_val, 1)
            avg_train_loss = total_loss / max(n_batches, 1)
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f}")
            wandb.log({
                "gnosis/epoch": epoch,
                "gnosis/train_loss_epoch": avg_train_loss,
                "gnosis/val_loss": avg_val_loss,
            })

            # Save checkpoint
            unwrapped = accelerator.unwrap_model(model)
            save_path = os.path.join(args.output_dir, f"epoch_{epoch}")
            unwrapped.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"Saved checkpoint to {save_path}")

    # Save final
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"[Gnosis SFT] Training complete. Saved to {args.output_dir}")
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="checkpoints/gnosis_head")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()
    train_gnosis(args)
