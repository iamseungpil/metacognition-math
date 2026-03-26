"""SimpleCorrectnessProbe Training (Phase 2 — Simple Version).

Trains a lightweight MLP probe on hidden states from Qwen3 Meta SFT model.
No Gnosis model patching needed — uses standard HF model with output_hidden_states.

Architecture: mean-pooled last-layer hidden (4096-D) → MLP → P(correct)
Loss: BCE(P(correct), is_correct_label)

Usage:
  python src/training/probe_sft.py \
    --model_path checkpoints/qwen3_meta_sft \
    --data_path rollouts/rollouts_final.parquet \
    --output_dir checkpoints/simple_probe_qwen3
"""
import argparse
import os
import json

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, accuracy_score
import wandb


class SimpleCorrectnessProbe(nn.Module):
    """MLP probe: mean-pooled hidden states → correctness probability."""

    def __init__(self, hidden_dim=4096, intermediate_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim, intermediate_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(intermediate_dim // 2, 1),
        )

    def forward(self, hidden_states, attention_mask=None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)
        logits = self.net(pooled).squeeze(-1)
        return torch.sigmoid(logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="checkpoints/simple_probe_qwen3")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max_samples", type=int, default=10000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model (standard, no Gnosis patch needed)
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        use_cache=False,
    )
    model.eval()
    model.cuda()

    # Get hidden_dim from model
    hidden_dim = model.config.hidden_size
    print(f"Hidden dim: {hidden_dim}")

    # Load data
    df = pd.read_parquet(args.data_path)
    if args.max_samples and len(df) > args.max_samples:
        # Balance correct/incorrect
        correct = df[df["is_correct"] == True].sample(n=min(args.max_samples // 2, len(df[df["is_correct"] == True])), random_state=42)
        incorrect = df[df["is_correct"] == False].sample(n=min(args.max_samples // 2, len(df[df["is_correct"] == False])), random_state=42)
        df = pd.concat([correct, incorrect]).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Data: {len(df)} samples, {df['is_correct'].sum()} correct ({df['is_correct'].mean()*100:.1f}%)")

    # Step 1: Extract hidden states (one forward pass per sample)
    print("Extracting hidden states...")
    all_hidden = []
    all_labels = []

    with torch.no_grad():
        for idx in range(len(df)):
            row = df.iloc[idx]
            text = f"Question: {row['question']}\n\nAnswer: {row['completion']}"
            enc = tokenizer(text, max_length=args.max_length, truncation=True, return_tensors="pt")
            input_ids = enc["input_ids"].cuda()
            attention_mask = enc["attention_mask"].cuda()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1]  # (1, S, D)

            # Mean pool
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            all_hidden.append(pooled.cpu().to(torch.float16))
            all_labels.append(1.0 if row["is_correct"] else 0.0)

            if (idx + 1) % 100 == 0:
                print(f"  Extracted {idx + 1}/{len(df)}")

    all_hidden = torch.cat(all_hidden, dim=0)  # (N, D)
    all_labels = torch.tensor(all_labels)  # (N,)
    print(f"Hidden states: {all_hidden.shape}, Labels: {all_labels.shape}")

    # Save extracted hidden states
    torch.save({"hidden": all_hidden, "labels": all_labels}, os.path.join(args.output_dir, "hidden_states.pt"))
    print("Saved hidden states")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Step 2: Train probe
    print("Training probe...")
    probe = SimpleCorrectnessProbe(hidden_dim=hidden_dim).cuda()
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=0.01)

    # Train/val split
    n_val = min(int(len(all_labels) * 0.1), 1000)
    val_hidden = all_hidden[:n_val].float().cuda()
    val_labels = all_labels[:n_val].cuda()
    train_hidden = all_hidden[n_val:].float().cuda()
    train_labels = all_labels[n_val:].cuda()

    wandb.init(project="metacot-math", name="simple-probe-qwen3", config=vars(args))

    best_auroc = 0.0
    for epoch in range(args.epochs):
        # Train
        probe.train()
        perm = torch.randperm(len(train_labels))
        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(train_labels), args.batch_size):
            batch_idx = perm[i:i + args.batch_size]
            h = train_hidden[batch_idx]
            y = train_labels[batch_idx]

            probs = probe(h.unsqueeze(1))  # (B, 1, D) → probe expects (B, S, D)
            loss = F.binary_cross_entropy(probs, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(probe.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        # Eval
        probe.eval()
        with torch.no_grad():
            val_probs = probe(val_hidden.unsqueeze(1))
            val_loss = F.binary_cross_entropy(val_probs, val_labels).item()

            probs_np = val_probs.cpu().numpy()
            labels_np = val_labels.cpu().numpy()

            auroc = roc_auc_score(labels_np, probs_np)
            acc = accuracy_score(labels_np, (probs_np >= 0.5).astype(float))

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_loss={val_loss:.4f} AUROC={auroc:.4f} acc={acc:.4f}")

        wandb.log({
            "probe/train_loss": avg_loss,
            "probe/val_loss": val_loss,
            "probe/auroc": auroc,
            "probe/accuracy": acc,
            "probe/epoch": epoch,
        })

        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(probe.state_dict(), os.path.join(args.output_dir, "best_probe.pt"))
            with open(os.path.join(args.output_dir, "best_metrics.json"), "w") as f:
                json.dump({"auroc": auroc, "accuracy": acc, "val_loss": val_loss, "epoch": epoch}, f, indent=2)

    print(f"Best AUROC: {best_auroc:.4f}")
    print(f"Probe saved to {args.output_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
