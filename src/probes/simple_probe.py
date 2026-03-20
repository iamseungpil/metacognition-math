"""Simple hidden-state probe baseline (control group for Gnosis comparison).

A lightweight MLP that predicts correctness from the mean-pooled last-layer
hidden state. Much simpler than Gnosis: no attention maps, no dilated conv,
no set transformers. Serves as a baseline to show how much Gnosis's
architectural choices matter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional
import json
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score


class SimpleCorrectnessProbe(nn.Module):
    """MLP probe: mean-pooled hidden states → correctness probability."""

    def __init__(self, hidden_dim: int = 3584, intermediate_dim: int = 512):
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

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            hidden_states: (B, S, D) last-layer hidden states
            attention_mask: (B, S) binary mask
        Returns:
            probs: (B,) correctness probability
        """
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = hidden_states.mean(dim=1)

        logits = self.net(pooled).squeeze(-1)
        return torch.sigmoid(logits)


class HiddenStateDataset(Dataset):
    """Dataset of cached hidden states + correctness labels."""

    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        manifest_path = self.data_dir / f"{split}_manifest.json"
        with open(manifest_path) as f:
            self.manifest = json.load(f)

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx):
        entry = self.manifest[idx]
        hs = torch.load(
            self.data_dir / entry["hidden_state_file"],
            map_location="cpu",
            weights_only=True,
        )
        return {
            "hidden_states": hs,
            "label": torch.tensor(entry["is_correct"], dtype=torch.float32),
            "problem_id": entry["problem_id"],
        }


def collate_hidden_states(batch):
    """Collate variable-length hidden states with padding."""
    max_len = max(item["hidden_states"].shape[0] for item in batch)
    hidden_dim = batch[0]["hidden_states"].shape[1]
    B = len(batch)

    padded_hs = torch.zeros(B, max_len, hidden_dim)
    masks = torch.zeros(B, max_len)
    labels = torch.zeros(B)

    for i, item in enumerate(batch):
        seq_len = item["hidden_states"].shape[0]
        padded_hs[i, :seq_len] = item["hidden_states"]
        masks[i, :seq_len] = 1.0
        labels[i] = item["label"]

    return {
        "hidden_states": padded_hs,
        "attention_mask": masks,
        "labels": labels,
    }


def train_simple_probe(
    data_dir: str,
    output_dir: str,
    hidden_dim: int = 3584,
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "cuda",
    wandb_run=None,
):
    """Train the simple probe and evaluate."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_ds = HiddenStateDataset(data_dir, "train")
    val_ds = HiddenStateDataset(data_dir, "val")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_hidden_states, num_workers=4,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=collate_hidden_states, num_workers=4,
    )

    model = SimpleCorrectnessProbe(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auroc = 0.0

    for epoch in range(epochs):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            hs = batch["hidden_states"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            probs = model(hs, mask)
            loss = F.binary_cross_entropy(probs, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validate
        model.eval()
        all_probs, all_labels = [], []
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                hs = batch["hidden_states"].to(device)
                mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                probs = model(hs, mask)
                loss = F.binary_cross_entropy(probs, labels)
                val_losses.append(loss.item())

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs >= 0.5).astype(float)

        auroc = roc_auc_score(all_labels, all_probs)
        acc = accuracy_score(all_labels, preds)
        f1 = f1_score(all_labels, preds)

        metrics = {
            "epoch": epoch,
            "train_loss": np.mean(train_losses),
            "val_loss": np.mean(val_losses),
            "val_auroc": auroc,
            "val_accuracy": acc,
            "val_f1": f1,
            "val_prob_mean": float(all_probs.mean()),
            "val_prob_std": float(all_probs.std()),
        }

        print(
            f"Epoch {epoch}: train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} auroc={auroc:.4f} "
            f"acc={acc:.4f} f1={f1:.4f}"
        )

        if wandb_run:
            wandb_run.log({"simple_probe/" + k: v for k, v in metrics.items()})

        if auroc > best_auroc:
            best_auroc = auroc
            torch.save(model.state_dict(), output_path / "best_probe.pt")
            with open(output_path / "best_metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

    print(f"Best AUROC: {best_auroc:.4f}")
    return best_auroc


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=3584)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    import wandb
    run = wandb.init(project="metacot-math", name="simple-probe-baseline")
    train_simple_probe(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        wandb_run=run,
    )
    wandb.finish()
