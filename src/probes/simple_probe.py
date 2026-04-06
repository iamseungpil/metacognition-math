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

    def __init__(self, hidden_dim: int, intermediate_dim: int = 512):
        super().__init__()
        self.register_buffer("temperature", torch.ones(1))
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

    def _pool(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return hidden_states.mean(dim=1)

    def logits(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        pooled = self._pool(hidden_states, attention_mask)
        return self.net(pooled).squeeze(-1)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Args:
            hidden_states: (B, S, D) last-layer hidden states
            attention_mask: (B, S) binary mask
        Returns:
            probs: (B,) correctness probability
        """
        logits = self.logits(hidden_states, attention_mask)
        return torch.sigmoid(logits / self.temperature.clamp(min=1e-3))


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
        target = entry.get("target_prob")
        if target is None:
            target = float(entry["is_correct"])
        return {
            "hidden_states": hs,
            "label": torch.tensor(target, dtype=torch.float32),
            "problem_id": entry["problem_id"],
            "sample_kind": entry.get("sample_kind", "full"),
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


def _compute_ece(probs, labels, n_bins=15):
    """Expected Calibration Error — measures probability calibration quality."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_acc = labels[mask].mean()
        bin_conf = probs[mask].mean()
        ece += mask.sum() / len(probs) * abs(bin_acc - bin_conf)
    return float(ece)


def calibrate_temperature(model, val_loader, device="cuda", lr=0.01, max_iter=50):
    """Learn a temperature parameter to improve probe calibration (post-hoc)."""
    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=lr, max_iter=max_iter)

    all_logits, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            hs = batch["hidden_states"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            # Get raw logits (before sigmoid)
            if mask is not None:
                m = mask.unsqueeze(-1).float()
                pooled = (hs * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
            else:
                pooled = hs.mean(dim=1)
            logits = model.net(pooled).squeeze(-1)
            all_logits.append(logits)
            all_labels.append(labels)

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    def closure():
        optimizer.zero_grad()
        scaled = torch.sigmoid(all_logits / temperature.clamp(min=1e-3))
        loss = F.binary_cross_entropy(scaled, all_labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    calibrated_temp = float(temperature.detach().item())
    model.temperature.fill_(calibrated_temp)
    return calibrated_temp


def _compute_probe_metrics(probs, labels):
    labels = np.asarray(labels, dtype=float)
    probs = np.asarray(probs, dtype=float)
    metrics = {
        "val_brier": float(np.mean((probs - labels) ** 2)),
        "val_mae": float(np.mean(np.abs(probs - labels))),
        "val_ece": _compute_ece(probs, labels, n_bins=15),
        "val_prob_mean": float(probs.mean()),
        "val_prob_std": float(probs.std()),
        "val_target_mean": float(labels.mean()),
        "val_target_std": float(labels.std()),
    }

    is_binary = np.all(np.isin(labels, [0.0, 1.0]))
    if is_binary and len(np.unique(labels)) >= 2:
        preds = (probs >= 0.5).astype(float)
        metrics["val_auroc"] = float(roc_auc_score(labels, probs))
        metrics["val_accuracy"] = float(accuracy_score(labels, preds))
        metrics["val_f1"] = float(f1_score(labels, preds))
    else:
        metrics["val_auroc"] = None
        metrics["val_accuracy"] = None
        metrics["val_f1"] = None

    return metrics


def train_simple_probe(
    data_dir: str,
    output_dir: str,
    hidden_dim: Optional[int] = None,
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

    if hidden_dim is None:
        sample = train_ds[0]["hidden_states"]
        hidden_dim = int(sample.shape[-1])

    model = SimpleCorrectnessProbe(hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_brier = float("inf")
    best_state_dict = None

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
        metrics = {
            "epoch": epoch,
            "train_loss": np.mean(train_losses),
            "val_loss": np.mean(val_losses),
        }
        metrics.update(_compute_probe_metrics(all_probs, all_labels))
        brier = metrics["val_brier"]
        auroc = metrics["val_auroc"]
        auroc_text = f" auroc={auroc:.4f}" if auroc is not None else ""

        print(
            f"Epoch {epoch}: train_loss={metrics['train_loss']:.4f} "
            f"val_loss={metrics['val_loss']:.4f} brier={brier:.4f}{auroc_text}"
        )

        if wandb_run:
            wandb_run.log({"simple_probe/" + k: v for k, v in metrics.items()})

        if brier < best_brier:
            best_brier = brier
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = metrics.copy()

    if best_state_dict is None:
        raise RuntimeError("probe training produced no checkpoint")

    model.load_state_dict(best_state_dict)
    temperature = calibrate_temperature(model, val_loader, device=device)

    model.eval()
    calibrated_probs, calibrated_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            hs = batch["hidden_states"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            probs = model(hs, mask)
            calibrated_probs.extend(probs.cpu().numpy())
            calibrated_labels.extend(labels.cpu().numpy())

    calibrated_probs = np.array(calibrated_probs)
    calibrated_labels = np.array(calibrated_labels)
    calibrated_metrics = _compute_probe_metrics(calibrated_probs, calibrated_labels)
    best_metrics.update({
        "temperature": temperature,
        "calibrated_val_brier": calibrated_metrics["val_brier"],
        "calibrated_val_mae": calibrated_metrics["val_mae"],
        "calibrated_val_ece": calibrated_metrics["val_ece"],
        "calibrated_val_auroc": calibrated_metrics["val_auroc"],
        "calibrated_val_accuracy": calibrated_metrics["val_accuracy"],
        "calibrated_val_f1": calibrated_metrics["val_f1"],
    })

    torch.save(
        {
            "state_dict": model.state_dict(),
            "temperature": temperature,
        },
        output_path / "best_probe.pt",
    )
    with open(output_path / "best_metrics.json", "w") as f:
        json.dump(best_metrics, f, indent=2)

    print(f"Best Brier: {best_brier:.4f}")
    return best_brier


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=None)
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
