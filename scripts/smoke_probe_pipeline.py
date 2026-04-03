#!/usr/bin/env python3
"""Smoke-check a trained simple probe artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ece(probs, labels, n_bins=15):
    if not probs:
        return None
    total = len(probs)
    ece = 0.0
    for idx in range(n_bins):
        lo = idx / n_bins
        hi = (idx + 1) / n_bins
        bucket = [
            (p, y) for p, y in zip(probs, labels)
            if (lo <= p < hi) or (idx == n_bins - 1 and lo <= p <= hi)
        ]
        if not bucket:
            continue
        bucket_probs = [p for p, _ in bucket]
        bucket_labels = [y for _, y in bucket]
        ece += len(bucket) / total * abs(mean(bucket_probs) - mean(bucket_labels))
    return float(ece)


def _evaluate_probe(model, dataset, collate_hidden_states, max_samples=None):
    import torch

    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    if n == 0:
        raise RuntimeError("empty evaluation split for probe smoke")

    probs = []
    labels = []
    for start in range(0, n, 16):
        batch_items = [dataset[i] for i in range(start, min(start + 16, n))]
        batch = collate_hidden_states(batch_items)
        with torch.no_grad():
            batch_probs = model(batch["hidden_states"], batch["attention_mask"])
        probs.extend(float(x) for x in batch_probs.tolist())
        labels.extend(float(x) for x in batch["labels"].tolist())

    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / len(probs)
    mae = sum(abs(p - y) for p, y in zip(probs, labels)) / len(probs)
    return {
        "n_eval_samples": len(probs),
        "recomputed_val_brier": float(brier),
        "recomputed_val_mae": float(mae),
        "recomputed_val_ece": _ece(probs, labels),
        "prob_min": min(probs),
        "prob_max": max(probs),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=4)
    parser.add_argument("--eval-max-samples", type=int, default=256)
    args = parser.parse_args()

    import torch
    from src.probes.simple_probe import HiddenStateDataset, collate_hidden_states, SimpleCorrectnessProbe
    from src.probes.retrain import infer_hidden_dim

    probe_dir = Path(args.probe_dir)
    probe_path = probe_dir / "best_probe.pt"
    metrics_path = probe_dir / "best_metrics.json"
    cache_dir = probe_dir / "hidden_states_cache"
    train_manifest = cache_dir / "train_manifest.json"
    val_manifest = cache_dir / "val_manifest.json"

    for path in [probe_path, metrics_path, train_manifest, val_manifest]:
        if not path.exists():
            raise FileNotFoundError(path)

    train_ds = HiddenStateDataset(str(cache_dir), "train")
    val_ds = HiddenStateDataset(str(cache_dir), "val")
    if len(train_ds) == 0:
        raise RuntimeError("empty train manifest")
    if len(val_ds) == 0:
        raise RuntimeError("empty val manifest")

    batch = collate_hidden_states([train_ds[i] for i in range(min(args.max_samples, len(train_ds)))])
    hidden_dim = args.hidden_dim or infer_hidden_dim(data_dir=str(cache_dir))
    model = SimpleCorrectnessProbe(hidden_dim=hidden_dim)
    temperature = None
    state = torch.load(probe_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        temperature = state.get("temperature")
        state = state["state_dict"]
    incompatible = model.load_state_dict(state, strict=False)
    allowed_missing = {"temperature"}
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing - allowed_missing:
        raise RuntimeError(f"unexpected missing probe keys: {sorted(missing)}")
    if unexpected:
        raise RuntimeError(f"unexpected extra probe keys: {sorted(unexpected)}")
    if temperature is not None:
        model.temperature.fill_(float(temperature))
    model.eval()

    with torch.no_grad():
        probs = model(batch["hidden_states"], batch["attention_mask"])
    if probs.ndim != 1:
        raise RuntimeError(f"unexpected probe output shape: {tuple(probs.shape)}")
    if torch.any(probs < 0) or torch.any(probs > 1):
        raise RuntimeError("probe probabilities out of range")

    metrics = json.loads(metrics_path.read_text())
    recomputed = _evaluate_probe(
        model,
        val_ds,
        collate_hidden_states,
        max_samples=args.eval_max_samples if args.eval_max_samples > 0 else None,
    )
    if recomputed["recomputed_val_brier"] < 0.0:
        raise RuntimeError("invalid probe brier")

    print(json.dumps({
        "probe_path": str(probe_path),
        "n_smoke_samples": int(probs.shape[0]),
        "hidden_dim": hidden_dim,
        "stored_metric_keys": sorted(metrics.keys()),
        "stored_val_brier": metrics.get("val_brier"),
        "stored_val_ece": metrics.get("val_ece"),
        "stored_calibrated_val_brier": metrics.get("calibrated_val_brier"),
        **recomputed,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
