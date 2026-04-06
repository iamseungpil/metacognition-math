"""Re-train Gnosis probe on SFT model's hidden states.

After SFT changes the model, hidden state distribution shifts.
Probe must be re-trained on the new model's representations.

Also supports extracting hidden states at <|meta|> positions
for step-level probe training.
"""
import json
from pathlib import Path
import random
import re
from typing import Optional

from tqdm import tqdm

from src.metacot.prompt import META_START, META_END
from src.training.tokenizer_utils import ensure_meta_tokens_not_special

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional for helper-only tests
    pd = None

try:
    import torch
except ImportError:  # pragma: no cover - optional for helper-only tests
    torch = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:  # pragma: no cover - optional for helper-only tests
    AutoTokenizer = None
    AutoModelForCausalLM = None


def _iter_meta_prefix_texts(text: str) -> list[tuple[int, str]]:
    """Return prefixes ending at each meta block.

    Probe reward is computed on prefixes, not full completions. We therefore
    cache prefix-level hidden states during retraining so the probe sees the
    same object at train and reward time.
    """
    pattern = re.compile(
        rf"{re.escape(META_START)}(.*?){re.escape(META_END)}",
        re.IGNORECASE | re.DOTALL,
    )
    return [(idx, text[:match.end()]) for idx, match in enumerate(pattern.finditer(text))]


def _combine_prompt_and_completion(prompt_text: str, completion_fragment: str) -> str:
    prompt_text = (prompt_text or "").rstrip("\n")
    if not prompt_text:
        return completion_fragment
    return prompt_text + completion_fragment


def _group_train_val_split(
    manifest: list[dict],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Split by problem_id to avoid leakage across train/val."""
    grouped: dict[str, list[dict]] = {}
    for entry in manifest:
        grouped.setdefault(str(entry["problem_id"]), []).append(entry)

    group_ids = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(group_ids)

    n_val_groups = max(1, int(round(len(group_ids) * val_fraction))) if group_ids else 0
    val_group_ids = set(group_ids[:n_val_groups])

    train_manifest, val_manifest = [], []
    for problem_id, entries in grouped.items():
        target = val_manifest if problem_id in val_group_ids else train_manifest
        target.extend(entries)
    return train_manifest, val_manifest


def _normalize_probability(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        value = 1.0 if value else 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if value > 1.0:
        value /= 100.0
    return max(0.0, min(1.0, value))


def _decode_jsonish(value):
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return None
    return value


def _lookup_prefix_target(row, prefix_idx: int) -> tuple[Optional[float], str]:
    candidate_keys = [
        "meta_prefix_target_probs",
        "prefix_target_probs",
        "probe_target_probs",
        "prefix_correct_probs",
        "empirical_prefix_accuracy",
        "prefix_accuracy",
        "meta_prefix_accuracy",
    ]
    for key in candidate_keys:
        raw = _decode_jsonish(row.get(key))
        if raw is None:
            continue
        if isinstance(raw, dict):
            value = raw.get(prefix_idx)
            if value is None:
                value = raw.get(str(prefix_idx))
        elif (
            hasattr(raw, "__len__")
            and hasattr(raw, "__getitem__")
            and not isinstance(raw, (str, bytes, dict))
            and 0 <= prefix_idx < len(raw)
        ):
            value = raw[prefix_idx]
        else:
            continue
        normalized = _normalize_probability(value)
        if normalized is not None:
            return normalized, key
    return None, "missing_prefix_target"


def _resolve_probe_target(row, sample_kind: str, prefix_idx: Optional[int]) -> tuple[Optional[float], str]:
    if sample_kind == "full":
        return _normalize_probability(bool(row["is_correct"])), "trajectory_final_correctness"
    if prefix_idx is None:
        return None, "missing_prefix_index"
    return _lookup_prefix_target(row, prefix_idx)


def infer_hidden_dim(model_path: Optional[str] = None, data_dir: Optional[str] = None) -> int:
    """Infer hidden size from model config or cached hidden states."""
    if model_path is not None:
        config_path = Path(model_path) / "config.json"
        if config_path.exists():
            config = json.loads(config_path.read_text())
            for key in ("hidden_size", "n_embd", "d_model"):
                value = config.get(key)
                if isinstance(value, int) and value > 0:
                    return value

    if data_dir is not None:
        if torch is None:
            raise RuntimeError("torch is required to infer hidden_dim from cached hidden states")
        cache_dir = Path(data_dir)
        for manifest_name in ("train_manifest.json", "val_manifest.json"):
            manifest_path = cache_dir / manifest_name
            if not manifest_path.exists():
                continue
            manifest = json.loads(manifest_path.read_text())
            if not manifest:
                continue
            sample_path = cache_dir / manifest[0]["hidden_state_file"]
            sample = torch.load(sample_path, map_location="cpu", weights_only=True)
            if sample.ndim != 2:
                raise RuntimeError(f"unexpected hidden state shape: {tuple(sample.shape)}")
            return int(sample.shape[-1])

    raise RuntimeError("failed to infer hidden_dim from model_path or data_dir")


def extract_hidden_states_from_sft_model(
    model_path: str,
    rollouts_path: str,
    output_dir: str,
    num_samples: int = 5000,
    max_len: int = 2048,
    device: str = "cuda",
):
    """Extract hidden states from SFT model for probe re-training.

    Extracts at two levels:
    1. Full-sequence pooled (for overall correctness prediction)
    2. At <|meta|> positions (for step-level prediction during RL)
    """
    if pd is None or torch is None or AutoTokenizer is None or AutoModelForCausalLM is None:
        raise ImportError("pandas, torch, and transformers are required for probe extraction")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    hs_dir = output_path / "hidden_states"
    hs_dir.mkdir(exist_ok=True)

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    old_vocab_size = len(tokenizer)
    ensure_meta_tokens_not_special(tokenizer, [META_START, META_END])
    num_added = max(0, len(tokenizer) - old_vocab_size)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,
    ).to(device).eval()

    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # Load rollouts
    df = pd.read_parquet(rollouts_path)
    correct = df[df["is_correct"]].sample(
        n=min(num_samples // 2, df["is_correct"].sum()), random_state=42,
    )
    incorrect = df[~df["is_correct"]].sample(
        n=min(num_samples // 2, (~df["is_correct"]).sum()), random_state=42,
    )
    selected = pd.concat([correct, incorrect]).sample(frac=1, random_state=42)
    print(f"Extracting from {len(selected)} samples ({len(correct)} correct, {len(incorrect)} incorrect)")

    manifest = []
    saved = 0
    for idx, (_, row) in enumerate(tqdm(selected.iterrows(), total=len(selected))):
        prompt_text = str(row.get("prompt_text", "") or "")
        completion_text = row["completion"]
        full_text = _combine_prompt_and_completion(prompt_text, completion_text)
        sequence_specs = [("full", None, full_text)]
        sequence_specs.extend(
            ("meta_prefix", prefix_idx, _combine_prompt_and_completion(prompt_text, prefix_text))
            for prefix_idx, prefix_text in _iter_meta_prefix_texts(completion_text)
        )

        for sample_kind, prefix_idx, sample_text in sequence_specs:
            target_prob, target_source = _resolve_probe_target(row, sample_kind, prefix_idx)
            enc = tokenizer(sample_text, return_tensors="pt", truncation=True, max_length=max_len)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                last_hidden = outputs.hidden_states[-1][0].cpu().to(torch.float16)

            del outputs, input_ids, attention_mask

            hs_file = f"hs_{saved:06d}.pt"
            torch.save(last_hidden, hs_dir / hs_file)

            manifest.append({
                "idx": saved,
                "source_row_idx": idx,
                "problem_id": row["problem_id"],
                "is_correct": bool(row["is_correct"]),
                "category": row["category"],
                "hidden_state_file": f"hidden_states/{hs_file}",
                "seq_len": last_hidden.shape[0],
                "sample_kind": sample_kind,
                "prefix_index": prefix_idx,
                "target_prob": target_prob,
                "target_source": target_source,
            })
            saved += 1

        if (idx + 1) % 250 == 0:
            torch.cuda.empty_cache()

    train_manifest, val_manifest = _group_train_val_split(manifest, val_fraction=0.1, seed=42)
    with open(output_path / "train_manifest.json", "w") as f:
        json.dump(train_manifest, f)
    with open(output_path / "val_manifest.json", "w") as f:
        json.dump(val_manifest, f)
    with open(output_path / "manifest_stats.json", "w") as f:
        json.dump(
            {
                "n_samples": len(manifest),
                "n_train": len(train_manifest),
                "n_val": len(val_manifest),
                "sample_kind_counts": {
                    "full": sum(1 for m in manifest if m["sample_kind"] == "full"),
                    "meta_prefix": sum(1 for m in manifest if m["sample_kind"] == "meta_prefix"),
                },
                "target_source_counts": {
                    key: sum(1 for m in manifest if m["target_source"] == key)
                    for key in sorted({m["target_source"] for m in manifest})
                },
            },
            f,
            indent=2,
        )

    print(f"Done! Train: {len(train_manifest)}, Val: {len(val_manifest)}, Total cached: {len(manifest)}")
    del model
    torch.cuda.empty_cache()


def retrain_probe(
    model_path: str,
    rollouts_path: str,
    output_dir: str,
    hidden_dim: Optional[int] = None,
    num_samples: int = 5000,
    epochs: int = 10,
    device: str = "cuda",
    require_prefix_targets: bool = True,
    min_prefix_targets: int = 128,
):
    """Full pipeline: extract hidden states → train probe."""
    if torch is None:
        raise ImportError("torch is required for probe retraining")
    from src.probes.simple_probe import train_simple_probe
    hs_dir = Path(output_dir) / "hidden_states_cache"

    print("Step 1: Extracting hidden states from SFT model...")
    extract_hidden_states_from_sft_model(
        model_path=model_path,
        rollouts_path=rollouts_path,
        output_dir=str(hs_dir),
        num_samples=num_samples,
        device=device,
    )

    resolved_hidden_dim = hidden_dim or infer_hidden_dim(model_path=model_path, data_dir=str(hs_dir))
    print(f"Resolved hidden_dim={resolved_hidden_dim}")

    train_manifest = json.loads((hs_dir / "train_manifest.json").read_text())
    val_manifest = json.loads((hs_dir / "val_manifest.json").read_text())
    full_manifest = train_manifest + val_manifest
    usable_prefix_targets = [
        entry for entry in full_manifest
        if entry.get("sample_kind") == "meta_prefix" and entry.get("target_prob") is not None
    ]
    if require_prefix_targets and len(usable_prefix_targets) < min_prefix_targets:
        raise RuntimeError(
            "probe retraining is gated: prefix-conditioned future-success targets are missing "
            f"or too sparse ({len(usable_prefix_targets)} < {min_prefix_targets}). "
            "Do not launch E6/E7 until multi-rollout prefix targets are prepared."
        )

    print("Step 2: Training probe...")
    import wandb
    run = wandb.init(project="metacot-math", name="probe-retrain", reinit=True)
    best_brier = train_simple_probe(
        data_dir=str(hs_dir),
        output_dir=output_dir,
        hidden_dim=resolved_hidden_dim,
        epochs=epochs,
        wandb_run=run,
    )
    wandb.finish()
    print(f"Probe re-trained. Best Brier: {best_brier:.4f}")
    return best_brier


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--rollouts-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    retrain_probe(
        model_path=args.model_path,
        rollouts_path=args.rollouts_path,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        num_samples=args.num_samples,
        epochs=args.epochs,
    )
